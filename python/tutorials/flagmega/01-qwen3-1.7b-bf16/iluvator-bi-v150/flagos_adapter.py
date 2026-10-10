"""FlagOS vLLM worker extension for native prefill and FlagMega decode.

Native attention and the compiled entry share strided views of one packed KV
allocation. This measured ABI supports BF16, one request, eager execution, and
TRITON_ATTN. Sampling, scheduling and page allocation stay with vLLM.
"""
from pathlib import Path
import hashlib

import torch

from triton.flagmega.runtime import load
from cache_bridge import semantic_layer_views


class FlagMegaExecutor:
    def __init__(self, artifact, config, device):
        if (not config.model_config.enforce_eager or config.scheduler_config.max_num_seqs != 1
                or config.parallel_config.tensor_parallel_size != 1
                or config.parallel_config.pipeline_parallel_size != 1
                or config.speculative_config is not None or config.lora_config is not None
                or config.model_config.dtype != torch.bfloat16
                or config.cache_config.enable_prefix_caching
                or config.kv_transfer_config is not None):
            raise ValueError('Requires eager BF16, batch=1, TP=PP=1, no prefix cache/LoRA/speculation/KV transfer')
        self.model = load(artifact, device=str(device))
        if self.model.result_kind != 'tensor_state':
            raise ValueError('FlagOS integration requires a logits-only artifact')
        self.config = self.model.state_config
        hf = config.model_config.hf_config
        if (hf.num_hidden_layers, hf.num_key_value_heads, hf.head_dim) != (
                self.config.num_layers, self.config.num_kv_heads, self.config.head_dim):
            raise ValueError('Native model and compiled KV geometries differ')
        self.state = self.model.create_state()
        self.tokens = torch.zeros((1,), dtype=torch.int32, device=device)
        self.logits = self.model.create_outputs()
        self.placeholder = torch.empty((1, hf.hidden_size), dtype=torch.bfloat16, device=device)
        self.active = False
        self.decode_calls = 0
        self.prefill_calls = 0
        self.source_hash = hashlib.sha256((Path(artifact)/'generated_kernels.py').read_bytes()).hexdigest()

    def bind(self, runner):
        from vllm.model_executor.models.utils import extract_layer_index
        from vllm.v1.worker.utils import bind_kv_cache
        cfg, cache = self.config, runner.kv_cache_config
        if len(cache.kv_cache_groups) != 1 or runner.shared_kv_cache_layers:
            raise ValueError('Expected one unshared attention cache group')
        group = cache.kv_cache_groups[0]
        spec = group.kv_cache_spec
        if (cache.num_blocks != cfg.num_blocks or tuple(runner._kernel_block_sizes) != (cfg.block_size,)
                or spec.block_size != cfg.block_size or spec.num_kv_heads != cfg.num_kv_heads
                or spec.head_size != cfg.head_dim or spec.dtype != torch.bfloat16):
            raise ValueError('Native KV allocation differs from compiled ABI')
        names = sorted(group.layer_names, key=extract_layer_index)
        if [extract_layer_index(x) for x in names] != list(range(cfg.num_layers)):
            raise ValueError('Native KV layers differ from compiled ABI')
        for name in names:
            layer = runner.compilation_config.static_forward_context[name]
            if layer.attn_backend.get_name() != 'TRITON_ATTN':
                raise ValueError('This cache bridge requires TRITON_ATTN runtime-stride addressing')
        # The native backend consumes runtime strides. Its page stride is
        # num_layers times the per-layer contiguous stride, with no data copy.
        self.views = tuple(view.permute(1, 0, 2, 3, 4)
                           for view in semantic_layer_views(self.state.kv_caches, cfg))
        self.model.prepare(self.tokens, self.state, output=self.logits)
        self.model.run_into(self.logits, self.tokens, self.state)
        torch.cuda.synchronize()
        self.state.kv_caches.zero_()
        self.state.seq_lens.zero_()
        self.state.slot_mapping.zero_()
        runner.kv_caches.clear()
        bind_kv_cache(dict(zip(names, self.views, strict=True)),
                      runner.compilation_config.static_forward_context, runner.kv_caches)
        for name, view in zip(names, self.views, strict=True):
            actual = runner.compilation_config.static_forward_context[name].kv_cache
            if (actual.data_ptr(), actual.shape, actual.stride()) != (view.data_ptr(), view.shape, view.stride()):
                raise ValueError('Native attention did not retain the strided packed KV view')

    def decode(self, tokens, metadata):
        if (tokens.shape != (1,) or metadata.seq_lens.shape != (1,)
                or metadata.max_query_len != 1 or metadata.num_actual_tokens != 1):
            raise ValueError('Expected one query and one sequence')
        width = metadata.block_table.shape[1]
        if metadata.block_table.shape[0] != 1 or width > self.config.num_blocks:
            raise ValueError('Native page table exceeds compiled ABI')
        self.tokens.copy_(tokens)
        torch.sub(metadata.seq_lens, 1, out=self.state.seq_lens)
        self.state.block_table.zero_()
        self.state.block_table[:, :width].copy_(metadata.block_table)
        self.model.run_into(self.logits, self.tokens, self.state)
        self.decode_calls += 1
        self.active = True
        return self.placeholder

    def stats(self):
        return {'decode_calls': self.decode_calls, 'native_prefill_calls': self.prefill_calls,
                'source_sha256': self.source_hash, 'semantic_hash': self.model.manifest['semantic_hash'],
                'resource': self.model.resource_report, 'kv_copy_bytes_per_decode': 0}


class FlagMegaExtension:
    def flagmega_install(self, artifact):
        if hasattr(self, 'flagmega_executor'):
            raise ValueError('Install the compiled executor exactly once before requests')
        executor = FlagMegaExecutor(artifact, self.vllm_config, self.device)
        runner = self.model_runner
        executor.bind(runner)
        self.flagmega_executor = executor
        native_forward = runner.model.forward
        native_logits = runner.model.compute_logits

        def forward(*args, **kwargs):
            from vllm.forward_context import get_forward_context
            metadata = get_forward_context().attn_metadata
            tokens = kwargs.get('input_ids', args[0] if args else None)
            executor.active = False
            if metadata and tokens is not None and tokens.numel() == 1:
                first = next(iter(metadata.values())) if isinstance(metadata, dict) else metadata
                return executor.decode(tokens, first)
            executor.prefill_calls += 1
            return native_forward(*args, **kwargs)

        def logits(hidden, *args, **kwargs):
            return executor.logits if executor.active else native_logits(hidden, *args, **kwargs)

        runner.model.forward = forward
        runner.model.compute_logits = logits
        return executor.stats()

    def flagmega_stats(self):
        return self.flagmega_executor.stats()
