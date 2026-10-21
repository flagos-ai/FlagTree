"""FlagOS vLLM worker extension for native prefill and FlagMega decode.

Native attention and the compiled entry share strided views of one packed KV
allocation. This measured ABI supports BF16, one request, eager vLLM execution,
and TRITON_ATTN. Sampling, scheduling and page allocation stay with vLLM.

vLLM stays eager because its own capture cannot drive this path (see
`_record`); decode instead replays a graph this extension records and owns, so
the device work is prerecorded while the Python handshake still runs per step.
"""
from pathlib import Path

import torch

from triton.flagmega.runtime import load
from cache_bridge import semantic_layer_views
from decode_inputs import FusedDecodeInputs


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
        self.prepared_decode = False
        self.graph = None
        self.graph_key = None
        self.graph_replays = 0
        self.graph_recordings = 0
        self.decode_calls = 0
        self.prefill_calls = 0
        self.codegen_identity = artifact_codegen_identity(artifact, self.model.manifest)
        self.source_hash = self.codegen_identity['source_sha256']

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
        # Replaying requires every address the recording touches to be the one
        # it was recorded against: vLLM's three source tensors plus the page
        # table width. A changed key re-records rather than replaying stale
        # addresses.
        key = (tokens.data_ptr(), metadata.seq_lens.data_ptr(),
               metadata.block_table.data_ptr(), width)
        if self.graph is None or self.graph_key != key:
            # Recording is one-way: `_record` rebinds the compiler global
            # scratch to its capture stream, after which an eager `run_into`
            # on vLLM's default stream raises RuntimeContractError. There is
            # no eager fallback once this has run.
            self._record(tokens, metadata, width, key)
        self.graph.replay()
        self.graph_replays += 1
        self.decode_calls += 1
        self.active = True
        return self.placeholder

    def _stage(self, tokens, metadata, width):
        self.tokens.copy_(tokens)
        torch.sub(metadata.seq_lens, 1, out=self.state.seq_lens)
        self.state.block_table.zero_()
        self.state.block_table[:, :width].copy_(metadata.block_table)

    def _record(self, tokens, metadata, width, key):
        # Staging is four single-element kernels whose launch cost (0.46 ms) is
        # comparable to a quarter of the megakernel itself at this scale, so
        # the recording covers staging and the launch together.
        #
        # This graph is owned here rather than taken from vLLM's capture
        # because vLLM returns a weak ref to the capture-time output on every
        # replay and never re-enters Python, which strands both the
        # `placeholder`/`active` handshake and the native logits head. Driving
        # the replay from inside `decode` keeps that handshake intact: Python
        # still runs per step, only the device work is prerecorded.
        #
        # Replay is faster than eager launching at every measured context, and
        # at prompt 2048 it is faster in both of the two latency modes that
        # workload exhibits (-0.8 ms/request in the fast mode, -5.8 in the
        # slow one). The modes themselves are a device-side bistability
        # present without this recording; both paths reach the same ~1678 ms
        # floor, but eager happens to enter it more often at 2048, so the
        # aggregate median there favours eager. See ITERATION.md section 98.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Compiler global scratch holds the TLE grid-barrier counters and
            # binds to its first launch stream, so it must be prepared on the
            # stream that will own the recording. `prepare` also allocates and
            # synchronizes, neither of which is legal while capturing.
            self.model.prepare(self.tokens, self.state, output=self.logits)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self._stage(tokens, metadata, width)
            self.model.run_into(self.logits, self.tokens, self.state)
        self.graph, self.graph_key = graph, key
        self.graph_recordings += 1
        torch.cuda.synchronize()

    def stats(self):
        return {'decode_calls': self.decode_calls, 'native_prefill_calls': self.prefill_calls,
                'graph_replays': self.graph_replays,
                'graph_recordings': self.graph_recordings,
                'fused_decode_input_calls': getattr(getattr(self, 'decode_inputs', None), 'calls', 0),
                'codegen': self.codegen_identity,
                'source_sha256': self.source_hash, 'semantic_hash': self.model.manifest['semantic_hash'],
                'resource': self.model.resource_report, 'kv_copy_bytes_per_decode': 0}


def artifact_codegen_identity(artifact, manifest):
    """Identify the source and library actually selected by the loader."""
    from triton.flagmega.artifacts.manifest import hash_file, resolve_artifact_path
    root = Path(artifact)
    codegen = manifest['codegen']
    identity = {'backend': manifest['codegen_backend'], 'source': codegen['source'],
                'source_sha256': hash_file(resolve_artifact_path(root, codegen['source']))}
    if codegen.get('library'):
        identity.update(library=codegen['library'],
                        library_sha256=hash_file(resolve_artifact_path(root, codegen['library'])))
    return identity


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
            if (metadata and tokens is not None and tokens.numel() == 1
                    and executor.prepared_decode):
                first = next(iter(metadata.values())) if isinstance(metadata, dict) else metadata
                return executor.decode(tokens, first)
            executor.prefill_calls += 1
            return native_forward(*args, **kwargs)

        def logits(hidden, *args, **kwargs):
            return executor.logits if executor.active else native_logits(hidden, *args, **kwargs)

        runner.model.forward = forward
        runner.model.compute_logits = logits
        executor.decode_inputs = FusedDecodeInputs(runner)
        native_prepare_inputs = runner._prepare_inputs

        def prepare_inputs(scheduler_output, num_scheduled_tokens):
            executor.prepared_decode = executor.decode_inputs.supports(scheduler_output)
            if executor.prepared_decode:
                return executor.decode_inputs.prepare(scheduler_output, num_scheduled_tokens)
            return native_prepare_inputs(scheduler_output, num_scheduled_tokens)

        runner._prepare_inputs = prepare_inputs
        return executor.stats()

    def flagmega_stats(self):
        return self.flagmega_executor.stats()
