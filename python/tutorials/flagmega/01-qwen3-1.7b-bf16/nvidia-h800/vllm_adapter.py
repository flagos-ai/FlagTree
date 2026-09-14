"""Out-of-tree worker for the pinned vLLM nncase branch's executor protocol.

No vLLM files are patched. The existing slot is named ``nncase_executor``;
this implementation loads only a FlagMega artifact, never a PyNTT package.
Native vLLM owns prefill, scheduler, page allocation and sampling.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch

from triton.flagmega.runtime import load
from vllm.v1.worker.gpu_worker import Worker
from cache_bridge import preserve_cache_slot, semantic_layer_views
from accuracy import require_same_tokens


class FlagMegaDecodeExecutor:
    def __init__(self, artifact, vllm_config, device):
        self.device = torch.device(device)
        self.model = load(artifact, device=str(self.device))
        self.config = self.model.state_config
        self.state = self.model.create_state()
        self.input_ids = torch.zeros((1,), dtype=torch.int32, device=device)
        outputs = self.model.create_outputs()
        self.logits, self.token = outputs if isinstance(outputs, tuple) else (outputs, None)
        self.graph = None
        self.decode_calls = 0
        self.validation = []
        self.reference_model = None
        cfg = vllm_config
        hf = cfg.model_config.hf_config
        if (cfg.scheduler_config.max_num_seqs != 1 or cfg.parallel_config.tensor_parallel_size != 1
                or cfg.parallel_config.pipeline_parallel_size != 1 or cfg.speculative_config is not None
                or cfg.lora_config is not None or cfg.model_config.dtype != torch.bfloat16):
            raise ValueError("This measured profile requires BF16, batch/concurrency=1, TP=PP=1, no LoRA/speculation")
        if (hf.num_hidden_layers != self.config.num_layers or hf.num_key_value_heads != self.config.num_kv_heads
                or hf.head_dim != self.config.head_dim or hf.vocab_size != self.logits.shape[-1]):
            raise ValueError("FlagMega artifact and native prefill model have different geometry")
        if cfg.model_config.max_model_len > (self.config.num_blocks - 1) * self.config.block_size:
            raise ValueError("Reserve one vLLM null page; requested context exceeds executable capacity")
        if cfg.cache_config.enable_prefix_caching:
            raise ValueError("This benchmark requires prefix caching disabled for both engines")

    @staticmethod
    def is_decode_step(input_batch, max_query_len, num_scheduled_tokens):
        if input_batch.num_reqs == 0 or max_query_len != 1:
            return False
        scheduled = np.asarray(num_scheduled_tokens[:input_batch.num_reqs])
        return bool(np.all(scheduled == 1) and np.all(
            input_batch.num_computed_tokens_cpu[:input_batch.num_reqs]
            >= input_batch.num_prompt_tokens[:input_batch.num_reqs]))

    def allocate_kv_cache_tensors(self, kv_cache_config):
        from vllm.model_executor.models.utils import extract_layer_index
        from vllm.v1.kv_cache_interface import AttentionSpec

        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1 or not isinstance(groups[0].kv_cache_spec, AttentionSpec):
            raise ValueError("Expected a single homogeneous attention cache group")
        spec = groups[0].kv_cache_spec
        cfg = self.config
        if (kv_cache_config.num_blocks != cfg.num_blocks or spec.block_size != cfg.block_size
                or spec.num_kv_heads != cfg.num_kv_heads or spec.head_size != cfg.head_dim
                or spec.dtype != torch.bfloat16):
            raise ValueError("vLLM KV allocation differs from final IR ABI")
        names = sorted(groups[0].layer_names, key=extract_layer_index)
        if [extract_layer_index(name) for name in names] != list(range(cfg.num_layers)):
            raise ValueError("Cache layer names do not match model layers")
        expected_bytes = cfg.num_blocks * 2 * cfg.block_size * cfg.num_kv_heads * cfg.head_dim * 2
        allocations = kv_cache_config.kv_cache_tensors
        if (len(allocations) != len(names) or any(
                len(value.shared_by) != 1 or value.size != expected_bytes for value in allocations)
                or {value.shared_by[0] for value in allocations} != set(names)):
            raise ValueError("Unexpected shared cache allocation or byte extent")
        self.views = semantic_layer_views(self.state.kv_caches, cfg)
        return dict(zip(names, self.views, strict=True))

    def _prepare(self):
        if self.token is None:
            self.model.prepare(self.input_ids, self.state, output=self.logits)
        else:
            self.model.prepare(self.input_ids, self.state, logits=self.logits, next_token=self.token)

    def _launch(self):
        if self.token is None:
            self.model.run_into(self.logits, self.input_ids, self.state)
        else:
            self.model.run_into(self.logits, self.token, self.input_ids, self.state)

    def bind_kv_cache(self, kv_caches, kv_cache_config):
        if len(kv_caches) != self.config.num_layers:
            raise ValueError("Cache layer count mismatch")
        for actual, expected in zip(kv_caches, self.views, strict=True):
            if (actual.data_ptr() != expected.data_ptr() or actual.shape != expected.shape
                    or actual.stride() != expected.stride()):
                raise ValueError("Native attention is not bound to the expected zero-copy view")
        self._prepare()
        self._launch()
        torch.cuda.synchronize(self.device)
        self.state.seq_lens.zero_()
        # Compiler scratch is stream-owned. A fresh prepared instance binds it
        # to the graph capture stream, not the earlier eager warmup stream.
        self._prepare()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._launch()
        self.state.kv_caches.zero_()
        self.state.seq_lens.zero_()

    def execute_decode(self, input_ids, attn_metadata):
        if not isinstance(attn_metadata, dict) or not attn_metadata:
            raise ValueError("Expected native FlashAttention metadata")
        metadata = next(iter(attn_metadata.values()))
        if any(value is not metadata for value in attn_metadata.values()):
            raise ValueError("Every layer must use the same attention metadata")
        if (metadata.num_actual_tokens != 1 or metadata.max_query_len != 1
                or tuple(metadata.seq_lens.shape) != (1,) or tuple(input_ids.shape) != (1,)
                or input_ids.dtype != torch.int32 or input_ids.device != self.device):
            raise ValueError("Only one-token, one-sequence decode is supported")
        width = metadata.block_table.shape[1]
        if metadata.block_table.shape[0] != 1 or width > self.config.num_blocks:
            raise ValueError("Native block table does not fit the executable ABI")
        self.input_ids.copy_(input_ids)
        # Native seq_lens includes the current query; FlagMega takes past length
        # and advances it. Never overwrite the scheduler's own metadata.
        torch.sub(metadata.seq_lens, 1, out=self.state.seq_lens)
        self.state.block_table[:, :width].copy_(metadata.block_table)
        reference = None
        native_slots = None
        if self.reference_model is not None:
            # Validation only, never part of a reported performance run.
            position = int(metadata.seq_lens.item()) - 1
            physical = int(metadata.block_table[0, position // self.config.block_size].item())
            offset = position % self.config.block_size
            # Restore every layer's current K/V slot after the reference. A
            # missing/misaddressed candidate cache write must remain observable.
            with preserve_cache_slot(self.state.kv_caches[physical, :, :, offset]):
                hidden = self.reference_model(input_ids=input_ids, positions=self.state.seq_lens.to(torch.int64),
                                              intermediate_tensors=None, inputs_embeds=None)
                reference = self.reference_model.compute_logits(hidden).float().clone()
                native_slots = self.state.kv_caches[physical, :, :, offset].clone()
        # FlagMega's slot_mapping is a logical token position, not vLLM's
        # physical slot. Its QKV kernel computes it and uses the page table.
        if self.graph is None:
            raise RuntimeError("KV cache binding/graph preparation did not run")
        self.graph.replay()
        self.decode_calls += 1
        if reference is not None:
            cosine = torch.nn.functional.cosine_similarity(self.logits.flatten(), reference.flatten(), dim=0).item()
            rmse = torch.mean((self.logits - reference).square()).sqrt().item()
            native_token = reference.argmax().item()
            token = self.logits.argmax().item()
            actual_slots = self.state.kv_caches[physical, :, :, offset]
            slot_cosines = torch.nn.functional.cosine_similarity(
                actual_slots.flatten(2).float(), native_slots.flatten(2).float(), dim=2).cpu().tolist()
            self.validation.append({"cosine": cosine, "rmse": rmse,
                                    "native_token": native_token, "flagmega_token": token,
                                    "length": int(metadata.seq_lens.item()),
                                    "kv_cosine_by_layer": slot_cosines,
                                    "page_table": metadata.block_table.cpu().tolist()})
            if not torch.isfinite(self.logits).all() or token != native_token:
                failure_path = os.environ.get("FLAGMEGA_TUTORIAL_FAILURE_REPORT")
                if failure_path:
                    Path(failure_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(failure_path).write_text(json.dumps(self.validation, indent=2) + "\n")
                    torch.save({"input_ids": input_ids.cpu(), "logits": self.logits.cpu(),
                                "native_logits": reference.cpu(), "native_slots": native_slots.cpu(),
                                "kv_caches": self.state.kv_caches.cpu(),
                                "block_table": self.state.block_table.cpu(),
                                "past_length": int(metadata.seq_lens.item()) - 1},
                               Path(failure_path).with_suffix(".pt"))
                if not torch.isfinite(self.logits).all():
                    raise AssertionError("Nonfinite FlagMega logits")
                require_same_tokens([token], [native_token], context=f"same-history decode length {metadata.seq_lens.item()}")
        return self.logits


class TutorialWorker(Worker):
    def load_model(self):
        super().load_model()
        artifact = os.environ.get("FLAGMEGA_TUTORIAL_ARTIFACT")
        if not artifact:
            return
        runner = self.model_runner
        if not hasattr(runner, "nncase_executor") or runner.nncase_executor is not None:
            raise RuntimeError("This adapter requires the documented vLLM decode-executor hook, unused")
        if os.environ.get("VLLM_ATTENTION_BACKEND") != "FLASH_ATTN":
            raise ValueError("This profile uses native FlashAttention prefill")
        runner.nncase_executor = FlagMegaDecodeExecutor(artifact, self.vllm_config, self.device)
        if os.environ.get("FLAGMEGA_TUTORIAL_VALIDATE_NATIVE") == "1":
            runner.nncase_executor.reference_model = runner.model

    def tutorial_stats(self):
        executor = self.model_runner.nncase_executor
        if executor is None:
            return {"decode_executor": "native_vllm"}
        return {"decode_executor": "flagmega", "decode_calls": executor.decode_calls,
                "validation": executor.validation,
                "resources": executor.model.resource_report}
