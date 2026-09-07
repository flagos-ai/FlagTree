# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Single-sequence paged-attention artifact execution and CUDA graph ownership.

Prefill is a causal scan using the compiled one-token entry, not a separate
framework forward or a claimed batched-prefill implementation.
"""

import hashlib
from contextlib import nullcontext
from pathlib import Path
import time

from .sampling import SamplingConfig, sample_logits


class ArtifactBackend:
    def __init__(self, model, *, sampling=SamplingConfig(), cuda_graph=True, warmups=3):
        import torch
        self.model = model
        self.device = torch.device(model.device)
        self.sampling = sampling
        self.graph = None
        self.length = 0
        self.closed = False
        self._has_logits = False
        self.load_ms = 0.0
        if model.result_kind not in {"tensor_state", "logits_token_state"}:
            raise ValueError("Text serving requires a token-input, logits-output paged-attention artifact")
        from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig
        if not isinstance(getattr(model, "state_config", None), PagedAttentionStateConfig):
            raise ValueError("Text serving currently requires the paged-attention state ABI")
        self.capacity = model.state_config.max_sequence_length
        self.state = model.create_state()
        self.input_ids = torch.zeros((1,), dtype=torch.int32, device=self.device)
        outputs = model.create_outputs()
        self.logits, self.token = outputs if model.result_kind == "logits_token_state" else (outputs, None)
        vocab = model.ir_module.metadata.get("vocab_size")
        if not isinstance(vocab, int) or tuple(self.logits.shape) != (1, vocab):
            raise ValueError("Artifact must return full vocabulary logits, not layer hidden states")
        self.vocab_size = vocab
        self.generator = torch.Generator(device=self.device).manual_seed(sampling.seed)
        self._builtin_token = self.token is not None
        if self.token is None:
            self.token = torch.empty((1,), dtype=torch.int32, device=self.device)
        self._cpu_token = torch.empty((1,), dtype=torch.int32, device="cpu",
                                      pin_memory=self.device.type == "cuda")
        if cuda_graph and self.device.type != "cuda":
            raise ValueError("CUDA graph execution requires a CUDA device; select eager explicitly")
        if warmups < 0:
            raise ValueError("warmups must be nonnegative")
        started = time.perf_counter()
        self._prepare()
        self._launch()
        self.synchronize()
        if self.state.sequence_length != 1:
            raise ValueError("Artifact must advance the paged-attention state once per input token")
        self.reset()
        if cuda_graph:
            # Prepared compiler scratch is stream-owned; the capture needs its
            # own binding, separate from the eager compilation launch.
            self._prepare()
            self.graph = torch.cuda.CUDAGraph()
            with self._device_context(), torch.cuda.graph(self.graph):
                self._launch()
        for _ in range(warmups):
            self.state.seq_lens.zero_()
            self._replay()
        self.reset()
        self.synchronize()
        self.prepare_ms = (time.perf_counter() - started) * 1000

    @classmethod
    def load(cls, artifact, *, device="cuda:0", **kwargs):
        from triton.flagmega.runtime import load
        started = time.perf_counter()
        model = load(artifact, device=device)
        elapsed = (time.perf_counter() - started) * 1000
        try:
            result = cls(model, **kwargs)
        except BaseException:
            model.close()
            raise
        result.load_ms = elapsed
        return result

    def _prepare(self):
        with self._device_context():
            self._prepare_on_device()

    def _device_context(self):
        import torch
        return torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext()

    def _prepare_on_device(self):
        if self._builtin_token:
            self.model.prepare(self.input_ids, self.state, logits=self.logits, next_token=self.token)
        else:
            self.model.prepare(self.input_ids, self.state, output=self.logits)

    def _launch(self):
        with self._device_context():
            self._launch_on_device()

    def _launch_on_device(self):
        if self._builtin_token:
            self.model.run_into(self.logits, self.token, self.input_ids, self.state)
        else:
            self.model.run_into(self.logits, self.input_ids, self.state)
            if self.sampling.temperature == 0:
                self.token.copy_(self.logits.argmax(dim=-1))

    def _replay(self):
        with self._device_context():
            if self.graph is None:
                self._launch()
            else:
                self.graph.replay()

    def append(self, token):
        self._require_open()
        if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < self.vocab_size:
            raise ValueError("Token ID is outside the artifact vocabulary")
        if self.length >= self.capacity:
            raise ValueError("Context capacity exhausted; reset the conversation")
        self.input_ids.fill_(token)
        self._replay()
        self.length += 1
        self._has_logits = True

    def sample(self):
        self._require_open()
        if not self._has_logits:
            raise ValueError("Sampling requires logits from the current prefix")
        if self.sampling.temperature != 0:
            self.token.copy_(sample_logits(self.logits, self.sampling, generator=self.generator))
        self._cpu_token.copy_(self.token, non_blocking=self.device.type == "cuda")
        self.synchronize()
        return int(self._cpu_token.item())

    def rewind(self, length):
        self._require_open()
        if not 0 <= length <= self.length:
            raise ValueError("Cannot rewind beyond the cached prefix")
        # Past cache bytes remain unreachable until overwritten. This operation
        # is specific to the paged-attention ABI, not recurrent state rewind.
        self.state.seq_lens.fill_(length)
        self.length = length
        self._has_logits = False

    def reset(self):
        self._require_open()
        self.state.kv_caches.zero_()
        self.state.seq_lens.zero_()
        self.state.slot_mapping.zero_()
        self.length = 0
        self._has_logits = False

    def synchronize(self):
        if self.device.type == "cuda":
            import torch
            torch.cuda.current_stream(self.device).synchronize()

    def info(self):
        import torch
        self._require_open()
        artifact = Path(self.model.artifact)
        serving_source = hashlib.sha256()
        for source in sorted(Path(__file__).parent.glob("*.py")):
            serving_source.update(source.name.encode())
            serving_source.update(source.read_bytes())
        return {"backend": "flagmega", "prefill": "compiled_token_scan", "vllm": False,
                "device": str(self.device),
                "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
                "cuda_graph": self.graph is not None, "capacity": self.capacity,
                "num_layers": self.model.state_config.num_layers, "vocab_size": self.vocab_size,
                "load_ms": self.load_ms, "prepare_ms": self.prepare_ms,
                "allocated_bytes": torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else None,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
                "artifact_source_sha256": hashlib.sha256((artifact / "generated_kernels.py").read_bytes()).hexdigest(),
                "artifact_semantic_hash": self.model.manifest["semantic_hash"],
                "serving_source_sha256": serving_source.hexdigest(),
                "numerical_contract": self.model.ir_module.metadata.get("numerical_contract"),
                "resource": self.model.resource_report, "torch": torch.__version__}

    def _require_open(self):
        if self.closed:
            raise ValueError("Generation backend is closed")

    def close(self):
        if not self.closed:
            self.synchronize()
            self.graph = None
            self.model.close()
            self.state = self.input_ids = self.logits = self.token = self._cpu_token = None
            self.generator = None
            self.closed = True
