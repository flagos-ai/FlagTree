"""Tutorial-owned, batch-one execution of the mixed GDN/attention state ABI."""

import hashlib
from pathlib import Path
from time import perf_counter

import torch

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn._paged_attention_state import paged_attention_state_config_from_type
from triton.flagmega.runtime import load
from triton.flagmega.serving import SamplingConfig
from triton.flagmega.serving.sampling import sample_logits


class ArtifactBackend:
    def __init__(self, artifact, *, cuda_graph=True, sampling=SamplingConfig()):
        from agent_optimizations import shared_router_collective
        started = perf_counter()
        self.model = load(artifact, device="cuda:0")
        self.device = torch.device("cuda:0")
        self.sampling = sampling
        self.values = {}
        for argument in self.model.external_arguments:
            buffer = self.model.buffer_plan.buffer_map[argument["buffer"]]
            lanes = buffer.dtype.lanes if isinstance(buffer.dtype, fm.VectorType) else ()
            dtype = buffer.dtype.elem_type if lanes else buffer.dtype
            self.values[buffer.id] = torch.zeros((*buffer.shape, *lanes),
                                                dtype=getattr(torch, dtype.value), device=self.device)
        self.arguments = tuple(self.values[a["buffer"]] for a in self.model.external_arguments)
        outputs = self.model.buffer_plan.function_map[self.model.ir_module.entry].outputs
        self.logits = self.values[outputs[0][1][0]]
        self.token = self.values[outputs[1][1][0]]
        self.input_ids = self.values["input_ids"]
        self.config = paged_attention_state_config_from_type(self.model.ir_module.node_map["paged_state"].type)
        self.capacity = self.config.max_sequence_length
        self.vocab_size = self.model.ir_module.metadata["vocab_size"]
        if self.model.ir_module.metadata.get("num_hidden_layers") != 40:
            raise ValueError("The request benchmark requires all 40 decoder layers")
        self.generator = torch.Generator(device=self.device).manual_seed(sampling.seed)
        self.cpu_token = torch.empty_like(self.token, device="cpu", pin_memory=True)
        self.length = 0
        self.graph = None
        self.reset()
        self.model.prepare(*self.arguments)
        self._launch()
        self.synchronize()
        if self.values["paged_state.seq_lens"].item() != 1:
            raise ValueError("Artifact must advance its state exactly once per token")
        self.reset()
        if cuda_graph:
            self.model.prepare(*self.arguments)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._launch()
            self.reset()
        self.prepare_ms = (perf_counter() - started) * 1000

    def _launch(self):
        self.values["paged_state.slot_mapping"].copy_(self.values["paged_state.seq_lens"])
        self.model.run_into(*self.arguments)

    def append(self, token):
        if type(token) is not int or not 0 <= token < self.vocab_size or self.length >= self.capacity:
            raise ValueError("Invalid token or exhausted context capacity")
        self.input_ids.fill_(token)
        if self.graph is None:
            self._launch()
        else:
            self.graph.replay()
        self.length += 1

    def sample(self):
        if self.sampling.temperature != 0:
            self.token.copy_(sample_logits(self.logits, self.sampling, generator=self.generator))
        self.cpu_token.copy_(self.token, non_blocking=True)
        self.synchronize()
        return int(self.cpu_token.item())

    def rewind(self, length):
        # Recurrent state cannot be rewound by changing the attention length.
        # This benchmark starts each request from zero and never reuses prefixes.
        if length != 0 or self.length != 0:
            raise ValueError("Mixed recurrent state requires a fresh request; prefix reuse is unsupported")

    def reset(self):
        for name, value in self.values.items():
            if name.startswith(("gdn_state.", "paged_state.")):
                value.zero_()
        table = self.values["paged_state.block_table"]
        table.copy_(torch.arange(table.numel(), dtype=torch.int32, device=self.device).reshape_as(table))
        self.values["paged_state.query_start_loc"].copy_(torch.tensor([0, 1], dtype=torch.int32, device=self.device))
        self.length = 0

    def synchronize(self):
        torch.cuda.current_stream(self.device).synchronize()

    def info(self):
        artifact = Path(self.model.artifact)
        return {"backend": "flagmega", "prefill": "compiled_token_scan", "vllm": False,
                "gpu": torch.cuda.get_device_name(), "cuda_graph": self.graph is not None,
                "capacity": self.capacity, "num_layers": 40, "vocab_size": self.vocab_size,
                "prepare_ms": self.prepare_ms,
                "artifact_source_sha256": hashlib.sha256((artifact / self.model.manifest["codegen"]["source"]).read_bytes()).hexdigest(),
                "artifact_semantic_hash": self.model.manifest["semantic_hash"],
                "backend_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "numerical_contract": self.model.ir_module.metadata.get("numerical_contract"),
                "torch": torch.__version__}

    def close(self):
        self.synchronize()
        self.graph = None
        self.model.close()
        self.values.clear()
        self.arguments = ()
