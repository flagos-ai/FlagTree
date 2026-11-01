"""One-H800 execution of compiled batched prefill and single-token decode."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn._paged_attention_state import paged_attention_state_config_from_type
from triton.flagmega.runtime import SharedReadonlyPool, load
from triton.flagmega.serving import SamplingConfig
from triton.flagmega.serving.sampling import sample_logits


class PhaseBundleBackend:
    def __init__(self, artifact, *, cuda_graph=True, sampling=SamplingConfig()):
        started = perf_counter()
        self.root = Path(artifact).resolve()
        bundle = json.loads((self.root / "bundle.json").read_text())
        self.imported_semantic_hashes = bundle.get("imported_semantic_hashes", {})
        if bundle["schema"] != "flagmega.qwen35-phase-bundle/v1" or "decode" not in bundle["phases"]:
            raise ValueError("Expected a compiled Qwen3.5 prefill/decode bundle")
        self.device = torch.device("cuda:0")
        self.sampling = sampling
        self.pool = SharedReadonlyPool(bundle["rdata_nbytes"], device=self.device)
        self.phases = {}
        common = {}
        shared_fields = {"gdn_state.convolution", "gdn_state.recurrent", "paged_state.kv_caches",
                         "paged_state.seq_lens", "paged_state.block_table"}
        for name, relative in bundle["phases"].items():
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Phase artifact is outside its bundle")
            model = load(path, device="cuda:0", shared_rdata=self.pool)
            if model.ir_module.metadata.get("num_hidden_layers") != 40:
                raise ValueError("The request benchmark requires all 40 layers")
            values = {}
            for argument in model.external_arguments:
                buffer = model.buffer_plan.buffer_map[argument["buffer"]]
                lanes = buffer.dtype.lanes if isinstance(buffer.dtype, fm.VectorType) else ()
                dtype = buffer.dtype.elem_type if lanes else buffer.dtype
                shape = (*buffer.shape, *lanes)
                if buffer.id in common:
                    value = common[buffer.id]
                    if tuple(value.shape) != shape or value.dtype != getattr(torch, dtype.value):
                        raise ValueError("Compiled phases disagree on their persistent state ABI")
                else:
                    value = torch.zeros(shape, dtype=getattr(torch, dtype.value), device=self.device)
                    if buffer.id in shared_fields:
                        common[buffer.id] = value
                values[buffer.id] = value
            arguments = tuple(values[a["buffer"]] for a in model.external_arguments)
            outputs = model.buffer_plan.function_map[model.ir_module.entry].outputs
            self.phases[name] = {"model": model, "values": values, "arguments": arguments,
                                 "logits": values[outputs[0][1][0]], "token": values[outputs[1][1][0]],
                                 "graph": None, "tokens": values["input_ids"].numel()}
        self.state = common
        decoder = self.phases["decode"]
        self.config = paged_attention_state_config_from_type(decoder["model"].ir_module.node_map["paged_state"].type)
        self.capacity = self.config.max_sequence_length
        self.vocab_size = decoder["model"].ir_module.metadata["vocab_size"]
        self.generator = torch.Generator(device=self.device).manual_seed(sampling.seed)
        self.cpu_token = torch.empty_like(decoder["token"], device="cpu", pin_memory=True)
        self.length = 0
        self.active = decoder
        for phase in self.phases.values():
            self.reset()
            phase["model"].prepare(*phase["arguments"])
            self._launch(phase)
            self.synchronize()
            if int(self.state["paged_state.seq_lens"].item()) != phase["tokens"]:
                raise ValueError("Compiled phase did not advance exactly its query length")
            if cuda_graph:
                self.reset()
                phase["model"].prepare(*phase["arguments"])
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._launch(phase)
                phase["graph"] = graph
        self.reset()
        self.prepare_ms = (perf_counter() - started) * 1000

    def _launch(self, phase):
        if phase["tokens"] == 1:
            phase["values"]["paged_state.slot_mapping"].copy_(self.state["paged_state.seq_lens"])
        phase["model"].run_into(*phase["arguments"])

    def _run(self, phase):
        if phase["graph"] is None:
            self._launch(phase)
        else:
            phase["graph"].replay()
        self.active = phase

    def prefill(self, tokens):
        tokens = tuple(tokens)
        if self.length or any(type(t) is not int or not 0 <= t < self.vocab_size for t in tokens):
            raise ValueError("Batched prefill requires a fresh state and valid tokens")
        phase = self.phases.get(f"prefill-{len(tokens)}")
        if phase is None:
            raise ValueError(f"Bundle has no compiled prefill for {len(tokens)} tokens")
        phase["values"]["input_ids"].copy_(torch.tensor(tokens, device=self.device, dtype=torch.int32)
                                           .reshape_as(phase["values"]["input_ids"]))
        self._run(phase)
        self.length = len(tokens)

    def append(self, token):
        if type(token) is not int or not 0 <= token < self.vocab_size or self.length >= self.capacity:
            raise ValueError("Invalid token or exhausted context capacity")
        phase = self.phases["decode"]
        phase["values"]["input_ids"].fill_(token)
        self._run(phase)
        self.length += 1

    def sample(self):
        if self.sampling.temperature != 0:
            self.active["token"].copy_(sample_logits(self.active["logits"], self.sampling, generator=self.generator))
        self.cpu_token.copy_(self.active["token"], non_blocking=True)
        self.synchronize()
        return int(self.cpu_token.item())

    def rewind(self, length):
        if length != 0 or self.length != 0:
            raise ValueError("Mixed recurrent state requires a fresh request")

    def reset(self):
        for value in self.state.values():
            value.zero_()
        table = self.state["paged_state.block_table"]
        table.copy_(torch.arange(table.numel(), device=self.device, dtype=torch.int32).reshape_as(table))
        for phase in self.phases.values():
            values = phase["values"]
            values["paged_state.query_start_loc"].copy_(torch.tensor([0, phase["tokens"]], device=self.device,
                                                                   dtype=torch.int32))
            # The paged ABI carries one logical chunk base for every query.
            values["paged_state.slot_mapping"].zero_()
        self.length = 0
        self.active = self.phases["decode"]

    def synchronize(self):
        torch.cuda.current_stream(self.device).synchronize()

    @property
    def logits(self):
        return self.active["logits"]

    def info(self):
        return {"backend": "flagmega", "prefill": "compiled_batched", "vllm": False,
                "gpu": torch.cuda.get_device_name(), "cuda_graph": self.phases["decode"]["graph"] is not None,
                "capacity": self.capacity, "num_layers": 40, "vocab_size": self.vocab_size,
                "prepare_ms": self.prepare_ms, "shared_rdata_nbytes": self.pool.nbytes,
                "numerical_contract": self.phases["decode"]["model"].ir_module.metadata.get("numerical_contract"),
                "artifact_semantic_hashes": {n: p["model"].manifest["semantic_hash"] for n, p in self.phases.items()},
                "imported_semantic_hashes": self.imported_semantic_hashes,
                "artifact_source_sha256": {n: p["model"].codegen["source_sha256"] for n, p in self.phases.items()},
                "backend_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "torch": torch.__version__}

    def close(self):
        self.synchronize()
        for phase in self.phases.values():
            phase["graph"] = None
            phase["model"].close()
        self.phases.clear()
        self.active = None
        self.state.clear()
        self.pool.tensor = None
