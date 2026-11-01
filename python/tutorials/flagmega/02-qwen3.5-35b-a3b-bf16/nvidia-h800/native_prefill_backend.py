"""Decode continuation from verified, native vLLM prefill snapshots."""

import hashlib
import json
from pathlib import Path

import torch

from artifact_backend import ArtifactBackend
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetState, GatedDeltaNetStateConfig
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionState


class NativePrefillBackend(ArtifactBackend):
    def __init__(self, artifact, reference, checkpoint, **kwargs):
        self.reference_path = Path(reference).resolve()
        report = json.loads(self.reference_path.read_text())
        if report.get("schema") != "flagmega.prepared-decode-reference/v1" or report.get("performance_valid") is not False:
            raise ValueError("Expected untimed native vLLM prefill snapshots")
        self.references = {r["prefix_length"]: r for r in report["records"]}
        self.snapshots = {}
        for length, row in self.references.items():
            path = self.reference_path.parent / row["state"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["state_sha256"]:
                raise ValueError("Native prefill snapshot hash differs")
            value = torch.load(path, map_location="cpu", weights_only=True)
            if (value["prefix_length"] != length or value["initial_token"] != row["initial_token"]
                    or set(value["layers"]) != set(range(40))):
                raise ValueError("Native prefill must cover the complete prefix and all 40 layers")
            self.snapshots[length] = value
        self.pending_native_token = None
        self.native_prefix_length = 0
        super().__init__(artifact, **kwargs)
        cfg = json.loads((Path(checkpoint) / "config.json").read_text())["text_config"]
        self.layer_types = cfg["layer_types"]
        self.gdn = GatedDeltaNetState(self.values["gdn_state.convolution"], self.values["gdn_state.recurrent"],
            GatedDeltaNetStateConfig(num_layers=self.layer_types.count("linear_attention"),
                num_key_heads=cfg["linear_num_key_heads"], num_value_heads=cfg["linear_num_value_heads"],
                key_head_dim=cfg["linear_key_head_dim"], value_head_dim=cfg["linear_value_head_dim"],
                conv_kernel_size=cfg["linear_conv_kernel_dim"], hidden_size=cfg["hidden_size"]))
        self.paged = PagedAttentionState(*(self.values[f"paged_state.{name}"] for name in
            ("kv_caches", "query_start_loc", "seq_lens", "slot_mapping", "block_table")), config=self.config)
        self.state_roundtrip_exact = {}

    def prefill(self, tokens):
        tokens = tuple(tokens)
        row = self.references.get(len(tokens))
        if row is None or tokens != tuple(row["prompt_token_ids"]):
            raise ValueError("Prompt differs from the native prefill snapshot")
        self.restore(len(tokens))

    def restore(self, length):
        self.reset()
        initial = self.snapshots[length]
        linear = attention = 0
        for index, kind in enumerate(self.layer_types):
            state = initial["layers"][index]
            if state["kind"] != kind:
                raise ValueError("Native layer kind differs from the compiled decoder")
            if kind == "linear_attention":
                self.gdn.update_convolution_layer(state["convolution"], linear)
                self.gdn.update_recurrent_layer(state["recurrent"], linear)
                linear += 1
            else:
                for field in ("key", "value"):
                    self.paged.update(state[field], cache_kind=field, layer_id=attention, advance_sequence=False)
                attention += 1
        self.paged.seq_lens.fill_(length)
        self.paged.slot_mapping.fill_(length)
        self.length = self.native_prefix_length = length
        self.pending_native_token = initial["initial_token"]
        if length not in self.state_roundtrip_exact:
            from accuracy import logical_states, state_errors
            errors = state_errors(logical_states(self.gdn, self.paged, self.layer_types, length), initial["layers"])
            if any(not field["exact"] for row in errors for field in row["fields"].values()):
                raise ValueError("Native prefill state conversion changed numeric values")
            self.state_roundtrip_exact[length] = True
        self.synchronize()

    def append(self, token):
        self.pending_native_token = None
        super().append(token)

    def sample(self):
        return self.pending_native_token if self.pending_native_token is not None else super().sample()

    def info(self):
        from triton._C import libtriton
        return {**super().info(), "prefill": "native_vllm_snapshot", "prefill_timed": False,
                "resources": self.model.resource_report,
                "compiler_lib_sha256": hashlib.sha256(Path(libtriton.__file__).read_bytes()).hexdigest(),
                "native_reference_sha256": hashlib.sha256(self.reference_path.read_bytes()).hexdigest(),
                "native_state_sha256": {str(n): r["state_sha256"] for n, r in self.references.items()},
                "native_prefill_backend_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
