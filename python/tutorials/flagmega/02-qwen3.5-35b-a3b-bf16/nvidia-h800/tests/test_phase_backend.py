"""The compiled paged ABI stores a chunk base, not a per-token slot list."""

import importlib.util
from pathlib import Path

import torch


def test_reset_preserves_scalar_slot_abi_for_batched_prefill():
    path = Path(__file__).resolve().parents[1] / "phase_backend.py"
    spec = importlib.util.spec_from_file_location("qwen35_phase_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = module.PhaseBundleBackend.__new__(module.PhaseBundleBackend)
    backend.device = torch.device("cpu")
    backend.state = {
        "gdn_state.convolution": torch.ones(3),
        "gdn_state.recurrent": torch.ones(4),
        "paged_state.kv_caches": torch.ones(5),
        "paged_state.seq_lens": torch.tensor([37], dtype=torch.int32),
        "paged_state.block_table": torch.zeros((1, 16), dtype=torch.int32),
    }
    backend.phases = {
        name: {"tokens": tokens, "values": {
            "paged_state.query_start_loc": torch.zeros(2, dtype=torch.int32),
            "paged_state.slot_mapping": torch.tensor([37], dtype=torch.int64),
        }} for name, tokens in (("decode", 1), ("prefill-32", 32))
    }
    backend.length = 37
    backend.reset()
    assert backend.length == 0
    assert backend.active is backend.phases["decode"]
    for phase in backend.phases.values():
        assert phase["values"]["paged_state.query_start_loc"].tolist() == [0, phase["tokens"]]
        assert phase["values"]["paged_state.slot_mapping"].tolist() == [0]
    for name, value in backend.state.items():
        if name != "paged_state.block_table":
            assert not value.count_nonzero()
    assert backend.state["paged_state.block_table"].tolist() == [list(range(16))]
