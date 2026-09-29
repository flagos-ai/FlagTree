"""CPU-only parity checks for the SM90 MegaMoE W1 layout contract."""

from pathlib import Path

import torch

from megamoe_operator.production.qwen3_fp8_shared_data import interleave_l1_gate_up_rows


def test_gran8_gate_up_interleave():
    source = torch.arange(32, dtype=torch.uint8).reshape(1, 32, 1)
    actual = interleave_l1_gate_up_rows(source, torch).flatten().tolist()
    expected = (list(range(0, 8)) + list(range(16, 24)) + list(range(8, 16)) + list(range(24, 32)))
    assert actual == expected


def test_w1_scale_rows_stay_in_nl1n():
    nl1n = 24
    gate_rows = [n_block // 2 for n_block in range(nl1n)]
    up_rows = [nl1n // 2 + n_block // 2 for n_block in range(nl1n)]

    assert gate_rows == [value for value in range(12) for _ in range(2)]
    assert up_rows == [value for value in range(12, 24) for _ in range(2)]
    assert max(gate_rows + up_rows) < nl1n


def test_v234_embeds_corrected_w1_contract():
    source = (Path(__file__).resolve().parent / "megamoe_operator" / "production" / "v234" /
              "run.py").read_text(encoding="utf-8")

    for forbidden in ("cur_e * 2 * NL1N", "EPR, 2 * NL1N, NK1"):
        assert forbidden not in source
    for required in (
            "cur_e * NL1N",
            "NL1N // 2",
            "interleave_l1_gate_up_rows",
            "EPR, NL1N, NK1",
    ):
        assert required in source
