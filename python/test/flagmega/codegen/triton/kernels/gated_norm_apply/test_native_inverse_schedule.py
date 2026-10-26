# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Keep the selected sqrt/div inverse at a BF16 rounding-sensitive boundary."""

from dataclasses import replace

import pytest
import torch
import triton
import triton.language as tl
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError, IRVerificationError
from triton.flagmega.runtime import load
from triton.flagmega.targets import NvidiaSm90Target
from python.test.flagmega.devices import is_nvidia_sm90


@triton.jit
def _independent_reference(X, Gate, Weight, Stats, Out, Approx):
    row = tl.arange(0, 16)
    col = tl.arange(0, 128)
    offsets = row[:, None] * 128 + col[None, :]
    x = tl.load(X + offsets).to(tl.float32)
    gate = tl.load(Gate + offsets).to(tl.float32)
    weight = tl.load(Weight + col).to(tl.float32)
    square = tl.sum(x * x, 1)
    variance = square / 128 + 1e-5
    inverse = 1.0 / tl.sqrt(variance)
    approximate = tl.rsqrt(variance)
    out = ((x * inverse[:, None]) * weight[None, :]) * tl.sigmoid(gate)
    approx = ((x * approximate[:, None]) * weight[None, :]) * tl.sigmoid(gate)
    tl.store(Stats + row, square)
    tl.store(Out + offsets, out)
    tl.store(Approx + offsets, approx)


def _module(dtype):
    class Graph(fm.Module):
        def forward(self):
            kind = fm.tensor_type("bfloat16", (16, 128))
            x = self.input("x", kind, id="x")
            gate = self.input("gate", kind, id="gate")
            weight = self.input("weight", fm.tensor_type("bfloat16", (128,)), id="weight")
            bias = self.input("bias", fm.tensor_type("bfloat16", (128,)), id="bias")
            stats = self.input("stats", fm.tensor_type("float32", (1, 16, 1)), id="stats")
            out = fm.F.nn.gated_norm_apply(x, stats, weight, bias, gate, axis=-1,
                epsilon=1e-5, use_mean=False, activation="sigmoid", output_dtype=dtype)
            self.function("main", (x, gate, weight, bias, stats), (out,))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _compile(tmp_path, dtype, algorithm="sqrt_div"):
    base = NvidiaSm90Target().triton_implementation_model
    catalog = replace(base, implementations=tuple(replace(item,
        parameters={**item.parameters, "compute_num_warps": 8, "inverse_std_algorithm": algorithm})
        if item.family == "gated_norm_apply" else item for item in base.implementations))
    compiler = Compiler()
    compiler.target = NvidiaSm90Target(triton_implementation_model=catalog)
    module = compiler.compile(_module(dtype)).module
    restored = fm.load_module(fm.emit_module(module, tmp_path / "selected.py"))
    assert restored.semantic_hash == module.semantic_hash
    return write_artifact(restored, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_selected_sqrt_div_matches_independent_reference(tmp_path, dtype):
    if not is_nvidia_sm90():
        pytest.skip("requires SM90")
    generator = torch.Generator().manual_seed(20)
    values = {name: torch.randn((16, 128), generator=generator).bfloat16().cuda() for name in ("x", "gate")}
    values["weight"] = torch.randn((128,), generator=generator).bfloat16().cuda()
    values["bias"] = torch.zeros_like(values["weight"])
    values["stats"] = torch.empty((1, 16, 1), device="cuda")
    expected = torch.empty((16, 128), dtype=getattr(torch, dtype), device="cuda")
    approximate = torch.empty_like(expected)
    _independent_reference[(1,)](values["x"], values["gate"], values["weight"], values["stats"],
                                 expected, approximate, num_warps=8)
    assert torch.count_nonzero(expected != approximate), "fixture must distinguish the inverse algorithms"
    runtime = load(_compile(tmp_path, dtype), device="cuda:0")
    try:
        output = torch.empty_like(expected)
        inputs = tuple(values[spec["value"]] for spec in runtime.external_arguments if spec["role"] != "result")
        runtime.prepare(*inputs, output=output)
        runtime.run_into(output, *inputs)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("algorithm", ["unknown", True, None, []])
def test_unsupported_inverse_algorithm_is_rejected(tmp_path, algorithm):
    with pytest.raises((CodegenError, IRVerificationError), match="inverse_std_algorithm"):
        _compile(tmp_path, "bfloat16", algorithm)
