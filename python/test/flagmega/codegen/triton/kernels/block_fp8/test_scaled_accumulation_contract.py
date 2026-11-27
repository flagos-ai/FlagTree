# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Selected scale promotion preserves SM90's combined-scale FMA rounding."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton.kernel_call_renderers import _scaled_accumulation_mode
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.targets import NvidiaSm90Target

from python.test.flagmega.codegen.triton.kernels.block_fp8.test_dequantized_runtime import _run


class Projection(fm.Module):
    def __init__(self, glu):
        super().__init__(dialect="high_level", stage="imported", entry="main")
        self.glu = glu

    def forward(self):
        x = self.input("x", fm.tensor_type("bfloat16", (1, 256)), id="x")
        w = self.input("w", fm.tensor_type("float8_e4m3fn", (16, 256)), id="w")
        s = self.input("s", fm.tensor_type("float32", (1, 2)), id="s")
        attrs = {"weight_block_n": 16, "weight_block_k": 128, "activation_quantization": "dynamic_fp8_sm90"}
        if self.glu:
            up = self.input("up", w.type, id="up")
            up_scale = self.input("up_scale", s.type, id="up_scale")
            result = fm.F.nn.matmul_glu(x, w, up, s, up_scale, activation="silu", **attrs)
            self.function("main", (x, w, up, s, up_scale), (result,))
        else:
            result = fm.F.math.block_scaled_matmul(x, w, s, **attrs)
            self.function("main", (x, w, s), (result,))


def _compile(tmp_path, glu, mode):
    family = "matmul_glu" if glu else "block_fp8"
    identity = f"tir.{family}.mma"
    compiler = Compiler()
    base = compiler.target.triton_implementation_model
    catalog = replace(
        base,
        implementations=tuple(
            replace(value, parameters={**value.parameters, "scaled_accumulation": mode})
            if value.id == identity
            else value
            for value in base.implementations
        ),
    )
    compiler.target = NvidiaSm90Target(triton_implementation_model=catalog)
    proposed = compiler.compile(Projection(glu).build(), stop_after="propose-tir").module
    choices = [
        (point.id, identity)
        for point in proposed.selection_points
        if any(candidate.id == identity for candidate in point.candidates)
    ]
    assert choices
    selected = apply_plan(proposed, override_plan(proposed, choices))
    compiled = compiler.compile(selected).module
    restored = fm.load_module(fm.emit_module(compiled, tmp_path / "selected.py"))
    assert restored.semantic_hash == compiled.semantic_hash
    assert any(
        d.dispatch.microkernel.parameters.get("scaled_accumulation") == mode for d in restored.kernel_definitions
    )
    return write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)


@pytest.mark.parametrize("glu", [False, True])
@pytest.mark.parametrize("mode", ["compensated", "fma"])
def test_selected_scaled_accumulation_preserves_rounding_boundary(tmp_path, glu, mode):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    x = torch.ones((1, 256), dtype=torch.bfloat16)
    w = torch.ones((16, 256))
    w[:, 128:] = -1
    s = torch.tensor([[1.0, np.nextafter(np.float32(1.0), np.float32(2.0))]], dtype=torch.float32)
    values = {"x": x, "w": w.to(torch.float8_e4m3fn), "s": s}
    # Quantization maps all 128 ones to 448. Their raw FP8 dot is exact.
    sa = np.float32(1.0 / 448.0)
    raw = np.float32(128 * 448)
    first = np.float32(raw * sa)
    if mode == "fma":
        combined_scale = np.float32(sa * s[0, 1].item())
        # A product of these FP32 operands and the cancellation fit exactly
        # in FP64; one FP32 cast models the final FMA rounding independently.
        expected = np.float32(float(first) - float(raw) * float(combined_scale))
    else:
        expected = np.float32(first - np.float32(first * s[0, 1].item()))
    gold = torch.full((1, 16), float(expected), dtype=torch.bfloat16)
    if glu:
        values = {
            "x": x,
            "w": values["w"],
            "up": torch.ones((16, 256)).to(torch.float8_e4m3fn),
            "s": s,
            "up_scale": torch.full((1, 2), 1.0 / 256.0),
        }
        gold = torch.nn.functional.silu(gold)
    actual = _run(_compile(tmp_path, glu, mode), values, (1, 16), "bfloat16")
    torch.testing.assert_close(actual, gold, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode,variant,quantization,block_k",
    [
        ("unknown", "mma", "dynamic_fp8_sm90", 128),
        (True, "mma", "dynamic_fp8_sm90", 128),
        ("fma", "simt", "dynamic_fp8_sm90", 128),
        ("fma", "mma", "dynamic_fp8", 128),
        ("fma", "mma", "dynamic_fp8_sm90", 16),
    ],
)
def test_scaled_promotion_rejects_unsupported_selected_contracts(mode, variant, quantization, block_k):
    with pytest.raises(CodegenError, match="scaled"):
        _scaled_accumulation_mode(
            {"parameters": {"scaled_accumulation": mode}, "variant": variant}, quantization, block_k
        )
