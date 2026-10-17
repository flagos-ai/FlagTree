# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A native mHC transition preserves its FP32 carry and explicit FMA order."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.mhc_post import MhcPost
from triton.flagmega.ir.ops.nn.mhc_pre import MhcPre

from python.test.flagmega.ir.ops.nn.test_mhc import _pre_attrs


class PostCarry(fm.Module):
    def __init__(self, hc=4, hidden=8, dtype="float32", mode="value_fma_first"):
        super().__init__(dialect="high_level", stage="imported", entry="main")
        self.hc, self.hidden, self.dtype, self.mode = hc, hidden, dtype, mode

    def forward(self):
        types = {"value": fm.tensor_type("bfloat16", (1, self.hidden)),
                 "residual": fm.tensor_type("bfloat16", (1, self.hc, self.hidden)),
                 "post": fm.tensor_type("float32", (1, self.hc)),
                 "comb": fm.tensor_type("float32", (1, self.hc, self.hc))}
        values = tuple(self.input(name, value, id=name) for name, value in types.items())
        result = fm.F.nn.mhc_post(*values, output_dtype=self.dtype, accumulation_mode=self.mode, name="carry")
        self.function("main", values, (result,))


class CollapseCarry(fm.Module):
    def __init__(self, *, lanes=(), input_dtype="bfloat16", reduction="sequential_fma"):
        super().__init__(dialect="high_level", stage="imported", entry="main")
        self.lanes, self.input_dtype, self.reduction = lanes, input_dtype, reduction

    def forward(self):
        types = {"streams": fm.tensor_type("float32", (1, 4, 8)),
                 "projection": fm.tensor_type("float32", (24, 32)),
                 "base": fm.tensor_type("float32", (24,)), "scale": fm.tensor_type("float32", (3,))}
        values = tuple(self.input(name, value, id=name) for name, value in types.items())
        result = fm.F.nn.mhc_pre(*values, **_pre_attrs(), collapsed_dtype="float32",
                                collapse_input_dtype=self.input_dtype, collapse_reduction=self.reduction,
                                output_lanes=self.lanes, name="pre")
        self.function("main", values, fm.F.tensors.get_items(result, 0, 1, 2, name_prefix="field"))


def test_value_first_fma_preserves_the_unrounded_residual():
    value = torch.full((1, 8), 1.5, dtype=torch.bfloat16)
    residual = torch.zeros((1, 4, 8), dtype=torch.bfloat16)
    residual[:, 0] = -1.5
    post = torch.full((1, 4), 1 + 2 ** -23)
    comb = torch.zeros((1, 4, 4))
    comb[:, 0] = 1.
    attrs = {"output_dtype": "float32", "accumulation_mode": "value_fma_first"}
    actual = MhcPost.evaluate(SimpleNamespace(attrs=attrs), (value, residual, post, comb),
                              SimpleNamespace(torch=torch))
    expected = torch.full((1, 4, 8), 3 * 2 ** -24)
    torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)
    assert actual.dtype == torch.float32


def test_post_accumulates_all_residuals_in_source_order():
    # Ascending sources give 2; reversing sources 1..3 gives 3. A parallel
    # reduction is not the declared left-to-right accumulation either.
    value = torch.zeros((1, 8), dtype=torch.bfloat16)
    residual = torch.ones((1, 4, 8), dtype=torch.bfloat16)
    post = torch.zeros((1, 4))
    comb = torch.tensor([2 ** 24, 1., -(2 ** 24), 2.]).view(1, 4, 1).expand(1, 4, 4)
    attrs = {"output_dtype": "float32", "accumulation_mode": "value_fma_first"}
    actual = MhcPost.evaluate(SimpleNamespace(attrs=attrs), (value, residual, post, comb),
                              SimpleNamespace(torch=torch))
    torch.testing.assert_close(actual, torch.full_like(actual, 2.), rtol=0, atol=0)


def test_projection_keeps_the_unrounded_input_when_only_collapse_rounds():
    generator = torch.Generator().manual_seed(178)
    streams = torch.randn((1, 4, 8), generator=generator)
    inputs = {"streams": streams, "projection": torch.randn((24, 32), generator=generator),
              "base": torch.randn(24, generator=generator), "scale": torch.randn(3, generator=generator)}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    original = evaluator.run(CollapseCarry(input_dtype=None, reduction="sum").build(), inputs)
    rounded = evaluator.run(CollapseCarry(reduction="sum").build(), inputs)
    cast_all = evaluator.run(CollapseCarry(input_dtype=None, reduction="sum").build(),
                             {**inputs, "streams": streams.bfloat16().float()})
    for value, expected in zip(rounded[:2], original[:2], strict=True):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert any(not torch.equal(a, b) for a, b in zip(cast_all[:2], original[:2], strict=True))
    assert not torch.equal(rounded[2], original[2])


def test_collapse_uses_ordered_fma_after_the_requested_stream_rounding():
    streams = torch.zeros((1, 4, 8))
    streams[:, 0] = 1.5 + 2 ** -20
    streams[:, 1] = -1.5
    inputs = {"streams": streams, "projection": torch.zeros((24, 32)),
              "base": torch.zeros(24), "scale": torch.zeros(3)}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    result = evaluator.run(CollapseCarry().build(), inputs)[2]
    torch.testing.assert_close(result, torch.full_like(result, 2 ** -25), rtol=0, atol=0)
    separately_rounded = evaluator.run(CollapseCarry(reduction="sum").build(), inputs)[2]
    assert torch.count_nonzero(separately_rounded) == 0


@pytest.mark.parametrize("attrs", [{"output_dtype": "int32"}, {"output_dtype": "float64"},
                                    {"accumulation_mode": "bad"}, {"accumulation_mode": []}])
def test_post_rejects_unknown_precision_contracts(attrs):
    with pytest.raises(IRSchemaError):
        MhcPost.normalize_attrs(attrs)


@pytest.mark.parametrize("attrs", [{"collapse_input_dtype": "int32"}, {"collapse_input_dtype": "float64"},
                                    {"collapse_reduction": "bad"}, {"collapse_reduction": []}])
def test_pre_rejects_unknown_collapse_contracts(attrs):
    with pytest.raises(IRSchemaError):
        MhcPre.normalize_attrs({**_pre_attrs(), **attrs})


@pytest.mark.parametrize("kind", ["post", "pre"])
def test_ordered_precision_contracts_round_trip(tmp_path, kind):
    module = PostCarry().build() if kind == "post" else CollapseCarry(lanes=(2, 4)).build()
    path = fm.emit_module(module, tmp_path / f"{kind}.py")
    assert fm.load_module(path).semantic_hash == module.semantic_hash
    check = """
import sys
from triton.flagmega.ir import load_module
module = load_module(sys.argv[1])
assert module.semantic_hash == sys.argv[2]
"""
    subprocess.run([sys.executable, "-c", check, str(path), module.semantic_hash], check=True)


def test_post_default_attributes_remain_empty():
    module = PostCarry(dtype=None, mode="residual_then_value").build()
    assert dict(module.node_map["carry"].attrs) == {}
    assert module.node_map["carry"].type.dtype == fm.DType.BFLOAT16
