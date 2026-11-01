# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Residual rounding belongs to mHC post without an intermediate activation."""

import importlib.util
from pathlib import Path

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.mhc_post import MhcPost


class PostRounding(fm.Module):
    def __init__(self, *, explicit=False, dtype="bfloat16", mode="value_fma_first", hidden=8,
                 output_dtype="float32", tokens=2):
        super().__init__(dialect="high_level", stage="imported", entry="main")
        self.explicit, self.dtype, self.mode, self.hidden = explicit, dtype, mode, hidden
        self.output_dtype = output_dtype
        self.tokens = tokens

    def forward(self):
        value = self.input("value", fm.tensor_type("bfloat16", (self.tokens, self.hidden)), id="value")
        residual = self.input("residual", fm.tensor_type("float32", (self.tokens, 4, self.hidden)), id="residual")
        post = self.input("post", fm.tensor_type("float32", (self.tokens, 4)), id="post")
        comb = self.input("comb", fm.tensor_type("float32", (self.tokens, 4, 4)), id="comb")
        rounded = fm.F.tensors.cast(residual, self.dtype) if self.explicit else residual
        attrs = {} if self.explicit or self.dtype is None else {"residual_input_dtype": self.dtype}
        result = fm.F.nn.mhc_post(value, rounded, post, comb, output_dtype=self.output_dtype,
                                 accumulation_mode=self.mode, name="mixed", **attrs)
        self.function("main", (value, residual, post, comb), (result,))


@pytest.mark.parametrize("dtype", ("bfloat16", "float16", "float32"))
@pytest.mark.parametrize("mode", ("value_fma_first", "residual_then_value"))
def test_input_rounding_matches_original_conversion_and_preserves_resume(tmp_path, dtype, mode):
    generator = torch.Generator().manual_seed(197)
    residual = torch.randn((2, 4, 8), generator=generator)
    residual[0, 0, :4] = torch.tensor([1.00390625, -1.00390625, 1.003905, 2.0078125])
    inputs = {"value": torch.randn((2, 8), generator=generator).bfloat16(), "residual": residual,
              "post": torch.randn((2, 4), generator=generator), "comb": torch.randn((2, 4, 4), generator=generator)}
    module = PostRounding(dtype=dtype, mode=mode).build()
    reference = PostRounding(explicit=True, dtype=dtype, mode=mode).build()
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(module, inputs), evaluator.run(reference, inputs), rtol=0, atol=0)
    assert not any(n.op == "tensors.cast" for n in module.nodes)
    restored = fm.load_module(fm.emit_module(module, tmp_path / "rounded.py"))
    assert restored.semantic_hash == module.semantic_hash


def test_input_rounding_preserves_the_source_based_default_output_dtype():
    module = PostRounding(output_dtype=None).build()
    inputs = {"value": torch.zeros((2, 8), dtype=torch.bfloat16),
              "residual": torch.full((2, 4, 8), 1.00390625),
              "post": torch.zeros((2, 4)), "comb": torch.eye(4).expand(2, 4, 4)}
    actual, = TorchEvaluator(DictWeightResolver({})).run(module, inputs)
    assert module.node_map["mixed"].type.dtype == fm.DType.FLOAT32
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, inputs["residual"].bfloat16().float(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ("int32", "float64", [], True))
def test_input_rounding_rejects_unsupported_requested_precision(dtype):
    with pytest.raises(IRSchemaError, match="residual_input_dtype"):
        MhcPost.normalize_attrs({"residual_input_dtype": dtype})


def test_native_profile_places_residual_conversion_in_the_consumer():
    tutorial = Path(__file__).parents[5] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    spec = importlib.util.spec_from_file_location("_mhc_input_rounding_native_profile", tutorial / "local_optimizations/native_profile.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    source = PostRounding(dtype=None).build()
    result = profile._apply_projection_profile(source)
    post = result.node_map["mixed"]
    assert post.inputs[1] == "residual"
    assert post.attrs["residual_input_dtype"] == "bfloat16"
    assert not any(n.op == "tensors.cast" and n.inputs == ("residual",) for n in result.nodes)
