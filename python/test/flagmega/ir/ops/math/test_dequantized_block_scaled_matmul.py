# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Stored FP8 weights can implement an unquantized BF16 projection contract."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.math.block_scaled_matmul import BlockScaledMatMul
from triton.flagmega.ir.ops.math.packed_block_scaled_matmul import PackedBlockScaledMatMul


class DequantizedProjection(fm.Module):
    def __init__(self, *, packed_weights=False, lanes=(), consumer_lanes=(), n=32, k=128,
                 dtype="bfloat16", quantization="none", weight_dtype="bfloat16", constants=False,
                 block_n=16, block_k=128):
        super().__init__(dialect="high_level", stage="imported", entry="main")
        self.packed_weights, self.lanes, self.consumer_lanes = packed_weights, lanes, consumer_lanes
        self.n, self.k, self.dtype = n, k, dtype
        self.quantization, self.weight_dtype, self.constants = quantization, weight_dtype, constants
        self.block_n, self.block_k = block_n, block_k

    def forward(self):
        value = self.input("value", fm.tensor_type(self.dtype, (1, self.k)), id="value")
        weight_type = (fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), (self.n, self.k // 32))
                       if self.packed_weights else fm.tensor_type("float8_e4m3fn", (self.n, self.k)))
        scale_type = fm.tensor_type("float32", ((self.n + self.block_n - 1) // self.block_n,
                                               (self.k + self.block_k - 1) // self.block_k))
        operands = []
        for name, value_type in (("weight", weight_type), ("scale", scale_type)):
            operands.append(self.weight(name, value_type, source="memory", key=name, id=name) if self.constants
                            else self.input(name, value_type, id=name))
        attrs = {"weight_block_n": self.block_n, "weight_block_k": self.block_k, "output_lanes": self.lanes,
                 "activation_quantization": self.quantization, "weight_dequant_dtype": self.weight_dtype,
                 "name": "projection"}
        result = (fm.F.math.packed_block_scaled_matmul(value, *operands, k_pack=2, k_vector=16, **attrs)
                  if self.packed_weights else fm.F.math.block_scaled_matmul(value, *operands, **attrs))
        if self.consumer_lanes:
            result = fm.F.tensors.pack(result, self.consumer_lanes, axes=(-1,) * len(self.consumer_lanes))
        self.function("main", (value,) if self.constants else (value, *operands), (result,))


def boundary_operands(n=32, k=128, *, boundary="activation", dtype="bfloat16", packed=False):
    value = torch.zeros((1, k), dtype=getattr(torch, dtype))
    weight = torch.zeros((n, k))
    if boundary == "activation":
        value[0, 0], value[0, 1] = 1.203125, .81640625
        weight[:, 1] = 1.
        scales = torch.ones(((n + 15) // 16, (k + 127) // 128))
        expected = .81640625
    else:
        value[0, 0] = 129.
        weight[:, 0] = 1.
        scales = torch.full(((n + 15) // 16, (k + 127) // 128), 1.00390625)
        expected = 129.
    weight = weight.to(torch.float8_e4m3fn)
    if packed:
        weight = weight.reshape(n, k // 32, 2, 16)
    return {"value": value, "weight": weight, "scale": scales}, expected


def _evaluate(value, weight, scales, **precision):
    attrs = {"weight_block_n": 16, "weight_block_k": 128,
             "activation_quantization": "none", "weight_dequant_dtype": "bfloat16", **precision}
    node = SimpleNamespace(attrs=attrs, type=fm.tensor_type("bfloat16", (1, weight.shape[0])))
    context = SimpleNamespace(torch=torch, torch_dtype=lambda dtype: getattr(torch, dtype.value))
    return BlockScaledMatMul.evaluate(node, (value, weight, scales), context)


def test_unquantized_activation_is_preserved_with_fp8_storage():
    value = torch.zeros((1, 128), dtype=torch.bfloat16)
    value[0, 0], value[0, 1] = 1.203125, .81640625
    weight = torch.zeros((32, 128))
    weight[:, 1] = 1.
    actual = _evaluate(value, weight.to(torch.float8_e4m3fn), torch.ones((2, 1)))
    torch.testing.assert_close(actual, torch.full_like(actual, .81640625), rtol=0, atol=0)


def test_weight_rounds_before_multiplication_not_just_before_the_final_store():
    value = torch.zeros((1, 128), dtype=torch.bfloat16)
    value[0, 0] = 129.
    weight = torch.zeros((32, 128))
    weight[:, 0] = 1.
    # The FP32 weight is halfway between BF16 1 and 1.0078125, and rounds
    # to 1. Multiplying first would give 129.50390625 and then round to 130.
    actual = _evaluate(value, weight.to(torch.float8_e4m3fn), torch.full((2, 1), 1.00390625))
    torch.testing.assert_close(actual, torch.full_like(actual, 129.), rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("lanes", [(), (8,), (2, 4)])
def test_precision_and_output_layout_survive_construction_and_evaluation(packed, lanes):
    module = DequantizedProjection(packed_weights=packed, lanes=lanes).build()
    values, expected = boundary_operands(packed=packed)
    output, = TorchEvaluator(DictWeightResolver({})).run(module, values)
    torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)
    assert module.node_map["projection"].attrs["activation_quantization"] == "none"
    assert module.node_map["projection"].attrs["weight_dequant_dtype"] == "bfloat16"


@pytest.mark.parametrize("definition", [BlockScaledMatMul, PackedBlockScaledMatMul])
@pytest.mark.parametrize("invalid", [{"activation_quantization": "bad"}, {"activation_quantization": []},
                                      {"weight_dequant_dtype": "float32"}, {"weight_dequant_dtype": "int32"},
                                      {"weight_dequant_dtype": []}])
def test_rejects_unknown_precision_contracts(definition, invalid):
    attrs = {"weight_block_n": 16, "weight_block_k": 128, **invalid}
    if definition is PackedBlockScaledMatMul:
        attrs.update(k_pack=2, k_vector=16)
    with pytest.raises(IRSchemaError):
        definition.normalize_attrs(attrs)


@pytest.mark.parametrize("encoding", ["IQ4_XS", "Q4_K"])
def test_native_gguf_cannot_silently_ignore_precision_overrides(encoding):
    with pytest.raises(IRSchemaError, match="GGUF"):
        BlockScaledMatMul.normalize_attrs({"weight_block_n": 16, "weight_block_k": 128,
                                           "weight_encoding": encoding, "activation_quantization": "none"})


@pytest.mark.parametrize("packed", [False, True])
def test_default_attributes_and_python_round_trip(tmp_path, packed):
    default = DequantizedProjection(packed_weights=packed, quantization="dynamic_fp8", weight_dtype=None).build()
    assert "activation_quantization" not in default.node_map["projection"].attrs
    assert "weight_dequant_dtype" not in default.node_map["projection"].attrs
    module = DequantizedProjection(packed_weights=packed, lanes=(2, 4)).build()
    path = fm.emit_module(module, tmp_path / "dequantized.py")
    assert fm.load_module(path).semantic_hash == module.semantic_hash
    check = "from triton.flagmega.ir import load_module; import sys; assert load_module(sys.argv[1]).semantic_hash == sys.argv[2]"
    subprocess.run([sys.executable, "-c", check, str(path), module.semantic_hash], check=True)


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_precision_override_rejects_non_floating_activation(dtype):
    with pytest.raises(IRSchemaError):
        DequantizedProjection(dtype=dtype).build()


@pytest.mark.parametrize("dtype,quantization,weight_dtype,mma", [
    ("bfloat16", "none", "bfloat16", True),
    ("float16", "none", "float16", True),
    ("float32", "none", "bfloat16", False),
    ("bfloat16", "none", None, False),
    ("bfloat16", "dynamic_fp8", "bfloat16", False),
    ("bfloat16", "dynamic_fp8", None, True),
])
def test_candidates_require_matching_tensor_core_operand_precision(dtype, quantization, weight_dtype, mma):
    from triton.flagmega.compiler import Compiler

    module = DequantizedProjection(dtype=dtype, quantization=quantization, weight_dtype=weight_dtype).build()
    proposed = Compiler().compile(module, stop_after="propose-tir").module
    points = [point for point in proposed.selection_points
              if any(candidate.id.startswith("tir.block_fp8.") for candidate in point.candidates)]
    assert len(points) == 1
    candidates = {candidate.id for candidate in points[0].candidates}
    assert "tir.block_fp8.simt" in candidates
    assert ("tir.block_fp8.mma" in candidates) == mma
