# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The native reciprocal quantizer has a distinct, explicit rounding contract."""

import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.options import CompileOptions


def inputs():
    value = torch.zeros((1, 128), dtype=torch.bfloat16)
    value[0, 0], value[0, 1] = .050537109375, .134765625
    weight = torch.zeros((32, 128), dtype=torch.float32)
    weight[:, 0] = 1.
    return value, weight.to(torch.float8_e4m3fn), torch.ones((2, 1))


class Sm90Projection(fm.Module):
    def __init__(self, *, family="linear", packed=False, dtype="bfloat16", mode="dynamic_fp8_sm90"):
        super().__init__(dialect="high_level", stage="imported", entry="main")
        self.family, self.packed, self.dtype, self.mode = family, packed, dtype, mode

    def forward(self):
        value = self.input("value", fm.tensor_type(self.dtype, (1, 128)), id="value")
        wt = (fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), (32, 4)) if self.packed
              else fm.tensor_type("float8_e4m3fn", (32, 128)))
        scale_type = fm.tensor_type("float32", (2, 1))
        parameters = [value]
        attrs = {"weight_block_n": 16, "weight_block_k": 128, "activation_quantization": self.mode,
                 "name": "projection"}
        if self.packed:
            attrs.update(k_pack=2, k_vector=16)
        weight = self.input("weight", wt, id="weight")
        if self.family == "linear":
            scale = self.input("scale", scale_type, id="scale")
            parameters += [weight, scale]
            result = (fm.F.math.packed_block_scaled_matmul(value, weight, scale, **attrs) if self.packed
                      else fm.F.math.block_scaled_matmul(value, weight, scale, **attrs))
        else:
            up = self.input("up", wt, id="up")
            scale = self.input("scale", scale_type, id="scale")
            up_scale = self.input("up_scale", scale_type, id="up_scale")
            parameters += [weight, up, scale, up_scale]
            result = (fm.F.nn.packed_matmul_glu(*parameters, activation="silu", **attrs) if self.packed
                      else fm.F.nn.matmul_glu(*parameters, activation="silu", **attrs))
        self.function("main", tuple(parameters), (result,))


def operands(*, family="linear", packed=False):
    value, weight, scale = inputs()
    if packed:
        weight = weight.reshape(32, 4, 2, 16)
    if family == "linear":
        return {"value": value, "weight": weight, "scale": scale}
    return {"value": value, "weight": weight, "up": weight.clone(), "scale": scale, "up_scale": scale.clone()}


@pytest.mark.parametrize("packed", (False, True))
def test_explicit_sm90_activation_quantization_keeps_the_native_fp8_boundary(packed):
    value, weight, scale = inputs()
    types = (fm.tensor_type("bfloat16", (1, 128)),
             fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), (32, 4)) if packed
             else fm.tensor_type("float8_e4m3fn", (32, 128)), fm.tensor_type("float32", (2, 1)))
    nodes = tuple(fm.Node(name, "builtin.var", (), kind) for name, kind in zip(("value", "weight", "scale"), types))
    op = fm.get_definition("math.packed_block_scaled_matmul" if packed else "math.block_scaled_matmul")
    attrs = {"weight_block_n": 16, "weight_block_k": 128}
    if packed:
        attrs.update(k_pack=2, k_vector=16)
        weight = weight.reshape(32, 4, 2, 16)
    prepared = op.prepare(nodes, attrs)
    node = fm.Node("projection", op.op_name, tuple(value.id for value in nodes), prepared.result_type,
                   prepared.effect, {**prepared.attrs, "activation_quantization": "dynamic_fp8_sm90"})
    context = SimpleNamespace(torch=torch, torch_dtype=lambda dtype: getattr(torch, dtype.value))
    output = op.evaluate(node, (value, weight, scale), context)
    torch.testing.assert_close(output, torch.full_like(output, .052978515625), atol=0, rtol=0)
    # The existing RN contract remains a separate choice.
    original = op.evaluate(replace(node, attrs=prepared.attrs), (value, weight, scale), context)
    torch.testing.assert_close(original, torch.full_like(original, .048095703125), atol=0, rtol=0)


@pytest.mark.parametrize("family", ("linear", "glu"))
@pytest.mark.parametrize("packed", (False, True))
def test_sm90_quantization_survives_normal_compilation(family, packed):
    module = Sm90Projection(family=family, packed=packed).build()
    expected = torch.full((1, 32), .052978515625, dtype=torch.bfloat16)
    if family == "glu":
        expected = torch.nn.functional.silu(expected) * expected
    actual, = TorchEvaluator(DictWeightResolver({})).run(module, operands(family=family, packed=packed))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    proposed = Compiler().compile(module, stop_after="propose-tir").module
    assert not any(node.op == "tensors.pack" for node in proposed.nodes)
    calls = [node for node in proposed.nodes if "activation_quantization" in node.attrs]
    assert calls and all(node.attrs["activation_quantization"] == "dynamic_fp8_sm90" for node in calls)
    assert any(point.candidates for point in proposed.selection_points)


@pytest.mark.parametrize("family", ("linear", "glu"))
@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("dtype", ("float16", "float32", "int32"))
def test_sm90_reciprocal_quantization_rejects_other_activation_dtypes(family, packed, dtype):
    with pytest.raises(IRSchemaError):
        Sm90Projection(family=family, packed=packed, dtype=dtype).build()


@pytest.mark.parametrize("family", ("linear", "glu"))
def test_sm90_quantization_is_rejected_by_other_targets(family):
    with pytest.raises(CodegenError):
        Compiler(CompileOptions(target="nvidia-sm120")).compile(Sm90Projection(family=family).build())


@pytest.mark.parametrize("family", ("linear", "glu"))
def test_original_canonical_attributes_do_not_gain_a_default_quantization_mode(family):
    module = Sm90Projection(family=family, mode="dynamic_fp8").build()
    assert all("activation_quantization" not in node.attrs for node in module.nodes)


@pytest.mark.parametrize("family", ("linear", "glu"))
def test_resumed_raw_tir_cannot_change_the_approximation_target(family):
    from triton.flagmega.codegen.triton.tir_package import describe_tir_package

    module = Compiler().compile(Sm90Projection(family=family).build()).module
    changed = replace(module, metadata={**module.metadata, "codegen_template_target": {
        "platform": "nvidia", "architecture": "sm120"}})
    with pytest.raises(CodegenError, match="SM90 reciprocal"):
        describe_tir_package(changed)


def test_all_four_constructors_round_trip_in_a_fresh_process(tmp_path):
    paths = []
    for family in ("linear", "glu"):
        for packed in (False, True):
            module = Sm90Projection(family=family, packed=packed).build()
            path = tmp_path / f"{family}-{packed}.py"
            path.write_text(fm.module_source(module))
            paths.extend((str(path), module.semantic_hash))
    subprocess.run([sys.executable, "-c", '''
import sys
from triton.flagmega.ir import load_module
for path, expected in zip(sys.argv[1::2], sys.argv[2::2]):
    module = load_module(path)
    assert module.semantic_hash == expected
    assert any(n.attrs.get("activation_quantization") == "dynamic_fp8_sm90" for n in module.nodes)
''', *paths], check=True)


@pytest.mark.parametrize("op_name,encoding_key", (("math.block_scaled_matmul", "weight_encoding"),
    ("math.packed_block_scaled_matmul", "weight_encoding"), ("nn.matmul_glu", "gate_weight_encoding"),
    ("nn.packed_matmul_glu", "up_weight_encoding")))
def test_native_gguf_templates_cannot_ignore_sm90_quantization(op_name, encoding_key):
    definition = fm.get_definition(op_name)
    attrs = {"weight_block_n": 128, "weight_block_k": 128, "activation_quantization": "dynamic_fp8_sm90"}
    if "glu" in op_name:
        attrs["activation"] = "silu"
    if "packed" in op_name:
        attrs.update(k_pack=2, k_vector=16)
    assert definition.normalize_attrs(attrs)["activation_quantization"] == "dynamic_fp8_sm90"
    with pytest.raises(IRSchemaError, match="require FP8 storage"):
        definition.normalize_attrs({**attrs, encoding_key: "IQ4_XS"})
