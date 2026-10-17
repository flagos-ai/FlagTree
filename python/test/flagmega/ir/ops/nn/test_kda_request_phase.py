# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A KDA numerical phase is explicit input, independent of state contents."""

from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.kda_gated_step import KdaGatedStep

from python.test.flagmega.codegen.triton.kernels.kda_gated_recurrent_core.test_kda_gated_recurrent_runtime import (
    _module,
    _values,
)


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_explicit_phase_uses_its_static_numerical_contract(tmp_path, prefill, packed):
    attrs = {"heads": 2, "dim": 128, "query_scale": 128 ** -.5,
             "output_lanes": (8,) if packed else ()}
    module = _module(request_phase=True, **attrs)
    reloaded = fm.load_module(fm.emit_module(module, tmp_path / "phase.py"))
    assert reloaded.semantic_hash == module.semantic_hash
    node, = [node for node in module.nodes if node.op == "nn.kda_gated_step"]
    assert len(node.inputs) == 9 and node.inputs[-1] == "prefill"
    assert node.effect.kind == "read_write"
    values = _values(torch, heads=2, dim=128)
    assert values["state"].count_nonzero() > 0
    evaluator = TorchEvaluator(DictWeightResolver({}))
    actual = evaluator.run(reloaded, {**values, "prefill": torch.tensor(prefill)})
    expected = evaluator.run(_module(arithmetic="chunk_bf16" if prefill else "recurrent", **attrs), values)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    alternative = evaluator.run(_module(arithmetic="recurrent" if prefill else "chunk_bf16", **attrs), values)
    assert any(torch.count_nonzero(a != b) for a, b in zip(actual, alternative, strict=True))


@pytest.mark.parametrize("dtype,shape", [("bool", (1,)), ("int32", ())])
def test_phase_rejects_nonscalar_or_nonboolean_inputs(dtype, shape):
    source = _module(request_phase=True, heads=2, dim=128)
    node, = [node for node in source.nodes if node.op == "nn.kda_gated_step"]
    inputs = tuple(source.node_map[value] for value in node.inputs)
    with pytest.raises(IRSchemaError, match="scalar BOOL"):
        KdaGatedStep.infer_type((*inputs[:-1], replace(inputs[-1], type=fm.tensor_type(dtype, shape))), node.attrs)


def test_runtime_phase_validates_both_algorithms():
    with pytest.raises(IRSchemaError, match="head_k = head_v = 128"):
        _module(request_phase=True, dim=4)


def test_phase_is_broadcast_when_heads_are_partitioned():
    placement = fm.Placement((2,), "c", "c")
    module = _module(request_phase=True, heads=4, dim=128, placement=placement,
                     head_policy=fm.SBP.split_contiguous((0,)), output_lanes=(8,))
    node, = [node for node in module.nodes if node.op == "nn.kda_gated_step"]
    inputs = tuple(module.node_map[value] for value in node.inputs)
    assert node.type.fields[0].axis_policies[1] == fm.SBP.split_contiguous((0,))
    assert inputs[-1].type.axis_policies == ()
    other = fm.Placement((2,), "x", "b")
    with pytest.raises(IRSchemaError, match="broadcast"):
        KdaGatedStep.infer_type((*inputs[:-1], replace(inputs[-1], type=fm.DistributedType(
            fm.tensor_type("bool", ()), (), other))), node.attrs)
