# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A boundary snapshot need only materialize the consumer's actual owners."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import PyNttDistributedReshardRealizationPolicy
from triton.flagmega.passes.functions import propagate_post_auto_distributed_function_boundary_layouts


def _module(*, extra_raw_use=False, extra_cast_use=False, output_cast=False, output_view=False, output_parameter=False,
            same_owner_source=False, remote_owner=False, cast_element=fm.DType.FLOAT32):
    placement = fm.Placement((8, 16), "yx", "bb")
    tensor = fm.tensor_type(fm.VectorType(fm.DType.FLOAT32, (4, )), (1, 512))
    full = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement)
    input_split = replace(full, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((0, ), 64)))
    owner = replace(full, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 4)))
    if remote_owner:
        owner = replace(owner, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((1, 0), 4)))
    if same_owner_source:
        input_split = owner
    cast_dtype = fm.VectorType(cast_element, (2, 2))
    cast_owner = replace(owner, tensor=fm.tensor_type(cast_dtype, (1, 512)))

    class Graph(fm.Module):

        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            parameter = self.input("parameter", full, id="parameter")
            local = fm.F.distributed.boxing(parameter, input_split, name="local")
            local_use = fm.F.math.silu(local, name="local_use")
            cast = fm.F.tensors.bitcast(parameter, cast_dtype, name="cast")
            view = fm.F.distributed.sharded_view(cast, cast_owner, name="owner")
            residual = fm.F.math.add(view, view, name="residual")
            outputs = [local_use, residual]
            if extra_raw_use:
                outputs.append(fm.F.math.silu(parameter, name="extra_raw"))
            if extra_cast_use:
                outputs.append(fm.F.math.silu(cast, name="extra_cast"))
            if output_cast:
                outputs.append(cast)
            if output_view:
                outputs.append(view)
            if output_parameter:
                outputs.append(parameter)
            value = self.input("value", full, id="value")
            call = fm.F.builtin.call(value, result_type=fm.TupleType(tuple(node.type for node in outputs)),
                                     callee="layer", name="call")
            self.function("main", (value, ), (call, ))
            self.function("layer", (parameter, ), tuple(outputs), attrs={"reusable": True, "noinline": True})

    return Graph().build(), owner, cast_owner


def _run(module):
    return propagate_post_auto_distributed_function_boundary_layouts(module, PyNttDistributedReshardRealizationPolicy())


def test_raw_restore_materializes_only_owner_demand_before_bitcast():
    original, owner_type, cast_type = _module()
    rewritten = _run(original)
    restored = [node for node in rewritten.nodes if node.metadata.get("boundary_layout") == "raw_parameter_restore"]
    assert len(restored) == 1
    restore = restored[0]
    assert restore.type == owner_type
    assert restore.op == "distributed.boxing"  # Snapshot, not a source alias.
    assert restore.inputs == ("parameter", )
    assert rewritten.node_map["cast"].type == cast_type
    assert rewritten.node_map["cast"].inputs == (restore.id, )
    assert "owner" not in rewritten.node_map
    assert rewritten.node_map["residual"].inputs == ("cast", "cast")
    assert rewritten.node_map["parameter"].type == original.node_map["local"].type
    assert _run(rewritten) is rewritten


@pytest.mark.parametrize("extra", [
    "extra_raw_use", "extra_cast_use", "output_cast", "output_view", "output_parameter", "same_owner_source",
    "remote_owner"
])
def test_other_raw_consumers_and_function_results_keep_full_snapshot(extra):
    original, _, _ = _module(**{extra: True})
    rewritten = _run(original)
    restore = next(node for node in rewritten.nodes if node.metadata.get("boundary_layout") == "raw_parameter_restore")
    assert restore.type == original.node_map["parameter"].type
    assert restore.op == "distributed.boxing"
    assert rewritten.node_map["cast"].type == original.node_map["cast"].type


def test_numeric_bit_reinterpretation_is_not_owner_demand_propagation():
    original, _, _ = _module(cast_element=fm.DType.INT32)
    rewritten = _run(original)
    restore = next(node for node in rewritten.nodes if node.metadata.get("boundary_layout") == "raw_parameter_restore")
    assert restore.type == original.node_map["parameter"].type


def test_snapshot_survives_when_policy_would_choose_an_alias():
    from triton.flagmega.passes.auto_distributed.realization import (
        DistributedReshardRealization,
        DistributedReshardRealizationContext,
        DistributedReshardSourceKind,
        DistributedReshardUsageKind,
    )
    original, owner, _ = _module()
    policy = PyNttDistributedReshardRealizationPolicy()
    assert policy.classify(
        DistributedReshardRealizationContext(
            original.node_map["local"].type,
            owner,
            DistributedReshardSourceKind.FUNCTION_PARAMETER,
            DistributedReshardUsageKind.INTERNAL,
        )) is DistributedReshardRealization.SHARDED_VIEW
    rewritten = _run(original)
    restore = next(node for node in rewritten.nodes if node.metadata.get("boundary_layout") == "raw_parameter_restore")
    assert restore.op == "distributed.boxing"
    assert restore.id != restore.inputs[0]
    assert fm.local_tensor_type(restore.type).shape == fm.tensor_type("float32", (1, 4)).shape
    assert restore.type.tensor.dtype == fm.VectorType(fm.DType.FLOAT32, (4, ))


def test_vector_lane_order_and_values_are_exactly_preserved():
    import torch
    from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator

    original, _, _ = _module()
    rewritten = _run(original)
    value = torch.arange(2048, dtype=torch.float32).reshape(1, 512, 4)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    before = evaluator.run(original, {"value": value})[0]
    after = evaluator.run(rewritten, {"value": value})[0]
    assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(before, after, strict=True))
