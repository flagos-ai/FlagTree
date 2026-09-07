# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.passes.functions import lift_parameter_constant_transforms


def _module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="packing_candidates")
    logical = fm.tensor_type("bfloat16", (4, 4))
    physical = fm.tensor_type("bfloat16", (2, 2, 2, 2))
    first = builder.weight(
        "layers.0.weight",
        logical,
        source="unit.safetensors",
        key="layers.0.weight",
        id="first_weight",
        metadata={"rdata_group": {"name": "unit.layer.weight", "index": 0, "count": 2}},
    )
    second = builder.weight(
        "layers.1.weight",
        logical,
        source="unit.safetensors",
        key="layers.1.weight",
        id="second_weight",
        metadata={"rdata_group": {"name": "unit.layer.weight", "index": 1, "count": 2}},
    )
    parameter = builder.var(
        "weight",
        logical,
        id="pack_weight",
        metadata={"function_parameter": "pack"},
    )
    packed = builder.call(
        "tensors.reshape",
        (parameter,),
        physical,
        id="pack_physical",
        attrs={"shape": (2, 2, 2, 2)},
        metadata={
            "packed_from": "pack_weight",
            "packed_for": "pack",
            "packed_layout": "unit_2x2",
        },
    )
    first_call = builder.call(
        "builtin.call",
        (first,),
        physical,
        id="first_call",
        attrs={"callee": "pack"},
    )
    second_call = builder.call(
        "builtin.call",
        (second,),
        physical,
        id="second_call",
        attrs={"callee": "pack"},
    )
    builder.function("main", (), (first_call, second_call))
    builder.function(
        "pack",
        (parameter,),
        (packed,),
        attrs={"calling_convention": "device", "reusable": True},
    )
    return fm.verify_module(builder.build(entry="main"))


def test_lift_parameter_transform_refines_one_callee_abi_and_every_call():
    result = fm.verify_module(lift_parameter_constant_transforms(_module()))

    assert "pack_weight" not in result.node_map
    assert "pack_physical" not in result.node_map
    parameter_id = result.function_map["pack"].parameters[0]
    assert result.node_map[parameter_id].metadata["lifted_parameter_transform"] == "pack_physical"
    assert result.function_map["pack"].outputs == (parameter_id,)
    for call_id, weight_id in (
        ("first_call", "first_weight"),
        ("second_call", "second_weight"),
    ):
        clone_id = f"{call_id}.pack_physical"
        assert result.node_map[call_id].inputs == (clone_id,)
        assert result.node_map[clone_id].inputs == (weight_id,)
        assert result.node_map[clone_id].metadata["lifted_from_function"] == "pack"


def test_lifted_call_graph_preserves_reference_evaluation():
    result = lift_parameter_constant_transforms(_module())
    first = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    second = first + 32

    outputs = TorchEvaluator(DictWeightResolver({
        "layers.0.weight": first,
        "layers.1.weight": second,
    })).run(result, {})

    torch.testing.assert_close(outputs[0], first.reshape(2, 2, 2, 2))
    torch.testing.assert_close(outputs[1], second.reshape(2, 2, 2, 2))


def _multi_source_module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="packing_candidates")
    value_type = fm.tensor_type("bfloat16", (2, 2))
    weights = {}
    for layer in range(2):
        for role in ("left", "right"):
            node_id = f"layer_{layer}_{role}"
            weights[node_id] = builder.weight(
                node_id,
                value_type,
                source="unit.safetensors",
                key=node_id,
                id=node_id,
                metadata={
                    "rdata_group": {
                        "name": f"unit.layer.{role}",
                        "index": layer,
                        "count": 2,
                    }
                },
            )
    left = builder.var("left", value_type, id="fuse_left")
    right = builder.var("right", value_type, id="fuse_right")
    fused = builder.call(
        "math.add",
        (left, right),
        value_type,
        id="fuse_result",
        metadata={
            "packed_from": (left.id, right.id),
            "packed_layout": "fused_pair",
        },
    )
    calls = tuple(
        builder.call(
            "builtin.call",
            (weights[f"layer_{layer}_left"], weights[f"layer_{layer}_right"]),
            value_type,
            id=f"layer_{layer}_call",
            attrs={"callee": "fuse"},
        )
        for layer in range(2)
    )
    builder.function("main", (), calls)
    builder.function("fuse", (left, right), (fused,))
    return fm.verify_module(builder.build(entry="main"))


def test_multi_source_lift_derives_model_independent_composite_rdata_group():
    lifted = fm.verify_module(lift_parameter_constant_transforms(_multi_source_module()))
    clones = tuple(lifted.node_map[f"layer_{layer}_call.fuse_result"] for layer in range(2))

    names = {value.metadata["rdata_group"]["name"] for value in clones}
    assert names == {"composite[unit.layer.left,unit.layer.right]"}
    assert [value.metadata["rdata_group"]["index"] for value in clones] == [0, 1]
    assert all("qwen" not in value.lower() for value in names)

    frozen = freeze_constant_islands(lifted)
    plan = fm.make_buffer_plan(frozen)
    grouped = tuple(
        value for value in plan.buffers
        if value.rdata_group == next(iter(names))
    )
    assert {value.group_index for value in grouped} == {0, 1}


def test_multi_source_lift_rejects_misaligned_rdata_group_indexes():
    source = _multi_source_module()
    edited = replace(
        source,
        nodes=tuple(
            replace(
                node,
                metadata={
                    "rdata_group": {
                        "name": "unit.layer.right",
                        "index": 1,
                        "count": 2,
                    }
                },
            )
            if node.id == "layer_0_right"
            else node
            for node in source.nodes
        ),
    )

    with pytest.raises(IRVerificationError, match="one integer layer index/count"):
        lift_parameter_constant_transforms(fm.verify_module(edited))
