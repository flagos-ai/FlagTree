# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.auto_distributed import (
    PyNttDistributedReshardRealizationPolicy,
)
from triton.flagmega.passes.functions import (
    propagate_post_auto_distributed_function_boundary_layouts,
)


def _module():
    placement = fm.Placement((2, 2), "yx", "bb")
    tensor = fm.tensor_type("float32", (4, 16))
    broadcast = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    split = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
        placement,
    )

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            parameter = self.input("parameter", broadcast, id="parameter")
            local = fm.F.distributed.boxing(parameter, split, name="local")
            weight = self.weight(
                "weight",
                broadcast.tensor,
                source="unit.safetensors",
                key="weight",
                id="weight",
            )
            value = fm.F.distributed.sharded_view(
                weight, broadcast, name="weight.broadcast")
            call = fm.F.builtin.call(
                value, result_type=split, callee="layer", name="call")
            self.function("main", (), (call,))
            self.function(
                "layer",
                (parameter,),
                (local,),
                attrs={"reusable": True, "noinline": True},
            )

    return Graph().build()


def _run(module):
    return propagate_post_auto_distributed_function_boundary_layouts(
        module,
        PyNttDistributedReshardRealizationPolicy(),
    )


def test_agent_editable_identity_override_keeps_distributed_abi():
    original = _module()
    overridden = replace(
        original,
        metadata={
            "distributed_function_boundary_layout_choices": {
                "layer": "identity",
            },
        },
    )

    rewritten = _run(overridden)

    assert rewritten is overridden
    assert rewritten.node_map["parameter"].type == original.node_map["parameter"].type
    assert "local" in rewritten.node_map


def test_invalid_agent_override_reports_valid_candidate_ids():
    overridden = replace(
        _module(),
        metadata={
            "distributed_function_boundary_layout_choices": {
                "layer": "missing",
            },
        },
    )

    with pytest.raises(IRVerificationError, match="expected one of.*identity.*internal"):
        _run(overridden)


def test_distributed_cp_sat_decision_is_serialized_in_editable_module_metadata():
    rewritten = _run(_module())

    assert rewritten.metadata[
        "distributed_function_boundary_layout_decisions"
    ] == ({
        "function": "layer",
        "candidate": "internal",
        "solver": "ortools-cp-sat",
    },)
