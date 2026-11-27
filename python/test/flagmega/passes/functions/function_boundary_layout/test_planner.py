# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.functions import propagate_function_boundary_layouts


class _PackUnpackLayer(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        tensor_type = fm.tensor_type("float32", (4, 16))
        parameter = self.input("parameter", tensor_type, id="parameter")
        packed = fm.F.tensors.pack(
            parameter, (4,), axes=(1,), name="packed"
        )
        result = fm.F.tensors.unpack(packed, axes=(1,), name="result")
        value = self.input("value", tensor_type, id="value")
        call = fm.F.builtin.call(
            value, result_type=tensor_type, callee="layer", name="call"
        )
        self.function("main", (value,), (call,))
        self.function("layer", (parameter,), (result,), attrs={"reusable": True})


def test_agent_editable_identity_override_keeps_original_function_abi():
    original = _PackUnpackLayer().build()
    overridden = replace(
        original,
        metadata={"function_boundary_layout_choices": {"layer": "identity"}},
    )

    rewritten = propagate_function_boundary_layouts(overridden)

    assert rewritten is overridden
    assert rewritten.node_map["parameter"].type == original.node_map["parameter"].type
    assert "packed" in rewritten.node_map
    assert "result" in rewritten.node_map


def test_invalid_agent_layout_override_is_rejected_with_candidate_ids():
    original = _PackUnpackLayer().build()
    overridden = replace(
        original,
        metadata={"function_boundary_layout_choices": {"layer": "missing"}},
    )

    with pytest.raises(IRVerificationError, match="expected one of.*identity.*internal"):
        propagate_function_boundary_layouts(overridden)


def test_cp_sat_decision_is_serialized_in_editable_module_metadata():
    rewritten = propagate_function_boundary_layouts(_PackUnpackLayer().build())

    decisions = rewritten.metadata["function_boundary_layout_decisions"]
    assert decisions == ({
        "function": "layer",
        "candidate": "internal",
        "solver": "ortools-cp-sat",
    },)
