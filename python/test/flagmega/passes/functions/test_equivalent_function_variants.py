# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Function finalization must preserve reuse after layouts converge."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import PyNttDistributedReshardRealizationPolicy
from triton.flagmega.passes.functions import propagate_post_auto_distributed_function_boundary_layouts
from triton.flagmega.passes.norm_stats import finalize_norm_stats_bindings


def _module(*, partial_variant=False, variant_op="silu", special=True):
    placement = fm.Placement((2, 2), "yx", "bb")
    value_type = fm.DistributedType(
        fm.tensor_type("float32", (4, 16)),
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    variant_type = (
        replace(value_type, partial=fm.SBP.partial((0, 1)))
        if partial_variant else value_type
    )

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            parameter = self.input("value", value_type, id="parameter")
            output = fm.F.math.silu(parameter, name="output")
            other = self.input("value", variant_type, id="other_parameter")
            local = (
                fm.F.distributed.boxing(other, value_type, name="materialized")
                if partial_variant else other
            )
            other_output = (
                fm.F.math.mul(local, local, name="other_output")
                if variant_op == "mul" else fm.F.math.silu(local, name="other_output")
            )
            value = self.input("value", value_type, id="value")
            other_value = self.input("other_value", variant_type, id="other_value")
            call = fm.F.builtin.call(
                value, result_type=value_type, callee="layer", name="call")
            other_call = fm.F.builtin.call(
                other_value, result_type=value_type, callee="variant", name="other_call")
            self.function("main", (value, other_value), (call, other_call))
            self.function("layer", (parameter,), (output,), attrs={"reusable": True, "noinline": True})
            self.function("variant", (other,), (other_output,), attrs={
                "reusable": True,
                "noinline": True,
                **({"specialized_from": "layer", "norm_stats_boundary_specialization": (0,)} if special else {}),
            })

    return Graph().build()


def test_finalization_merges_variants_when_post_distribution_layouts_converge():
    original = _module(partial_variant=True)
    # ABI mismatch must prevent premature merging.
    assert len(finalize_norm_stats_bindings(original).functions) == 3
    propagated = propagate_post_auto_distributed_function_boundary_layouts(
        original, PyNttDistributedReshardRealizationPolicy())
    assert propagated.node_map["parameter"].type == propagated.node_map["other_parameter"].type
    result = finalize_norm_stats_bindings(propagated)

    assert tuple(f.name for f in result.functions) == ("main", "layer")
    assert result.node_map["other_call"].attrs["callee"] == "layer"
    assert result.node_map["other_call"].inputs == propagated.node_map["other_call"].inputs
    assert "other_output" not in result.node_map
    assert "other_parameter" not in result.node_map
    fm.verify_module(result)
    assert finalize_norm_stats_bindings(result) is result


def test_merges_copied_selection_points_without_leaving_dangling_records():
    module = _module()
    candidates = (fm.Candidate("small", {"block": 16}), fm.Candidate("large", {"block": 32}))
    points = tuple(
        fm.SelectionPoint(f"choice.{owner}", "unit", candidates, "small", owner=owner)
        for owner in ("output", "other_output")
    )
    records = tuple(fm.SelectionRecord(p.id, "small", "agent", "unit") for p in points)
    result = finalize_norm_stats_bindings(replace(module, selection_points=points, selections=records))

    assert len(result.functions) == 2
    assert result.selection_points == points[:1]
    assert result.selections == records[:1]
    fm.verify_module(result)


@pytest.mark.parametrize("difference", ("operator", "metadata", "function_attr", "selection", "not_specialized"))
def test_keeps_variants_with_different_compilation_contracts(difference):
    module = _module(variant_op="mul" if difference == "operator" else "silu", special=difference != "not_specialized")
    if difference == "metadata":
        module = replace(module, nodes=tuple(
            replace(n, metadata={"target_schedule": "different"}) if n.id == "other_output" else n
            for n in module.nodes
        ))
    elif difference == "function_attr":
        module = replace(module, functions=tuple(
            replace(f, attrs={**dict(f.attrs), "noinline": False}) if f.name == "variant" else f
            for f in module.functions
        ))
    elif difference == "selection":
        candidates = (fm.Candidate("small"), fm.Candidate("large"))
        points = tuple(fm.SelectionPoint(f"choice.{owner}", "unit", candidates, "small", owner=owner)
                       for owner in ("output", "other_output"))
        module = replace(module, selection_points=points, selections=(
            fm.SelectionRecord(points[0].id, "small", "agent", "unit"),
            fm.SelectionRecord(points[1].id, "large", "agent", "unit"),
        ))
    result = finalize_norm_stats_bindings(module)
    assert len(result.functions) == 3
    assert result.node_map["other_call"].attrs["callee"] == "variant"


def test_preserves_unrelated_editable_functions():
    module = _module()
    unused = fm.Function("editable", ("other_value",), ("other_value",))
    result = finalize_norm_stats_bindings(replace(module, functions=(*module.functions, unused)))
    assert "editable" in result.function_map
    assert result.function_map["editable"] == unused
    assert "variant" not in result.function_map


def test_different_call_effect_resources_prevent_merging():
    module = _module()
    tensor = module.node_map["value"].type
    helper_parameter = fm.Node("helper_parameter", "builtin.var", (), tensor, attrs={"name": "helper"})
    nodes = (helper_parameter, *(
        replace(n, op="builtin.call", attrs={"callee": "helper"}, effect=fm.effect("write", resource))
        if (resource := {"output": "cache_a", "other_output": "cache_b"}.get(n.id)) else n
        for n in module.nodes
    ))
    module = replace(module, nodes=nodes, functions=(
        *module.functions, fm.Function("helper", (helper_parameter.id,), (helper_parameter.id,)),
    ))
    fm.verify_module(module)
    assert "variant" in finalize_norm_stats_bindings(module).function_map


def test_free_variables_with_same_name_and_type_are_not_alpha_renamed():
    module = _module()
    module = replace(module, functions=tuple(
        replace(f, parameters=()) if f.name != "main" else f for f in module.functions
    ), nodes=tuple(
        replace(n, inputs=()) if n.op == "builtin.call" else n for n in module.nodes
    ))
    fm.verify_module(module)
    assert "variant" in finalize_norm_stats_bindings(module).function_map


def test_positional_parameters_cannot_be_permuted_even_when_types_agree():
    module = _module()
    tensor = module.node_map["value"].type
    extra = fm.Node("extra", "builtin.var", (), tensor, attrs={"name": "extra"})
    other_extra = replace(extra, id="other_extra")
    functions = tuple(
        replace(f, parameters=(*f.parameters, "extra" if f.name == "layer" else "other_extra"))
        if f.name != "main" else f for f in module.functions
    )
    nodes = (extra, other_extra, *(
        replace(n, inputs=(*n.inputs, "value")) if n.op == "builtin.call" else
        replace(n, inputs=("other_extra",)) if n.id == "other_output" else n
        for n in module.nodes
    ))
    module = replace(module, nodes=nodes, functions=functions)
    fm.verify_module(module)
    assert "variant" in finalize_norm_stats_bindings(module).function_map
