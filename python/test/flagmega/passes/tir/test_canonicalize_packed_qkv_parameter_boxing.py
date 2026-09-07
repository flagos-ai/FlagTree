# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

"""Regression tests for packed-QKV ABI propagation through selected boxing.

AutoDistribution may initially express a reusable function's Q/K/V parameter
contract as three parameter-to-local boxing calls.  Canonicalization must
absorb those calls into the one fused caller-side readonly-data recipe; it may
not leave newly-created semantic compute after TIR selection.
"""

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.passes.tir.canonicalize_packed_qkv_weights import (
    canonicalize_packed_qkv_weights,
)

from .test_canonicalize_packed_qkv_weights import (
    _selected_module,
    _with_distributed_weights,
)


def _boxing_prim_function(name, source_type, result_type):
    dispatch = fm.KernelDispatch(
        semantic_op="distributed.boxing",
        semantic_candidate="semantic.distributed.boxing",
        arguments=("input",),
        outputs=("result",),
        reads=("input",),
        writes=("result",),
    )
    return fm.PrimFunction(
        name,
        "triton",
        (
            fm.PrimParameter("input", source_type, fm.PrimParameterRole.INPUT),
            fm.PrimParameter("result", result_type, fm.PrimParameterRole.OUTPUT),
        ),
        fm.Sequential((dispatch,)),
        fm.Return((fm.ReturnBinding(fm.ValueRef("result", result_type), "result"),)),
    )


def _selected_module_with_parameter_boxing():
    module = _with_distributed_weights(_selected_module())
    replacements = {}
    boxing_functions = []
    adapters = []
    for projection in ("q", "k", "v"):
        parameter_id = f"decode_{projection}"
        parameter = module.node_map[parameter_id]
        selected_type = parameter.type
        assert isinstance(selected_type, fm.DistributedType)
        source_type = fm.DistributedType(
            selected_type.tensor,
            tuple(fm.SBP.broadcast() for _ in selected_type.axis_policies),
            selected_type.placement,
        )
        replacements[parameter_id] = replace(parameter, type=source_type)
        weight_id = f"{projection}_weight"
        replacements[weight_id] = replace(
            module.node_map[weight_id], type=source_type
        )
        function_name = f"boxing_{projection}_weight"
        adapter = fm.Node(
            f"{parameter_id}.to_selected_layout",
            "tir.call",
            (parameter_id,),
            selected_type,
            attrs={"callee": function_name},
        )
        adapters.append(adapter)
        boxing_functions.append(
            _boxing_prim_function(function_name, source_type, selected_type)
        )

    kernel_call = module.node_map["decode_qkv"]
    replacements[kernel_call.id] = replace(
        kernel_call,
        inputs=(
            kernel_call.inputs[0],
            *(adapter.id for adapter in adapters),
            *kernel_call.inputs[4:],
        ),
    )
    nodes = []
    for node in module.nodes:
        if node.id == kernel_call.id:
            nodes.extend(adapters)
        nodes.append(replacements.get(node.id, node))
    return fm.verify_module(replace(
        module,
        nodes=tuple(nodes),
        prim_functions=(*module.prim_functions, *boxing_functions),
    ))


def test_parameter_boxing_is_absorbed_without_post_selection_compute_nodes():
    module = freeze_constant_islands(_selected_module_with_parameter_boxing())

    result = canonicalize_packed_qkv_weights(module)

    decode = result.function_map["decode_layer"]
    assert decode.parameters == ("decode_value", "decode_q.qkv_fused")
    assert not {
        node.id
        for node in result.nodes
        if node.id.endswith(".to_selected_layout")
    }
    assert not {
        node.op
        for node in result.nodes
        if node.op in {"distributed.materialize_local_shards", "tensors.concat"}
    }
    fused_asset = result.node_map[result.node_map["decode_call"].inputs[1]]
    assert fused_asset.op == "builtin.const_asset"
    recipe = next(
        value
        for value in result.constant_recipes
        if value.id == str(fused_asset.attrs["recipe"])
    )
    assert sum(
        node.op == "distributed.materialize_local_shards"
        for node in recipe.nodes
    ) == 3
    assert sum(node.op == "tensors.concat" for node in recipe.nodes) == 1
    fm.verify_module(result)
