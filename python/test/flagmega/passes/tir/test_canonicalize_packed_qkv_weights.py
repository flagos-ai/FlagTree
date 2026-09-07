# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
import inspect
from pathlib import Path

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.evaluator import DictWeightResolver, materialize_constant_assets
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.passes.tir.canonicalize_packed_qkv_weights import (
    CanonicalizePackedQKVWeightsPass,
    canonicalize_packed_qkv_weights,
)
from triton.flagmega.passes.tir import canonicalize_packed_qkv_weights as canonicalize
from triton.flagmega.passes.tir.bind_prim_function_buffers import (
    bind_prim_function_buffers,
)


def _packed_type(k: int, n: int):
    return fm.tensor_type(
        fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)),
        (k, n),
    )


def _var(name, value_type, *, function):
    return fm.Node(
        name,
        "builtin.var",
        (),
        value_type,
        attrs={"name": name},
        metadata={"function_parameter": function},
    )


def _weight(name, value_type):
    return fm.Node(
        name,
        "builtin.weight",
        (),
        value_type,
        attrs={"name": name, "source": "memory", "key": name},
    )


def _selected_module(*, independent_q_use=False, repeat_decode_call=False):
    value_type = fm.tensor_type("bfloat16", (2, 32))
    q_type = _packed_type(2, 8)
    kv_type = _packed_type(2, 4)
    q_output = fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16, (8,)), (2, 8))
    kv_output = fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16, (8,)), (2, 4))
    result_type = fm.TupleType((q_output, kv_output, kv_output))
    none_type = fm.NoneType()

    decode_value = _var("decode_value", value_type, function="decode_layer")
    decode_q = _var("decode_q", q_type, function="decode_layer")
    decode_k = _var("decode_k", kv_type, function="decode_layer")
    decode_v = _var("decode_v", kv_type, function="decode_layer")
    none = fm.Node("none", "builtin.none", (), none_type)

    argument_names = (
        "input",
        "q_weight",
        "k_weight",
        "v_weight",
        "q_bias",
        "k_bias",
        "v_bias",
        "q_input_scale",
        "k_input_scale",
        "v_input_scale",
        "q_weight_scale",
        "k_weight_scale",
        "v_weight_scale",
    )
    argument_types = (
        value_type,
        q_type,
        kv_type,
        kv_type,
        *(none_type for _ in range(9)),
    )
    output_names = ("result_0", "result_1", "result_2")
    dispatch = fm.KernelDispatch(
        semantic_op="ntt.packed_qkv_parallel_linear",
        candidate="tir.packed_qkv.reference",
        arguments=argument_names,
        outputs=output_names,
        semantic_attrs={
            "num_heads": 8,
            "num_kv_heads": 4,
            "output_data_type": "bfloat16",
            "rhs_layout": "k_major",
        },
        reads=argument_names,
        writes=output_names,
    )
    parameters = tuple(
        fm.PrimParameter(name, value_type, fm.PrimParameterRole.INPUT)
        for name, value_type in zip(argument_names, argument_types)
    ) + tuple(
        fm.PrimParameter(name, value_type, fm.PrimParameterRole.OUTPUT)
        for name, value_type in zip(output_names, result_type.fields)
    )
    kernel = fm.PrimFunction(
        "kernel_packed_qkv",
        "triton",
        parameters,
        fm.Sequential((dispatch,)),
        fm.Return(tuple(
            fm.ReturnBinding(fm.ValueRef(name, value_type), name)
            for name, value_type in zip(output_names, result_type.fields)
        )),
    )

    kernel_call = fm.Node(
        "decode_qkv",
        "tir.call",
        (
            decode_value.id,
            decode_q.id,
            decode_k.id,
            decode_v.id,
            *(none.id for _ in range(9)),
        ),
        result_type,
        attrs={"callee": kernel.name},
    )
    decode_outputs = (kernel_call.id,)
    if independent_q_use:
        decode_outputs += (decode_q.id,)
    decode = fm.Function(
        "decode_layer",
        (decode_value.id, decode_q.id, decode_k.id, decode_v.id),
        decode_outputs,
        {"reusable": True},
    )

    main_value = _var("main_value", value_type, function="main")
    q_weight = _weight("q_weight", q_type)
    k_weight = _weight("k_weight", kv_type)
    v_weight = _weight("v_weight", kv_type)
    decode_call = fm.Node(
        "decode_call",
        "tir.call",
        (main_value.id, q_weight.id, k_weight.id, v_weight.id),
        result_type if not independent_q_use else fm.TupleType((result_type, q_type)),
        attrs={"callee": decode.name},
    )
    repeated_call = replace(decode_call, id="decode_call_again")
    main = fm.Function(
        "main",
        (main_value.id,),
        (
            (decode_call.id, repeated_call.id)
            if repeat_decode_call
            else (decode_call.id,)
        ),
    )
    module = fm.IRModule(
        dialect="semantic_tir",
        stage="selected_tir",
        nodes=(
            decode_value,
            decode_q,
            decode_k,
            decode_v,
            none,
            kernel_call,
            main_value,
            q_weight,
            k_weight,
            v_weight,
            decode_call,
            *((repeated_call,) if repeat_decode_call else ()),
        ),
        functions=(main, decode),
        entry="main",
        prim_functions=(kernel,),
    )
    return fm.verify_module(module)


def _with_distributed_weights(module):
    placement = fm.Placement((2, 2), "yx", "bb")
    policies = (
        fm.SBP.split_contiguous((0,)),
        fm.SBP.split_contiguous((1,)),
    )
    replacements = {}
    for name in ("q_weight", "k_weight", "v_weight"):
        node = module.node_map[name]
        replacements[name] = replace(
            node,
            type=fm.DistributedType(node.type, policies, placement),
        )
    for name in ("decode_q", "decode_k", "decode_v"):
        source = replacements[name.removeprefix("decode_") + "_weight"]
        replacements[name] = replace(module.node_map[name], type=source.type)
    kernel = module.prim_functions[0]
    parameters = list(kernel.parameters)
    for parameter_index, source_name in zip(
        (1, 2, 3),
        ("q_weight", "k_weight", "v_weight"),
    ):
        parameters[parameter_index] = replace(
            parameters[parameter_index], type=replacements[source_name].type
        )
    return replace(
        module,
        nodes=tuple(replacements.get(node.id, node) for node in module.nodes),
        prim_functions=(replace(kernel, parameters=tuple(parameters)),),
    )


def _with_selected_distribution_contract(module):
    kernel = module.prim_functions[0]
    dispatch = fm.kernel_dispatch_of(kernel)
    assert dispatch is not None
    selected = replace(
        dispatch,
        semantic_parameters={
            **dict(dispatch.semantic_parameters),
            "distribution": {
                "placement": {"hierarchy": (2, 2), "axis_names": "yx"},
                "input_types": kernel.runtime_parameter_types,
                "output_type": kernel.runtime_return_type,
            },
        },
    )
    return replace(
        module,
        prim_functions=(
            replace(kernel, body=fm.Sequential((selected,))),
        ),
    )


def test_canonicalization_rewrites_reusable_graph_and_leaf_kernel_abis():
    result = canonicalize_packed_qkv_weights(_selected_module())

    decode = result.function_map["decode_layer"]
    assert decode.parameters == ("decode_value", "decode_q.qkv_fused")
    fused_parameter = result.node_map[decode.parameters[1]]
    assert fused_parameter.type == _packed_type(2, 16)

    main_call = result.node_map["decode_call"]
    assert main_call.inputs == ("main_value", "decode_call.qkv_fused")
    fused_weight = result.node_map[main_call.inputs[1]]
    assert fused_weight.op == "tensors.concat"
    assert fused_weight.inputs == ("q_weight", "k_weight", "v_weight")
    assert fused_weight.attrs == {"axis": 1}

    kernel_call = result.node_map["decode_qkv"]
    assert kernel_call.inputs[:2] == ("decode_value", "decode_q.qkv_fused")
    assert kernel_call.inputs[2:] == ("none",) * 9
    kernel = result.prim_function_map["kernel_packed_qkv"]
    assert tuple(value.name for value in kernel.runtime_parameters[:2]) == (
        "input",
        "q_weight_qkv_fused",
    )
    assert len(kernel.runtime_parameters) == 11
    dispatch = fm.kernel_dispatch_of(kernel)
    assert dispatch is not None
    assert dispatch.semantic_op == "ntt.packed_qkv_parallel_linear_fused_rhs"
    assert dispatch.arguments[:2] == ("input", "q_weight_qkv_fused")
    assert dispatch.semantic_attrs["projection_n_capacities"] == (64, 32, 32)
    fm.verify_module(result)


def test_canonicalization_rejects_an_independent_weight_reference():
    with pytest.raises(IRVerificationError, match="independent reference"):
        canonicalize_packed_qkv_weights(_selected_module(independent_q_use=True))


def test_canonicalization_reuses_one_derived_weight_for_repeated_calls():
    result = canonicalize_packed_qkv_weights(
        _selected_module(repeat_decode_call=True)
    )

    fused = tuple(
        node for node in result.nodes
        if node.metadata.get("canonicalized_from")
        == ("q_weight", "k_weight", "v_weight")
    )
    assert len(fused) == 1
    assert result.node_map["decode_call"].inputs[1] == fused[0].id
    assert result.node_map["decode_call_again"].inputs[1] == fused[0].id


def test_canonicalization_python_checkpoint_round_trips(tmp_path):
    result = canonicalize_packed_qkv_weights(_selected_module())
    checkpoint = fm.emit_module(result, tmp_path / "canonical_qkv.py")

    resumed = fm.load_module(checkpoint)
    assert resumed.semantic_hash == result.semantic_hash


def test_canonicalization_refreezes_one_editable_fused_constant_recipe(tmp_path):
    frozen = freeze_constant_islands(_selected_module())
    assert frozen.metadata["constant_phase"] == "frozen"

    result = canonicalize_packed_qkv_weights(frozen)
    main_call = result.node_map["decode_call"]
    fused_asset = result.node_map[main_call.inputs[1]]
    assert fused_asset.op == "builtin.const_asset"
    recipe = next(
        value for value in result.constant_recipes
        if value.id == fused_asset.attrs["recipe"]
    )
    fused_node = recipe.node_map[str(fused_asset.attrs["output"])]
    assert fused_node.op == "tensors.concat"
    assert fused_node.inputs == ("q_weight", "k_weight", "v_weight")
    assert not {
        node.id for node in result.nodes
        if node.id in {"q_weight", "k_weight", "v_weight"}
        and node.op == "builtin.const_asset"
    }

    shape = (2, 8, 8, 2, 8)
    q = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.bfloat16).reshape(shape)
    kv = torch.zeros((2, 4, 8, 2, 8), dtype=torch.bfloat16)
    assets = materialize_constant_assets(
        result,
        DictWeightResolver({"q_weight": q, "k_weight": kv, "v_weight": kv}),
    )
    torch.testing.assert_close(
        assets[fused_asset.id],
        torch.cat((q, kv, kv), dim=1),
    )
    checkpoint = fm.emit_module(result, tmp_path / "frozen_canonical_qkv.py")
    assert fm.load_module(checkpoint).semantic_hash == result.semantic_hash


def test_pass_object_uses_nncase_pass_name_and_contract():
    transform = CanonicalizePackedQKVWeightsPass()

    assert transform.name == "CanonicalizePackedQKVWeights"
    assert transform.run(_selected_module()).node_map["decode_call"].inputs == (
        "main_value",
        "decode_call.qkv_fused",
    )


def test_canonicalizer_uses_named_parameter_info_instead_of_abi_numbers():
    source = Path(inspect.getsourcefile(canonicalize) or "").read_text(
        encoding="utf-8"
    )

    assert "PackedQKVParallelLinear.q_weight" in source
    assert "PackedQKVParallelLinear.k_weight" in source
    assert "PackedQKVParallelLinear.v_weight" in source
    assert "(1, 2, 3)" not in source
    assert "!= 13" not in source


def test_canonicalization_rejects_non_k_major_selected_kernel():
    module = _selected_module()
    kernel = module.prim_functions[0]
    dispatch = fm.kernel_dispatch_of(kernel)
    assert dispatch is not None
    incompatible = replace(
        dispatch,
        semantic_attrs={**dict(dispatch.semantic_attrs), "rhs_layout": "n_major"},
    )
    module = replace(
        module,
        prim_functions=(replace(kernel, body=fm.Sequential((incompatible,))),),
    )

    with pytest.raises(IRVerificationError, match="requires K-major"):
        canonicalize_packed_qkv_weights(module)


def test_canonicalization_rejects_unfused_entry_weight_parameters():
    module = _selected_module()
    decode = module.function_map["decode_layer"]
    module = replace(module, entry=decode.name)

    with pytest.raises(IRVerificationError, match="Entry ABI cannot expose"):
        canonicalize_packed_qkv_weights(module)


def test_canonicalization_materializes_distributed_sources_owner_major():
    module = _with_distributed_weights(_selected_module())

    result = canonicalize_packed_qkv_weights(module)

    call = result.node_map["decode_call"]
    fused = result.node_map[call.inputs[1]]
    assert fused.op == "tensors.concat"
    assert fused.attrs == {"axis": 2}
    assert fused.type == fm.tensor_type(
        fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)),
        (4, 1, 8),
    )
    sources = tuple(result.node_map[value] for value in fused.inputs)
    assert all(
        value.op == "distributed.materialize_local_shards" for value in sources
    )
    assert tuple(value.inputs[0] for value in sources) == (
        "q_weight",
        "k_weight",
        "v_weight",
    )
    fm.verify_module(result)


def test_canonicalization_rewrites_the_selected_distributed_kernel_abi():
    module = _with_selected_distribution_contract(
        _with_distributed_weights(_selected_module())
    )

    result = canonicalize_packed_qkv_weights(module)
    kernel = result.prim_functions[0]
    dispatch = fm.kernel_dispatch_of(kernel)
    assert dispatch is not None
    distribution = dispatch.parameters["distribution"]
    fused_index = tuple(
        value.name for value in kernel.runtime_parameters
    ).index("q_weight_qkv_fused")

    assert len(distribution["input_types"]) == len(kernel.runtime_parameters)
    assert distribution["input_types"] == kernel.runtime_parameter_types
    assert distribution["input_types"][fused_index] == fm.tensor_type(
        fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)),
        (4, 1, 8),
    )
    assert len(kernel.runtime_parameters) == 11
    bind_prim_function_buffers(result)


def test_distributed_fused_rhs_is_a_real_frozen_materialization():
    frozen = freeze_constant_islands(
        _with_distributed_weights(_selected_module())
    )
    result = canonicalize_packed_qkv_weights(frozen)
    fused_asset = result.node_map[result.node_map["decode_call"].inputs[1]]
    recipe = next(
        value for value in result.constant_recipes
        if value.id == fused_asset.attrs["recipe"]
    )
    assert sum(
        node.op == "distributed.materialize_local_shards"
        for node in recipe.nodes
    ) == 3

    q = torch.arange(2 * 8 * 8 * 2 * 8, dtype=torch.bfloat16).reshape(
        2, 8, 8, 2, 8
    )
    k = torch.arange(2 * 4 * 8 * 2 * 8, dtype=torch.bfloat16).reshape(
        2, 4, 8, 2, 8
    )
    v = -k
    assets = materialize_constant_assets(
        result,
        DictWeightResolver({"q_weight": q, "k_weight": k, "v_weight": v}),
    )
    expected = []
    for row in range(2):
        for column in range(2):
            expected.append(torch.cat((
                q[row:row + 1, column * 4:(column + 1) * 4],
                k[row:row + 1, column * 2:(column + 1) * 2],
                v[row:row + 1, column * 2:(column + 1) * 2],
            ), dim=1))
    torch.testing.assert_close(
        assets[fused_asset.id],
        torch.stack(expected),
    )
