# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import inspect
from pathlib import Path

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates import (
    PackedQKVSemanticTIRCandidateProvider,
)
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.compiler import Compiler
from triton.flagmega.passes.constants import freeze_constant_islands


def _packed_qkv_graph() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="ntt", stage="frozen_constants")
    value_type = fm.tensor_type("bfloat16", (2, 32))
    packed_dtype = fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8))
    q_weight_type = fm.tensor_type(packed_dtype, (2, 8))
    kv_weight_type = fm.tensor_type(packed_dtype, (2, 4))
    output_dtype = fm.VectorType(fm.DType.BFLOAT16, (8,))
    result_type = fm.TupleType((
        fm.tensor_type(output_dtype, (2, 8)),
        fm.tensor_type(output_dtype, (2, 4)),
        fm.tensor_type(output_dtype, (2, 4)),
    ))
    value = builder.var("value", value_type, id="value")
    weights = tuple(
        builder.weight(
            name,
            value_type,
            source="weights",
            key=name,
            id=name,
        )
        for name, value_type in (
            ("q_weight", q_weight_type),
            ("k_weight", kv_weight_type),
            ("v_weight", kv_weight_type),
        )
    )
    none = builder.call("builtin.none", (), fm.NoneType(), id="none")
    qkv = builder.call(
        "ntt.packed_qkv_parallel_linear",
        (value, *weights, *(none for _ in range(9))),
        result_type,
        id="qkv",
        attrs={
            "num_heads": 8,
            "num_kv_heads": 4,
            "output_data_type": "bfloat16",
            "rhs_layout": "k_major",
        },
    )
    builder.function("main", (value,), (qkv,))
    return freeze_constant_islands(builder.build(entry="main"))


def test_packed_qkv_proposal_is_semantic_and_has_no_target_variant():
    output = fm.tensor_type(
        fm.VectorType(fm.DType.BFLOAT16, (8,)), (1, 16)
    )
    node = fm.Node(
        "qkv",
        "ntt.packed_qkv_parallel_linear",
        (),
        fm.TupleType((output, output, output)),
        attrs={"rhs_layout": "k_major"},
    )

    proposal = PackedQKVSemanticTIRCandidateProvider().propose(node, None)

    assert proposal is not None
    assert proposal.selection_kind == "semantic_tir"
    assert tuple(value.id for value in proposal.candidates) == (
        "semantic.ntt.packed_qkv_parallel_linear",
    )
    assert proposal.candidates[0].parameters == {}


def test_semantic_provider_has_no_machine_catalog_or_physical_geometry():
    source = Path(
        inspect.getsourcefile(PackedQKVSemanticTIRCandidateProvider) or ""
    ).read_text(encoding="utf-8").lower()

    for spelling in (
        "implementation_model",
        "context.implementations",
        "nvidia",
        "sm90",
        "block_k",
        "block_n",
        "mesh_size",
        "num_stages",
    ):
        assert spelling not in source


def test_tir_proposal_and_lowering_preserve_an_unselected_semantic_dispatch():
    target = NvidiaSm90Target()
    graph = _packed_qkv_graph()
    proposed = target.propose_tir(graph)
    point = next(value for value in proposed.selection_points if value.id == "tir.qkv")
    lowered = target.lower_to_tir(proposed)
    dispatch = fm.kernel_dispatch_for_call(lowered, lowered.node_map["qkv"])

    assert point.kind == "semantic_tir"
    assert point.default_candidate == "semantic.ntt.packed_qkv_parallel_linear"
    assert dispatch.semantic_op == "ntt.packed_qkv_parallel_linear"
    assert dispatch.semantic_candidate == point.default_candidate
    assert dispatch.microkernel is None
    assert "codegen_package_plan" not in lowered.metadata


def test_default_selects_truthful_packed_qkv_after_fused_rhs_canonicalization():
    compiled = Compiler().compile(_packed_qkv_graph()).module
    function = next(
        value for value in compiled.kernel_definitions
        if fm.kernel_dispatch_of(value) is not None
    )
    dispatch = fm.kernel_dispatch_of(function)

    assert compiled.stage == "bufferized_tir"
    assert dispatch.semantic_op == "ntt.packed_qkv_parallel_linear_fused_rhs"
    assert dispatch.semantic_candidate == "semantic.ntt.packed_qkv_parallel_linear"
    assert dispatch.microkernel.implementation == (
        "tir.qkv_parallel_linear.packed_fused_gemv"
    )
    assert dispatch.microkernel.facts["portable_triton"] is True
    assert not dispatch.microkernel.requires
    assert compiled.selection_map[f"microkernel.{function.name}"].candidate_id == (
        dispatch.microkernel.implementation
    )
    assert compiled.metadata["launch_contract"]["num_warps"] == 4
