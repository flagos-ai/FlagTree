# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.kernel_dispatch import selected_kernel_node
from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir.tir.serialization import tir_from_data, tir_to_data


def _dispatch(
    *, microkernel: fm.TIRMicroKernelSelection | None = None
) -> fm.KernelDispatch:
    return fm.T.kernel_dispatch(
        semantic_op="ntt.packed_qkv_parallel_linear_fused_rhs",
        semantic_candidate="ntt.packed_qkv_parallel_linear",
        arguments=("input", "weight"),
        outputs=("output",),
        semantic_parameters={"rhs_layout": "k_major"},
        semantic_facts={"m": 1, "k": 2048, "n": 6144},
        microkernel=microkernel,
        reads=("input", "weight"),
        writes=("output",),
    )


def _module(dispatch: fm.KernelDispatch) -> fm.IRModule:
    input_type = fm.tensor_type("bfloat16", (1, 2048))
    weight_type = fm.tensor_type("bfloat16", (2048, 6144))
    output_type = fm.tensor_type("bfloat16", (1, 6144))
    function = fm.T.prim_function(
        "packed_qkv",
        "triton",
        (
            fm.T.prim_parameter("input", input_type, fm.T.PrimParameterRole.INPUT),
            fm.T.prim_parameter("weight", weight_type, fm.T.PrimParameterRole.INPUT),
            fm.T.prim_parameter("output", output_type, fm.T.PrimParameterRole.OUTPUT),
        ),
        fm.T.sequential((dispatch,)),
        fm.T.return_((fm.T.return_binding(fm.T.value_ref("output", output_type), "output"),)),
    )
    builder = fm.IRBuilder(dialect="semantic_tir", stage="tir_canonicalized")
    source = builder.var("source", input_type, id="source")
    weight = builder.weight(
        "weight", weight_type, source="weights.safetensors", key="weight", id="weight"
    )
    builder.prim_function(function)
    result = builder.call(
        "tir.call", (source, weight), output_type, id="result", attrs={"callee": "packed_qkv"}
    )
    builder.function("main", (source,), (result,))
    return builder.build(
        entry="main",
        selection_points=(fm.SelectionPoint(
            "tir.result",
            "tir",
            (fm.Candidate("ntt.packed_qkv_parallel_linear"),),
            "ntt.packed_qkv_parallel_linear",
            "result",
        ),),
        selections=(fm.SelectionRecord(
            "tir.result",
            "ntt.packed_qkv_parallel_linear",
            "default-policy",
            "test",
        ),),
    )


def _microkernel(implementation: str = "triton.packed_qkv.blocked") -> fm.TIRMicroKernelSelection:
    return fm.T.microkernel_selection(
        implementation=implementation,
        family="packed_qkv_parallel_linear",
        variant="blocked",
        parameters={"block_m": 16, "block_n": 128},
        facts={"estimated_cycles": 42},
        requires=("tensor_core",),
    )


def test_semantic_dispatch_is_valid_before_microkernel_selection():
    dispatch = _dispatch()

    assert dispatch.semantic_candidate == "ntt.packed_qkv_parallel_linear"
    assert dispatch.microkernel is None
    assert dispatch.candidate == dispatch.semantic_candidate

    with pytest.raises(CodegenError, match="no selected microkernel"):
        selected_kernel_node(_module(dispatch), "result")


def test_legacy_dispatch_constructor_upgrades_target_data():
    dispatch = fm.T.kernel_dispatch(
        semantic_op="ntt.matmul",
        candidate="triton.matmul.blocked",
        arguments=("lhs", "rhs"),
        outputs=("output",),
        parameters={"family": "matmul", "variant": "blocked", "block_m": 16},
        facts={"requires": "tensor_core", "estimated_cycles": 7},
    )

    assert dispatch.semantic_candidate == "triton.matmul.blocked"
    assert dispatch.microkernel is not None
    assert dispatch.microkernel.requires == ("tensor_core",)
    assert dispatch.candidate == "triton.matmul.blocked"
    assert dispatch.parameters["block_m"] == 16


def test_split_tir_data_round_trip_and_legacy_data_upgrade():
    dispatch = _dispatch(microkernel=_microkernel())
    restored = tir_from_data(tir_to_data(dispatch))

    assert restored == dispatch
    assert restored.microkernel.implementation == "triton.packed_qkv.blocked"

    legacy = tir_from_data({
        "kind": "kernel_dispatch",
        "semantic_op": "ntt.matmul",
        "candidate": "triton.matmul.blocked",
        "arguments": {"$tuple": ["lhs", "rhs"]},
        "outputs": {"$tuple": ["output"]},
        "parameters": {"family": "matmul", "variant": "blocked"},
        "facts": {"requires": "tensor_core"},
    })
    assert legacy.microkernel is not None
    assert legacy.microkernel.requires == ("tensor_core",)


def test_editable_python_round_trip_exposes_both_selection_layers(tmp_path: Path):
    module = _module(_dispatch(microkernel=_microkernel()))
    checkpoint = fm.emit_module(module, tmp_path / "tir_selected.py")
    loaded = fm.load_module(checkpoint)
    source = checkpoint.read_text(encoding="utf-8")

    assert loaded.semantic_hash == module.semantic_hash
    assert "T.microkernel_selection(" in source
    assert "semantic_candidate='ntt.packed_qkv_parallel_linear'" in source


def test_semantic_and_microkernel_parameters_must_not_conflict():
    dispatch = _dispatch(microkernel=_microkernel())
    conflicting = fm.T.kernel_dispatch(
        semantic_op=dispatch.semantic_op,
        semantic_candidate=dispatch.semantic_candidate,
        arguments=dispatch.arguments,
        outputs=dispatch.outputs,
        semantic_parameters={"block_m": 32},
        microkernel=dispatch.microkernel,
    )

    with pytest.raises(IRSchemaError, match="conflicting values"):
        _ = conflicting.resolved_parameters
