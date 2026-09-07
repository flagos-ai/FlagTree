# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.implementation import (
    TritonImplementation,
    TritonImplementationModel,
)
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError, IRVerificationError

import pytest


def _block_fp8_module():
    builder = fm.IRBuilder(dialect="high_level", stage="frozen_constants")
    lhs = builder.var("lhs", fm.tensor_type("bfloat16", (1, 128)), id="lhs")
    weight = builder.weight(
        "weight",
        fm.tensor_type("float8_e4m3fn", (128, 128)),
        source="weights",
        key="weight",
        id="weight",
    )
    scale = builder.weight(
        "scale",
        fm.tensor_type("float32", (1, 1)),
        source="weights",
        key="scale",
        id="scale",
    )
    output = builder.call(
        "math.block_scaled_matmul",
        (lhs, weight, scale),
        fm.tensor_type("bfloat16", (1, 128)),
        id="output",
        attrs={"weight_block_n": 128, "weight_block_k": 128},
    )
    builder.function("main", (lhs,), (output,))
    return builder.build(entry="main")


def test_target_injected_model_owns_variant_availability_geometry_and_default():
    model = TritonImplementationModel(
        implementations=(TritonImplementation(
            "tir.block_fp8.simt",
            "block_fp8",
            "simt",
            {"tile_n": 7},
            requires=("fp8",),
        ),),
        preferences={"block_fp8": ("tir.block_fp8.simt",)},
        name="unit-machine",
    )
    target = NvidiaSm90Target(triton_implementation_model=model)

    proposed = target.propose_tir(_block_fp8_module())
    point = next(value for value in proposed.selection_points if value.id == "tir.output")

    assert tuple(value.id for value in point.candidates) == ("tir.block_fp8.simt",)
    assert point.default_candidate == "tir.block_fp8.simt"
    assert point.candidates[0].parameters["tile_n"] == 7


def test_selected_tir_records_and_revalidates_the_concrete_implementation_model():
    selected = Compiler().compile(
        _block_fp8_module(), stop_after="lower-tir"
    ).module
    default = NvidiaSm90Target().triton_implementation_model
    snapshot = selected.metadata["target_implementation_model"]

    assert snapshot == default.snapshot()
    changed = TritonImplementationModel(
        implementations=default.implementations,
        preferences=default.preferences,
        name="same-machine-new-implementation-table",
    )
    with pytest.raises(IRVerificationError, match="implementation-model snapshot differs"):
        NvidiaSm90Target(triton_implementation_model=changed).verify(selected)


def test_selected_tir_records_and_revalidates_target_owned_package_plan():
    selected = Compiler().compile(
        _block_fp8_module(), stop_after="lower-tir"
    ).module
    package_plan = dict(selected.metadata["codegen_package_plan"])

    assert package_plan["schema"] == "flagmega.triton-package-plan/v1"
    assert package_plan["kind"] == "tir_call_graph"
    assert package_plan["profile"] == "nvidia_sm90"
    assert package_plan["calls"][0]["call"] == "output"
    assert package_plan["calls"][0]["family"] == "block_fp8"
    assert package_plan["kernel_templates"] == ({
        "kernel": "block_fp8",
        "variant": "mma",
    },)
    assert package_plan["launch"]["num_warps"] == 4
    tampered = replace(
        selected,
        metadata={
            **dict(selected.metadata),
            "codegen_package_plan": {**package_plan, "profile": "edited"},
        },
    )
    with pytest.raises(IRVerificationError, match="package plan differs"):
        NvidiaSm90Target().verify(tampered)


def test_generic_provider_enumerates_target_contract_without_knowing_candidate_id():
    builder = fm.IRBuilder(dialect="high_level", stage="frozen_constants")
    lhs = builder.var("lhs", fm.tensor_type("bfloat16", (1, 128)), id="lhs")
    rhs = builder.weight(
        "rhs",
        fm.tensor_type("bfloat16", (64, 128)),
        source="weights",
        key="rhs",
        id="rhs",
    )
    output = builder.call(
        "math.matmul",
        (lhs, rhs),
        fm.tensor_type("bfloat16", (1, 64)),
        id="output",
        attrs={"transpose_a": False, "transpose_b": True},
    )
    builder.function("main", (lhs,), (output,))
    module = builder.build(entry="main")
    model = TritonImplementationModel(
        implementations=(TritonImplementation(
            "machine.linear.alpha",
            "dense_matmul",
            "custom_kernel",
            {"block_k": 37, "tile_n": 1},
            contract={
                "input_kind": "logical",
                "epilogue": "none",
                "vectorization_kind": "scalar",
                "distribution_kind": "canonical",
            },
        ),),
        preferences={"dense_matmul": ("machine.linear.alpha",)},
        name="unit-renamed-catalog",
    )

    proposed = NvidiaSm90Target(triton_implementation_model=model).propose_tir(module)
    point = next(value for value in proposed.selection_points if value.id == "tir.output")

    assert tuple(candidate.id for candidate in point.candidates) == (
        "machine.linear.alpha",
    )
    assert point.candidates[0].parameters == {
        "family": "dense_matmul",
        "variant": "custom_kernel",
        "block_k": 37,
        "tile_n": 1,
        "distribution_schedule": {"kind": "canonical"},
        "vector_schedule": {
            "contract": {
                "kind": "scalar",
                "axes": (),
                "lanes": (),
                "lane_count": 1,
            },
            "lowering": "output_tile",
            "physical": {"tile_n": 1},
        },
    }


def test_model_rejects_preference_that_crosses_family_boundary():
    implementation = TritonImplementation(
        "machine.add", "elementwise", "add",
        contract={"semantic_op": "math.add"},
    )

    with pytest.raises(CodegenError, match="another family"):
        TritonImplementationModel(
            implementations=(implementation,),
            preferences={"dense_matmul": ("machine.add",)},
            name="invalid",
        )


def test_catalog_resource_claims_have_first_class_contracts():
    model = NvidiaSm90Target().triton_implementation_model

    for implementation in model.implementations:
        tokens = set(implementation.variant.split("_"))
        if "smem" in tokens or "shared" in tokens:
            assert implementation.shared_workspaces, implementation.id
        if "pipeline" in tokens:
            assert implementation.transfer_pipeline is not None, implementation.id
        if "async_copy" in implementation.requires:
            assert implementation.transfer_pipeline is not None, implementation.id
