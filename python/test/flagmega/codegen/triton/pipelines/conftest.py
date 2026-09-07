# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.selection import override_plan


_PIPELINE = "tir.dense_matmul.tensor_descriptor_smem_pipeline_gemv"
_AUX_PIPELINE = (
    "tir.dense_matmul.tensor_descriptor_smem_pipeline_aux_gemv"
)


def _compile_pipeline_module(
    *,
    reusable: bool,
    implementation: str = _PIPELINE,
    worker_depth: int = 1,
    wrapper_depth: int = 0,
) -> fm.IRModule:
    if worker_depth < 1:
        raise ValueError("worker_depth must be positive")
    if not reusable and worker_depth != 1:
        raise ValueError("worker_depth is supported only for reusable modules")
    if wrapper_depth < 0 or (wrapper_depth and not reusable):
        raise ValueError("wrapper_depth requires a reusable module and must be nonnegative")
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("bfloat16", (1, 128))
    weight_type = fm.tensor_type("bfloat16", (128, 128))

    if reusable:
        worker_value = builder.var(
            "worker_value", value_type, id="worker_value"
        )
        worker_weights = tuple(
            builder.var(
                "worker_weight" if worker_depth == 1 else f"worker_weight_{index}",
                weight_type,
                id=(
                    "worker_weight"
                    if worker_depth == 1
                    else f"worker_weight_{index}"
                ),
            )
            for index in range(worker_depth)
        )
        projection = worker_value
        projection_ids = []
        for index, worker_weight in enumerate(worker_weights):
            projection_id = (
                "projection" if worker_depth == 1 else f"projection_{index}"
            )
            projection = builder.call(
                "math.matmul",
                (projection, worker_weight),
                value_type,
                id=projection_id,
                attrs={"transpose_a": False, "transpose_b": True},
            )
            projection_ids.append(projection_id)
        builder.function(
            "worker",
            (worker_value, *worker_weights),
            (projection,),
            attrs={
                "calling_convention": "device",
                "noinline": True,
                "reusable": True,
            },
        )
        callee = "worker"
        for index in range(wrapper_depth):
            name = f"wrapper_{index}"
            parameters = tuple(
                builder.var(f"{name}_arg_{parameter_index}", parameter.type,
                            id=f"{name}_arg_{parameter_index}")
                for parameter_index, parameter in enumerate((worker_value, *worker_weights))
            )
            result = builder.call(
                "builtin.call", parameters, value_type, id=f"{name}_call",
                attrs={"callee": callee},
            )
            builder.function(name, parameters, (result,), attrs={
                "calling_convention": "device", "noinline": True, "reusable": True,
            })
            callee = name
        value = builder.var("value", value_type, id="value")
        first_weights = tuple(
            builder.weight(
                "first_weight" if worker_depth == 1 else f"first_weight_{index}",
                weight_type,
                source="memory",
                key=(
                    "first_weight"
                    if worker_depth == 1
                    else f"first_weight_{index}"
                ),
                id=(
                    "first_weight"
                    if worker_depth == 1
                    else f"first_weight_{index}"
                ),
            )
            for index in range(worker_depth)
        )
        second_weights = tuple(
            builder.weight(
                "second_weight" if worker_depth == 1 else f"second_weight_{index}",
                weight_type,
                source="memory",
                key=(
                    "second_weight"
                    if worker_depth == 1
                    else f"second_weight_{index}"
                ),
                id=(
                    "second_weight"
                    if worker_depth == 1
                    else f"second_weight_{index}"
                ),
            )
            for index in range(worker_depth)
        )
        first = builder.call(
            "builtin.call",
            (value, *first_weights),
            value_type,
            id="first",
            attrs={"callee": callee},
        )
        second = builder.call(
            "builtin.call",
            (first, *second_weights),
            value_type,
            id="second",
            attrs={"callee": callee},
        )
        builder.function("main", (value,), (second,))
        vector_points = tuple(
            f"vectorization.{projection_id}"
            for projection_id in projection_ids
        )
        packing_points = tuple(
            f"packing.{projection_id}"
            for projection_id in projection_ids
        )
        tir_points = tuple(f"tir.{projection_id}" for projection_id in projection_ids)
    else:
        value = builder.var("value", value_type, id="value")
        weight = builder.weight(
            "weight",
            weight_type,
            source="memory",
            key="weight",
            id="weight",
        )
        projection = builder.call(
            "math.matmul",
            (value, weight),
            value_type,
            id="projection",
            attrs={"transpose_a": False, "transpose_b": True},
        )
        builder.function("main", (value,), (projection,))
        vector_points = ("vectorization.projection",)
        packing_points = ("packing.projection",)
        tir_points = ("tir.projection",)

    compiler = Compiler()
    proposed_vector = compiler.compile(
        builder.build(entry="main"), stop_after="propose-vectorization"
    ).module
    vectorized = compiler.run_stage(
        proposed_vector,
        "apply-vectorization",
        plan=override_plan(
            proposed_vector,
            tuple(
                (point, "vectorization.matmul.n")
                for point in vector_points
            ),
        ),
    ).module
    proposed_packing = Compiler().compile(
        vectorized, stop_after="propose-packing"
    ).module
    available_packing_points = {
        point.id for point in proposed_packing.selection_points
    }
    logical = Compiler().run_stage(
        proposed_packing,
        "apply-packing",
        plan=override_plan(
            proposed_packing,
            tuple(
                (point, "packing.logical")
                for point in packing_points
                if point in available_packing_points
            ),
        ),
    ).module
    proposed_tir = Compiler().compile(
        logical, stop_after="propose-tir"
    ).module
    selected_tir = Compiler().run_stage(
        proposed_tir,
        "lower-tir",
        plan=override_plan(
            proposed_tir,
            tuple((point, implementation) for point in tir_points),
        ),
    ).module
    return Compiler().compile(selected_tir).module


@pytest.fixture
def compile_pipeline_module():
    return _compile_pipeline_module
