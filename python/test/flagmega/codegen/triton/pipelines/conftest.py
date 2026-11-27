# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
from dataclasses import replace

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
    program_dag: bool = False,
    whole_program: bool = False,
) -> fm.IRModule:
    if worker_depth < 1:
        raise ValueError("worker_depth must be positive")
    if not reusable and worker_depth != 1:
        raise ValueError("worker_depth is supported only for reusable modules")
    if wrapper_depth < 0 or (wrapper_depth and not reusable):
        raise ValueError("wrapper_depth requires a reusable module and must be nonnegative")
    # Author the supported logical DenseMatMul schedule at its typed resume
    # boundary. AutoPacking now changes RHS storage to an N-major vector
    # layout, which is a different descriptor contract from this fixture.
    builder = fm.IRBuilder(dialect="high_level", stage="vector_contracts_lowered")
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
                metadata={"selected_vectorization": "vectorization.matmul.n",
                          "selected_vector_axes": (1,), "selected_vector_lanes": (8,)},
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
            metadata={"selected_vectorization": "vectorization.matmul.n",
                      "selected_vector_axes": (1,), "selected_vector_lanes": (8,)},
        )
        builder.function("main", (value,), (projection,))
        tir_points = ("tir.projection",)

    compiler = Compiler()
    proposed_tir = compiler.compile(builder.build(entry="main"), stop_after="propose-tir").module
    selected_tir = Compiler().run_stage(
        proposed_tir,
        "lower-tir",
        plan=override_plan(
            proposed_tir,
            tuple((point, implementation) for point in tir_points),
        ),
    ).module
    if program_dag or whole_program:
        from triton.flagmega.passes import FunctionalPass, PassManager
        from triton.flagmega.passes.functions import function_nodes

        selected_tir = Compiler().compile(selected_tir, stop_after="select-microkernels").module
        calls = [n.id for n in function_nodes(selected_tir, selected_tir.function_map[selected_tir.entry])
                 if n.op in {"tir.call", "builtin.call", "tir.kernel"}]
        symbols = {call: f"program_{index}" for index, call in enumerate(calls)}
        programs = tuple(dict(symbol=symbols[call], calls=(call,), num_warps=4,
                              depends_on=(symbols[calls[index-1]],) if index else ())
                         for index, call in enumerate(calls))
        if whole_program:
            programs = (dict(symbol="whole", calls=tuple(calls), num_warps=4),)
        selected_tir = PassManager().add(FunctionalPass("TestPoolRecordPrograms", lambda m: replace(
            m, metadata={**m.metadata, "kernel_program_partition": programs}))).run(selected_tir).module
    return Compiler().compile(selected_tir).module


@pytest.fixture
def compile_pipeline_module():
    return _compile_pipeline_module
