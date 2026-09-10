# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
# Shared pytest fixtures for descriptor code generation tests.

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateProviderRegistry,
    build_search_graph,
    solve_search_graph,
)
from triton.flagmega.passes.auto_distributed.materializer import (
    distribution_selection_state,
)
from triton.flagmega.selection import override_plan


def _two_call_descriptor_module(*, transpose_b: bool = False) -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("bfloat16", (1, 128))
    weight_type = fm.tensor_type("bfloat16", (128, 128))

    worker_value = builder.var("worker_value", value_type, id="worker_value")
    worker_weight = builder.var("worker_weight", weight_type, id="worker_weight")
    projection = builder.call(
        "math.matmul",
        (worker_value, worker_weight),
        value_type,
        id="projection",
        attrs={"transpose_a": False, "transpose_b": transpose_b},
    )
    builder.function(
        "worker",
        (worker_value, worker_weight),
        (projection,),
        attrs={"calling_convention": "device", "noinline": True, "reusable": True},
    )

    value = builder.var("value", value_type, id="value")
    first_weight = builder.var("first_weight", weight_type, id="first_weight")
    second_weight = builder.var("second_weight", weight_type, id="second_weight")
    first = builder.call(
        "builtin.call", (value, first_weight), value_type,
        id="first", attrs={"callee": "worker"},
    )
    second = builder.call(
        "builtin.call", (first, second_weight), value_type,
        id="second", attrs={"callee": "worker"},
    )
    builder.function("main", (value, first_weight, second_weight), (second,))

    compiler = Compiler()
    proposed_vector = compiler.compile(
        builder.build(entry="main"), stop_after="propose-vectorization"
    ).module
    vectorized = compiler.run_stage(
        proposed_vector,
        "apply-vectorization",
        plan=override_plan(
            proposed_vector,
            (("vectorization.projection", "vectorization.matmul.n"),),
        ),
    ).module
    proposed_tir = Compiler().compile(
        vectorized, stop_after="propose-tir"
    ).module
    selected_tir = Compiler().run_stage(
        proposed_tir,
        "lower-tir",
        plan=override_plan(
            proposed_tir,
            (("tir.projection", "tir.dense_matmul.tensor_descriptor_gemv"),),
        ),
    ).module
    return Compiler().compile(selected_tir).module


def _packed_descriptor_pipeline_module(
    implementation: str = (
        "tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv"
    ),
    *, k: int = 2048, n: int = 2048,
    output_policy: fm.SBPSplit | None = None,
    compiler: Compiler | None = None,
) -> fm.IRModule:
    class DenseProjection(fm.Module):
        def __init__(self):
            super().__init__(
                dialect="high_level", stage="imported", entry="main"
            )

        def forward(self):
            value = self.input(
                "value",
                fm.tensor_type("bfloat16", (1, k)),
                id="value",
            )
            weight = self.weight(
                "weight",
                fm.tensor_type("bfloat16", (n, k)),
                source="memory",
                key="weight",
                id="weight",
            )
            projection = fm.F.math.matmul(
                value,
                weight,
                transpose_b=True,
                name="projection",
            )
            self.function("main", (value,), (projection,))

    compiler = compiler or Compiler()
    source = DenseProjection().build()
    if output_policy is not None:
        from triton.flagmega.passes.auto_distributed.auto_distributed import AutoDistributedPass

        distribution = compiler.compile(source, stop_after="propose-distribution").module
        graph = AutoDistributedPass._build_graph(distribution, compiler.target)
        projection = next(bucket for bucket in graph.buckets if distribution.node_map[bucket.node_id].op == "ntt.packed_matmul")
        candidate = next(candidate for candidate in projection.candidates
                         if candidate.return_type.partial is None
                         and candidate.return_type.axis_policies == (fm.SBP.broadcast(), output_policy))
        result = solve_search_graph(graph, fixed_selections={projection.node_id: candidate.id})
        _, records = distribution_selection_state(result, policy=compiler.target.distribution_policy.identity)
        source = compiler.run_stage(distribution, "auto-distributed", plan=override_plan(
            distribution, tuple((r.point_id, r.candidate_id) for r in records))).module
    proposed = compiler.compile(source, stop_after="propose-tir").module
    point = next(
        value
        for value in proposed.selection_points
        if value.id == "tir.projection.vectorized.compute"
    )
    assert implementation in {
        candidate.id for candidate in point.candidates
    }
    selected = compiler.run_stage(
        proposed,
        "lower-tir",
        plan=override_plan(proposed, ((point.id, implementation),)),
    ).module
    return compiler.compile(selected).module


def _packed_glu_descriptor_pipeline_module(
    implementation: str = (
        "tir.dense_matmul_glu."
        "packed_tensor_descriptor_smem_pipeline_full_lhs_gemv"
    ),
    *, reduction_extent: int = 2048, output_extent: int = 2048,
    compiler: Compiler | None = None,
    round_activation: bool = True,
) -> fm.IRModule:
    class DenseGlu(fm.Module):
        def __init__(self):
            super().__init__(
                dialect="high_level", stage="imported", entry="main"
            )

        def forward(self):
            value = self.input(
                "value",
                fm.tensor_type("bfloat16", (1, reduction_extent)),
                id="value",
            )
            gate = self.weight(
                "gate",
                fm.tensor_type("bfloat16", (output_extent, reduction_extent)),
                source="memory",
                key="gate",
                id="gate",
            )
            up = self.weight(
                "up",
                fm.tensor_type("bfloat16", (output_extent, reduction_extent)),
                source="memory",
                key="up",
                id="up",
            )
            output = fm.F.nn.dense_matmul_glu(
                value,
                gate,
                up,
                activation="silu",
                round_activation=round_activation,
                name="glu",
            )
            self.function("main", (value,), (output,))

    compiler = compiler or Compiler()
    proposed = compiler.compile(
        DenseGlu().build(), stop_after="propose-tir"
    ).module
    eligible = tuple(
        value
        for value in proposed.selection_points
        if implementation in {candidate.id for candidate in value.candidates}
    )
    assert eligible, tuple(
        (value.id, tuple(candidate.id for candidate in value.candidates))
        for value in proposed.selection_points
    )
    point = eligible[0]
    selected = compiler.run_stage(
        proposed,
        "lower-tir",
        plan=override_plan(proposed, ((point.id, implementation),)),
    ).module
    return compiler.compile(selected).module


def _packed_norm_stats_pipeline_module(
    implementation: str, *, reduction_extent: int = 2048, output_extent: int = 2048,
    compiler: Compiler | None = None,
    output_policy: fm.SBPSplit | None = None,
    wide: bool = False,
    addend_cast_dtypes: tuple[str, ...] = (),
    projection_output_data_type: str = "bfloat16",
) -> fm.IRModule:
    class DenseResidualNorm(fm.Module):
        def __init__(self):
            super().__init__(
                dialect="high_level", stage="imported", entry="main"
            )

        def forward(self):
            value = self.input(
                "value",
                fm.tensor_type("bfloat16", (1, reduction_extent)),
                id="value",
            )
            residual = self.input(
                "residual",
                fm.tensor_type("float32" if wide else "bfloat16", (1, output_extent)),
                id="residual",
            )
            weight = self.weight(
                "weight",
                fm.tensor_type("bfloat16", (output_extent, reduction_extent)),
                source="memory",
                key="weight",
                id="weight",
            )
            scale = self.weight(
                "scale",
                fm.tensor_type("bfloat16", (output_extent,)),
                source="memory",
                key="scale",
                id="scale",
            )
            bias = self.weight(
                "bias",
                fm.tensor_type("bfloat16", (output_extent,)),
                source="memory",
                key="bias",
                id="bias",
            )
            projection = fm.F.math.matmul(
                value,
                weight,
                transpose_b=True,
                name="projection",
            )
            if wide:
                projection = fm.F.tensors.cast(projection, dtype="float32", name="wide_projection")
            combined = fm.F.math.add(
                projection, residual, name="combined"
            )
            stats = fm.F.nn.norm_stats(
                combined,
                axis=-1,
                use_mean=False,
                name="stats",
            )
            normalized = fm.F.nn.norm_apply(
                combined,
                stats,
                scale,
                bias,
                axis=-1,
                epsilon=1e-6,
                use_mean=False,
                name="normalized",
            )
            self.function("main", (value, residual), (normalized,))

    compiler = compiler or Compiler()
    source = DenseResidualNorm().build()
    if output_policy is not None:
        from triton.flagmega.passes.auto_distributed.auto_distributed import AutoDistributedPass

        distribution = compiler.compile(source, stop_after="propose-distribution").module
        graph = AutoDistributedPass._build_graph(distribution, compiler.target)
        projection = next(bucket for bucket in graph.buckets if distribution.node_map[bucket.node_id].op == "ntt.packed_matmul")
        projection_candidate = next(candidate for candidate in projection.candidates
            if candidate.return_type.partial is None
            and candidate.return_type.axis_policies == (fm.SBP.broadcast(), output_policy))
        combine = next(bucket for bucket in graph.buckets if distribution.node_map[bucket.node_id].op == "ntt.matmul_norm_stats_combine")
        combine_candidate = next(candidate for candidate in combine.candidates
            if candidate.input_types == (projection_candidate.return_type, projection_candidate.return_type))
        result = solve_search_graph(graph, fixed_selections={
            projection.node_id: projection_candidate.id, combine.node_id: combine_candidate.id,
        })
        _, records = distribution_selection_state(result, policy=compiler.target.distribution_policy.identity)
        source = compiler.run_stage(distribution, "auto-distributed", plan=override_plan(
            distribution, tuple((r.point_id, r.candidate_id) for r in records))).module
    proposed = compiler.compile(source, stop_after="propose-tir").module
    if addend_cast_dtypes or wide:
        # Build the explicit epilogue contract for kernel-unit coverage. Graph
        # discovery/private-use legality is exercised in the rewrite tests.
        proposed = fm.verify_module(replace(proposed, nodes=tuple(
            replace(node, attrs=fm.get_definition(node.op).normalize_attrs({
                **node.attrs, "addend_cast_dtypes": addend_cast_dtypes,
                "output_data_type": projection_output_data_type}))
            if node.op == "ntt.matmul_norm_stats" else node for node in proposed.nodes)))
    point = next(
        value
        for value in proposed.selection_points
        if implementation in {candidate.id for candidate in value.candidates}
    )
    selected = compiler.run_stage(
        proposed,
        "lower-tir",
        plan=override_plan(proposed, ((point.id, implementation),)),
    ).module
    return compiler.compile(selected).module


def _packed_qkv_mma_pipeline_module(
    implementation: str = (
        "tir.qkv_parallel_linear.packed_partial_mma_smem_pipeline"
    ),
    *, output_partition: bool = False, target=None,
) -> fm.IRModule:
    class PackedQKV(fm.Module):
        def __init__(self):
            super().__init__(
                dialect="high_level", stage="imported", entry="main"
            )

        def forward(self):
            value = self.input(
                "value",
                fm.tensor_type("bfloat16", (1, 2048)),
                id="value",
            )
            q_weight = self.weight(
                "q_weight",
                fm.tensor_type("bfloat16", (2048, 2048)),
                source="memory",
                key="q_weight",
                id="q_weight",
            )
            k_weight = self.weight(
                "k_weight",
                fm.tensor_type("bfloat16", (2048, 1024)),
                source="memory",
                key="k_weight",
                id="k_weight",
            )
            v_weight = self.weight(
                "v_weight",
                fm.tensor_type("bfloat16", (2048, 1024)),
                source="memory",
                key="v_weight",
                id="v_weight",
            )
            none = fm.F.builtin.none(name="none")
            qkv = fm.F.nn.qkv_parallel_linear(
                value,
                q_weight,
                k_weight,
                v_weight,
                none,
                none,
                none,
                none,
                none,
                none,
                none,
                none,
                none,
                num_heads=16,
                num_kv_heads=8,
                output_data_type="bfloat16",
                name="qkv",
            )
            q, k, v = fm.F.tensors.get_items(qkv, 0, 1, 2, name_prefix="qkv")
            self.function("main", (value,), (q, k, v))

    # This fixture exercises the split-K physical contract.  Q/K/V are direct
    # program outputs in this deliberately small graph, so AutoDistribution's
    # correct default is an output split rather than the split-K layout used by
    # the decoder layer.  Select split-K explicitly and let CP-SAT recompute
    # every coupled structural choice; depending on an unrelated default here
    # made this codegen UT change shape when distribution costs evolved.
    compiler = Compiler()
    if target is not None:
        compiler.target = target
    distribution_proposal = compiler.compile(
        PackedQKV().build(), stop_after="propose-distribution"
    ).module
    qkv_point = next(
        value
        for value in distribution_proposal.selection_points
        if value.id == "distribution.qkv.packed_projection"
    )
    split_k_candidate = next(
        candidate.id
        for candidate in qkv_point.candidates
        if (
            ".out_tuple_d_b_s_bc_h0_1_b2__d_b_s_bc_h0_1_b1__d_b_s_bc_h0_1_b1" in candidate.id
            if output_partition else
            candidate.parameters["reason"] == "packed-qkv-output-K-sbp-partial"
            and ".out_tuple_d_b_s_bc_h1_b8_partial_p_sum_h0" in candidate.id
        )
    )
    registry = DistributedCandidateProviderRegistry()
    compiler.target.register_auto_distributed_candidate_providers(registry)
    graph = build_search_graph(
        distribution_proposal,
        compiler.target.distributed_placements(distribution_proposal)[0],
        registry,
        compiler.target.distributed_reshard_realization_policy(),
        compiler.target.distributed_reshard_cost_model(),
        compiler.target.distributed_operation_cost_model(),
    )
    result = solve_search_graph(
        graph,
        fixed_selections={"qkv.packed_projection": split_k_candidate},
    )
    _, records = distribution_selection_state(
        result,
        policy=compiler.target.distribution_policy.identity,
    )
    distributed = compiler.run_stage(
        distribution_proposal,
        "auto-distributed",
        plan=override_plan(
            distribution_proposal,
            tuple((record.point_id, record.candidate_id) for record in records),
        ),
    ).module
    proposed = compiler.compile(
        distributed, stop_after="propose-microkernels"
    ).module
    eligible = tuple(
        value
        for value in proposed.selection_points
        if implementation in {candidate.id for candidate in value.candidates}
    )
    assert eligible, tuple(
        (value.id, tuple(candidate.id for candidate in value.candidates))
        for value in proposed.selection_points
    )
    point = eligible[0]
    selected = compiler.run_stage(
        proposed,
        "select-microkernels",
        plan=override_plan(proposed, ((point.id, implementation),)),
    ).module
    return compiler.compile(selected).module


@pytest.fixture(scope="module")
def packed_qkv_mma_aligned_pipeline_module():
    from dataclasses import replace
    from triton.flagmega.targets import NvidiaSm90Target
    from triton.flagmega.targets.nvidia import sm90_triton_implementation_model

    model = sm90_triton_implementation_model()
    model = replace(model, implementations=tuple(replace(implementation, shared_workspaces=tuple(
        replace(value, alignment_bytes=max(value.alignment_bytes, 2048))
        if value.matrix_compatible else value for value in implementation.shared_workspaces))
        for implementation in model.implementations))
    return _packed_qkv_mma_pipeline_module(target=NvidiaSm90Target(triton_implementation_model=model))


@pytest.fixture(scope="module")
def packed_qkv_mma_descriptor_table_pipeline_module() -> fm.IRModule:
    return _packed_qkv_mma_pipeline_module(
        "tir.qkv_parallel_linear."
        "packed_partial_mma_descriptor_table_smem_pipeline"
    )


@pytest.fixture
def two_call_descriptor_module():
    return _two_call_descriptor_module


@pytest.fixture
def compile_packed_projection():
    return _packed_descriptor_pipeline_module


@pytest.fixture(scope="module")
def packed_descriptor_pipeline_module():
    return _packed_descriptor_pipeline_module()


@pytest.fixture(scope="module")
def packed_descriptor_table_pipeline_module():
    return _packed_descriptor_pipeline_module(
        "tir.dense_matmul."
        "packed_tensor_descriptor_table_smem_pipeline_gemv"
    )


@pytest.fixture(scope="module")
def packed_glu_descriptor_pipeline_module():
    return _packed_glu_descriptor_pipeline_module()


@pytest.fixture(scope="module")
def packed_glu_portable_pipeline_module():
    return _packed_glu_descriptor_pipeline_module(
        "tir.dense_matmul_glu.packed_k_major_gemv_tn16"
    )


@pytest.fixture(scope="module")
def packed_glu_direct_lhs_descriptor_pipeline_module():
    return _packed_glu_descriptor_pipeline_module(
        "tir.dense_matmul_glu.packed_tensor_descriptor_smem_pipeline_gemv"
    )


@pytest.fixture(scope="module")
def packed_glu_inline_descriptor_pipeline_module():
    return _packed_glu_descriptor_pipeline_module(
        "tir.dense_matmul_glu."
        "packed_tensor_descriptor_smem_pipeline_inline_gemv"
    )


@pytest.fixture(scope="module")
def packed_glu_paired_inline_descriptor_pipeline_module():
    return _packed_glu_descriptor_pipeline_module(
        "tir.dense_matmul_glu."
        "packed_tensor_descriptor_paired_smem_pipeline_inline_gemv"
    )


@pytest.fixture(scope="module")
def packed_glu_paired_table_inline_descriptor_pipeline_module():
    return _packed_glu_descriptor_pipeline_module(
        "tir.dense_matmul_glu."
        "packed_tensor_descriptor_table_paired_smem_pipeline_inline_gemv"
    )


@pytest.fixture(scope="module")
def packed_glu_paired_full_lhs_descriptor_pipeline_module():
    return _packed_glu_descriptor_pipeline_module(
        "tir.dense_matmul_glu."
        "packed_tensor_descriptor_paired_smem_pipeline_full_lhs_gemv"
    )


@pytest.fixture(scope="module")
def packed_glu_full_lhs_inline_descriptor_pipeline_module():
    return _packed_glu_descriptor_pipeline_module(
        "tir.dense_matmul_glu."
        "packed_tensor_descriptor_smem_pipeline_full_lhs_inline_gemv"
    )


@pytest.fixture(scope="module")
def packed_norm_stats_descriptor_pipeline_module():
    return _packed_norm_stats_pipeline_module(
        "tir.dense_matmul."
        "packed_tensor_descriptor_smem_pipeline_gemv_norm_stats"
    )


@pytest.fixture(scope="module")
def packed_norm_stats_descriptor_table_pipeline_module():
    return _packed_norm_stats_pipeline_module(
        "tir.dense_matmul."
        "packed_tensor_descriptor_table_smem_pipeline_gemv_norm_stats"
    )


@pytest.fixture(scope="module")
def packed_norm_stats_portable_pipeline_module():
    return _packed_norm_stats_pipeline_module(
        "tir.dense_matmul.packed_k_major_gemv_norm_stats"
    )


@pytest.fixture(scope="module")
def packed_qkv_mma_pipeline_module():
    return _packed_qkv_mma_pipeline_module()


@pytest.fixture(scope="module")
def packed_qkv_simt_pipeline_module():
    return _packed_qkv_mma_pipeline_module(
        "tir.qkv_parallel_linear.packed_gemv_smem_pipeline", output_partition=True,
    )
