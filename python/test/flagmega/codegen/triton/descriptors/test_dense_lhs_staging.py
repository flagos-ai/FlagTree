# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Consumer-owned full-row staging for the ordinary single-result GEMV."""

import ast
from copy import deepcopy
from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package
from triton.flagmega.codegen.triton.kernel_call_renderers import _dense_matmul_call
from triton.flagmega.codegen.triton.runtime_binding import describe_function_runtime_binding
from triton.flagmega.errors import CodegenError
from triton.flagmega.selection import override_plan
from triton.flagmega.targets import NvidiaSm90Target

BASE = "tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv"
STAGED = BASE + "_lhs8192_async"
DONOR = "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_norm_stats_lhs8192"
REGISTERED = BASE + "_lhs8192_async"


def test_plain_async_candidate_is_registered_without_changing_default():
    model = NvidiaSm90Target().triton_implementation_model
    assert REGISTERED in {value.id for value in model.implementations}
    base = model.implementation(BASE)
    candidate = model.implementation(REGISTERED)
    assert candidate.variant == base.variant
    assert candidate.parameters == {
        **base.parameters, "lhs_stage_extent": 8192, "lhs_copy_tile": 1024, "lhs_copy_kind": "async"
    }
    assert candidate.contract == {
        **base.contract, "required_input_dtype": "bfloat16", "required_output_dtype": "bfloat16"
    }
    assert candidate.requires == (*base.requires, "async_copy")
    assert candidate.shared_workspaces == (*base.shared_workspaces, model.implementation(DONOR).shared_workspaces[1])
    assert candidate.transfer_pipeline == replace(base.transfer_pipeline, consumer_shared_workspace_indices=(1, ))
    assert REGISTERED not in model.preferences["dense_matmul"]


def make_module(implementation=STAGED, *, k=6144):

    class ProducedRow(fm.Module):

        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, k)), id="value")
            # Materialize a real owned-pool row instead of assuming alignment
            # or lifetime for an external pointer passed by the test runner.
            prepared = fm.F.math.add(value, value, name="prepared")
            weight = self.weight("weight", fm.tensor_type("bfloat16", (2048, k)), source="memory", key="weight",
                                 id="weight")
            result = fm.F.math.matmul(prepared, weight, transpose_b=True, name="projection")
            self.function("main", (value, ), (result, ))

    compiler = Compiler()
    compiler.target = NvidiaSm90Target()
    # Exercise the same complete local row and output ownership as the down
    # kernel. Automatic distribution may otherwise choose a split-K fixture.
    from triton.flagmega.passes.auto_distributed.auto_distributed import AutoDistributedPass
    from triton.flagmega.passes.auto_distributed import solve_search_graph
    from triton.flagmega.passes.auto_distributed.materializer import distribution_selection_state

    distribution = compiler.compile(ProducedRow().build(), stop_after="propose-distribution").module
    graph = AutoDistributedPass._build_graph(distribution, compiler.target)
    projection = next(bucket for bucket in graph.buckets
                      if distribution.node_map[bucket.node_id].op == "ntt.packed_matmul")
    full_row = next(candidate for candidate in projection.candidates
                    if candidate.return_type.partial is None and candidate.return_type.axis_policies == (
                        fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 2)))
    result = solve_search_graph(graph, fixed_selections={projection.node_id: full_row.id})
    _, records = distribution_selection_state(result, policy=compiler.target.distribution_policy.identity)
    distributed = compiler.run_stage(
        distribution, "auto-distributed",
        plan=override_plan(distribution, tuple((record.point_id, record.candidate_id) for record in records))).module
    proposed = compiler.compile(distributed, stop_after="propose-tir").module
    point = next(point for point in proposed.selection_points if point.id == "tir.projection.vectorized.compute")
    assert implementation in {candidate.id for candidate in point.candidates}
    selected = compiler.run_stage(proposed, "lower-tir", plan=override_plan(proposed,
                                                                            ((point.id, implementation), ))).module
    return compiler.compile(selected).module


@pytest.fixture(scope="module")
def staged_module():
    return make_module()


@pytest.fixture(scope="module")
def ordinary_module():
    return make_module(BASE)


def test_plain_dense_stages_true_k_and_waits_before_shared_read(staged_module, tmp_path):
    path = tmp_path / "plain.py"
    fm.emit_module(staged_module, path)
    package = describe_tir_package(fm.load_module(path))
    call = next(call for call in package["render_calls"] if call.get("implementation") == STAGED)
    assert call["local_k_capacity"] == 6144
    assert call["lhs_stage_extent"] == 8192
    assert call["descriptor_kind"] == "single"
    assert call["pipeline_contract"]["capacity"] == 4
    source = render_tir_package(package, "plain-staging-test")
    functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    consumer = ast.get_source_segment(source, functions[call["symbol"] + "__consumer"])
    stage = ast.get_source_segment(source, functions[call["symbol"] + "__consumer_stage"])
    assert "tl.static_range(0, 6144, 1024)" in consumer
    assert "is_async=True" in consumer
    assert consumer.index("async_commit_group") < consumer.index(".wait(dense_pipeline_sequence)")
    assert consumer.index(".wait(dense_pipeline_sequence)") < consumer.index("async_wait_group")
    assert consumer.index("async_wait_group") < consumer.index("__consumer_stage(")
    assert "dense_lhs_pointer = tle.gpu.local_ptr(" in stage
    assert "tl.load(dense_lhs_pointer)" in stage
    assert "dense_source +" not in stage


@pytest.mark.parametrize("mutation",
                         ("capacity", "copy_tile", "dtype", "input_dtype", "workspace_owner", "missing_workspace",
                          "external", "stride", "copy_kind", "alignment", "pool_offset", "scope_stride"))
def test_plain_dense_rejects_unproven_staging(staged_module, mutation):
    binding = describe_function_runtime_binding(staged_module)
    raw = deepcopy(next(call for call in binding["call_abi"]["kernel_calls"] if call.get("implementation") == STAGED))
    if mutation == "capacity":
        raw["parameters"]["lhs_stage_extent"] = 4096
    elif mutation == "copy_tile":
        raw["parameters"]["lhs_copy_tile"] = 4096
    elif mutation == "dtype":
        raw["shared_workspaces"][1]["dtype"] = "float32"
    elif mutation == "input_dtype":
        raw["inputs"][0]["buffers"][0]["abi"]["scalar_dtype"] = "float16"
    elif mutation == "workspace_owner":
        raw["transfer_pipeline"]["consumer_shared_workspace_indices"] = ()
    elif mutation == "missing_workspace":
        raw["shared_workspaces"] = raw["shared_workspaces"][:1]
    elif mutation == "external":
        raw["inputs"][0]["buffers"][0]["abi"]["pooled"] = False
    elif mutation == "stride":
        raw["inputs"][0]["buffers"][0]["abi"]["scalar_storage_strides"] = (12288, 2)
    elif mutation == "copy_kind":
        raw["parameters"]["lhs_copy_kind"] = "automatic"
    elif mutation == "alignment":
        raw["inputs"][0]["buffers"][0]["abi"]["alignment_bytes"] = 2
    elif mutation == "pool_offset":
        raw["inputs"][0]["buffers"][0]["abi"]["pool_byte_offset"] = 2
    elif mutation == "scope_stride":
        raw["inputs"][0]["buffers"][0]["abi"]["pool_scope_stride_bytes"] = 2
        raw["inputs"][0]["buffers"][0]["abi"]["pool_scope_index"] = "0"
    with pytest.raises(CodegenError, match="LHS"):
        _dense_matmul_call(raw)


def test_plain_default_keeps_global_loads_and_no_copy(ordinary_module):
    package = describe_tir_package(ordinary_module)
    call = next(call for call in package["render_calls"] if call.get("implementation") == BASE)
    assert "lhs_stage_extent" not in call
    assert not call["pipeline_consumer_workspaces"]
    source = render_tir_package(package, "plain-staging-test")
    functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    consumer = ast.get_source_segment(source, functions[call["symbol"] + "__consumer"])
    stage = ast.get_source_segment(source, functions[call["symbol"] + "__consumer_stage"])
    assert "dense_source +" in stage
    assert "async_commit_group" not in consumer


def test_plain_async_preserves_original_six_stage_arithmetic(staged_module, ordinary_module):
    stages = []
    for module, implementation in ((ordinary_module, BASE), (staged_module, REGISTERED)):
        package = describe_tir_package(module)
        call = next(call for call in package["render_calls"] if call.get("implementation") == implementation)
        assert (call["block_k"], call["reduction_group"], call["reduction_unroll"]) == (1024, 32, 32)
        source = render_tir_package(package, "plain-arithmetic-test").replace(call["symbol"], "gemv")
        functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        stage = functions["gemv__consumer_stage"]
        arithmetic = [ast.dump(node) for node in ast.walk(stage) if isinstance(node, (ast.AugAssign, ast.Return))]
        reduction_loops = [ast.dump(node.iter) for node in ast.walk(stage) if isinstance(node, ast.For)]
        consumer = ast.get_source_segment(source, functions["gemv__consumer"])
        assert "0, 6144, 1024" in consumer
        assert "tl.sum(dense_partial, axis=1)" in consumer
        assert "dense_accumulator.to(tl.bfloat16)" in consumer
        stages.append((arithmetic, reduction_loops))
    assert stages[0] == stages[1]


def test_plain_async_requires_capability(staged_module):
    from triton.flagmega.targets.nvidia.capability import Sm90Capability
    from triton.flagmega.targets.selection import CapabilitySelectionPolicy

    point = next(point for point in staged_module.selection_points if point.id == "tir.projection.vectorized.compute")
    filtered = CapabilitySelectionPolicy().add_defaults(replace(staged_module, selection_points=(),
                                                                selections=()), (point, ), "capability regression",
                                                        capability=Sm90Capability(supports_async_copy=False),
                                                        target_name="nvidia-sm90", policy_version="test")
    assert BASE in {candidate.id for candidate in filtered.selection_points[0].candidates}
    assert REGISTERED not in {candidate.id for candidate in filtered.selection_points[0].candidates}


def direct_norm_stats_staged_module():
    """An explicit typed epilogue fixture, independent of graph fusion discovery."""
    from .test_packed_dense_norm_stats_candidate import _module

    module = _module(reduction_extent=6144)
    # Keep only the semantic operation and its real operands. The original
    # candidate-only fixture also describes graph-discovery hint nodes.
    nodes = tuple(
        replace(
            node, attrs=fm.get_definition(node.op).normalize_attrs({"name": node.id} if node.op ==
                                                                   "builtin.var" else node.attrs))
        for node in module.nodes
        if node.id not in {"residual_add", "norm_consumer"})
    module = fm.verify_module(
        replace(
            module,
            stage="tuple_boxing_lowered",
            nodes=nodes,
            functions=(fm.Function("main", ("lhs", "rhs", "addend"), ("result", )), ),
            metadata={"auto_distribution": {"placement": nodes[0].type.placement.to_data()}},
        ))
    compiler = Compiler()
    proposed = compiler.run_stage(module, "propose-tir").module
    selected = compiler.run_stage(proposed, "lower-tir", plan=override_plan(proposed, (("tir.result", DONOR), ))).module
    return compiler.compile(selected).module


def test_direct_norm_stats_staging_preserves_explicit_epilogue(tmp_path):
    module = direct_norm_stats_staged_module()
    path = tmp_path / "norm_stats.py"
    fm.emit_module(module, path)
    package = describe_tir_package(fm.load_module(path))
    call, = package["render_calls"]
    assert call["semantic_op"] == "ntt.matmul_norm_stats"
    assert call["implementation"] == DONOR
    assert call["lhs_stage_extent"] == 8192
    assert call["lhs_copy_kind"] == "sync"
    assert call["local_k_capacity"] == 6144
    assert call["reduction_group"] == 128
    assert call["reduction_unroll"] == 4
    source = render_tir_package(package, "norm-stats-staging-test")
    ast.parse(source)
    assert "tl.range(0, 6144, 1024)" in source
    assert "loop_unroll_factor=4" in source
