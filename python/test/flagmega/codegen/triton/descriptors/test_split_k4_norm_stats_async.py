# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Normal typed split-K4 projection epilogue with a published owned LHS."""
from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.selection import override_plan
from triton.flagmega.targets import NvidiaSm90Target

from .test_packed_dense_norm_stats_candidate import _module


def test_normal_split_k4_async_norm_stats_preserves_projection_boundary(tmp_path):
    order = "split_k4_k32_mma_k16_pairtree"
    name = "tir.dense_matmul.test_split_k4_normstats_async"
    model = NvidiaSm90Target().triton_implementation_model
    donor = model.implementation(
        "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_norm_stats_lhs8192")
    variant = replace(donor, id=name,
        parameters={**donor.parameters, "accumulation_mode": "split_k4_mma",
                    "mma_block_k": 128, "lhs_copy_kind": "async"},
        facts={**donor.facts, "split_k4_ordered_mma_accumulation": True})
    target = NvidiaSm90Target(triton_implementation_model=replace(model,
        implementations=(*model.implementations, variant),
        preferences={**model.preferences, "dense_matmul": (name, *model.preferences["dense_matmul"])}))

    # Assemble the already packed semantic epilogue. A scalar high-level
    # projection adds a bitcast and does not exercise this fusion boundary.
    module = _module(reduction_extent=6144)
    nodes = module.node_map
    value = replace(nodes["lhs"], id="value", attrs={"name": "value"})
    prepared = fm.get_definition("math.add").prepare((value, value), {})
    lhs = fm.Node("lhs", "math.add", ("value", "value"), prepared.result_type,
                  prepared.effect, prepared.attrs)
    rhs = replace(nodes["rhs"], attrs={"name": "rhs"})
    addend = replace(nodes["addend"], attrs={"name": "addend"})
    prepared = fm.get_definition("ntt.matmul_norm_stats").prepare((lhs, rhs, addend),
        {**nodes["result"].attrs, "reduction_order": order, "output_data_type": "bfloat16"})
    result = replace(nodes["result"], attrs=prepared.attrs,
                     type=prepared.result_type, effect=prepared.effect)
    module = fm.verify_module(replace(module, stage="tuple_boxing_lowered",
        nodes=(value, lhs, rhs, addend, result),
        functions=(fm.Function("main", ("value", "rhs", "addend"), ("result",)),),
        metadata={"auto_distribution": {"placement": lhs.type.placement.to_data()}}))

    compiler = Compiler()
    compiler.target = target
    proposed = compiler.run_stage(module, "propose-tir").module
    point = next(p for p in proposed.selection_points if p.id == "tir.result")
    assert [c.id for c in point.candidates] == [name]
    selected = compiler.run_stage(proposed, "lower-tir",
        plan=override_plan(proposed, ((point.id, name),))).module
    final = compiler.compile(selected).module
    path = tmp_path / "norm_stats.py"
    fm.emit_module(final, path)
    package = describe_tir_package(fm.load_module(path))
    call = next(c for c in package["render_calls"] if c.get("implementation") == name)
    assert call["semantic_op"] == "ntt.matmul_norm_stats"
    assert call["accumulation_mode"] == "split_k4_mma"
    assert call["lhs_copy_kind"] == "async"
    assert call["lhs_stage_extent"] == 8192 and call["local_k_capacity"] == 6144
    source = render_tir_package(package, "split-k4-normal-normstats")
    consumer = source[source.index("def " + call["symbol"] + "__consumer("):]
    assert consumer.index("async_wait_group") < consumer.index("__split_k4_stage(")
    assert consumer.index("__split_k4_finalize(dense_accumulator)") < consumer.index(
        "dense_projection = dense_accumulator.to(tl.bfloat16).to(tl.float32)")
