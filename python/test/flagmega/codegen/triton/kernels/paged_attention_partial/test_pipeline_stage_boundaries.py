# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import ast

from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.codegen.triton.tir_package import _mesh_context


def test_mma_tile_helpers_inline_inside_attention_op_roles():
    operands = (
        "query", "block_table", "slot_mapping", "layer_id", "key_descriptor",
        "value_descriptor", "partial_max", "partial_sum", "partial_accumulator",
    )
    call = {
        **{name: name for name in operands},
        "symbol": "attention",
        "signature": ", ".join(operands),
        "family": "paged_attention_partial",
        "variant": "mma_tma_smem_pipeline",
        "consumer_warps": 8,
        "head_dim": 128,
        "query_token": "0",
        "query_dim_stride": 8,
        "query_lanes": 8,
        "accumulator_dim_stride": 1,
        "q_head_group_size": 2,
        "block_n": 64,
        "block_size": 256,
        "head": "shard_coord0",
        "context_shard": "shard_coord1",
        "context_shards": 4,
        "scale": 128 ** -0.5,
        "descriptor_block_shape": (1, 1, 64, 1, 128),
        "pipeline_consumer_parameters": ("key_reader", "value_reader"),
        "pipeline_producer_parameters": ("key_writer", "value_writer"),
        "pipeline_channels": (
            {"writer_parameter": "key_writer", "reader_parameter": "key_reader"},
            {"writer_parameter": "value_writer", "reader_parameter": "value_reader"},
        ),
    }
    source = TritonTemplateRegistry().render(
        "kernels/paged_attention_partial/platforms/nvidia/sm90/mma_tma_smem_pipeline.py.jinja",
        {
            **_mesh_context({"hierarchy": (2, 4), "hierarchy_levels": "bb", "name": "yx"}),
            "distributed_entry": True,
            "render_calls": (call,),
        },
    )
    functions = {
        node.name: node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    for stage in ("qk_stage", "value_stage"):
        assert ast.unparse(functions[f"attention__{stage}"].decorator_list[0]) == "triton.jit"
    for role in ("producer", "consumer"):
        assert ast.unparse(functions[f"attention__{role}"].decorator_list[0]) == "triton.jit(noinline=True)"
    producer = functions["attention__producer"]
    assigned = {node.id for node in ast.walk(producer)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    assert not {"key_descriptor", "value_descriptor"} & assigned
    assert {"attention_key_tensor_map", "attention_value_tensor_map"} <= assigned
