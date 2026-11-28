# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A published generic-global KV cache needs a bridge before async reads."""

import ast
import pytest

from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.codegen.triton.tir_package import _mesh_context


def render_attention(**overrides):
    operands = ("query", "block_table", "slot_mapping", "layer_id", "key_descriptor", "value_descriptor", "partial_max",
                "partial_sum", "partial_accumulator")
    call = {
        **{name: name
           for name in operands},
        "symbol":
        "attention",
        "signature":
        ", ".join(operands),
        "family":
        "paged_attention_partial",
        "variant":
        "mma_tma_smem_pipeline",
        "consumer_warps":
        8,
        "head_dim":
        128,
        "query_token":
        "0",
        "query_dim_stride":
        8,
        "query_lanes":
        8,
        "accumulator_dim_stride":
        1,
        "q_head_group_size":
        2,
        "local_query_heads":
        1,
        "query_head_stride":
        128,
        "block_n":
        64,
        "block_size":
        256,
        "head":
        "shard_coord0",
        "context_shard":
        "shard_coord1",
        "context_shards":
        4,
        "scale":
        128**-0.5,
        "descriptor_block_shape": (1, 1, 64, 1, 128),
        "pipeline_consumer_parameters": ("key_reader", "value_reader"),
        "pipeline_producer_parameters": ("key_writer", "value_writer"),
        "pipeline_channels": ({"writer_parameter": "key_writer", "reader_parameter":
                               "key_reader"}, {"writer_parameter": "value_writer", "reader_parameter": "value_reader"}),
        **overrides,
    }
    source = TritonTemplateRegistry().render(
        "kernels/paged_attention_partial/platforms/nvidia/sm90/mma_tma_smem_pipeline.py.jinja", {
            **_mesh_context({"hierarchy": (2, 4), "hierarchy_levels": "bb", "name": "yx"}), "distributed_entry": True,
            "render_calls": (call, )
        })
    return source


def calls(node, qualified_name):
    return [
        value for value in ast.walk(node) if isinstance(value, ast.Call) and ast.unparse(value.func) == qualified_name
    ]


@pytest.mark.parametrize("options",
                         [{}, {"trip_scoped_pipe": True}, {"trip_scoped_pipe": True, "trip_scoped_dynamic": True}])
def test_published_kv_data_fence_dominates_both_tma_issuers(options):
    source = render_attention(**options)
    compile(source, "normal_attention_render.py", "exec")
    functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    producer = functions["attention__producer"]
    fences = calls(producer, "tle.gpu.fence_proxy_async_global")
    assert len(fences) == 1, "Generic KV publication has no async-global data proxy bridge"
    fence, = fences
    # An unconditional role-level operation covers each lane that might be
    # elected separately to issue K or V. It is not inside a tile/phase branch.
    assert any(isinstance(stmt, ast.Expr) and stmt.value is fence for stmt in producer.body)
    assert not fence.args and not fence.keywords
    descriptors = calls(producer, "tle.gpu.tensor_map_fenceproxy_acquire")
    copies = calls(producer, "tle.gpu.copy")
    assert len(descriptors) == 2 and len(copies) == 2
    assert all(fence.lineno < node.lineno for node in (*descriptors, *copies))
    assert {ast.unparse(node.args[0]) for node in descriptors} == {"key_descriptor", "value_descriptor"}
    for name, function in functions.items():
        if name != "attention__producer":
            assert not calls(function, "tle.gpu.fence_proxy_async_global")
