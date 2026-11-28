# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One owner may hold a whole GQA group, and the MMA tile must publish it.

``m16n8k16`` computes ``q_head_group_size`` rows whether or not they are
owned, so the SM90 paged-attention partial always allocated a
``(group, head_dim)`` accumulator and then masked every row but row 0 off.
That was correct only while ownership happened to give each participant one
query head.  When the context split follows the attention work instead of the
mesh shape, a GQA participant owns its whole KV group, and the implementation
contract, the renderer and the template all have to admit that:

* the catalog key ``requires_local_query_heads_within_one_kv_group`` replaces
  the exact ``required_local_query_heads: 1``, so the MMA implementation stays
  in the candidate list instead of being silently filtered out and replaced by
  the scalar ``decode_t16`` fallback;
* the renderer proves the owned rows are addressable inside the owner's own
  partial-state span before it lets the template publish them;
* the template masks ``row < local_query_heads`` rather than ``row == 0`` and
  offsets the query load by the owner-local head stride.

Everything here is CPU-only: types, strides, candidate ids and emitted text.
"""

import pathlib
import re

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _paged_attention_partial_call, )
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.codegen.triton.tir_package import _mesh_context
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError
from triton.flagmega.evaluator import PagedAttentionStateConfig
from triton.flagmega.ir.python_ir import emit_module
from triton.flagmega.options import CompileOptions
from triton.flagmega.stages import get_stage

HEAD_DIM = 128
LANES = 8
MMA_TEMPLATE = ("kernels/paged_attention_partial/platforms/nvidia/sm90/"
                "mma_tma_smem_pipeline.py.jinja")

# --------------------------------------------------------------------------
# selection, through the real stages on a packed GQA decode module
# --------------------------------------------------------------------------


class _PackedDecodeAttention(fm.Module):
    """A packed BF16x8 decode attention, the shape selection actually sees.

    The query arrives already packed because the MMA contract is stated on the
    physical BF16x8 decode ABI; an unpacked query is a different machine
    profile and is served by another implementation.
    """

    def __init__(self, *, query_heads: int, kv_heads: int):
        super().__init__(dialect="high_level", stage="stats_threaded", entry="main")
        self.query_heads = query_heads
        self.cache = PagedAttentionStateConfig(28, kv_heads, HEAD_DIM, block_size=256, num_blocks=32)

    def forward(self):
        query = self.input(
            "query",
            fm.tensor_type(
                fm.vector_type("bfloat16", (LANES, )),
                (1, self.query_heads, HEAD_DIM // LANES),
            ),
        )
        state = self.input("state", self.cache.ref_type)
        layer = self.input("layer", fm.tensor_type("int32", ()))
        attention = fm.F.nn.paged_attention(
            query,
            state,
            layer,
            scale=HEAD_DIM**-0.5,
            layout=("seq", "head", "dim"),
            hidden_size=self.query_heads * HEAD_DIM,
            name="attention",
            metadata={"source": "unit"},
        )
        self.function("main", (query, state, layer), (attention, ))


def _selected_partial(tmp_path, *, query_heads, kv_heads, hierarchy):
    module = _PackedDecodeAttention(query_heads=query_heads, kv_heads=kv_heads).build()
    compiler = Compiler(CompileOptions(
        target="nvidia-sm90",
        placement=fm.Placement(hierarchy, "yx", "bb"),
    ))
    module = get_stage("decompose-paged-attention").run(module, compiler.target)
    selected = compiler.compile(module, stop_after="select-microkernels").module
    emitted = pathlib.Path(tmp_path) / "selected.py"
    emit_module(selected, emitted)
    found = set(re.findall(
        r"implementation='(tir\.paged_attention_partial\.[A-Za-z_0-9]+)'",
        emitted.read_text(),
    ))
    assert len(found) == 1, found
    return found.pop()


def test_mma_is_selected_when_one_owner_holds_a_whole_gqa_group(tmp_path):
    # Qwen3-1.7B decode on one H800: 16 query heads over 8 KV heads on a
    # (8, 16) mesh.  Context splits 16 ways, so each participant owns the
    # whole group of 2 query heads of one KV head.
    assert _selected_partial(tmp_path, query_heads=16, kv_heads=8,
                             hierarchy=(8, 16)) == "tir.paged_attention_partial.mma_tma_smem_pipeline"


def test_mma_is_not_selected_when_owned_heads_cross_two_kv_groups(tmp_path):
    # 32 query heads over 8 KV heads on a (4, 16) mesh leaves 8 owner-local
    # heads against a group of 4, so one KV stream cannot feed the tile.  The
    # contract key is a real predicate, not an unconditional admission.
    assert _selected_partial(tmp_path, query_heads=32, kv_heads=8,
                             hierarchy=(4, 16)) != "tir.paged_attention_partial.mma_tma_smem_pipeline"


# --------------------------------------------------------------------------
# renderer, on the owner-local ABI the bufferizer produces
# --------------------------------------------------------------------------


def _abi(
    local_shape,
    *,
    logical_shape=None,
    lanes=1,
    dtype="float32",
    itemsize=4,
    storage="workspace",
    storage_kind="canonical_global",
    coordinate_space="canonical_global",
    owner_stride=0,
    partial_axes=None,
    distributed=True,
    strides=None,
):
    local_shape = tuple(local_shape)
    logical_shape = tuple(logical_shape or local_shape)
    if strides is None:
        packed, stride = [], 1
        for extent in reversed(local_shape):
            packed.append(stride)
            stride *= extent
        strides = tuple(value * lanes for value in reversed(packed))
    return {
        "storage":
        storage,
        "storage_kind":
        storage_kind,
        "pool_byte_offset":
        0,
        "scalar_dtype":
        dtype,
        "scalar_itemsize":
        itemsize,
        "logical_shape":
        logical_shape,
        "local_capacity_shape":
        local_shape,
        "active_shape_expressions":
        tuple(str(v) for v in local_shape),
        "logical_coordinate_expressions":
        tuple(f"local_coord_{axis}" for axis in range(len(local_shape))),
        "scalar_storage_strides":
        tuple(strides),
        "scalar_lane_shape": () if lanes == 1 else (lanes, ),
        "scalar_lane_count":
        lanes,
        "component_stride_scalar_elements":
        owner_stride,
        "coordinate_space":
        coordinate_space,
        "distributed_type": ({
            "placement": {"hierarchy": (8, 16)},
            "axis_policies": tuple({"kind": "broadcast"} for _ in local_shape),
            "partial": ({"axes": partial_axes, "reduce_op": "sum"} if partial_axes else None),
        } if distributed else None),
    }


def _binding(name, abi, *, value_kind="pointer", argument=None):
    return {
        "formal": name,
        "actual": name,
        "runtime_argument": name if argument is None else argument,
        "runtime_value_kind": value_kind,
        "abi": abi,
    }


def _parameter(formal, *bindings):
    return {"formal": formal, "buffers": bindings}


def _stage_shape(num_stages, block_n):
    return (num_stages, 1, 1, block_n, 1, HEAD_DIM)


def _mma_raw(*, local_query_heads=2, query_heads=16, kv_heads=8, block_n=64):
    """The MMA call record for one owner holding ``local_query_heads`` heads."""

    num_stages = 2
    query = _abi(
        (1, local_query_heads, HEAD_DIM // LANES),
        logical_shape=(1, query_heads, HEAD_DIM // LANES),
        lanes=LANES,
        dtype="bfloat16",
        itemsize=2,
    )
    cache = _abi(
        (32, 28, 2, 256, kv_heads, HEAD_DIM // LANES),
        lanes=LANES,
        dtype="bfloat16",
        itemsize=2,
        storage="input",
        storage_kind="compact_local",
        coordinate_space="local",
        distributed=False,
    )
    state = (
        cache,
        _abi((2, ), dtype="int32", storage="input", distributed=False),
        _abi((1, ), dtype="int32", storage="input", distributed=False),
        _abi((1, ), dtype="int64", itemsize=8, storage="input", distributed=False),
        _abi((1, 32), dtype="int32", storage="input", distributed=False),
    )
    scalar = _abi((), dtype="int32", storage="scalar", distributed=False)
    statistics = _abi(
        (1, local_query_heads, 1),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        owner_stride=local_query_heads,
        partial_axes=(1, ),
    )
    accumulator = _abi(
        (1, local_query_heads, HEAD_DIM),
        storage_kind="compact_per_owner",
        coordinate_space="local",
        owner_stride=local_query_heads * HEAD_DIM,
        partial_axes=(1, ),
    )
    return {
        "variant":
        "mma_tma_smem_pipeline",
        "semantic_attrs": {
            "layout": ("seq", "head", "dim"),
            "split_hierarchy_axis": 1,
            "split_count": 16,
            "scale": HEAD_DIM**-0.5,
        },
        "parameters": {
            "block_m": 1,
            "block_n": block_n,
            "block_k": HEAD_DIM,
            "num_stages": num_stages,
            "consumer_warps": 8,
            "worker_width": 32,
            "producer_warps": 1,
            "producer_registers": 24,
        },
        "transfer_pipeline": {"capacity": num_stages},
        "shared_workspaces": (
            {"shape": _stage_shape(num_stages, block_n)},
            {"shape": _stage_shape(num_stages, block_n)},
        ),
        "inputs": (
            _parameter("q", _binding("q", query)),
            _parameter(
                "state",
                *(_binding(f"state.{index}", abi) for index, abi in enumerate(state)),
            ),
            _parameter(
                "layer_id",
                _binding("layer_id", scalar, value_kind="immediate", argument="0"),
            ),
        ),
        "outputs": (
            _parameter("result_0", _binding("partial_max", statistics)),
            _parameter("result_1", _binding("partial_sum", statistics)),
            _parameter("result_2", _binding("partial_accumulator", accumulator)),
        ),
        "workspaces": (),
    }


@pytest.mark.parametrize("local_query_heads", [1, 2])
def test_mma_renderer_passes_the_owned_row_count_and_head_stride(local_query_heads):
    call = _paged_attention_partial_call(_mma_raw(local_query_heads=local_query_heads))

    assert call["q_head_group_size"] == 2
    assert call["local_query_heads"] == local_query_heads
    # The query head stride is read from the owner-local ABI, not assumed to
    # be head_dim: a packed BF16x8 query stores head_dim scalars per head.
    assert call["query_head_stride"] == HEAD_DIM


def test_mma_renderer_rejects_owned_heads_crossing_a_kv_group():
    with pytest.raises(CodegenError, match="whole sub-block of one KV group"):
        _paged_attention_partial_call(_mma_raw(local_query_heads=4, query_heads=16, kv_heads=8))


def test_mma_renderer_rejects_a_statistics_stride_the_template_cannot_address():
    raw = _mma_raw()
    statistics = raw["outputs"][0]["buffers"][0]["abi"]
    statistics["scalar_storage_strides"] = (4, 2, 1)

    with pytest.raises(CodegenError, match="partial_max head rows with stride 1"):
        _paged_attention_partial_call(raw)


def test_mma_renderer_rejects_an_accumulator_stride_the_template_cannot_address():
    raw = _mma_raw()
    accumulator = raw["outputs"][2]["buffers"][0]["abi"]
    accumulator["scalar_storage_strides"] = (512, 256, 1)

    with pytest.raises(CodegenError, match="partial_accumulator head rows with stride 128"):
        _paged_attention_partial_call(raw)


def test_mma_renderer_rejects_a_partially_active_head_axis():
    # The template carries no per-row active predicate, only the owned-row
    # mask, so a padded head axis must not reach it.
    raw = _mma_raw()
    statistics = raw["outputs"][0]["buffers"][0]["abi"]
    statistics["active_shape_expressions"] = ("1", "min(2, 16 - shard_coord_1)", "1")

    with pytest.raises(CodegenError, match="wholly active on its head axis"):
        _paged_attention_partial_call(raw)


# --------------------------------------------------------------------------
# template, on the fields the renderer passes
# --------------------------------------------------------------------------


def _render_mma(call_overrides):
    operands = (
        "query",
        "block_table",
        "slot_mapping",
        "layer_id",
        "key_descriptor",
        "value_descriptor",
        "partial_max",
        "partial_sum",
        "partial_accumulator",
    )
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
        HEAD_DIM,
        "query_token":
        "0",
        "query_dim_stride":
        LANES,
        "query_lanes":
        LANES,
        "accumulator_dim_stride":
        1,
        "q_head_group_size":
        2,
        "block_n":
        64,
        "block_size":
        256,
        "head":
        "shard_coord0",
        "context_shard":
        "shard_coord1",
        "context_shards":
        16,
        "scale":
        HEAD_DIM**-0.5,
        "descriptor_block_shape": (1, 1, 64, 1, HEAD_DIM),
        "pipeline_consumer_parameters": ("key_reader", "value_reader"),
        "pipeline_producer_parameters": ("key_writer", "value_writer"),
        "pipeline_channels": (
            {"writer_parameter": "key_writer", "reader_parameter": "key_reader"},
            {"writer_parameter": "value_writer", "reader_parameter": "value_reader"},
        ),
        **call_overrides,
    }
    return TritonTemplateRegistry().render(
        MMA_TEMPLATE,
        {
            **_mesh_context({"hierarchy": (8, 16), "hierarchy_levels": "bb", "name": "yx"}),
            "distributed_entry": True,
            "render_calls": (call, ),
        },
    )


@pytest.mark.parametrize("local_query_heads", [1, 2])
def test_mma_template_publishes_exactly_the_owned_rows(local_query_heads):
    source = _render_mma({"local_query_heads": local_query_heads, "query_head_stride": HEAD_DIM})
    compile(source, "grouped_attention_render.py", "exec")

    # Every row predicate follows ownership.  At one owned head this is the
    # same row set as the old `== 0`, so the change is a generalisation.
    assert source.count(f"< {local_query_heads}") == 4
    assert "== 0" not in source
    # The query load reaches row r of this owner's own head block.
    assert "query_head_row = tl.arange(0, 2)" in source
    assert f"+ query_head_row[:, None] * {HEAD_DIM}" in source
    assert "tl.zeros((2, 1), tl.int32)" not in source
    # Statistics walk the head axis with the asserted unit stride, the
    # accumulator with head_dim.
    assert "partial_max + statistics_row" in source
    assert f"+ accumulator_row[:, None] * {HEAD_DIM}" in source
