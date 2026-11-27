# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""All-reduce writes every private destination, including broadcast owners."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from .test_partial_reduce_local_abi import _abi, _prepare


@pytest.mark.parametrize("sharing_scope", ["chip", "block"])
def test_plain_result_addressing_uses_physical_sharing_not_distributed_storage_kind(sharing_scope):
    source = _abi((1, 48), local_shape=(1, 3), storage_kind="compact_per_owner", coordinate_space="local",
                  coordinates=("local_coord_0", "local_coord_1 + shard_coord_1 * 3"),
                  axis_policies=({"kind": "broadcast"}, {"kind": "split", "stages":
                                                         ({"hierarchy_axes":
                                                           (1, ), "distribution": {"kind": "contiguous", "granularity": 3}}, )}), partial_axes=(0, ), owner_stride=3)
    result = _abi((1, 48), storage_kind="compact_local")
    result.update(distributed_type=None, memory_sharing_scope=sharing_scope)
    if sharing_scope == "block":
        leaf = _prepare((source,), (result,), ("gather_reduce_scatter",))["leaves"][0]
        assert leaf["writer_active"] == "True"
        assert leaf["capacity"] == 48
        assert "boxing_offsets" in leaf["source_offset"]
    else:
        leaf = _prepare((source, ), (result, ), ("gather_reduce_scatter", ))["leaves"][0]
        assert leaf["writer_active"] == "(shard_y == 0)"
        assert "shard_x" in leaf["result_offset"]
        assert leaf["capacity"] == 3


@pytest.mark.parametrize("storage_kind", ["compact_local", "compact_per_owner", "replicated_local"])
def test_matching_compact_result_writes_all_owners_without_canonical_writer_filter(storage_kind):
    source = _abi((1, 2), storage_kind="compact_per_owner", coordinate_space="local", partial_axes=(0, 1),
                  owner_stride=8, lane_count=4)
    result = _abi((1, 2), storage_kind=storage_kind,
                  coordinate_space="canonical_global" if storage_kind == "replicated_local" else "local",
                  owner_stride=8 if storage_kind == "compact_per_owner" else 0, lane_count=4)
    leaf = _prepare((source, ), (result, ), ("gather_reduce_scatter", ))["leaves"][0]
    assert leaf["writer_active"] == "True"
    assert leaf["partial_owner_count"] == 128
    assert "% 4" in leaf["result_offset"]


@pytest.mark.parametrize("partial_axes", [(0, 1), (0, ), (1, )])
@pytest.mark.parametrize("destination", ["preserved", "broadcast", "resharded"])
@pytest.mark.parametrize("split_kind", ["cyclic", "contiguous"])
def test_partial_reduce_into_block_local_result_then_local_consumer(tmp_path, partial_axes, destination, split_kind):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    placement = fm.Placement((2, 4), "yx", "bb")
    preserved_axes = tuple(axis for axis in range(2) if axis not in partial_axes)
    owners = 1
    for axis in partial_axes:
        owners *= placement.hierarchy[axis]
    rows = 9 if split_kind == "cyclic" else 12
    tensor = fm.tensor_type("float32", (rows, owners * 3))
    if not preserved_axes:
        outer = fm.SBP.broadcast()
    elif split_kind == "cyclic":
        outer = fm.SBP.split_block_cyclic(preserved_axes, 1)
    else:
        count = placement.hierarchy[preserved_axes[0]]
        outer = fm.SBP.split_contiguous(preserved_axes, rows // count)
    distributed = fm.DistributedType(tensor, (outer, fm.SBP.split_contiguous(partial_axes, 3)), placement)

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", tensor)
            local = fm.F.distributed.force_boxing(value, distributed)
            partial = fm.F.math.reduce_sum(local, axes=(1, ), keep_dims=True)
            if destination == "preserved":
                result_policies = partial.type.axis_policies
            elif destination == "broadcast":
                result_policies = (fm.SBP.broadcast(), fm.SBP.broadcast())
            else:
                result_policies = (fm.SBP.split_block_cyclic(partial_axes, 2), fm.SBP.broadcast())
            materialized_type = fm.DistributedType(partial.type.tensor, result_policies, placement)
            reduced = fm.F.distributed.force_boxing(partial, materialized_type, name="all_reduce")
            consumed = fm.F.math.silu(reduced)
            result = fm.F.distributed.force_boxing(consumed, consumed.type.tensor)
            self.function("main", (value, ), (result, ))

    module = Compiler().compile(
        Graph(
            dialect="distributed",
            stage="frozen_constants",
            entry="main",
            metadata={"auto_distribution": {"placement": placement.to_data()}},
        ).build()).module
    descriptor = fm.verify_buffer_plan(module).buffer_map["all_reduce"]
    assert descriptor.distributed_storage_kind == fm.DistributedBufferStorageKind.COMPACT_LOCAL
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    value = (torch.arange(rows * owners * 3, device="cuda").reshape(rows, owners * 3) % 13 - 6).float()
    expected = torch.nn.functional.silu(value.sum(-1, keepdim=True)).cpu()
    runtime.prepare(value)
    for _ in range(3):
        torch.testing.assert_close(runtime.run(value).cpu(), expected, rtol=2e-6, atol=2e-6)


def test_packed_split_k_projection_reduces_vector_packets_into_private_broadcast(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    mesh = fm.Placement((8, 16), "yx", "bb")
    broad = fm.SBP.broadcast()
    value_type = fm.tensor_type("bfloat16", (1, 2048))
    weight_type = fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (128, 64))

    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", value_type)
            weight = self.input("weight", weight_type)
            value_b = fm.F.distributed.boxing(value, fm.DistributedType(value_type, (broad, broad), mesh))
            activated = fm.F.math.silu(value_b)
            lhs = fm.F.distributed.sharded_view(activated, fm.DistributedType(
                value_type, (broad, fm.SBP.split_contiguous((0,), 256)), mesh))
            rhs = fm.F.distributed.boxing(weight, fm.DistributedType(weight_type, (
                fm.SBP.split_contiguous((0,), 16), fm.SBP.split_contiguous((1,), 4)), mesh))
            none = fm.F.builtin.none()
            partial = fm.F.ntt.packed_matmul(lhs, rhs, none, none, output_data_type="float32", metadata={
                "selected_vectorization": "vectorization.matmul.n",
                "selected_vector_axes": (1, 1), "selected_vector_lanes": (2, 4),
            })
            reduced = fm.F.distributed.force_boxing(partial, fm.DistributedType(
                partial.type.tensor, (broad, broad), mesh), name="all_reduce")
            scalar = fm.F.tensors.bitcast(reduced, "float32")
            output = fm.F.math.silu(scalar)
            self.function("main", (value, weight), (fm.F.distributed.force_boxing(output, output.type.tensor),))

    module = Compiler().compile(Graph(dialect="distributed", stage="frozen_constants", entry="main",
                                     metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()).module
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
                   device="cuda:0")
    generator = torch.Generator(device="cuda").manual_seed(20260913)
    value = torch.randn((1, 2048), device="cuda", generator=generator).bfloat16()
    weight = (torch.randn((512, 2048), device="cuda", generator=generator) / 32).bfloat16()
    packed = weight.reshape(64, 8, 128, 2, 8).permute(2, 0, 1, 3, 4).contiguous()
    expected = torch.nn.functional.silu(torch.nn.functional.silu(value).float() @ weight.float().T)
    runtime.prepare(value, packed)
    for _ in range(3):
        torch.testing.assert_close(runtime.run(value, packed).cpu(), expected.cpu(), rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("lanes", [1, 8])
def test_qkv_collective_routes_flat_partials_to_head_aligned_results(tmp_path, lanes):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    mesh = fm.Placement((2, 4), "yx", "bb")
    broad = fm.SBP.broadcast()
    heads = (5, 3, 3)

    class Graph(fm.Module):
        def forward(self):
            inputs, partials, destinations = [], [], []
            for index, count in enumerate(heads):
                tensor = fm.tensor_type("float32", (2, count * 128))
                source = self.input(f"value{index}", tensor)
                inputs.append(source)
                local = fm.F.distributed.force_boxing(source, fm.DistributedType(tensor, (
                    fm.SBP.split_contiguous((0,), 1), fm.SBP.split_block_cyclic((1,), 32)), mesh))
                partial = fm.F.math.reduce_sum(local, axes=(0,), keep_dims=True)
                if lanes != 1:
                    partial = fm.F.tensors.pack(partial, axes=(-1,), lanes=(lanes,))
                partials.append(partial)
                destinations.append(fm.DistributedType(partial.type.tensor, (
                    broad, fm.SBP.split_block_cyclic((1 if index == 0 else 0,), 128 // lanes)), mesh))
            combined = fm.F.ntt.packed_qkv_parallel_linear_combine(
                fm.F.builtin.tuple(*partials), fm.TupleType(tuple(destinations)), name="collective")
            outputs = []
            for index, count in enumerate(heads):
                field = fm.F.tensors.get_item(combined, index)
                view = fm.F.tensors.reshape(field, shape=(1, count, 128 // lanes), name=f"heads{index}")
                if lanes != 1:
                    view = fm.F.tensors.bitcast(view, "float32")
                consumed = fm.F.math.silu(view)
                outputs.append(fm.F.distributed.force_boxing(consumed, consumed.type.tensor))
            self.function("main", tuple(inputs), tuple(outputs))

    module = Compiler().compile(Graph(
        dialect="distributed", stage="distributed", entry="main",
        metadata={"auto_distribution": {"placement": mesh.to_data()}},
    ).build()).module
    buffers = fm.verify_buffer_plan(module).buffer_map
    for index in range(3):
        view = buffers[f"heads{index}"]
        assert view.alias_of is not None
        assert view.mem_span == buffers[view.alias_of].mem_span
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
                   device="cuda:0")
    values = tuple((torch.arange(2 * count * 128, device="cuda").reshape(2, count * 128) % 17 - 8).float()
                   for count in heads)
    expected = tuple(torch.nn.functional.silu(value.sum(0).reshape(1, count, 128))
                     for value, count in zip(values, heads))
    outputs = tuple(torch.empty_like(value) for value in expected)
    runtime.prepare(*values, *outputs)
    for _ in range(3):
        for output in outputs:
            output.fill_(float("nan"))
        runtime.run_into(*values, *outputs)
        for actual, reference in zip(outputs, expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-6)
