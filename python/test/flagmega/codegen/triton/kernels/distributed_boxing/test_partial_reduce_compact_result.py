# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""All-reduce writes every private destination, including broadcast owners."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError
from triton.flagmega.runtime import load
from .test_partial_reduce_local_abi import _abi, _prepare


@pytest.mark.parametrize("sharing_scope", ["chip", "block"])
def test_plain_result_addressing_uses_physical_sharing_not_distributed_storage_kind(sharing_scope):
    source = _abi((1, 48), local_shape=(1, 3), storage_kind="compact_per_owner", coordinate_space="local",
                  coordinates=("local_coord_0", "local_coord_1 + shard_coord_1 * 3"),
                  axis_policies=({"kind": "broadcast"}, {"kind": "split", "stages":
                                                         ({"hierarchy_axes":
                                                           (1, )}, )}), partial_axes=(0, ), owner_stride=3)
    result = _abi((1, 48), storage_kind="compact_local")
    result.update(distributed_type=None, memory_sharing_scope=sharing_scope)
    if sharing_scope == "block":
        # A private full tensor needs routed gathering of the other shards.
        with pytest.raises(CodegenError, match="different compact owner map"):
            _prepare((source, ), (result, ), ("gather_reduce_scatter", ))
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
def test_partial_reduce_into_block_local_result_then_local_consumer(tmp_path, partial_axes):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    placement = fm.Placement((2, 4), "yx", "bb")
    preserved_axes = tuple(axis for axis in range(2) if axis not in partial_axes)
    owners = 1
    for axis in partial_axes:
        owners *= placement.hierarchy[axis]
    tensor = fm.tensor_type("float32", (9, owners * 3))
    outer = fm.SBP.split_block_cyclic(preserved_axes, 1) if preserved_axes else fm.SBP.broadcast()
    distributed = fm.DistributedType(tensor, (outer, fm.SBP.split_contiguous(partial_axes, 3)), placement)

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", tensor)
            local = fm.F.distributed.force_boxing(value, distributed)
            partial = fm.F.math.reduce_sum(local, axes=(1, ), keep_dims=True)
            materialized_type = fm.DistributedType(partial.type.tensor, partial.type.axis_policies, placement)
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
    value = (torch.arange(9 * owners * 3, device="cuda").reshape(9, owners * 3) % 13 - 6).float()
    expected = torch.nn.functional.silu(value.sum(-1, keepdim=True)).cpu()
    runtime.prepare(value)
    for _ in range(3):
        torch.testing.assert_close(runtime.run(value).cpu(), expected, rtol=2e-6, atol=2e-6)
