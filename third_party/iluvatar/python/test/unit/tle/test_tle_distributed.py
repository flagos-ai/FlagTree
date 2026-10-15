"""Iluvatar-specific TLE distributed tests.

Shared Python semantics for device_mesh, MeshConfig, sharding,
make_sharded_tensor and shard_id live in
python/test/tle/unit/test_tle_distributed.py and are run directly from
iluvatar CI. Keep this file focused on backend-local codegen/runtime
differences to avoid duplicated tests drifting apart.

The comm_ptr (``device``/``node`` axis) path of ``tle.shard_id`` is covered by
test_tle_get_device_id.py, which checks the FlagCX lowering.
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from utils import compile_iluvatar

BLOCK_CLUSTER_MESH = tle.device_mesh(tle.MeshConfig(block_cluster=[("cluster_x", 2)]))


@triton.jit
def _shard_id_axis_kernel(out_ptr, mesh: tl.constexpr):
    pid = tl.program_id(0)
    sid = tle.shard_id(mesh, "cluster_x")
    tl.store(out_ptr + pid, sid)


class TestShardId:

    def test_shard_id_axis_codegen_uses_program_id(self):
        compiled = compile_iluvatar(
            _shard_id_axis_kernel,
            signature={"out_ptr": "*i32", "mesh": "constexpr"},
            constexprs={"mesh": BLOCK_CLUSTER_MESH},
        )
        ttir = compiled.asm["ttir"]
        assert "tt.get_program_id" in ttir, ttir
        assert "get_device_id" not in ttir, ttir

    def test_shard_id_cluster_axis_gpu(self):
        # Iluvatar currently infers mesh cluster launch via program_id math but
        # does not expose cluster_dims in kernel metadata like NVIDIA. Use a
        # grid matching launch_size instead of cluster-expanded CTA count.
        grid = 2
        out = torch.empty((grid, ), device="cuda", dtype=torch.int32)

        _shard_id_axis_kernel[(grid, )](
            out,
            mesh=BLOCK_CLUSTER_MESH,
            num_ctas=1,
            num_warps=4,
        )
        torch.cuda.synchronize()

        expected = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
