"""Iluvatar-specific coverage for tle.distributed_barrier.

The mesh/scope bookkeeping of distributed_barrier is builder-agnostic and is
already covered by python/test/tle/unit/test_tle_distributed.py, which iluvatar
CI runs directly. This file pins the backend-local part: which FlagCX runtime
call each barrier_kind lowers to, and how the paths corex cannot implement
(cooperative grid, CTA cluster) are reported.
"""
import re

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C.libtriton.tle import attr

from utils import compile_iluvatar

GRID_MESH = tle.device_mesh(tle.MeshConfig(block=[("block_x", 4)]))
CLUSTER_MESH = tle.device_mesh(tle.MeshConfig(block_cluster=[("cluster_x", 2)]))
# An explicit-space barrier requires a mesh declaring the topology axis it
# synchronizes over. The size does not reach the IR, so any value compiles.
DEVICE_MESH = tle.device_mesh({"device": 2})


@triton.jit
def _device_barrier_kernel(out, comm, mesh: tl.constexpr, barrier_kind: tl.constexpr):
    tle.distributed_barrier(mesh=mesh, device_dptr=comm, space="device", barrier_kind=barrier_kind)
    tl.store(out, 1)


@triton.jit
def _group_barrier_kernel(out, comm, mesh: tl.constexpr, group_kind: tl.constexpr):
    tle.distributed_barrier(mesh=mesh, device_dptr=comm, space="device", group_kind=group_kind)
    tl.store(out, 1)


@triton.jit
def _local_barrier_kernel(out):
    tle.distributed_barrier()
    tl.store(out, 1)


@triton.jit
def _mesh_barrier_kernel(out, mesh: tl.constexpr):
    tle.distributed_barrier(mesh=mesh)
    tl.store(out, 1)


def _compile_device_barrier(barrier_kind="sync"):
    return compile_iluvatar(
        _device_barrier_kernel,
        signature={"out": "*i32", "comm": "i64", "mesh": "constexpr", "barrier_kind": "constexpr"},
        constexprs={"mesh": DEVICE_MESH, "barrier_kind": barrier_kind},
    )


def _compile_group_barrier(group_kind):
    return compile_iluvatar(
        _group_barrier_kernel,
        signature={"out": "*i32", "comm": "i64", "mesh": "constexpr", "group_kind": "constexpr"},
        constexprs={"mesh": DEVICE_MESH, "group_kind": group_kind},
    )


class TestDeviceSpaceBarrier:

    def test_device_barrier_uses_iluvatar_tle_op(self):
        ttir = _compile_device_barrier().asm["ttir"]
        assert "iluvatar_tle.distributed_barrier" in ttir, ttir
        assert re.search(r"(?<!iluvatar_)\btle\.", ttir) is None, ttir

    def test_device_barrier_carries_device_space_attrs(self):
        ttir = _compile_device_barrier().asm["ttir"]
        assert 'space = "device"' in ttir, ttir
        assert 'barrier_type = "sync"' in ttir, ttir
        assert f"order = {int(attr.MemoryOrder.AcqRel)} : i32" in ttir, ttir
        assert f"memory_scope = {int(attr.SyncScope.System)} : i32" in ttir, ttir
        assert 'group_kind = "block"' in ttir, ttir

    def test_device_barrier_survives_to_ttgir(self):
        ttgir = _compile_device_barrier().asm["ttgir"]
        assert "distributed_barrier" in ttgir, ttgir

    @pytest.mark.parametrize(
        "barrier_kind, symbol",
        [
            ("sync", "flagcxIntraBarrierSyncS"),
            ("arrive", "flagcxIntraBarrierArriveS"),
            ("wait", "flagcxIntraBarrierWaitS"),
        ],
    )
    def test_device_barrier_lowers_to_flagcx_abi(self, barrier_kind, symbol):
        llir = _compile_device_barrier(barrier_kind).asm["llir"]
        assert "iluvatar_tle." not in llir, llir
        assert re.search(rf"@{symbol}\(|{symbol}\.exit", llir), llir

    def test_device_barrier_call_matches_corex_bitcode_abi(self):
        llir = _compile_device_barrier().asm["llir"]
        assert re.search(r"flagcxIntraBarrierSyncS\.exit", llir), llir
        assert re.search(r"inttoptr i64 %\d+ to ptr addrspace\(1\)", llir), llir
        assert re.search(r"load [^\n]*, ptr addrspace\(1\)", llir), llir

    # LLVM constant-folds the coop kind, index, multimem and order operands into
    # a specialized clone of the callee, so their literal values are gone by the
    # time llir is emitted. What survives is the cooperative group the folded
    # body selects, which is the property those operands exist to carry: a block
    # barrier synchronizes the whole CTA, a warp barrier must not.
    @pytest.mark.parametrize(
        "group_kind, expect_cta_barrier",
        [("block", True), ("warp", False), ("thread", False)],
    )
    def test_device_barrier_group_kind_selects_cooperative_group(self, group_kind, expect_cta_barrier):
        llir = _compile_group_barrier(group_kind).asm["llir"]
        assert ("bi.sl.barrier" in llir) is expect_cta_barrier, llir

    # flagcxCoopKind_t has no grid entry: 3 is tile_span. Trunk's mapping sends
    # "grid" to 3, which would quietly synchronize a tile span instead, so the
    # kinds the scalar barrier ABI cannot express are rejected outright.
    @pytest.mark.parametrize("group_kind", ["grid", "tile_span", "lanes"])
    def test_unsupported_group_kind_is_rejected(self, group_kind, capfd):
        with pytest.raises(RuntimeError):
            _compile_group_barrier(group_kind)
        assert "does not support group_kind" in capfd.readouterr().err


class TestLocalBarrierFallback:

    def test_plain_barrier_stays_cta_local(self):
        compiled = compile_iluvatar(_local_barrier_kernel, signature={"out": "*i32"})
        ttir = compiled.asm["ttir"]
        assert "iluvatar_tle.distributed_barrier" in ttir, ttir
        llir = compiled.asm["llir"]
        assert "iluvatar_tle." not in llir, llir
        assert "flagcxIntraBarrier" not in llir, llir


def _compile_mesh_barrier(mesh):
    return compile_iluvatar(
        _mesh_barrier_kernel,
        signature={"out": "*i32", "mesh": "constexpr"},
        constexprs={"mesh": mesh},
    )


class TestUnsupportedMeshBarriers:

    # The reason is reported as an MLIR diagnostic; the Python exception only
    # carries the generic "PassManager::run failed", so check stderr instead.
    def test_grid_mesh_barrier_is_rejected(self, capfd):
        with pytest.raises(RuntimeError):
            _compile_mesh_barrier(GRID_MESH)
        assert "cooperative grid launch" in capfd.readouterr().err

    def test_cluster_mesh_barrier_stays_cta_local(self):
        llir = _compile_mesh_barrier(CLUSTER_MESH).asm["llir"]
        assert "flagcxIntraBarrier" not in llir, llir

    def test_submesh_barrier_is_rejected(self, capfd):
        with pytest.raises(RuntimeError):
            _compile_mesh_barrier(CLUSTER_MESH[0:1])
        assert "CTA cluster launch" in capfd.readouterr().err
