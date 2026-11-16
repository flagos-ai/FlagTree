"""Compile coverage for the XPU TLE device-rank path."""

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from triton.backends.xpu.tle_supported import TLE_SUPPORTED_PRIMITIVES

pytestmark = pytest.mark.skipif(
    "shard_id" not in TLE_SUPPORTED_PRIMITIVES,
    reason="XPU FlagCX device bitcode is not available",
)

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))


@triton.jit
def _store_local_rank(out, comm, mesh: tl.constexpr):
    tl.store(out, tle.shard_id(mesh, "device", device_dptr=comm))


@triton.jit
def _store_implicit_local_rank(out):
    tl.store(out, tle._get_local_rank(None))


def test_shard_id_device_axis_compiles_through_llvm():
    source = triton.compiler.ASTSource(
        fn=_store_local_rank,
        signature={"out": "*i32", "comm": "i64", "mesh": "constexpr"},
        constexprs={"mesh": DEVICE_MESH},
    )
    compiled = triton.compile(source)
    assert compiled.metadata.xpu_uses_flagcx
    if compiled.asm["ttir"]:
        assert "triton_xpu.get_device_id" in compiled.asm["ttir"]
    if compiled.asm["llir"]:
        assert "triton_xpu.get_device_id" not in compiled.asm["llir"]
        assert "flagcxDevCommGetIntraRank" in compiled.asm["llir"]


def test_implicit_device_rank_uses_flagcx_context(monkeypatch):
    from triton.runtime import DistributedRtContext

    monkeypatch.setenv("FLAGTREE_LITE_DIST", "1")
    monkeypatch.setattr(DistributedRtContext, "_init_count", 1)
    source = triton.compiler.ASTSource(fn=_store_implicit_local_rank, signature={"out": "*i32"})
    compiled = triton.compile(source)
    assert compiled.metadata.flagcx_implicit_args
    assert "flagcxDevCommGetIntraRank" in compiled.asm["llir"]
