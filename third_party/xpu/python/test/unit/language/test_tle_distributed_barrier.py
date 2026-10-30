"""Compile coverage for the XPU FlagCX device barrier."""

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from triton.backends.xpu.tle_supported import TLE_SUPPORTED_PRIMITIVES

pytestmark = pytest.mark.skipif("distributed_barrier" not in TLE_SUPPORTED_PRIMITIVES,
                                reason="XPU FlagCX device bitcode is not available")

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))


@triton.jit
def _barrier(out, comm, mesh: tl.constexpr, kind: tl.constexpr):
    tle.distributed_barrier(mesh, comm, space="device", barrier_kind=kind)
    tl.store(out, 1)


@pytest.mark.parametrize("kind,symbol", [("sync", "flagcxIntraBarrierSyncS"), ("arrive", "flagcxIntraBarrierArriveS"),
                                         ("wait", "flagcxIntraBarrierWaitS")])
def test_device_barrier_uses_xputle_and_flagcx(kind, symbol):
    source = triton.compiler.ASTSource(
        fn=_barrier,
        signature={"out": "*i32", "comm": "i64", "mesh": "constexpr", "kind": "constexpr"},
        constexprs={"mesh": DEVICE_MESH, "kind": kind},
    )
    compiled = triton.compile(source)
    assert compiled.metadata.xpu_uses_flagcx
    if compiled.asm["ttir"]:
        assert "triton_xpu.distributed_barrier" in compiled.asm["ttir"]
    if compiled.asm["llir"]:
        assert "triton_xpu.distributed_barrier" not in compiled.asm["llir"]
        assert symbol in compiled.asm["llir"]
