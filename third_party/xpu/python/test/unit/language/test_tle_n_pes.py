"""Compile coverage for the XPU TLE peer-count path."""

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from triton.backends.xpu.tle_supported import TLE_SUPPORTED_PRIMITIVES

pytestmark = pytest.mark.skipif("n_pes" not in TLE_SUPPORTED_PRIMITIVES,
                                reason="XPU FlagCX device bitcode is not available")


@triton.jit
def _store_num_pes(out, comm):
    tl.store(out, tle.n_pes(comm))


@triton.jit
def _store_peer_rank(out, comm, mesh: tl.constexpr):
    rank = tle.shard_id(mesh, "device", device_dptr=comm)
    tl.store(out, (rank + 1) % tle.n_pes(comm))


def _compile(kernel, *, mesh=None):
    signature = {"out": "*i32", "comm": "i64"}
    constants = {}
    if mesh is not None:
        signature["mesh"] = "constexpr"
        constants["mesh"] = mesh
    return triton.compile(triton.compiler.ASTSource(fn=kernel, signature=signature, constexprs=constants))


def test_n_pes_compiles_through_llvm():
    compiled = _compile(_store_num_pes)
    assert compiled.metadata.xpu_uses_flagcx
    if compiled.asm["ttir"]:
        assert "triton_xpu.get_num_pes" in compiled.asm["ttir"]
    if compiled.asm["llir"]:
        assert "triton_xpu.get_num_pes" not in compiled.asm["llir"]
        assert "flagcxDevCommGetIntraSize" in compiled.asm["llir"]


@pytest.mark.skipif("shard_id" not in TLE_SUPPORTED_PRIMITIVES,
                    reason="XPU TLE device-rank lowering has not been enabled")
def test_n_pes_composes_with_shard_id():
    mesh = tle.device_mesh(tle.MeshConfig(device=2))
    compiled = _compile(_store_peer_rank, mesh=mesh)
    assert compiled.metadata.xpu_uses_flagcx
