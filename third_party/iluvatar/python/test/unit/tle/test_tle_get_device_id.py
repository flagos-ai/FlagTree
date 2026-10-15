import re

import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.experimental.tle.language.distributed import _get_local_rank

from utils import compile_iluvatar

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))


@triton.jit
def _get_device_id_kernel(out, comm):
    rank = _get_local_rank(comm)
    tl.store(out, rank)


@triton.jit
def _shard_id_kernel(out, comm, mesh: tl.constexpr):
    rank = tle.shard_id(mesh, 'device', device_dptr=comm)
    tl.store(out, rank)


def _compile_get_device_id():
    return compile_iluvatar(
        _get_device_id_kernel,
        signature={"out": "*i32", "comm": "i64"},
    )


def _compile_shard_id():
    return compile_iluvatar(
        _shard_id_kernel,
        signature={"out": "*i32", "comm": "i64", "mesh": "constexpr"},
        constexprs={"mesh": DEVICE_MESH},
    )


def test_get_device_id_uses_iluvatar_tle_op():
    ttir = _compile_get_device_id().asm["ttir"]
    assert "iluvatar_tle.get_device_id" in ttir, ttir
    assert re.search(r"(?<!iluvatar_)\btle\.", ttir) is None, ttir


def test_get_device_id_survives_to_ttgir():
    ttgir = _compile_get_device_id().asm["ttgir"]
    assert "get_device_id" in ttgir, ttgir


def test_get_device_id_lowers_to_flagcx_abi():
    llir = _compile_get_device_id().asm["llir"]
    assert "iluvatar_tle." not in llir, llir
    assert re.search(r"inttoptr i64 %\d+ to ptr addrspace\(1\)", llir), llir
    assert re.search(r"load i32, ptr addrspace\(1\)", llir), llir


def test_shard_id_device_axis_maps_to_get_device_id():
    ttir = _compile_shard_id().asm["ttir"]
    assert "iluvatar_tle.get_device_id" in ttir, ttir
