import re

import triton
import triton.language as tl
from triton.experimental.tle.language.distributed import _get_local_rank, n_pes

from utils import compile_iluvatar


@triton.jit
def _n_pes_kernel(out, comm):
    tl.store(out, n_pes(comm))


@triton.jit
def _peer_rank_kernel(out, comm):
    rank = _get_local_rank(comm)
    peer = (rank + 1) % n_pes(comm)
    tl.store(out, peer)


def _compile_n_pes():
    return compile_iluvatar(
        _n_pes_kernel,
        signature={"out": "*i32", "comm": "i64"},
    )


def _compile_peer_rank():
    return compile_iluvatar(
        _peer_rank_kernel,
        signature={"out": "*i32", "comm": "i64"},
    )


def test_n_pes_uses_iluvatar_tle_op():
    ttir = _compile_n_pes().asm["ttir"]
    assert "iluvatar_tle.get_num_pes" in ttir, ttir
    assert re.search(r"(?<!iluvatar_)\btle\.", ttir) is None, ttir


def test_n_pes_survives_to_ttgir():
    ttgir = _compile_n_pes().asm["ttgir"]
    assert "get_num_pes" in ttgir, ttgir


def test_n_pes_lowers_to_flagcx_abi():
    llir = _compile_n_pes().asm["llir"]
    assert "iluvatar_tle." not in llir, llir
    assert re.search(r"inttoptr i64 %\d+ to ptr addrspace\(1\)", llir), llir
    assert re.search(r"load i32, ptr addrspace\(1\)", llir), llir


def test_n_pes_composes_with_get_device_id():
    llir = _compile_peer_rank().asm["llir"]
    assert len(re.findall(r"load i32, ptr addrspace\(1\)", llir)) >= 2, llir
    assert re.search(r"\bsrem i32\b", llir), llir
