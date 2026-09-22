"""Iluvatar-specific coverage for tle.remote.

The shard_id/scope bookkeeping of tle.remote is builder-agnostic and is already
covered by python/test/tle/unit/test_tle_distributed.py, which iluvatar CI runs
directly. Trunk's python/test/tle/unit/test_tle_d2d_barrier.py pins the backend
part, but it asserts on compiled.asm['ptx'], which corex does not produce, so
this file pins the same properties against llir: which FlagCX runtime call the
device space lowers to, the address space that call has to use, and how the
spaces corex cannot implement are reported.
"""
import re

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from utils import compile_iluvatar


@triton.jit
def _remote_read_kernel(out_ptr, dmem, dtype: tl.constexpr):
    pid = tl.program_id(0)
    remote_mem = tle.remote(dmem, space="device", dtype=dtype, shard_id=1, offset=pid)
    tl.store(out_ptr + pid, tl.load(remote_mem))


@triton.jit
def _remote_no_offset_kernel(out_ptr, dmem, dtype: tl.constexpr):
    remote_mem = tle.remote(dmem, space="device", dtype=dtype, shard_id=1)
    tl.store(out_ptr, tl.load(remote_mem))


@triton.jit
def _cluster_remote_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.float32, nv_mma_shared_layout=False)
    local = tle.gpu.local_ptr(smem, (offs, ))
    tl.store(local, tl.zeros([BLOCK], tl.float32))
    tle.distributed_barrier()
    remote = tle.remote(local, 0, space="cluster")
    tl.store(out_ptr + offs, tl.load(remote))


def _compile_remote_read(dtype=tl.float32):
    return compile_iluvatar(
        _remote_read_kernel,
        signature={"out_ptr": "*fp32", "dmem": "i64", "dtype": "constexpr"},
        constexprs={"dtype": dtype},
    )


class TestDeviceSpaceRemote:

    def test_remote_uses_iluvatar_tle_op(self):
        ttir = _compile_remote_read().asm["ttir"]
        assert "iluvatar_tle.remote_pointers" in ttir, ttir
        assert re.search(r"(?<!iluvatar_)\btle\.", ttir) is None, ttir

    def test_remote_carries_device_space_attr(self):
        ttir = _compile_remote_read().asm["ttir"]
        assert 'space = "device"' in ttir, ttir

    def test_remote_survives_to_ttgir(self):
        ttgir = _compile_remote_read().asm["ttgir"]
        assert "remote_pointers" in ttgir, ttgir

    def test_remote_lowers_to_flagcx_abi(self):
        llir = _compile_remote_read().asm["llir"]
        assert "iluvatar_tle." not in llir, llir
        assert re.search(r"@flagcxGetIntraPointerC\(|flagcxGetIntraPointerC\.exit", llir), llir

    # Same address space trap as the device barrier: a call whose pointer
    # address space disagrees with the linked definition makes corex
    # mis-resolve the implicit kernel arguments, and the callee reads garbage
    # kernel state. Both ends are AS1, so the address space is pinned here
    # rather than left to the runtime test.
    def test_remote_call_matches_corex_bitcode_abi(self):
        llir = _compile_remote_read().asm["llir"]
        assert re.search(r"inttoptr i64 %\d+ to ptr addrspace\(1\)", llir), llir
        assert re.search(r"load [^\n]*, ptr addrspace\(1\)", llir), llir

    # The peer address comes back through the P2P mapping, so it is global
    # memory and the adapter returns it as AS1. The kernel loads from it
    # directly; anything that has to cast the result means the two sides
    # disagree about the address space again.
    def test_remote_result_is_already_global(self):
        llir = _compile_remote_read().asm["llir"]
        kernel = re.search(r"define[^\n]*@_remote_read_kernel\b.*?\n}", llir, re.S)
        assert kernel, llir
        assert re.search(r"@llvm\.bi\.load[^\n]*ptr addrspace\(1\)", kernel.group(0)), kernel.group(0)

    # The frontend offset is an element index while the FlagCX ABI takes bytes,
    # so the scale has to be derived from the pointee type. A constant offset is
    # folded into the specialized callee and leaves nothing to look at, so use a
    # runtime one; LLVM turns the multiply into a shift by log2(element size).
    # Getting this wrong reads the wrong element rather than failing, which is
    # what the two-device test in test_tle_remote_pointers_multi_gpu.py catches.
    @pytest.mark.parametrize("dtype, log2_elem_bytes", [(tl.float32, 2), (tl.float16, 1)])
    def test_offset_is_scaled_to_bytes(self, dtype, log2_elem_bytes):
        llir = _compile_remote_read(dtype).asm["llir"]
        assert re.search(rf"\b(?:shl|mul)\b[^\n]*, {log2_elem_bytes}\b", llir), llir

    def test_device_remote_without_offset_is_rejected(self):
        with pytest.raises(Exception, match="require an offset"):
            compile_iluvatar(
                _remote_no_offset_kernel,
                signature={"out_ptr": "*fp32", "dmem": "i64", "dtype": "constexpr"},
                constexprs={"dtype": tl.float32},
            )


class TestUnsupportedRemoteSpaces:

    # A corex cluster is always one CTA, so mapping a shared pointer onto a peer
    # CTA can only ever return the local pointer. Trunk lowers this with
    # nvvm.mapa; degrading it to a local read here would silently drop the
    # cross-CTA access, so it is rejected the same way submesh barriers are.
    # The reason is an MLIR diagnostic, so check stderr rather than the message.
    def test_cluster_remote_is_rejected(self, capfd):
        with pytest.raises(RuntimeError):
            compile_iluvatar(
                _cluster_remote_kernel,
                signature={"out_ptr": "*fp32", "BLOCK": "constexpr"},
                constexprs={"BLOCK": 64},
            )
        assert "CTA cluster launch" in capfd.readouterr().err
