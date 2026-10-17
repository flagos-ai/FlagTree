# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Linked FlagCX calls must agree with the C device-library signatures."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from triton.runtime import DistributedRtContext


_MESH = tl.constexpr(tle.device_mesh({"device": 4, "block": [("block", 1)]}))


@triton.jit
def _flagcx_abi(output, device_dptr: tl.constexpr, KIND: tl.constexpr):
    rank = tle.shard_id(_MESH, "device", device_dptr=device_dptr)
    node = tle.shard_id(_MESH, "node", device_dptr=device_dptr)
    peer = tle.remote(device_dptr, rank, space="device", dtype=tl.int32, offset=0)
    tl.store(output, tl.load(peer) + rank + node)
    tle.distributed_barrier(device_dptr=device_dptr, space="device", barrier_kind=KIND)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.require_tle("distributed_barrier")
@pytest.mark.parametrize("kind", ("arrive", "wait", "sync"))
def test_flagcx_device_library_calls_are_directly_inlinable(monkeypatch, kind):
    # Warmup only: the handles are representative 64-bit addresses, never
    # dereferenced. Executing this kernel requires a real four-rank context.
    monkeypatch.setattr(DistributedRtContext, "_instance", None)
    monkeypatch.setattr(DistributedRtContext, "_initialized", False)
    monkeypatch.setattr(DistributedRtContext, "_init_count", 0)
    context = DistributedRtContext(0x7F0000000E00, 0x7F0000001000)
    output = torch.empty(1, dtype=torch.int32, device="cuda")
    compiled = _flagcx_abi.warmup(output, context, kind, grid=(1,), num_warps=4)
    # Exact ABI agreement lets the library's alwaysinline routines inline and
    # specialize the constant cooperation kind. A pointer address-space or
    # return-type mismatch leaves incompatible, indirect calls after linking.
    assert not re.search(
        r"\bcall\b[^\n]*@flagcx(?:DevCommGet|GetIntraPointerC|IntraBarrier)",
        compiled.asm["llir"],
    )
