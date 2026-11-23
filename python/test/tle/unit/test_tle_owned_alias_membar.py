"""Shared byte aliases retain their physical origin across thread layouts."""

import pytest
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl


@triton.jit
def _owned_alias_cross_warp(source, output, WARPS: tl.constexpr):
    arena = tle.gpu.alloc([2048], tl.uint8, nv_mma_shared_layout=False)
    first = tle.gpu.alloc([512], tl.int32, alias=arena, alias_offset_bytes=0, nv_mma_shared_layout=False)
    second = tle.gpu.alloc([128], tl.int32, alias=arena, alias_offset_bytes=512, nv_mma_shared_layout=False)
    write_layout: tl.constexpr = tle.gpu.BlockEncoding([1], [32], [WARPS], [0])
    read_layout: tl.constexpr = tle.gpu.BlockEncoding([2], [32], [WARPS], [0])
    write_index = tle.gpu.set_layout(tl.arange(0, 128), write_layout)
    value = tl.load(source + write_index)
    # Delay the writers in later warps. The read layout assigns their values
    # to earlier warps, exposing a missing CTA retirement/publication edge.
    value = tl.inline_asm_elementwise(
        "{ .reg .u32 tid; .reg .u64 start, now, delta; .reg .pred late, wait; "
        "mov.u32 tid, %tid.x; setp.ge.u32 late, tid, 64; "
        "@!late bra DONE; mov.u64 start, %clock64; "
        "LOOP: mov.u64 now, %clock64; sub.u64 delta, now, start; "
        "setp.lt.u64 wait, delta, 200000; @wait bra LOOP; "
        "DONE: mov.b32 $0, $1; }", constraints="=r,r", args=[value], dtype=tl.int32, is_pure=False, pack=1)
    tl.store(tle.gpu.local_ptr(first, (write_index + 128, )), value)
    read_index = tle.gpu.set_layout(tl.arange(0, 128), read_layout)
    result = tl.load(tle.gpu.local_ptr(second, (read_index, )))
    tl.store(output + read_index, result)


@pytest.mark.skip_if_tle_unsupported("gpu.alloc", "gpu.local_ptr", "gpu.set_layout")
@pytest.mark.skipif(torch.version.hip is not None, reason="Uses NVIDIA PTX to delay writers across warps")
@pytest.mark.parametrize("num_warps", [4, 8])
def test_owned_byte_alias_cross_warp_raw(num_warps):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("The current alias regression uses the SM90 TLE pipeline")
    source = torch.arange(128, device="cuda", dtype=torch.int32) + 1000
    output = torch.empty_like(source)
    for _ in range(16):
        output.fill_(-1)
        _owned_alias_cross_warp[(1, )](source, output, num_warps, num_warps=num_warps)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, source, rtol=0, atol=0)
