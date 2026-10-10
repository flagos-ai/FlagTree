"""The standard Triton IR builder must expose TLE async-copy ops."""

import torch
import triton
import triton.language as tl
import triton._C.libtriton.ir as ir
import triton.experimental.tle.language as tle


def test_standard_ir_builder_exposes_iluvatar_async_copy_ops():
    assert hasattr(ir.builder, "create_async_copy_global_to_local")
    assert hasattr(ir.builder, "create_async_commit_group")
    assert hasattr(ir.builder, "create_async_wait_group")


@triton.jit
def _explicit_partial_wait_kernel(src, dst, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    stage0 = tle.gpu.alloc(
        [BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False)
    stage1 = tle.gpu.alloc(
        [BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False)

    tle.gpu.copy(src + offsets, stage0, [BLOCK], is_async=True)
    tle.gpu.async_commit_group()
    tle.gpu.copy(src + BLOCK + offsets, stage1, [BLOCK], is_async=True)
    tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(1)
    value0 = tl.load(tle.gpu.local_ptr(stage0, (offsets,)))
    tl.store(dst + offsets, value0 + 1.0)
    tle.gpu.async_wait_group(0)
    value1 = tl.load(tle.gpu.local_ptr(stage1, (offsets,)))
    tl.store(dst + BLOCK + offsets, value1 + 2.0)


def test_explicit_partial_wait_survives_pipeline_analysis():
    block = 64
    src = torch.arange(2 * block, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    compiled = _explicit_partial_wait_kernel.warmup(
        src, dst, BLOCK=block, grid=(1,), num_warps=4)

    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "ttg.async_wait {num = 1 : i32, tle.explicit_async_wait}" in ttgir
    assert "ttg.async_wait {num = 0 : i32, tle.explicit_async_wait}" in ttgir
    # BI-V150 drains the async-copy queue with waitcnt; cp.async wait-group
    # intrinsics are the NVIDIA lowering and are not emitted by this backend.
    assert llir.count("llvm.bi.sl.waitcnt(i64 8)") >= 2

    _explicit_partial_wait_kernel[(1,)](src, dst, BLOCK=block, num_warps=4)
    expected = torch.cat((src[:block] + 1.0, src[block:] + 2.0))
    torch.testing.assert_close(dst, expected, atol=0, rtol=0)
