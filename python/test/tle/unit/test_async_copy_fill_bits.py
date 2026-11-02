import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _stage_fill(src, dst, VALID: tl.constexpr, FILL_ID: tl.constexpr,
                EXPLICIT: tl.constexpr):
    indices = tl.arange(0, 128)
    arena = tle.gpu.alloc((128,), src.dtype.element_ty, nv_mma_shared_layout=False)
    pointers = tle.gpu.local_ptr(arena, (indices,), [128])
    # Use an integer case key: Python treats +0.0 and -0.0 as equal keys.
    if FILL_ID == 0:
        other = 0.0
    elif FILL_ID == 1:
        other = -0.0
    elif FILL_ID == 2:
        other = 1.0
    else:
        other = -float("inf")
    other = tl.full((128,), other, src.dtype.element_ty)
    if EXPLICIT:
        tle.gpu.copy(src + indices, arena, (128,), mask=indices < VALID,
                     other=other, is_async=True)
        tle.gpu.async_commit_group()
        tle.gpu.async_wait_group(0)
    else:
        values = tl.load(src + indices, indices < VALID, other=other)
        tl.store(pointers, values)
    tl.debug_barrier()
    tl.store(dst + indices, tl.load(pointers))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("fill_id", range(4))
@pytest.mark.parametrize("explicit", [False, True])
def test_async_staging_preserves_fill_bits(fill_id, dtype, explicit):
    patterns = {
        torch.float32: [0x00000000, 0x80000000, 0x3F800000, 0xFF800000],
        torch.float16: [0x0000, 0x8000, 0x3C00, 0xFC00],
        torch.bfloat16: [0x0000, 0x8000, 0x3F80, 0xFF80],
        torch.float64: [0x0000000000000000, 0x8000000000000000,
                        0x3FF0000000000000, 0xFFF0000000000000],
    }
    fill_bits = patterns[dtype][fill_id]
    width = 64 if dtype == torch.float64 else 32 if dtype == torch.float32 else 16
    bit_dtype = {16: torch.int16, 32: torch.int32, 64: torch.int64}[width]
    source = torch.full((128,), 7.5, device="cuda", dtype=dtype)
    output = torch.empty_like(source)
    signed_bits = fill_bits if fill_bits < 2**(width - 1) else fill_bits - 2**width
    expected = torch.full((128,), signed_bits, device="cuda", dtype=bit_dtype)
    # Explicit cp.async needs at least four bytes with a uniform mask.
    valid = 74 if explicit else 73
    expected[:valid] = source[:valid].view(bit_dtype)
    compiled = _stage_fill[(1,)](source, output, valid, fill_id, explicit, num_warps=4)
    torch.testing.assert_close(output.view(bit_dtype), expected, rtol=0, atol=0)
    if dtype == torch.float32 and (fill_bits == 0 or explicit):
        assert "cp.async" in compiled.asm["ptx"]
    if explicit and fill_bits == 0:
        assert "st.shared" not in compiled.asm["ptx"]


@triton.jit
def _stage_runtime_fill(src, dst, VALID: tl.constexpr,
                        MASK_AFTER_LOWERING: tl.constexpr):
    indices = tl.arange(0, 128)
    arena = tle.gpu.alloc((128,), src.dtype.element_ty, nv_mma_shared_layout=False)
    if MASK_AFTER_LOWERING:
        mask = (tl.program_id(0) & 15) * 128 + indices < 2048
    else:
        mask = indices < VALID
    tle.gpu.copy(src + indices, arena, (128,), mask=mask,
                 other=indices.to(src.dtype.element_ty), is_async=True)
    tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(0)
    tl.debug_barrier()
    pointers = tle.gpu.local_ptr(arena, (indices,), [128])
    tl.store(dst + indices, tl.load(pointers))


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize("mask_case", ["mixed", "empty", "full", "full_after_lowering"])
def test_async_staging_preserves_runtime_integer_fill(dtype, mask_case):
    alignment = 4 if dtype == torch.int8 else 2 if dtype == torch.int16 else 1
    valid = ((73 + alignment - 1) // alignment * alignment
             if mask_case == "mixed" else 0 if mask_case == "empty" else 128)
    source = torch.full((128,), 7, device="cuda", dtype=dtype)
    output = torch.empty_like(source)
    expected = torch.arange(128, device="cuda", dtype=dtype)
    expected[:valid] = 7
    compiled = _stage_runtime_fill[(1,)](source, output, valid,
                                       mask_case == "full_after_lowering", num_warps=4)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    if mask_case == "mixed":
        assert "cp.async" in compiled.asm["ptx"]
    if mask_case in ("full", "full_after_lowering"):
        assert "st.shared" not in compiled.asm["ptx"]
