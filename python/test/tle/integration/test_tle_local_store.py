# Copyright (c) 2025  XCoreSigma Inc. All rights reserved.

"""
TLE Local Store Integration Tests

Tests TLE local store functionality and bidirectional copy operations:
- Local store operations (tle.local_store)
- Copy function bidirectional support (tle.copy)
- Integration with Triton JIT
- Memory allocation and data transfer validation
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tle as tle


@triton.jit
def elementwise_add_kernel(
    a_ptr, b_ptr, c_ptr, xnumel, ynumel,
    xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,
    XBLOCK: tl.constexpr, YBLOCK: tl.constexpr,
):
    """
    Element-wise addition kernel using TLE pipeline

    This kernel demonstrates the complete TLE workflow:
    1. Allocate shared memory buffers
    2. Use pipeline for for-loop iteration
    3. copy data to shared memory
    4. Load data from shared memory for computation
    5. Store results back to global memory
    """
    pid = tl.program_id(0)

    # Calculate row offset for current program
    xoffs = pid * XBLOCK + tl.arange(0, XBLOCK)

    # Calculate global memory pointers
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    # Allocate shared memory buffers
    a_smem = tle.alloc([XBLOCK, YBLOCK], dtype=tl.float32, layout=None, scope=tle.smem)
    b_smem = tle.alloc([XBLOCK, YBLOCK], dtype=tl.float32, layout=None, scope=tle.smem)
    c_smem = tle.alloc([XBLOCK, YBLOCK], dtype=tl.float32, layout=None, scope=tle.smem)

    # Use TLE pipeline for block-wise processing
    for yoff in tle.pipeline(0, ynumel, YBLOCK, num_stages=2):
        # Calculate column offset for current block
        yoffs = tl.arange(0, YBLOCK) + yoff
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]

        # copy data to shared memory
        tle.copy(a_ptrs + ystride_a * yoffs[None, :], a_smem, [XBLOCK, YBLOCK])
        tle.copy(b_ptrs + ystride_b * yoffs[None, :], b_smem, [XBLOCK, YBLOCK])

        # Load data from shared memory
        aval = tle.local_load(a_smem)
        bval = tle.local_load(b_smem)

        c_val = aval + bval
        tle.local_store(c_smem, c_val)
        tle.copy(c_smem, c_ptrs + ystride_c * yoffs[None, :], [XBLOCK, YBLOCK])
        #tle.copy(c_smem, c_ptr, shape, c_stride[x,y], [XBLOCK, YBLOCK])


def elementwise_add(A, B, C, XBLOCK=32, YBLOCK=64):
    """
    Wrapper function to execute element-wise addition using TLE pipeline

    Args:
        A: Input tensor A (CUDA tensor)
        B: Input tensor B (CUDA tensor)
        C: Output tensor C (CUDA tensor)
        XBLOCK: Block size for X dimension
        YBLOCK: Block size for Y dimension
    """
    assert A.shape == B.shape == C.shape, "Input and output tensor shapes must match"
    xnumel, ynumel = A.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )

    return elementwise_add_kernel[grid](
        A, B, C, xnumel, ynumel,
        *A.stride(), *B.stride(), *C.stride(),
        XBLOCK, YBLOCK
    )


class TestTLELocalStore:
    """TLE Local Store Integration Tests"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
    def test_local_store_basic(self):
        """Test basic local store functionality with element-wise addition"""
        torch.manual_seed(42)  # Ensure reproducibility

        xnumel, ynumel = 512, 512
        XBLOCK, YBLOCK = 64, 64

        # Create test data
        a = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
        b = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
        c = torch.empty_like(a, device="cuda", dtype=torch.float32)

        # Execute TLE pipeline computation
        elementwise_add(a, b, c, XBLOCK, YBLOCK)

        # Verify results
        expected = a + b
        torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)



if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v", "-s"])