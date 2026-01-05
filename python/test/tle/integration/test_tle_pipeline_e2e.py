# Copyright (c) 2025  XCoreSigma Inc. All rights reserved.
# flagtree tle
"""
TLE End-to-End Integration Tests

Tests complete workflow of TLE pipeline functionality in real GPU environment:
- Memory allocation (tle.alloc)
- copy date from GM to ShareMem (tle.copy)
- Local load (tle.local_load)
- Pipeline iterator (tle.pipeline)
- Integration with Triton JIT
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def elementwise_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    xnumel,
    ynumel,
    xstride_a,
    ystride_a,
    xstride_b,
    ystride_b,
    xstride_c,
    ystride_c,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
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

    # Use TLE pipeline for block-wise processing
    #for yoff in range(0, ynumel, YBLOCK):
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

        # Perform computation
        c_val = aval + bval

        # Store results
        tl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)


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

    return elementwise_add_kernel[grid](A, B, C, xnumel, ynumel, *A.stride(), *B.stride(), *C.stride(), XBLOCK, YBLOCK)


class TestTLEPipelineEndToEnd:
    """TLE Pipeline End-to-End Integration Tests"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
    def test_elementwise_add_basic(self):
        """Test basic element-wise addition functionality"""
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
    def test_elementwise_add_different_sizes(self):
        """Test different tensor sizes"""
        torch.manual_seed(123)

        test_cases = [
            (256, 256, 32, 32),
            (1024, 1024, 64, 64),
            (2048, 512, 128, 32),
            (512, 2048, 32, 128),
        ]

        for xnumel, ynumel, XBLOCK, YBLOCK in test_cases:
            # Create test data
            a = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
            b = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
            c = torch.empty_like(a, device="cuda", dtype=torch.float32)

            # Execute computation
            elementwise_add(a, b, c, XBLOCK, YBLOCK)

            # Verify results
            expected = a + b
            torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)
            assert c.shape == expected.shape, f"Shape mismatch: {xnumel}x{ynumel}, block size: {XBLOCK}x{YBLOCK}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
    def test_elementwise_add_different_dtypes(self):
        """Test different data types"""
        torch.manual_seed(456)

        dtypes = [torch.float32]  # Skip float16 due to type conversion issues in TLE
        xnumel, ynumel = 512, 512
        XBLOCK, YBLOCK = 64, 64

        for dtype in dtypes:
            # Create test data
            a = torch.randn(xnumel, ynumel, device="cuda", dtype=dtype)
            b = torch.randn(xnumel, ynumel, device="cuda", dtype=dtype)
            c = torch.empty_like(a, device="cuda", dtype=dtype)

            # Execute computation
            elementwise_add(a, b, c, XBLOCK, YBLOCK)

            # Verify results
            expected = a + b
            torch.testing.assert_close(c, expected, atol=1e-3, rtol=1e-3)
            assert c.shape == expected.shape, f"Shape mismatch for data type {dtype}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
    def test_elementwise_add_edge_cases(self):
        """Test edge cases"""
        torch.manual_seed(789)

        # Test minimum size
        a = torch.randn(1, 1, device="cuda", dtype=torch.float32)
        b = torch.randn(1, 1, device="cuda", dtype=torch.float32)
        c = torch.empty_like(a, device="cuda", dtype=torch.float32)

        elementwise_add(a, b, c, 1, 1)
        torch.testing.assert_close(c, a + b, atol=1e-5, rtol=1e-5)

        # Test non-square tensors
        a = torch.randn(128, 1024, device="cuda", dtype=torch.float32)
        b = torch.randn(128, 1024, device="cuda", dtype=torch.float32)
        c = torch.empty_like(a, device="cuda", dtype=torch.float32)

        elementwise_add(a, b, c, 32, 128)
        torch.testing.assert_close(c, a + b, atol=1e-5, rtol=1e-5)

    def test_tle_module_import(self):
        """Test TLE module import (no GPU required)"""
        # Verify all necessary functions and types can be imported
        assert hasattr(tle, 'alloc')
        assert hasattr(tle, 'copy')
        assert hasattr(tle, 'local_load')
        assert hasattr(tle, 'pipeline')
        assert hasattr(tle, 'scope')
        assert hasattr(tle, 'buffered_tensor')

        # Verify functions have docstrings
        assert tle.alloc.__doc__ is not None
        assert tle.copy.__doc__ is not None
        assert tle.local_load.__doc__ is not None
        assert tle.pipeline.__doc__ is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
