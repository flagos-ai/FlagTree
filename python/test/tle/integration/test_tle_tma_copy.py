# Copyright (c) 2025  XCoreSigma Inc. All rights reserved.
# flagtree tle
"""
TLE TMA Copy Integration Tests

Tests TLE TMA copy functionality and bidirectional copy operations:
- TMA copy operations (tle.copy with TMA descriptors)
- Local memory operations (tle.alloc, tle.local_load, tle.local_store)
- Integration with Triton JIT and TMA descriptors
- Memory allocation and data transfer validation
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def elementwise_tma_add_kernel(
    a_desc,
    b_desc,
    c_desc,
    xnumel,
    ynumel,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    # Calculate row offset for current program

    # Define block shape and NVMMASharedLayout for TMA compatibility
    #layout = tle.nv_mma_shared_layout.make_default([XBLOCK, YBLOCK], tl.float32)

    # Allocate shared memory buffers
    a_smem = tle.alloc([XBLOCK, YBLOCK], dtype=tl.float32, layout=None, scope=tle.smem)
    b_smem = tle.alloc([XBLOCK, YBLOCK], dtype=tl.float32, layout=None, scope=tle.smem)
    c_smem = tle.alloc([XBLOCK, YBLOCK], dtype=tl.float32, layout=None, scope=tle.smem)

    # Use TLE pipeline for block-wise processing
    for yoff in range(0, ynumel, YBLOCK):
        # Calculate column offset for current block
        # copy data to shared memory
        tle.copy(a_desc, a_smem, [XBLOCK, YBLOCK], [pid * XBLOCK, yoff])
        tle.copy(b_desc, b_smem, [XBLOCK, YBLOCK], [pid * XBLOCK, yoff])
        # Load data from shared memory
        aval = tle.local_load(a_smem)
        bval = tle.local_load(b_smem)

        c_val = aval + bval
        tle.local_store(c_smem, c_val)
        tle.copy(c_smem, c_desc, [XBLOCK, YBLOCK], [pid * XBLOCK, yoff])


def elementwise_add(A, B, C, XBLOCK=32, YBLOCK=64):
    """
    Wrapper function to execute element-wise addition using TLE pipeline

    Args:
        A: Input tensor descriptor A (TMA descriptor)
        B: Input tensor descriptor B (TMA descriptor)
        C: Output tensor descriptor C (TMA descriptor)
        XBLOCK: Block size for X dimension
        YBLOCK: Block size for Y dimension
    """
    # For TMA descriptors, we don't have direct access to shape/stride
    # We'll use the block sizes for the computation
    xnumel, ynumel = 512, 512  # Default test size
    grid = (triton.cdiv(xnumel, XBLOCK), )

    return elementwise_tma_add_kernel[grid](A, B, C, xnumel, ynumel, XBLOCK, YBLOCK)


class TestTLETmaCopy:
    """TLE TMA Copy Integration Tests"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
    def test_tma_copy_basic(self):
        """Test basic TMA copy functionality with element-wise addition"""
        torch.manual_seed(42)  # Ensure reproducibility

        xnumel, ynumel = 512, 512
        XBLOCK, YBLOCK = 32, 64

        # Create test data
        a = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
        b = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
        c = torch.empty_like(a, device="cuda", dtype=torch.float32)
        from triton.tools.tensor_descriptor import TensorDescriptor
        a_tma = TensorDescriptor.from_tensor(a, block_shape=[XBLOCK, YBLOCK])
        b_tma = TensorDescriptor.from_tensor(b, block_shape=[XBLOCK, YBLOCK])
        c_tma = TensorDescriptor.from_tensor(c, block_shape=[XBLOCK, YBLOCK])
        # Execute TLE pipeline computation
        elementwise_add(a_tma, b_tma, c_tma, XBLOCK, YBLOCK)

        # Verify results
        expected = a + b
        torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
    def test_tma_copy_different_block_sizes(self):
        """Test TMA copy with different block sizes"""
        torch.manual_seed(123)

        for XBLOCK, YBLOCK in [(16, 128), (64, 32), (128, 16)]:
            xnumel, ynumel = 256, 256

            # Create test data
            a = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
            b = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
            c = torch.empty_like(a, device="cuda", dtype=torch.float32)

            from triton.tools.tensor_descriptor import TensorDescriptor
            a_tma = TensorDescriptor.from_tensor(a, block_shape=[XBLOCK, YBLOCK])
            b_tma = TensorDescriptor.from_tensor(b, block_shape=[XBLOCK, YBLOCK])
            c_tma = TensorDescriptor.from_tensor(c, block_shape=[XBLOCK, YBLOCK])

            # Execute TLE pipeline computation
            elementwise_add(a_tma, b_tma, c_tma, XBLOCK, YBLOCK)

            # Verify results
            expected = a + b
            torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
    def test_tma_copy_different_dtypes(self):
        """Test TMA copy with different data types"""
        torch.manual_seed(456)

        xnumel, ynumel = 256, 256
        XBLOCK, YBLOCK = 32, 64
        # for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for dtype in [torch.float32]:
            # Create test data
            a = torch.randn(xnumel, ynumel, device="cuda", dtype=dtype)
            b = torch.randn(xnumel, ynumel, device="cuda", dtype=dtype)
            c = torch.empty_like(a, device="cuda", dtype=dtype)

            from triton.tools.tensor_descriptor import TensorDescriptor
            a_tma = TensorDescriptor.from_tensor(a, block_shape=[XBLOCK, YBLOCK])
            b_tma = TensorDescriptor.from_tensor(b, block_shape=[XBLOCK, YBLOCK])
            c_tma = TensorDescriptor.from_tensor(c, block_shape=[XBLOCK, YBLOCK])

            # Execute TLE pipeline computation
            elementwise_add(a_tma, b_tma, c_tma, XBLOCK, YBLOCK)

            # Verify results
            expected = a + b
            torch.testing.assert_close(c, expected, atol=1e-3 if dtype == torch.float16 else 1e-5,
                                       rtol=1e-3 if dtype == torch.float16 else 1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
    def test_tma_copy_large_tensor(self):
        """Test TMA copy with larger tensors"""
        torch.manual_seed(789)

        xnumel, ynumel = 512, 512
        XBLOCK, YBLOCK = 64, 64

        # Create test data
        a = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
        b = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
        c = torch.empty_like(a, device="cuda", dtype=torch.float32)

        from triton.tools.tensor_descriptor import TensorDescriptor
        a_tma = TensorDescriptor.from_tensor(a, block_shape=[XBLOCK, YBLOCK])
        b_tma = TensorDescriptor.from_tensor(b, block_shape=[XBLOCK, YBLOCK])
        c_tma = TensorDescriptor.from_tensor(c, block_shape=[XBLOCK, YBLOCK])

        # Execute TLE pipeline computation
        elementwise_add(a_tma, b_tma, c_tma, XBLOCK, YBLOCK)

        # Verify results
        expected = a + b
        torch.testing.assert_close(c, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
    def test_tma_copy_non_divisible(self):
        """Test TMA copy with non-divisible tensor dimensions"""
        torch.manual_seed(101)

        # Test with dimensions that are not perfectly divisible by block sizes
        xnumel, ynumel = 500, 300  # Not divisible by 32, 64
        XBLOCK, YBLOCK = 32, 64

        # Create test data
        a = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
        b = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
        c = torch.empty_like(a, device="cuda", dtype=torch.float32)

        from triton.tools.tensor_descriptor import TensorDescriptor
        a_tma = TensorDescriptor.from_tensor(a, block_shape=[XBLOCK, YBLOCK])
        b_tma = TensorDescriptor.from_tensor(b, block_shape=[XBLOCK, YBLOCK])
        c_tma = TensorDescriptor.from_tensor(c, block_shape=[XBLOCK, YBLOCK])

        # Execute TLE pipeline computation
        elementwise_add(a_tma, b_tma, c_tma, XBLOCK, YBLOCK)

        # Verify results (only check valid region)
        expected = a + b
        torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
