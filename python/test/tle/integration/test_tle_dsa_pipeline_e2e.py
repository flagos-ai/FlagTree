# flagtree tle
"""TLE DSA pipeline end-to-end integration (backend-portable: txda/npu).

The pipeline uses on-chip buffers and ``tle.dsa.to_tensor`` to read them back.
Ascend does not expose ``tle.dsa.local_ptr``, so the portable form uses the
buffer-to-tensor view instead.
"""

import pytest

torch = pytest.importorskip("torch")

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from .._dsa_backend import get_dsa_backend

BACKEND, DEVICE, SPACE = get_dsa_backend()
IS_ASCEND = BACKEND in ("npu", "ascend")


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
    SPACE: tl.constexpr,
    IS_ASCEND: tl.constexpr,
):
    """
    Element-wise addition using TLE DSA buffers and a software pipeline.

    1. Allocate on-chip buffers.
    2. Iterate with ``tle.dsa.pipeline``.
    3. Copy GM -> on-chip with ``tle.dsa.copy``.
    4. Read the buffers through ``tle.dsa.to_tensor``.
    5. Store results back to global memory.
    """
    pid = tl.program_id(0)

    xoffs = pid * XBLOCK + tl.arange(0, XBLOCK)

    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    if IS_ASCEND:
        a_smem = tle.dsa.alloc([XBLOCK, YBLOCK], dtype=tl.float32, mem_addr_space=SPACE)
        b_smem = tle.dsa.alloc([XBLOCK, YBLOCK], dtype=tl.float32, mem_addr_space=SPACE)
    else:
        a_smem = tle.dsa.alloc([XBLOCK, YBLOCK], dtype=tl.float32)
        b_smem = tle.dsa.alloc([XBLOCK, YBLOCK], dtype=tl.float32)

    for yoff in tle.dsa.pipeline(0, ynumel, YBLOCK, num_stages=2):
        yoffs = tl.arange(0, YBLOCK) + yoff
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]

        tle.dsa.copy(a_ptrs + ystride_a * yoffs[None, :], a_smem, [XBLOCK, YBLOCK])
        tle.dsa.copy(b_ptrs + ystride_b * yoffs[None, :], b_smem, [XBLOCK, YBLOCK])

        aval = tle.dsa.to_tensor(a_smem)
        bval = tle.dsa.to_tensor(b_smem)
        c_val = aval + bval

        tl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)


def elementwise_add(A, B, C, XBLOCK=32, YBLOCK=64):
    """Execute element-wise addition using the TLE pipeline."""
    assert A.shape == B.shape == C.shape, "Input and output tensor shapes must match"
    xnumel, ynumel = A.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )

    return elementwise_add_kernel[grid](A, B, C, xnumel, ynumel, *A.stride(), *B.stride(), *C.stride(), XBLOCK,
                                        YBLOCK, SPACE=SPACE, IS_ASCEND=IS_ASCEND)


class TestTLEPipelineEndToEnd:
    """TLE Pipeline End-to-End Integration Tests"""

    def test_elementwise_add_basic(self):
        """Test basic element-wise addition functionality."""
        torch.manual_seed(42)

        xnumel, ynumel = 512, 512
        XBLOCK, YBLOCK = 64, 64

        a = torch.randn(xnumel, ynumel, device=DEVICE, dtype=torch.float32)
        b = torch.randn(xnumel, ynumel, device=DEVICE, dtype=torch.float32)
        c = torch.empty_like(a)

        elementwise_add(a, b, c, XBLOCK, YBLOCK)

        torch.testing.assert_close(c, a + b, atol=1e-5, rtol=1e-5)

    def test_elementwise_add_different_sizes(self):
        """Test different tensor sizes."""
        torch.manual_seed(123)

        test_cases = [
            (256, 256, 32, 32),
            (1024, 1024, 64, 64),
            (2048, 512, 128, 32),
            (512, 2048, 32, 128),
        ]

        for xnumel, ynumel, XBLOCK, YBLOCK in test_cases:
            a = torch.randn(xnumel, ynumel, device=DEVICE, dtype=torch.float32)
            b = torch.randn(xnumel, ynumel, device=DEVICE, dtype=torch.float32)
            c = torch.empty_like(a)

            elementwise_add(a, b, c, XBLOCK, YBLOCK)

            expected = a + b
            torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)
            assert c.shape == expected.shape, f"Shape mismatch: {xnumel}x{ynumel}, block size: {XBLOCK}x{YBLOCK}"

    def test_elementwise_add_different_dtypes(self):
        """Test different data types."""
        torch.manual_seed(456)

        dtypes = [torch.float32]  # float16 is not used by the dsa.copy path yet
        xnumel, ynumel = 512, 512
        XBLOCK, YBLOCK = 64, 64

        for dtype in dtypes:
            a = torch.randn(xnumel, ynumel, device=DEVICE, dtype=dtype)
            b = torch.randn(xnumel, ynumel, device=DEVICE, dtype=dtype)
            c = torch.empty_like(a)

            elementwise_add(a, b, c, XBLOCK, YBLOCK)

            expected = a + b
            torch.testing.assert_close(c, expected, atol=1e-3, rtol=1e-3)
            assert c.shape == expected.shape, f"Shape mismatch for data type {dtype}"

    def test_elementwise_add_edge_cases(self):
        """Test edge cases."""
        torch.manual_seed(789)

        a = torch.randn(1, 1, device=DEVICE, dtype=torch.float32)
        b = torch.randn(1, 1, device=DEVICE, dtype=torch.float32)
        c = torch.empty_like(a)

        elementwise_add(a, b, c, 1, 1)
        torch.testing.assert_close(c, a + b, atol=1e-5, rtol=1e-5)

        a = torch.randn(128, 1024, device=DEVICE, dtype=torch.float32)
        b = torch.randn(128, 1024, device=DEVICE, dtype=torch.float32)
        c = torch.empty_like(a)

        elementwise_add(a, b, c, 32, 128)
        torch.testing.assert_close(c, a + b, atol=1e-5, rtol=1e-5)

    def test_tle_module_import(self):
        """Test the TLE module surface for the active DSA backend."""
        assert hasattr(tle, "dsa")
        for name in ("alloc", "copy", "pipeline", "to_tensor", "to_buffer"):
            assert hasattr(tle.dsa, name), f"tle.dsa.{name} is missing"

        if IS_ASCEND:
            assert hasattr(tle.dsa, "workspace")
            assert hasattr(tle.dsa, "ascend")
            assert hasattr(tle.dsa.ascend, "UB")
        else:
            assert hasattr(tle.dsa, "local_ptr")
            assert hasattr(tle.dsa, "scope")
            assert hasattr(tle.dsa, "buffered_tensor")

        assert tle.dsa.alloc.__doc__ is not None
        assert tle.dsa.copy.__doc__ is not None
        assert tle.dsa.pipeline.__doc__ is not None
        assert tle.dsa.to_tensor.__doc__ is not None
        assert tle.dsa.to_buffer.__doc__ is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
