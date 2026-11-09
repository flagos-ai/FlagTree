#!/usr/bin/env python3
"""
Pytest tests for ubtuner decorator.

Tests cover:
1. UBTuner decorator detection and error handling
2. Decorator order validation (autotune outside ubtuner)
3. Two-chain kernel UB overflow scenarios
"""
import os
import shutil
import sys
import tempfile

import pytest

import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

import triton
import triton.language as tl

from triton.backends.ascend.runtime.ubtuner import ubtuner, get_origin_fn

# Set environment before imports
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

# Skip all tests if NPU is not available
pytestmark = pytest.mark.skipif(torch_npu is None or not (hasattr(torch, "npu") and torch.npu.is_available()),
                                reason="requires torch_npu runtime")

BLOCK = int(os.environ.get("BLOCK", "20480"))


# =============================================================================
# Test: Decorator order validation
# =============================================================================
class TestDecoratorOrder:
    """Test that decorator order is correctly validated."""

    def test_autotune_outside_ubtuner_allowed(self):
        """Test: @autotune outside @ubtuner is allowed."""

        @triton.autotune(
            configs=[triton.Config(kwargs={'N': 512})],
            key=['N'],
        )
        @ubtuner(key=['N'])
        @triton.jit
        def kernel_allowed(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
            offs = tl.arange(0, N)
            a = tl.load(a_ptr + offs)
            b = tl.load(b_ptr + offs)
            r = a * b
            tl.store(out_ptr + offs, r)

        # Should not raise any error when defining
        assert kernel_allowed is not None

    def test_ubtuner_outside_autotune_raises_error(self):
        """Test: @ubtuner outside @autotune should raise ValueError."""

        with pytest.raises(ValueError) as exc_info:

            @ubtuner(key=['N'])
            @triton.autotune(
                configs=[triton.Config(kwargs={'N': 512})],
                key=['N'],
            )
            @triton.jit
            def kernel_disallowed(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
                offs = tl.arange(0, N)
                a = tl.load(a_ptr + offs)
                b = tl.load(b_ptr + offs)
                r = a * b
                tl.store(out_ptr + offs, r)

        assert "Cannot apply @ubtuner decorator" in str(exc_info.value)
        assert "@autotune" in str(exc_info.value)

    def test_ubtuner_with_intermediate_decorator(self):
        """Test: @ubtuner with intermediate decorator should work when autotune is outermost."""

        def some_decorator(fn):
            return fn

        @triton.autotune(
            configs=[triton.Config(kwargs={'N': 512})],
            key=['N'],
        )
        @some_decorator
        @ubtuner(key=['N'])
        @triton.jit
        def kernel_with_intermediate(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
            offs = tl.arange(0, N)
            a = tl.load(a_ptr + offs)
            b = tl.load(b_ptr + offs)
            r = a * b
            tl.store(out_ptr + offs, r)

        assert kernel_with_intermediate is not None


# =============================================================================
# Base kernel for TestTwoChainKernel - defined once, decorated per test
# =============================================================================
@triton.jit
def _add_kernel_base(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    pid = tl.program_id(0)
    combined_block_start = pid * BLOCK_SIZE * ITERATIONS

    for i in range(0, ITERATIONS):
        curr_block_start = combined_block_start + i * BLOCK_SIZE
        offsets = curr_block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)


# =============================================================================
# Test: Two-chain kernel scenarios
# =============================================================================
class TestTwoChainKernel:
    """Test two-chain kernel UB overflow scenarios using add_kernel."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["TRITON_CACHE_DIR"] = tmpdir
            os.environ["TRITON_ENABLE_UBTUNER"] = "compile"
            shutil.rmtree(tmpdir, ignore_errors=True)
            yield
            # Cleanup after test
            shutil.rmtree(tmpdir, ignore_errors=True)
            os.environ.pop("TRITON_ENABLE_UBTUNER", None)
            os.environ.pop("TRITON_CACHE_DIR", None)

    def _create_tensors(self, size=BLOCK):
        """Create input and output tensors for add_kernel."""
        x = torch.rand(size, dtype=torch.float32).npu()
        y = torch.rand(size, dtype=torch.float32).npu()
        output = torch.empty(size, dtype=torch.float32).npu()
        expected = x.cpu() + y.cpu()
        return x, y, output, expected

    def _make_kernel(self, decorators):
        """Apply decorators to base kernel."""
        fn = get_origin_fn(_add_kernel_base)
        setattr(fn, '_ubtuned', False)  # 重置_add_kernel_base属性
        kernel = _add_kernel_base
        for dec in decorators:
            kernel = dec(kernel)
        return kernel

    def test_ubtuner_only_no_overflow(self):
        """Test: @ubtuner only - should not overflow (with memory scheduler)."""
        kernel = self._make_kernel([
            lambda fn: ubtuner(key=['BLOCK_SIZE'])(fn),
        ])
        x, y, output, expected = self._create_tensors()
        try:
            kernel[(1, )](x, y, output, output.numel(), BLOCK_SIZE=BLOCK, ITERATIONS=8, multibuffer=True)
            torch.npu.synchronize()
        except Exception as e:
            if "ub overflow" in str(e).lower():
                pytest.fail(f"UB overflow occurred with ubtuner: {e}")
            raise
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)

    def test_autotune_with_ubtuner_inner_no_overflow(self):
        """Test: @autotune (outer) + @ubtuner (inner) - should not overflow."""
        kernel = self._make_kernel([
            lambda fn: ubtuner(key=['BLOCK_SIZE'])(fn),
            lambda fn: triton.autotune(configs=[triton.Config(kwargs={'BLOCK_SIZE': BLOCK})], key=['BLOCK_SIZE'])(fn),
        ])
        x, y, output, expected = self._create_tensors()
        try:
            kernel[(1, )](x, y, output, output.numel(), ITERATIONS=8, multibuffer=True)
            torch.npu.synchronize()
        except Exception as e:
            if "ub overflow" in str(e).lower():
                pytest.fail(f"UB overflow occurred: {e}")
            raise
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)

    def test_autotune_only_overflow(self):
        """Test: @autotune only (without TRITON_ENABLE_UBTUNER) - should overflow."""
        os.environ.pop("TRITON_ENABLE_UBTUNER", None)
        kernel = self._make_kernel([
            lambda fn: triton.autotune(configs=[triton.Config(kwargs={'BLOCK_SIZE': BLOCK})], key=['BLOCK_SIZE'])(fn),
        ])
        x, y, output, _ = self._create_tensors()
        with pytest.raises(Exception) as exc_info:
            kernel[(1, )](x, y, output, output.numel(), ITERATIONS=8, multibuffer=True)
            torch.npu.synchronize()
        assert "ub overflow" in str(exc_info.value).lower()

    def test_autotune_with_triton_enable_ubtuner(self):
        """Test: @autotune with TRITON_ENABLE_UBTUNER=compile - should not overflow."""
        kernel = self._make_kernel([
            lambda fn: triton.autotune(configs=[triton.Config(kwargs={'BLOCK_SIZE': BLOCK})], key=['BLOCK_SIZE'])(fn),
        ])
        x, y, output, expected = self._create_tensors()
        try:
            kernel[(1, )](x, y, output, output.numel(), ITERATIONS=8, multibuffer=True)
            torch.npu.synchronize()
        except Exception as e:
            if "ub overflow" in str(e).lower():
                pytest.fail(f"UB overflow occurred with TRITON_ENABLE_UBTUNER=compile: {e}")
            raise
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    # Run pytest when executed directly
    sys.exit(pytest.main([__file__, "-v"]))
