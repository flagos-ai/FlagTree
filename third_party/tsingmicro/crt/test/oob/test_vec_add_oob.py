"""OOB tests for vector add kernel."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kernels"))
from conftest import run_kernel_script, DmaResult


def _create_test_script(n_elements, inject_pid_shift, inject_block_scale,
                        inject_mask_mode, block_size=256, tensor_size=None):
    """Generate a self-contained test script for vec_add_oob.

    Args:
        n_elements: passes to kernel, telling it how many elements to process.
        tensor_size: actual number of elements allocated. Defaults to
            n_elements. Set larger n_elements with smaller tensor_size to
            test OOB detection.
    """
    if tensor_size is None:
        tensor_size = n_elements
    kernels_dir = os.path.join(os.path.dirname(__file__), "kernels")
    script = f'''
import sys
sys.path.insert(0, "{kernels_dir}")
import os
for var in ("TX8_DEPS_ROOT", "LLVM_SYSPATH"):
    if var not in os.environ:
        raise RuntimeError(f"{{var}} is not set. Source your build environment first.")

os.environ["TRITON_DMA_CHECKING"] = "1"
os.environ["TRITON_DMA_CHECK_ABORT"] = "0"

import torch
import triton
DEVICE = triton.runtime.driver.active.get_active_torch_device()
from vec_add_oob import add_kernel_oob

n_elements = {n_elements}
block_size = {block_size}
tensor_size = {tensor_size}
grid = (triton.cdiv(n_elements, block_size),)
print(f"\\ngrid={{grid}}\\n")

x = torch.rand(tensor_size, dtype=torch.float32, device=DEVICE)
y = torch.rand(tensor_size, dtype=torch.float32, device=DEVICE)
output = torch.empty_like(x)

add_kernel_oob[grid](
    x, y, output, n_elements,
    inject_pid_shift={inject_pid_shift},
    inject_block_scale={inject_block_scale},
    inject_mask_mode={inject_mask_mode},
    BLOCK_SIZE=block_size,
)
'''
    return script


def main():
    """Generate test scripts for all vec_add OOB test cases."""
    cases = {
        "test_normal_no_oob": _create_test_script(1024, 0, 1, 0),
        "test_offset_oob_n_elements": _create_test_script(2048, 0, 1, 0, tensor_size=1024),
        "test_offset_calc_pid_shift": _create_test_script(1024, 1, 1, 1),
        "test_offset_calc_block_scale": _create_test_script(1024, 0, 2, 1),
        "test_mask_missing": _create_test_script(1024, 0, 1, 1, tensor_size=1000),
        "test_mask_wrong_cmp": _create_test_script(1025, 0, 1, 2, tensor_size=1024),
    }
    out_dir = os.path.dirname(__file__)
    for name, script in cases.items():
        out_path = os.path.join(out_dir, f"_generated_{name}.py")
        with open(out_path, "w") as f:
            f.write(script)
        print(f"Written to {out_path}")


if __name__ == "__main__":
    main()


class TestVecAddOOB:
    def test_normal_no_oob(self, dma_env, tmp_path):
        """Normal parameters: should not trigger OOB detection."""
        script = _create_test_script(1024, 0, 1, 0)
        result = run_kernel_script(script, tmp_path, 'TestVecAddOOB.test_normal_no_oob')
        assert result.oob_count >= 0, f"Failed to parse DMA result: {result}"
        assert result.passed, f"Normal kernel should not OOB: oob={result.oob_count}"

    def test_offset_oob_n_elements(self, dma_env, tmp_path):
        """n_elements > actual buffer: kernel tries to access beyond buffer."""
        script = _create_test_script(2048, 0, 1, 0, tensor_size=1024)
        result = run_kernel_script(script, tmp_path, 'TestVecAddOOB.test_offset_oob_n_elements')
        assert result.detected, \
            f"Expected OOB detection with n_elements=2048, tensor=1024: {result}"

    def test_offset_calc_pid_shift(self, dma_env, tmp_path):
        """pid+1 shifts all blocks, last block OOB.
        Uses mask_mode=1 (no mask) so shifted blocks actually access OOB addr."""
        script = _create_test_script(1024, 1, 1, 1)
        result = run_kernel_script(script, tmp_path, 'TestVecAddOOB.test_offset_calc_pid_shift')
        assert result.detected, \
            f"Expected OOB detection with pid_shift=1: {result}"

    def test_offset_calc_block_scale(self, dma_env, tmp_path):
        """BLOCK_SIZE*2 doubles stride, later blocks OOB.
        Uses mask_mode=1 (no mask) so scaled blocks actually access OOB addr."""
        script = _create_test_script(1024, 0, 2, 1)
        result = run_kernel_script(script, tmp_path, 'TestVecAddOOB.test_offset_calc_block_scale')
        assert result.detected, \
            f"Expected OOB detection with block_scale=2: {result}"

    def test_mask_missing(self, dma_env, tmp_path):
        """No mask: tail elements beyond buffer are not filtered."""
        script = _create_test_script(1024, 0, 1, 1, tensor_size=1000)
        result = run_kernel_script(script, tmp_path, 'TestVecAddOOB.test_mask_missing')
        assert result.detected, \
            f"Expected OOB detection without mask: {result}"

    def test_mask_wrong_cmp(self, dma_env, tmp_path):
        """<= instead of <: element at n_elements passes mask but is OOB."""
        script = _create_test_script(1025, 0, 1, 2, tensor_size=1024)
        result = run_kernel_script(script, tmp_path, 'TestVecAddOOB.test_mask_wrong_cmp')
        assert result.detected, \
            f"Expected OOB detection with wrong mask cmp: {result}"
