"""OOB tests for dropout kernel."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kernels"))
from conftest import run_kernel_script, DmaResult


def _create_test_script(n_elements, inject_mask_mode, inject_n_elements_scale, block_size=256, tensor_size=None):
    """Generate a self-contained test script for dropout_oob.

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
from dropout_oob import dropout_kernel_oob

n_elements = {n_elements}
block_size = {block_size}
tensor_size = {tensor_size}
grid = (triton.cdiv(n_elements, block_size),)
print(f"\\ngrid={{grid}}\\n")

p = 0.5
x = torch.rand(tensor_size, dtype=torch.float32, device=DEVICE)
x_keep = (torch.rand(tensor_size, device=DEVICE) > p).to(torch.int32)
output = torch.empty_like(x)

dropout_kernel_oob[grid](
    x, x_keep, output, n_elements, p,
    inject_mask_mode={inject_mask_mode},
    inject_n_elements_scale={inject_n_elements_scale},
    BLOCK_SIZE=block_size,
    seed=0,
)
'''
    return script


def main():
    """Generate test scripts for all dropout OOB test cases."""
    cases = {
        "test_normal_no_oob": _create_test_script(2000, 0, 1),
        "test_offset_n_elements_scale": _create_test_script(2000, 1, 2),
        "test_mask_missing": _create_test_script(2000, 1, 1, tensor_size=1000),
    }
    out_dir = os.path.dirname(__file__)
    for name, script in cases.items():
        out_path = os.path.join(out_dir, f"_generated_{name}.py")
        with open(out_path, "w") as f:
            f.write(script)
        print(f"Written to {out_path}")


if __name__ == "__main__":
    main()


class TestDropoutOOB:

    def test_normal_no_oob(self, dma_env, tmp_path):
        """Normal parameters: should not trigger OOB detection."""
        script = _create_test_script(2000, 0, 1)
        result = run_kernel_script(script, tmp_path, 'TestDropoutOOB.test_normal_no_oob')
        assert result.oob_count >= 0, f"Failed to parse DMA result: {result}"
        assert result.passed, f"Normal kernel should not OOB: oob={result.oob_count}"

    def test_offset_n_elements_scale(self, dma_env, tmp_path):
        """n_elements*2: kernel tries to access beyond buffer."""
        script = _create_test_script(2000, 1, 2)
        result = run_kernel_script(script, tmp_path, 'TestDropoutOOB.test_offset_n_elements_scale')
        assert result.detected, \
            f"Expected OOB detection with n_elements_scale=2, mask_mode=1: {result}"

    def test_mask_missing(self, dma_env, tmp_path):
        """No mask: tail elements beyond buffer are not filtered."""
        script = _create_test_script(2000, 1, 1, tensor_size=1000)
        result = run_kernel_script(script, tmp_path, 'TestDropoutOOB.test_mask_missing')
        assert result.detected, \
            f"Expected OOB detection without mask: {result}"
