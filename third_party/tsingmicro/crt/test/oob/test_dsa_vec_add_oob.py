"""OOB tests for DSA (ping-pong / pipeline / unroll) vector add kernel."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kernels"))
from conftest import run_kernel_script, DmaResult



def _create_test_script(n_elements, inject_n_elements_scale, inject_mask_mode,
                        inject_n_iter_scale, block_size=1024, tensor_size=None):
    """Generate a self-contained test script for dsa_vec_add_oob.

    OOB is triggered by:
      - inject_n_elements_scale > 1: inflates n_elements, making masks
        insufficient.
      - inject_mask_mode = 1: removes masks entirely.
      - inject_n_iter_scale > 1: doubles iterations to 8*scale with sub_sz=BLOCK_SIZE/8,
        causing extra iterations beyond BLOCK_SIZE to go OOB.
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
from dsa_vec_add_oob import add_kernel_dsa_oob

n_elements = {n_elements}
block_size = {block_size}
tensor_size = {tensor_size}
grid = (triton.cdiv(n_elements, block_size),)
print(f"\\ngrid={{grid}}\\n")

x = torch.rand(tensor_size, dtype=torch.float32, device=DEVICE)
y = torch.rand(tensor_size, dtype=torch.float32, device=DEVICE)
output = torch.empty_like(x)

add_kernel_dsa_oob[grid](
    x, y, output, n_elements,
    inject_n_elements_scale={inject_n_elements_scale},
    inject_mask_mode={inject_mask_mode},
    inject_n_iter_scale={inject_n_iter_scale},
    BLOCK_SIZE=block_size,
)
'''
    return script


def main():
    """Generate test scripts for all DSA vec_add OOB test cases."""
    cases = {
        "test_normal_no_oob": _create_test_script(1024, 1, 0, 1),
        "test_n_elements_inflated_oob": _create_test_script(2048, 2, 0, 1, tensor_size=1024),
        "test_mask_missing_oob": _create_test_script(1024, 1, 1, 1, tensor_size=1000),
        "test_n_iter_wrong_oob": _create_test_script(1024, 1, 1, 2),
    }
    out_dir = os.path.dirname(__file__)
    for name, script in cases.items():
        out_path = os.path.join(out_dir, f"_generated_{name}.py")
        with open(out_path, "w") as f:
            f.write(script)
        print(f"Written to {out_path}")


if __name__ == "__main__":
    main()


class TestDSAVecAddOOB:
    def test_normal_no_oob(self, dma_env, tmp_path):
        """Normal parameters: should not trigger OOB detection."""
        script = _create_test_script(1024, 1, 0, 1)
        result = run_kernel_script(script, tmp_path, 'TestDSAVecAddOOB.test_normal_no_oob')
        assert result.passed, f"Normal kernel should not OOB: oob={result.oob_count}"

    def test_n_elements_inflated_oob(self, dma_env, tmp_path):
        """Inflate n_elements: mask becomes too loose for actual tensor size."""
        script = _create_test_script(2048, 2, 0, 1, tensor_size=1024)
        result = run_kernel_script(script, tmp_path, 'TestDSAVecAddOOB.test_n_elements_inflated_oob')
        assert result.detected, \
            f"Expected OOB detection with n_elements_scale=2, tensor=1024: {result}"

    def test_mask_missing_oob(self, dma_env, tmp_path):
        """No mask: tail elements beyond buffer are not filtered."""
        script = _create_test_script(1024, 1, 1, 1, tensor_size=1000)
        result = run_kernel_script(script, tmp_path, 'TestDSAVecAddOOB.test_mask_missing_oob')
        assert result.detected, \
            f"Expected OOB detection without mask: {result}"

    def test_n_iter_wrong_oob(self, dma_env, tmp_path):
        """n_iter_scale=2: doubles iterations from 8 to 16.
        Iterations 8-15 access offsets beyond BLOCK_SIZE, going OOB.
        Combined with mask_mode=1 (no mask) to trigger OOB."""
        script = _create_test_script(1024, 1, 1, 2)
        result = run_kernel_script(script, tmp_path, 'TestDSAVecAddOOB.test_n_iter_wrong_oob')
        assert result.detected, \
            f"Expected OOB detection with n_iter_scale=2: {result}"
