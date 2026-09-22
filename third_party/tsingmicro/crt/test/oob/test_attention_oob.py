"""OOB tests for fused attention kernel."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kernels"))
from conftest import run_kernel_script, DmaResult


def _create_test_script(Z, H, N_CTX, HEAD_DIM, inject_N_CTX_scale, inject_stride_scale, BLOCK_M=64, BLOCK_N=32):
    """Generate a self-contained test script for attention_oob.

    The kernel uses block pointers (tl.make_block_ptr) for Q/K/V/O.
    OOB is triggered by:
      - inject_N_CTX_scale > 1: inflates N_CTX in block pointer shapes,
        causing loads/stores beyond actual tensor dimensions.
      - inject_stride_scale = 0: replaces all strides with 1, causing
        wrong address calculations.
    """
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
from attention_oob import _attn_fwd_oob

Z = {Z}
H = {H}
N_CTX = {N_CTX}
HEAD_DIM = {HEAD_DIM}
BLOCK_M = {BLOCK_M}
BLOCK_N = {BLOCK_N}

dtype = torch.float16

torch.manual_seed(42)
q = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
k = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
v = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
o = torch.empty_like(q)
M = torch.empty((Z, H, N_CTX), dtype=torch.float32, device=DEVICE)
sm_scale = 0.5

grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H, 1)
print(f"\\ngrid={{grid}}\\n")
_attn_fwd_oob[grid](
    q, k, v, sm_scale, M, o,
    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
    o.stride(0), o.stride(1), o.stride(2), o.stride(3),
    Z, H, N_CTX,
    inject_N_CTX_scale={inject_N_CTX_scale},
    inject_stride_scale={inject_stride_scale},
    HEAD_DIM=HEAD_DIM,
    BLOCK_M=BLOCK_M,
    BLOCK_N=BLOCK_N,
)
'''
    return script


def main():
    """Generate test scripts for all attention OOB test cases."""
    cases = {
        "test_normal_no_oob":
        _create_test_script(Z=1, H=2, N_CTX=64, HEAD_DIM=32, inject_N_CTX_scale=1, inject_stride_scale=1),
        "test_n_ctx_inflated_oob":
        _create_test_script(Z=1, H=2, N_CTX=64, HEAD_DIM=32, inject_N_CTX_scale=2, inject_stride_scale=1),
        "test_stride_corrupted_oob":
        _create_test_script(Z=1, H=2, N_CTX=64, HEAD_DIM=32, inject_N_CTX_scale=1, inject_stride_scale=10),
    }
    out_dir = os.path.dirname(__file__)
    for name, script in cases.items():
        out_path = os.path.join(out_dir, f"_generated_{name}.py")
        with open(out_path, "w") as f:
            f.write(script)
        print(f"Written to {out_path}")


if __name__ == "__main__":
    main()


class TestAttentionOOB:

    def test_normal_no_oob(self, dma_env, tmp_path):
        """Normal parameters: should not trigger OOB detection."""
        script = _create_test_script(Z=1, H=2, N_CTX=64, HEAD_DIM=32, inject_N_CTX_scale=1, inject_stride_scale=1)
        result = run_kernel_script(script, tmp_path, 'TestAttentionOOB.test_normal_no_oob')
        assert result.passed, f"Normal kernel should not OOB: oob={result.oob_count}"

    def test_n_ctx_inflated_oob(self, dma_env, tmp_path):
        """Inflate N_CTX by 2x: block pointers think there are 2x more rows,
        causing Q load and O store to go beyond actual tensor."""
        script = _create_test_script(Z=1, H=2, N_CTX=64, HEAD_DIM=32, inject_N_CTX_scale=2, inject_stride_scale=1)
        result = run_kernel_script(script, tmp_path, 'TestAttentionOOB.test_n_ctx_inflated_oob')
        assert result.detected, \
            f"Expected OOB detection with N_CTX_scale=2: {result}"

    def test_stride_corrupted_oob(self, dma_env, tmp_path):
        """Inflate strides by 10x: addresses jump far beyond buffer."""
        script = _create_test_script(Z=1, H=2, N_CTX=64, HEAD_DIM=32, inject_N_CTX_scale=1, inject_stride_scale=10)
        result = run_kernel_script(script, tmp_path, 'TestAttentionOOB.test_stride_corrupted_oob')
        assert result.detected, \
            f"Expected OOB detection with stride_scale=0: {result}"
