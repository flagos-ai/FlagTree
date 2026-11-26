"""OOB tests for gather/scatter kernel.

Covers 3 scenarios:
  1. Normal (all inject=0)
  2. Partial index OOB (index_oob=1)
  3. Full index OOB (index_oob_all=1)
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kernels"))
from conftest import run_kernel_script, DmaResult


def _create_test_script(M, N, K, inject_offs_scale=1, inject_index_oob=0, inject_index_oob_all=0, BLOCK_M=4):
    """Generate a self-contained test script for gather_scatter_oob.

    Args:
        M, N, K: matrix dimensions.
        inject_offs_scale: >1 scale Offs effective size to M * inject_offs_scale.
        inject_index_oob: adds M to loaded indices (partial OOB).
        inject_index_oob_all: adds M+16 to all loaded indices (full OOB).
        BLOCK_M: block size along M dimension.
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
from gather_scatter_oob import gather_scatter_kernel_oob

M = {M}
N = {N}
K = {K}
BLOCK_M = {BLOCK_M}

torch.manual_seed(42)
# Offs: row indices for gather (can be M+padding in size for some tests)
offs_size = max(M * {inject_offs_scale}, 1)
Offs = torch.arange(0, offs_size, dtype=torch.int32, device=DEVICE)

A = torch.randn(({M}, K), dtype=torch.float16, device=DEVICE)
B = torch.randn((K, N), dtype=torch.float16, device=DEVICE)
C = torch.empty(({M}, N), dtype=torch.float16, device=DEVICE)

grid = (triton.cdiv(offs_size, BLOCK_M),)
print(f"\\ngrid={{grid}}\\n")
gather_scatter_kernel_oob[grid](
    Offs, A, B, C,
    M, N, K,
    A.stride(0), A.stride(1),
    B.stride(0), B.stride(1),
    C.stride(0), C.stride(1),
    inject_index_oob={inject_index_oob},
    inject_index_oob_all={inject_index_oob_all},
    BLOCK_M=BLOCK_M,
)
'''
    return script


def main():
    """Generate test scripts for all gather_scatter OOB test cases."""
    cases = {
        "test_normal_no_oob": _create_test_script(M=64, N=32, K=128),
        "test_index_partial_oob": _create_test_script(M=64, N=32, K=128, inject_index_oob=1),
        "test_index_full_oob": _create_test_script(M=64, N=32, K=128, inject_index_oob_all=1),
        "test_indirect_index_oob": _create_test_script(M=64, N=32, K=128, inject_offs_scale=2),
    }
    out_dir = os.path.dirname(__file__)
    for name, script in cases.items():
        out_path = os.path.join(out_dir, f"_generated_{name}.py")
        with open(out_path, "w") as f:
            f.write(script)
        print(f"Written to {out_path}")


if __name__ == "__main__":
    main()


class TestGatherScatterOOB:
    # 1. Normal
    def test_normal_no_oob(self, dma_env, tmp_path):
        """Normal parameters: should not trigger OOB detection."""
        script = _create_test_script(M=64, N=32, K=128)
        result = run_kernel_script(script, tmp_path, 'TestGatherScatterOOB.test_normal_no_oob')
        assert result.passed, f"Normal kernel should not OOB: oob={result.oob_count}"

    # 2. Partial index OOB
    def test_index_partial_oob(self, dma_env, tmp_path):
        """Some indices shifted by +M, pointing beyond A rows."""
        script = _create_test_script(M=64, N=32, K=128, inject_index_oob=1)
        result = run_kernel_script(script, tmp_path, 'TestGatherScatterOOB.test_index_partial_oob')
        assert result.detected, \
            f"Expected OOB detection with index_oob=1: {result}"

    # 3. Full index OOB
    def test_index_full_oob(self, dma_env, tmp_path):
        """All indices shifted to M+16, pointing well beyond A rows."""
        script = _create_test_script(M=64, N=32, K=128, inject_index_oob_all=1)
        result = run_kernel_script(script, tmp_path, 'TestGatherScatterOOB.test_index_full_oob')
        assert result.detected, \
            f"Expected OOB detection with index_oob_all=1: {result}"

    # 4. indirect index OOB
    def test_indirect_index_oob(self, dma_env, tmp_path):
        """All indices shifted to M+16, pointing well beyond A rows."""
        script = _create_test_script(M=64, N=32, K=128, inject_offs_scale=2)
        result = run_kernel_script(script, tmp_path, 'TestGatherScatterOOB.test_indirect_index_oob')
        assert result.detected, \
            f"Expected OOB detection with index_oob_all=1: {result}"
