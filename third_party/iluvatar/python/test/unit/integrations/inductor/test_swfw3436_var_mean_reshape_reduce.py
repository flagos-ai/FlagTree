"""Regression for warp reduce after reshape to a non-contiguous lane-bit layout.

Inductor's fused var_mean over multiple dims (e.g. dims [1, 3]) can emit:

  load 3D tile -> tl.reshape([X, R0, R1] -> [X, R0*R1]) -> tl.reduce(..., 1)

On Iluvatar (warp=64) the reshape result may be a #linear layout whose
reduction-axis lane bits are interleaved with non-reduction bits
(e.g. col bits {0,1,2,5}, row bits {3,4}).  Old ReduceOpToLLVM::warpReduce
assumed a contiguous (numLaneToReduce, interleave) run and issued the wrong
shuffleXor masks, silently corrupting the reduction (paired duplicate
outputs).

These tests cover:
  1. torch.compile(var_mean) on the original inductor shape
  2. a minimal reshape+sum kernel that isolates the same layout
"""

import pytest
import torch
import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

# Shape that produces interleaved reduction-axis lane bits after reshape
# (threadsPerWarp=[4,2,8], order=[2,0,1] -> #linear with col bits {0,1,2,5}).
_X, _R0, _R1 = 4, 2, 8


def _var_mean_last(x: torch.Tensor):
    return torch.var_mean(x, dim=-1)


def _var_mean_dims_1_3(x: torch.Tensor):
    return torch.var_mean(x, dim=[1, 3])


@pytest.mark.parametrize(
    "fn,dims_desc",
    [
        (_var_mean_last, "dim=-1"),
        (_var_mean_dims_1_3, "dim=[1,3]"),
    ],
)
def test_compiled_var_mean_matches_eager(fn, dims_desc):
    """Compiled var_mean must match eager; dim=[1,3] is the bug trigger."""
    torch.manual_seed(0)
    # Matches the inductor AOT repro: f32[1, 2, 4, 8] contiguous in last dim.
    x = torch.randn(1, 2, 4, 8, device="cuda", dtype=torch.float32)

    eager_var, eager_mean = fn(x)
    compiled = torch.compile(fn)
    compiled_var, compiled_mean = compiled(x)

    torch.testing.assert_close(compiled_var, eager_var, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(compiled_mean, eager_mean, rtol=1e-5, atol=1e-5)


@triton.jit
def _reshape_sum_kernel(in_ptr, out_ptr, X: tl.constexpr, R0: tl.constexpr, R1: tl.constexpr):
    x = tl.arange(0, X)[:, None, None]
    r0 = tl.arange(0, R0)[None, :, None]
    r1 = tl.arange(0, R1)[None, None, :]
    # val[x, r0, r1] with strides that coalesce to order=[2, 0, 1].
    v = tl.load(in_ptr + (r1 + R1 * x + R1 * X * r0))
    v2 = tl.reshape(v, [X, R0 * R1])
    tl.store(out_ptr + tl.arange(0, X), tl.sum(v2, 1))


def test_reshape_sum_noncontiguous_lane_bits():
    """Minimal kernel: reshape then sum must not use wrong shuffleXor masks."""
    device = "cuda"
    # Physical [R0, X, R1]; kernel loads as v[x, r0, r1] = src[r0, x, r1].
    r0 = torch.arange(_R0, device=device)[:, None, None]
    x = torch.arange(_X, device=device)[None, :, None]
    r1 = torch.arange(_R1, device=device)[None, None, :]
    src = (100 * r0 + 10 * x + r1).to(torch.float32)

    ref = src.permute(1, 0, 2).reshape(_X, -1).sum(1)
    out = torch.empty(_X, device=device, dtype=torch.float32)
    _reshape_sum_kernel[(1, )](src, out, _X, _R0, _R1, num_warps=1)

    # Pre-fix symptom was paired duplicates, e.g. [136, 136, 456, 456]
    # instead of [856, 1016, 1176, 1336] for this patterned input.
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
