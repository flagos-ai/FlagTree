# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest
import torch
import torch_npu
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fixed Constants
# ---------------------------------------------------------------------------
_M_ROWS = 16  # Rows per program
_OFFS = 8  # Write window offset for two programs
_HALF = 12  # Mask threshold
_NUM_C = 24  # Rows of matrix C (= OFFS + M_ROWS, ensure pid=1 write window not out of bounds)

assert _OFFS < _HALF < _M_ROWS, "OFFS < HALF < M_ROWS to ensure True/False on both sides"
assert _NUM_C >= _OFFS + _M_ROWS, "NUM_C must accommodate upper bound of pid=1 write window"


# ---------------------------------------------------------------------------
# Triton Kernel
# ---------------------------------------------------------------------------
@triton.jit
def _copy_matrix_kernel(
    A_ptr,
    idx_ptr,
    C_ptr,
    idx_stride,
    A_row_stride,
    A_col_stride,
    C_row_stride,
    C_col_stride,
    BLOCK_N: tl.constexpr,
    HALF: tl.constexpr,
):
    """
    Discrete memory access + overlapping write window + runtime mask.

    pid=0 write window: rows [0,  15], mask=True when idx < HALF
    pid=1 write window: rows [8,  23], mask=True when idx >= HALF
    Overlap region    : rows [8,  15] -> triggers load-select-store RMW
    """
    program_id = tl.program_id(axis=0).to(tl.int64)
    N_id = tl.program_id(axis=1).to(tl.int64)

    OFFS: tl.constexpr = 8
    M_ROWS: tl.constexpr = 16

    N_BLOCK = N_id * BLOCK_N + tl.arange(0, BLOCK_N)  # shape: (BLOCK_N,)
    M_BLOCK = tl.arange(0, M_ROWS)  # shape: (M_ROWS,)

    # Discrete row indices (loaded at runtime -> mask cannot be statically analyzed)
    idx = tl.load(idx_ptr + program_id * idx_stride + M_BLOCK)

    # Runtime mask (generates scf.if -> compiler converts to load-select-store)
    if program_id == 0:
        mask = idx < HALF
    else:
        mask = idx >= HALF

    val = tl.load(
        A_ptr + idx[:, None] * A_row_stride + tl.arange(0, BLOCK_N)[None, :] * A_col_stride,
        mask=mask[:, None],
    )

    # Write to C (mask=False rows rely on load-select-store to preserve original values)
    tl.store(
        C_ptr + (OFFS * program_id + M_BLOCK[:, None]) * C_row_stride + N_BLOCK[None, :] * C_col_stride,
        val,
        mask=mask[:, None],
    )
    # C (24 × N)             Program 0                    Program 1
    # Row 0~7   ──────────── write value (mask=True)  ── Not involved
    # Row 8~11  ──────────── write value (mask=True)  ── Not written (mask=False, overwritten by P0 to 0)
    # Row 12~15 ──────────── Not written (mask=False) ── write value (mask=True)
    # Row 16~23 ──────────── Not involved             ── write value (mask=True)


# ---------------------------------------------------------------------------
# Helper: Construct discrete index vector
# ---------------------------------------------------------------------------
def _make_idx(device: str) -> torch.Tensor:
    """
    Construct 2x16 index matrix that meets mask distribution requirements.

    pid=0 row (idx0):
      First HALF=12 values ∈ [0, HALF)    -> mask=True
      Last  4      values ∈ [HALF, M_ROWS) -> mask=False
    pid=1 row (idx1):
      First  4      values ∈ [OFFS, OFFS+4)  -> mask=False
      Last HALF=12 values ∈ [HALF, HALF*2)  -> mask=True
    """

    def shuffle_quads(lst: list) -> list:
        """Reverse each group of 4 elements (ignore if less than 4)."""
        out = lst[:]
        for i in range(0, len(out) - 3, 4):
            out[i], out[i + 1], out[i + 2], out[i + 3] = \
                out[i + 3], out[i + 2], out[i + 1], out[i]
        return out

    num_false = _M_ROWS - _HALF  # = 4

    seg0_true = shuffle_quads(list(range(0, _HALF)))  # 12 values, < 12
    seg0_false = shuffle_quads(list(range(_HALF, _HALF + num_false)))  # 4 values, >= 12
    idx0 = seg0_true + seg0_false  # Total length 16

    seg1_false = shuffle_quads(list(range(_OFFS, _OFFS + num_false)))  # 4 values, < 12
    seg1_true = shuffle_quads(list(range(_HALF, _HALF + _HALF)))  # 12 values, >= 12
    idx1 = seg1_false + seg1_true  # Total length 16

    assert len(idx0) == _M_ROWS, f"idx0 length error: {len(idx0)}"
    assert len(idx1) == _M_ROWS, f"idx1 length error: {len(idx1)}"
    assert all(v < _HALF for v in seg0_true), "pid=0 True segment should all be < HALF"
    assert all(v >= _HALF for v in seg0_false), "pid=0 False segment should all be >= HALF"
    assert all(v < _HALF for v in seg1_false), "pid=1 False segment should all be < HALF"
    assert all(v >= _HALF for v in seg1_true), "pid=1 True segment should all be >= HALF"

    return torch.tensor([idx0, idx1], dtype=torch.int32, device=device)


# ---------------------------------------------------------------------------
# Dtype Mapping
# ---------------------------------------------------------------------------
_DTYPE_MAP = {
    'int32': torch.int32,
    'float32': torch.float32,
    'float16': torch.float16,
    'int16': torch.int16,
}


# ---------------------------------------------------------------------------
# Single Execution + Verification
# ---------------------------------------------------------------------------
def _run_once(BLOCK_N: int, dtype_str: str) -> None:
    """
    Execute kernel once and verify results.

    Expectations:
      C[0:HALF,  :] all 0  -- pid=0 writes rows [0,HALF) of A (all 0)
      C[HALF:NUM_C, :] all 1  -- pid=1 writes rows [HALF, NUM_C) of A (all 1)
    """
    dev = 'npu'
    td = _DTYPE_MAP[dtype_str]
    zero_val = 0.0 if dtype_str.startswith('float') else 0
    one_val = 1.0 if dtype_str.startswith('float') else 1

    # A: First HALF rows all 0, last HALF rows all 1
    A = torch.zeros((_NUM_C, BLOCK_N), dtype=td, device=dev)
    A[_HALF:, :] = one_val

    idx = _make_idx(dev)

    # C: Fill all with 2
    C = torch.full((_NUM_C, BLOCK_N), 2, dtype=td, device=dev)

    grid = (2, 1)
    _copy_matrix_kernel[grid](
        A_ptr=A,
        idx_ptr=idx,
        C_ptr=C,
        idx_stride=idx.stride(0),
        A_row_stride=A.stride(0),
        A_col_stride=A.stride(1),
        C_row_stride=C.stride(0),
        C_col_stride=C.stride(1),
        BLOCK_N=BLOCK_N,
        HALF=_HALF,
        enable_sync_block_lock=True,
    )

    # Verification
    assert torch.all(C[:_HALF] == zero_val), (
        f"[dtype={dtype_str}, BLOCK_N={BLOCK_N}] "
        f"C[:HALF] should all be {zero_val}, actual unique values: {C[:_HALF].unique().tolist()}")
    assert torch.all(C[_HALF:] == one_val), (
        f"[dtype={dtype_str}, BLOCK_N={BLOCK_N}] "
        f"C[HALF:] should all be {one_val}, actual unique values: {C[_HALF:].unique().tolist()}")


@pytest.mark.parametrize("param_list", [
    # --- int32 ---
    (16, 'int32'),
    (32, 'int32'),
    (64, 'int32'),
    # --- float32 ---
    (16, 'float32'),
    (32, 'float32'),
    (64, 'float32'),
])
def test_discrete_overlap_mask(param_list):
    """
    Verify no precision issues in discrete access + overlapping write window + runtime mask scenario.

    Race condition errors are probabilistic. Each parameter combination is executed 10 times
    to fully cover concurrent timing scenarios.
    If sync_block_lock fix is effective, all 10 runs pass; if race condition exists, assertion failure
    occurs with high probability.
    """
    BLOCK_N, dtype_str = param_list
    for _ in range(10):
        _run_once(BLOCK_N, dtype_str)


if __name__ == "__main__":
    configs = [
        (32, 'int32'),
        (32, 'float32'),
    ]
    for BLOCK_N, dtype_str in configs:
        print(f"Testing BLOCK_N={BLOCK_N}, dtype={dtype_str} ...", end=" ", flush=True)
        for _ in range(10):
            _run_once(BLOCK_N, dtype_str)
        print("PASS (10 rounds)")
    print("All tests passed.")
