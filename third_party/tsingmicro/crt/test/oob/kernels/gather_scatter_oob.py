"""Gather/scatter kernel with OOB error injection parameters.

Uses an Offs (indirection) array to load A rows, then matmul with B, store to C.
OOB can be triggered through: Offs array size, wrong index values.
"""
import triton
import triton.language as tl


@triton.jit
def gather_scatter_kernel_oob(
    Offs, A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    inject_index_oob: tl.constexpr,       # 0=normal, 1=indices >= M (partial OOB)
    inject_index_oob_all: tl.constexpr,   # 0=normal, 1=all indices = M + 16 (full OOB)
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)

    # Load row indices from Offs array
    ram = tl.load(Offs + rm)

    # --- OOB injection: shift indices to point out of bounds ---
    ram = ram + inject_index_oob * M + inject_index_oob_all * (M + 16)

    # Load A rows using the (possibly OOB) indices
    a = tl.load(A + ram[:, None] * stride_am + tl.arange(0, K)[None, :] * stride_ak)

    # Load B (not affected by injection — reference tile)
    b = tl.load(B + tl.arange(0, K)[:, None] * stride_bk + tl.arange(0, N)[None, :] * stride_bn)

    # Matmul
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # Store C using the (possibly OOB) indices
    tl.store(C + ram[:, None] * stride_cm + tl.arange(0, N)[None, :] * stride_cn, acc)
