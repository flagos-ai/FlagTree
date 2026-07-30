import re

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import pytest


def _is_cuda_backend():
    try:
        return triton.runtime.driver.active.get_current_target().backend == "cuda"
    except Exception:
        return False


# The merge is expressed in terms of the NVIDIA mma fragment layout, so these
# checks only apply on the cuda backend.
requires_cuda_backend = pytest.mark.skipif(not torch.cuda.is_available() or not _is_cuda_backend(),
                                           reason="requires an NVIDIA GPU backend")


@triton.jit
def segmented_dot_kernel(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, K_SEG: tl.constexpr,
                         NUM_SEG: tl.constexpr):
    # Accumulate one dot per K segment, the pattern concat_dot_fragments replaces.
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for s in tl.static_range(NUM_SEG):
        offs_k = s * K_SEG + tl.arange(0, K_SEG)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b, acc=acc)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def merged_dot_kernel(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, K_SEG: tl.constexpr,
                      NUM_SEG: tl.constexpr):
    # Merge the same segments into one fragment and issue a single dot. The
    # operand encoding comes from layout propagation off that dot, so the concat
    # sees dot_op tiles.
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)

    a0 = tl.load(a_ptr + offs_m[:, None] * K + (0 * K_SEG + tl.arange(0, K_SEG))[None, :])
    a1 = tl.load(a_ptr + offs_m[:, None] * K + (1 * K_SEG + tl.arange(0, K_SEG))[None, :])
    a_full = tle.concat_dot_fragments([a0, a1], dim=1)

    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    acc = tl.dot(a_full, b)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


def _shapes():
    M, N, K = 64, 64, 64
    num_seg = 2
    return M, N, K, K // num_seg, num_seg


@requires_cuda_backend
def test_concat_dot_fragments_matches_segmented_dot():
    M, N, K, k_seg, num_seg = _shapes()

    torch.manual_seed(0)
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    seg_out = torch.zeros((M, N), device='cuda', dtype=torch.float32)
    merged_out = torch.zeros((M, N), device='cuda', dtype=torch.float32)

    segmented_dot_kernel[(1, )](a, b, seg_out, M, N, K, k_seg, num_seg)
    merged_dot_kernel[(1, )](a, b, merged_out, M, N, K, k_seg, num_seg)

    assert torch.equal(merged_out, seg_out), \
        "concat_dot_fragments must be bit-exact with accumulating one dot per K segment"


@requires_cuda_backend
def test_concat_dot_fragments_keeps_dot_operand_encoding():
    # The op is only zero-cost while the dot_op encoding propagates through it:
    # a #blocked operand means it did not, and the merged fragment would then
    # round-trip through shared memory before reaching the mma.
    M, N, K, k_seg, num_seg = _shapes()
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    out = torch.zeros((M, N), device='cuda', dtype=torch.float32)

    compiled = merged_dot_kernel.warmup(a, b, out, M, N, K, k_seg, num_seg, grid=(1, ))
    ttgir = compiled.asm["ttgir"]

    concat = [ln for ln in ttgir.splitlines() if "tle.concat_dot_fragments" in ln]
    assert len(concat) == 1, f"expected exactly one concat, got {len(concat)}"
    assert "ttg.dot_op" in concat[0] and "#blocked" not in concat[0], \
        f"concat operands lost their dot_op encoding: {concat[0]}"

    # One mma for the whole K tile instead of one per segment. Match on the
    # result assignment so `warp_group_dot_wait` does not count as a dot.
    mma = re.findall(r"=\s+(ttng\.warp_group_dot|tt\.dot)\b", ttgir)
    assert len(mma) == 1, f"expected a single mma, got {mma}"

    # That the relabel emits no data movement is pinned at the IR level by
    # test_tle_concat_dot_fragments.mlir; PTX here is dominated by the B operand
    # staging, which is the same with or without the concat.


@requires_cuda_backend
def test_concat_dot_fragments_rejects_non_dot_operands(capfd):
    # Without a dot consuming the result the operands stay #blocked, which the
    # lowering rejects: the per-thread relabel is only valid for dot_op layouts.
    @triton.jit
    def blocked_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, TILE_N: tl.constexpr):
        offs_m = tl.arange(0, M)
        t0 = tle.extract_tile(
            tl.load(x_ptr + offs_m[:, None] * N + tl.arange(0, N)[None, :]),
            index=[0, 0],
            tile_shape=[M, TILE_N],
        )
        merged = tle.concat_dot_fragments([t0, t0], dim=1)
        tl.store(out_ptr + offs_m[:, None] * N + tl.arange(0, N)[None, :], merged)

    x = torch.zeros((32, 64), device='cuda', dtype=torch.float32)
    out = torch.zeros((32, 64), device='cuda', dtype=torch.float32)
    with pytest.raises(Exception):
        blocked_kernel[(1, )](x, out, 32, 64, 32)
    # Pin the reason: the raised error only says the pass manager failed, so
    # check the diagnostic itself, which MLIR writes to stderr.
    assert "expects dot_op encoded operands" in capfd.readouterr().err
