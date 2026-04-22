import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

_DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _recall(pred: torch.Tensor, ref: torch.Tensor) -> float:
    pred_set = set(pred[0].cpu().tolist())
    ref_set = set(ref[0].cpu().tolist())
    return len(pred_set & ref_set) / ref.shape[1]


@triton.jit
def _convert_to_uint32(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x80000000, tl.uint32)
    bits = tl.where(x < 0, ~bits, bits | sign_mask)
    return bits


@triton.jit
def _convert_to_uint16_hi8(x):
    h = x.to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
    bits = tl.where(x < 0, ~bits, bits | sign_mask)
    return (bits >> 8).to(tl.int32)


@triton.jit
def _minimal_topk_smem_overflow_fallback_fullscan(
    row_ptr,
    out_row,
    stride_xn,
    stride_outn,
    row_start,
    row_end,
    seq_len,
    hist_base_ptr,
    s_write_count_ptr,
    s_eq_count_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    RADIX_SIZE: tl.constexpr = 256
    CAND_ROUNDS: tl.constexpr = 4

    lane = tl.arange(0, BLOCK_SIZE)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    out_init_chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
    hist_clear_chunks: tl.constexpr = (RADIX_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_scan_tiles = tl.cdiv(seq_len, BLOCK_SIZE)

    tl.store(s_write_count_ptr, 0)
    tl.store(s_eq_count_ptr, 0)
    for t in tl.range(0, out_init_chunks):
        pos = t * BLOCK_SIZE + lane
        tl.store(out_row + pos * stride_outn, -1, mask=pos < TOPK)

    for t in tl.range(0, hist_clear_chunks):
        bins = t * BLOCK_SIZE + lane
        tl.store(hist_base_ptr + bins, 0, mask=bins < RADIX_SIZE)
    tl.debug_barrier()

    for t in tl.range(0, num_scan_tiles):
        offs = t * BLOCK_SIZE + lane
        valid = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        x = tl.load(row_ptr + offs * stride_xn, mask=valid, other=float("-inf"))
        digit = _convert_to_uint16_hi8(x)
        tl.atomic_add(
            hist_base_ptr + digit,
            ones,
            mask=valid,
            sem="relaxed",
            scope="cta",
        )
    tl.debug_barrier()

    radix_bins = tl.arange(0, RADIX_SIZE)
    zeros_radix = tl.zeros([RADIX_SIZE], dtype=tl.int32)
    counts = tl.load(hist_base_ptr + radix_bins)
    gt_exclusive, _ = tle.cumsum(counts, axis=0, reverse=True)
    cumsum_desc = gt_exclusive + counts
    threshold_mask = (cumsum_desc >= TOPK) & (gt_exclusive < TOPK)
    coarse_threshold_bin = tl.sum(
        tl.where(threshold_mask, radix_bins, zeros_radix),
        axis=0,
    )
    coarse_counts_gt = tl.sum(
        tl.where(threshold_mask, gt_exclusive, zeros_radix),
        axis=0,
    )
    gt_cursors = tl.where(radix_bins > coarse_threshold_bin, gt_exclusive, zeros_radix)
    tl.store(hist_base_ptr + radix_bins, gt_cursors)
    remaining = TOPK - coarse_counts_gt
    tl.store(s_write_count_ptr + zeros, coarse_counts_gt)
    tl.debug_barrier()

    for t in tl.range(0, num_scan_tiles):
        offs = t * BLOCK_SIZE + lane
        valid = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        idx = offs.to(tl.int32)
        x = tl.load(row_ptr + offs * stride_xn, mask=valid, other=float("-inf"))
        digit = _convert_to_uint16_hi8(x)

        take_gt = valid & (digit > coarse_threshold_bin)
        out_pos_gt = tl.atomic_add(
            hist_base_ptr + digit,
            ones,
            mask=take_gt,
            sem="relaxed",
            scope="cta",
        )
        tl.store(
            out_row + out_pos_gt * stride_outn,
            idx,
            mask=take_gt & (out_pos_gt < TOPK),
        )
    tl.debug_barrier()

    refine_prefix = tl.zeros((), dtype=tl.uint32)
    refine_mask = tl.zeros((), dtype=tl.uint32)
    for round_idx in tl.static_range(CAND_ROUNDS):
        if remaining > 0:
            for t in tl.range(0, hist_clear_chunks):
                bins = t * BLOCK_SIZE + lane
                tl.store(hist_base_ptr + bins, 0, mask=bins < RADIX_SIZE)
            tl.debug_barrier()

            shift: tl.constexpr = 24 - round_idx * 8
            for t in tl.range(0, num_scan_tiles):
                offs = t * BLOCK_SIZE + lane
                valid = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
                x = tl.load(row_ptr + offs * stride_xn, mask=valid, other=float("-inf"))
                coarse_digit = _convert_to_uint16_hi8(x)
                ordered = _convert_to_uint32(x)
                prefix_match = (ordered & refine_mask) == refine_prefix
                active = valid & (coarse_digit == coarse_threshold_bin) & prefix_match
                digit = ((ordered >> shift) & 0xFF).to(tl.int32)
                tl.atomic_add(
                    hist_base_ptr + digit,
                    ones,
                    mask=active,
                    sem="relaxed",
                    scope="cta",
                )
            tl.debug_barrier()

            counts = tl.load(hist_base_ptr + radix_bins)
            gt_exclusive, _ = tle.cumsum(counts, axis=0, reverse=True)
            cumsum_desc = gt_exclusive + counts
            base_write = tl.load(s_write_count_ptr)
            threshold_mask = (cumsum_desc >= remaining) & (gt_exclusive < remaining)
            threshold_bin = tl.sum(
                tl.where(threshold_mask, radix_bins, zeros_radix),
                axis=0,
            )
            counts_gt = tl.sum(
                tl.where(threshold_mask, gt_exclusive, zeros_radix),
                axis=0,
            )
            gt_cursors = tl.where(
                radix_bins > threshold_bin,
                base_write + gt_exclusive,
                zeros_radix,
            )
            tl.store(hist_base_ptr + radix_bins, gt_cursors)
            remaining = remaining - counts_gt
            tl.store(s_write_count_ptr + zeros, base_write + counts_gt)
            if round_idx == (CAND_ROUNDS - 1):
                tl.store(s_eq_count_ptr, 0)
            tl.debug_barrier()

            for t in tl.range(0, num_scan_tiles):
                offs = t * BLOCK_SIZE + lane
                valid = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
                idx = offs.to(tl.int32)
                x = tl.load(row_ptr + offs * stride_xn, mask=valid, other=float("-inf"))
                coarse_digit = _convert_to_uint16_hi8(x)
                ordered = _convert_to_uint32(x)
                prefix_match = (ordered & refine_mask) == refine_prefix
                active = valid & (coarse_digit == coarse_threshold_bin) & prefix_match
                digit = ((ordered >> shift) & 0xFF).to(tl.int32)

                take_gt = active & (digit > threshold_bin)
                out_pos_gt = tl.atomic_add(
                    hist_base_ptr + digit,
                    ones,
                    mask=take_gt,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    out_row + out_pos_gt * stride_outn,
                    idx,
                    mask=take_gt & (out_pos_gt < TOPK),
                )

                if remaining > 0:
                    take_eq = active & (digit == threshold_bin)
                    if round_idx == (CAND_ROUNDS - 1):
                        eq_pos = tl.atomic_add(
                            s_eq_count_ptr + zeros,
                            ones,
                            mask=take_eq,
                            sem="relaxed",
                            scope="cta",
                        )
                        take_eq_select = take_eq & (eq_pos < remaining)
                        out_pos_eq = tl.atomic_add(
                            s_write_count_ptr + zeros,
                            ones,
                            mask=take_eq_select,
                            sem="relaxed",
                            scope="cta",
                        )
                        tl.store(
                            out_row + out_pos_eq * stride_outn,
                            idx,
                            mask=take_eq_select & (out_pos_eq < TOPK),
                        )
            tl.debug_barrier()

            threshold_u32 = threshold_bin.to(tl.uint32)
            if round_idx == 0:
                refine_prefix = threshold_u32 << 24
                refine_mask = tl.full((), 0xFF000000, tl.uint32)
            elif round_idx == 1:
                refine_prefix = refine_prefix | (threshold_u32 << 16)
                refine_mask = tl.full((), 0xFFFF0000, tl.uint32)
            elif round_idx == 2:
                refine_prefix = refine_prefix | (threshold_u32 << 8)
                refine_mask = tl.full((), 0xFFFFFF00, tl.uint32)


@triton.jit
def _minimal_fallback_kernel(
    x_ptr,
    out_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    seq_len,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ASSUME_ALIGNED: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ptr = x_ptr + pid * stride_xm
    out_row = out_ptr + pid * stride_outm
    if ASSUME_ALIGNED:
        tl.assume(stride_xn == 1)
        tl.assume(stride_outn == 1)
        seq_len = tl.multiple_of(seq_len, BLOCK_SIZE)
    hist = tle.gpu.alloc([4096], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    write_cnt = tle.gpu.alloc([1], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    eq_cnt = tle.gpu.alloc([1], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    _minimal_topk_smem_overflow_fallback_fullscan(
        row_ptr,
        out_row,
        stride_xn,
        stride_outn,
        tl.zeros((), dtype=tl.int32),
        seq_len,
        seq_len,
        tle.gpu.local_ptr(hist, (0, )),
        tle.gpu.local_ptr(write_cnt, (0, )),
        tle.gpu.local_ptr(eq_cnt, (0, )),
        TOPK=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tle_topk_smem_fallback_fullscan_recall_seq262144():
    torch.manual_seed(1)
    seq_len = 262144
    topk = 2048
    x = torch.randn((1, seq_len), device=_DEVICE, dtype=torch.float32)
    out = torch.full((1, topk), -1, device=_DEVICE, dtype=torch.int32)
    ref = torch.topk(x, topk, dim=-1)[1]

    _minimal_fallback_kernel[(1, )](
        x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        seq_len=seq_len,
        TOPK=topk,
        BLOCK_SIZE=1024,
        ASSUME_ALIGNED=True,
        num_warps=32,
        num_stages=1,
    )
    assert _recall(out, ref) == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_warps", [16, 32])
def test_tle_topk_fallback_fullscan_stable_high_warps(num_warps):
    torch.manual_seed(1)
    seq_len = 262144
    topk = 2048
    x = torch.randn((1, seq_len), device=_DEVICE, dtype=torch.float32)
    ref = torch.topk(x, topk, dim=-1)[1]

    outputs = []
    for _ in range(3):
        out = torch.full((1, topk), -1, device=_DEVICE, dtype=torch.int32)
        _minimal_fallback_kernel[(1, )](
            x,
            out,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            out.stride(1),
            seq_len=seq_len,
            TOPK=topk,
            BLOCK_SIZE=1024,
            ASSUME_ALIGNED=True,
            num_warps=num_warps,
            num_stages=1,
        )
        outputs.append(out.clone())
        assert _recall(out, ref) == 1.0

    out_sets = [set(o[0].cpu().tolist()) for o in outputs]
    assert out_sets[0] == out_sets[1]
    assert out_sets[1] == out_sets[2]
