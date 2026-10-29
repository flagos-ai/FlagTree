"""apply_rotary_pos_emb: rotate the two halves of every head by position-dependent cos/sin.

For token t at position p = position_ids[t] and a head vector x = [x1 | x2] split into two
halves, the output is [x1 * cos[p] - x2 * sin[p] | x2 * cos[p] + x1 * sin[p]]. The same
rotation is applied to query and key, which may have different head counts.

Each program handles BLOCK_T tokens with all of their heads. Query, key and both outputs are
accessed through three-dimensional PartitionViews over [token, head, half], so tail tokens and
the heads past the head count fall outside the views and are dropped on store. The cos/sin
rows are the only data-dependent access and stay masked loads. The head dimension of the tile
is padded to a power of two, which keeps arbitrary head counts valid.
"""
import argparse

import torch
import triton
import triton.language as tl

import triton.experimental.tle as tle

# =============================================================================
#  Compile-time configuration
# =============================================================================
NUM_TOKENS = 4096
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 128
MAX_POSITIONS = 8192
TILE_ELEMENTS = 8192
MAX_BLOCK_T = 16
# A one-row tile does not compile, so the tile height is at least 2.
MIN_BLOCK_T = 2


# =============================================================================
#  Kernel
# =============================================================================
@triton.jit
def rotary_kernel(q_out, k_out, q, k, cos, sin, position_ids, num_tokens, cos_stride, NUM_Q: tl.constexpr,
                  NUM_K: tl.constexpr, Q_PAD: tl.constexpr, K_PAD: tl.constexpr, HEAD_DIM: tl.constexpr,
                  BLOCK_T: tl.constexpr):
    tile = tl.program_id(0)
    half: tl.constexpr = HEAD_DIM // 2
    rows = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    live = rows < num_tokens
    pos = tl.load(position_ids + rows, mask=live, other=0)
    table = pos[:, None] * cos_stride + tl.arange(0, half)[None, :]
    c = tl.load(cos + table, mask=live[:, None], other=0).to(tl.float32)[:, None, :]
    s = tl.load(sin + table, mask=live[:, None], other=0).to(tl.float32)[:, None, :]

    q_view = tle.make_partition_view(q, shape=[num_tokens, NUM_Q, HEAD_DIM], strides=[NUM_Q * HEAD_DIM, HEAD_DIM, 1],
                                     tile=[BLOCK_T, Q_PAD, half])
    qo_view = tle.make_partition_view(q_out, shape=[num_tokens, NUM_Q, HEAD_DIM],
                                      strides=[NUM_Q * HEAD_DIM, HEAD_DIM, 1], tile=[BLOCK_T, Q_PAD, half])
    q1 = tle.tile_load(q_view, index=(tile, 0, 0), dst_space=tle.language.dsa.ascend.UB).to(tl.float32)
    q2 = tle.tile_load(q_view, index=(tile, 0, 1), dst_space=tle.language.dsa.ascend.UB).to(tl.float32)
    tle.tile_store(qo_view, value=(q1 * c - q2 * s).to(q_out.type.element_ty), index=(tile, 0, 0),
                   src_space=tle.language.dsa.ascend.UB)
    tle.tile_store(qo_view, value=(q2 * c + q1 * s).to(q_out.type.element_ty), index=(tile, 0, 1),
                   src_space=tle.language.dsa.ascend.UB)

    k_view = tle.make_partition_view(k, shape=[num_tokens, NUM_K, HEAD_DIM], strides=[NUM_K * HEAD_DIM, HEAD_DIM, 1],
                                     tile=[BLOCK_T, K_PAD, half])
    ko_view = tle.make_partition_view(k_out, shape=[num_tokens, NUM_K, HEAD_DIM],
                                      strides=[NUM_K * HEAD_DIM, HEAD_DIM, 1], tile=[BLOCK_T, K_PAD, half])
    k1 = tle.tile_load(k_view, index=(tile, 0, 0), dst_space=tle.language.dsa.ascend.UB).to(tl.float32)
    k2 = tle.tile_load(k_view, index=(tile, 0, 1), dst_space=tle.language.dsa.ascend.UB).to(tl.float32)
    tle.tile_store(ko_view, value=(k1 * c - k2 * s).to(k_out.type.element_ty), index=(tile, 0, 0),
                   src_space=tle.language.dsa.ascend.UB)
    tle.tile_store(ko_view, value=(k2 * c + k1 * s).to(k_out.type.element_ty), index=(tile, 0, 1),
                   src_space=tle.language.dsa.ascend.UB)


# =============================================================================
#  Operator entry
# =============================================================================
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, block_t=None):
    """Returns (q_embed, k_embed) with the half-split rotation applied.

    q: [tokens, q_heads, head_dim], k: [tokens, kv_heads, head_dim], cos/sin:
    [max_positions, head_dim // 2], position_ids: [tokens] integer positions."""
    assert q.dim() == 3 and k.dim() == 3 and q.shape[0] == k.shape[0] and q.shape[2] == k.shape[2]
    assert q.shape[2] % 2 == 0 and cos.shape == sin.shape and cos.shape[1] == q.shape[2] // 2
    q, k = q.contiguous(), k.contiguous()
    num_tokens, num_q, head_dim = q.shape
    num_k = k.shape[1]
    q_pad, k_pad = triton.next_power_of_2(num_q), triton.next_power_of_2(num_k)
    if block_t is None:
        rows = max(q_pad, k_pad) * (head_dim // 2)
        block_t = max(MIN_BLOCK_T, min(MAX_BLOCK_T, TILE_ELEMENTS // rows))
        block_t = 1 << (block_t.bit_length() - 1)
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    grid = (triton.cdiv(num_tokens, block_t), )
    rotary_kernel[grid](q_out, k_out, q, k, cos, sin, position_ids, num_tokens, cos.stride(0), NUM_Q=num_q, NUM_K=num_k,
                        Q_PAD=q_pad, K_PAD=k_pad, HEAD_DIM=head_dim, BLOCK_T=block_t)
    return q_out, k_out


def reference(x, cos, sin, position_ids):
    c = cos[position_ids].float()[:, None, :]
    s = sin[position_ids].float()[:, None, :]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


def run_case(num_tokens, num_q, num_k, head_dim, max_positions, dtype, device):
    q = torch.randn(num_tokens, num_q, head_dim, dtype=dtype, device=device)
    k = torch.randn(num_tokens, num_k, head_dim, dtype=dtype, device=device)
    cos = torch.randn(max_positions, head_dim // 2, dtype=dtype, device=device)
    sin = torch.randn(max_positions, head_dim // 2, dtype=dtype, device=device)
    position_ids = torch.randint(0, max_positions, (num_tokens, ), dtype=torch.int64, device=device)

    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

    torch.testing.assert_close(q_out, reference(q, cos, sin, position_ids), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, reference(k, cos, sin, position_ids), atol=2e-2, rtol=2e-2)
    print(f"  tokens={num_tokens} q_heads={num_q} kv_heads={num_k} head_dim={head_dim} "
          f"{dtype}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="apply_rotary_pos_emb through TensorView")
    parser.add_argument("--tokens", type=int, default=NUM_TOKENS)
    parser.add_argument("--q-heads", type=int, default=NUM_Q_HEADS)
    parser.add_argument("--kv-heads", type=int, default=NUM_KV_HEADS)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)

    if args.no_check:
        print("Reference check skipped.")
        raise SystemExit(0)

    run_case(args.tokens, args.q_heads, args.kv_heads, args.head_dim, args.max_positions, torch.float16, device)
    run_case(args.tokens, args.q_heads, args.kv_heads, args.head_dim, args.max_positions, torch.bfloat16, device)
    # A token count that is not a multiple of the tile height leaves a partial tail block;
    # head counts that are not powers of two make the dense tile taller than the view.
    run_case(args.tokens - 5, args.q_heads, args.kv_heads, args.head_dim, args.max_positions, torch.float16, device)
    run_case(37, 6, 3, 64, 100, torch.float16, device)
    run_case(100, 28, 4, 128, 256, torch.float16, device)
    run_case(5, 4, 4, 32, 50, torch.float16, device)
    run_case(2048, 32, 8, 128, 4096, torch.float16, device)
    print("Test Passed!")
