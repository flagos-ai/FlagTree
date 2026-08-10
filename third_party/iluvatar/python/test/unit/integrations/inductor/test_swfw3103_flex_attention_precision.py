import math

import pytest
import torch
import triton
import triton.language as tl

try:
    from torch.nn.attention.flex_attention import flex_attention
except ImportError:
    flex_attention = None

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@triton.jit
def _score_debug_kernel(
    Q,
    K,
    BIAS,
    QK_OUT,
    BIAS_OUT,
    ADD_OUT,
    Q_LEN: tl.constexpr,
    HD: tl.constexpr,
    STRIDE_Z: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_M: tl.constexpr,
    STRIDE_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid % 4
    z = pid // 4
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HD)

    q = tl.load(
        Q + z * STRIDE_Z + h * STRIDE_H + offs_m[:, None] * STRIDE_M + offs_d[None, :] * STRIDE_K,
        mask=offs_m[:, None] < Q_LEN,
    )
    k = tl.load(
        K + z * STRIDE_Z + h * STRIDE_H + offs_n[:, None] * STRIDE_M + offs_d[None, :] * STRIDE_K,
        mask=offs_n[:, None] < Q_LEN,
        other=0.0,
    )
    qk = tl.dot(q, tl.trans(k), input_precision="ieee") * SM_SCALE

    m_idx = tl.minimum(offs_m, Q_LEN - 1)
    n_idx = tl.minimum(offs_n, Q_LEN - 1)
    bias = tl.load(BIAS + n_idx[None, :] + Q_LEN * m_idx[:, None]).to(tl.float32)
    add = qk + bias

    ptrs = pid * BLOCK_M * BLOCK_N + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    valid = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < Q_LEN)
    tl.store(QK_OUT + ptrs, qk, mask=valid)
    tl.store(BIAS_OUT + ptrs, bias, mask=valid)
    tl.store(ADD_OUT + ptrs, add, mask=valid)


def _clone_requires_grad(tensor: torch.Tensor, dtype=None) -> torch.Tensor:
    if dtype is None:
        dtype = tensor.dtype
    return tensor.detach().clone().to(dtype).requires_grad_(True)


def _assert_with_gold(eager: torch.Tensor, compiled: torch.Tensor, gold: torch.Tensor):
    ref_error = torch.mean(torch.square(eager.float() - gold.float())).sqrt()
    comp_error = torch.mean(torch.square(compiled.float() - gold.float())).sqrt()
    if ref_error.item() == 0.0:
        ref_error = torch.scalar_tensor(1e-5, device=gold.device)
    assert comp_error <= ref_error * 1.35 + 1e-7


@pytest.mark.skipif(flex_attention is None, reason="torch flex_attention is unavailable")
def test_absolute_2d_bias_compiled_matches_eager_and_gold():
    torch.manual_seed(3103)
    batch_size = 2
    num_heads = 4
    seq_length = 37
    head_dim = 16
    dtype = torch.bfloat16
    shape = (batch_size, num_heads, seq_length, head_dim)

    query_base = torch.randn(shape, device="cuda", dtype=dtype)
    key_base = torch.randn(shape, device="cuda", dtype=dtype)
    value_base = torch.randn(shape, device="cuda", dtype=dtype)
    bias_base = torch.randn((seq_length, seq_length), device="cuda", dtype=dtype)

    query_eager = _clone_requires_grad(query_base)
    key_eager = _clone_requires_grad(key_base)
    value_eager = _clone_requires_grad(value_base)
    bias_eager = _clone_requires_grad(bias_base)

    query_compiled = _clone_requires_grad(query_base)
    key_compiled = _clone_requires_grad(key_base)
    value_compiled = _clone_requires_grad(value_base)
    bias_compiled = _clone_requires_grad(bias_base)

    query_gold = _clone_requires_grad(query_base, torch.float32)
    key_gold = _clone_requires_grad(key_base, torch.float32)
    value_gold = _clone_requires_grad(value_base, torch.float32)
    bias_gold = _clone_requires_grad(bias_base, torch.float32)

    def eager_bias(score, b, h, q_idx, kv_idx):
        return score + bias_eager[q_idx, kv_idx]

    def compiled_bias(score, b, h, q_idx, kv_idx):
        return score + bias_compiled[q_idx, kv_idx]

    def gold_bias(score, b, h, q_idx, kv_idx):
        return score + bias_gold[q_idx, kv_idx]

    out_eager = flex_attention(query_eager, key_eager, value_eager, score_mod=eager_bias)
    out_compiled = torch.compile(flex_attention)(query_compiled, key_compiled, value_compiled, score_mod=compiled_bias)
    out_gold = flex_attention(query_gold, key_gold, value_gold, score_mod=gold_bias)

    grad = torch.randn_like(out_eager)
    eager_grads = torch.autograd.grad(out_eager, (query_eager, key_eager, value_eager, bias_eager), grad)
    compiled_grads = torch.autograd.grad(
        out_compiled,
        (query_compiled, key_compiled, value_compiled, bias_compiled),
        grad,
    )
    gold_grads = torch.autograd.grad(out_gold, (query_gold, key_gold, value_gold, bias_gold), grad.float())

    _assert_with_gold(out_eager, out_compiled, out_gold)
    for eager, compiled, gold in zip(eager_grads, compiled_grads, gold_grads, strict=True):
        _assert_with_gold(eager, compiled, gold)


def test_minimal_blocked_to_mma_bias_add_repro():
    seq = 37
    head_dim = 16
    block_m = 64
    block_n = 64
    torch.manual_seed(4321 + seq + head_dim)

    q = torch.randn((2, 4, seq, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, 4, seq, head_dim), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((seq, seq), device="cuda", dtype=torch.bfloat16)
    qk_out = torch.empty((8, block_m, block_n), device="cuda", dtype=torch.float32)
    bias_out = torch.empty((8, block_m, block_n), device="cuda", dtype=torch.float32)
    add_out = torch.empty((8, block_m, block_n), device="cuda", dtype=torch.float32)

    _score_debug_kernel[(8, )](
        q,
        k,
        bias,
        qk_out,
        bias_out,
        add_out,
        Q_LEN=seq,
        HD=head_dim,
        STRIDE_Z=4 * seq * head_dim,
        STRIDE_H=seq * head_dim,
        STRIDE_M=head_dim,
        STRIDE_K=1,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SM_SCALE=1.0 / math.sqrt(head_dim),
        num_warps=2,
        num_stages=1,
    )
    torch.cuda.synchronize()

    tri_qk = qk_out.view(2, 4, block_m, block_n)[:, :, :seq, :seq]
    tri_bias = bias_out.view(2, 4, block_m, block_n)[:, :, :seq, :seq]
    tri_add = add_out.view(2, 4, block_m, block_n)[:, :, :seq, :seq]

    ref_qk = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (1.0 / math.sqrt(head_dim))
    ref_bias = bias.float().view(1, 1, seq, seq).expand(2, 4, seq, seq)
    ref_add = ref_qk + ref_bias

    torch.testing.assert_close(tri_qk, ref_qk, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(tri_bias, ref_bias, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(tri_add, ref_add, atol=1e-2, rtol=1e-2)
