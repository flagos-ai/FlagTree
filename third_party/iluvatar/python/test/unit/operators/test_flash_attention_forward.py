import math
import pytest
import torch

import triton.ops


def _make_inputs(batch_size, nheads_q, nheads_k, seqlen_q, seqlen_k, headdim, dtype, device):
    q = torch.empty((batch_size, seqlen_q, nheads_q, headdim), dtype=dtype, device=device)
    k = torch.empty((batch_size, seqlen_k, nheads_k, headdim), dtype=dtype, device=device)
    v = torch.empty((batch_size, seqlen_k, nheads_k, headdim), dtype=dtype, device=device)
    q.normal_(mean=0.1, std=0.2)
    k.normal_(mean=0.4, std=0.2)
    v.normal_(mean=0.3, std=0.2)
    return q.requires_grad_(), k.requires_grad_(), v.requires_grad_()


def _make_varlen_inputs(lengths_q, lengths_k, nheads_q, nheads_k, headdim, dtype, device):
    assert len(lengths_q) == len(lengths_k)

    q_chunks = []
    k_chunks = []
    v_chunks = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    total_q = 0
    total_k = 0

    for seqlen_q, seqlen_k in zip(lengths_q, lengths_k):
        q = torch.empty((seqlen_q, nheads_q, headdim), dtype=dtype, device=device)
        k = torch.empty((seqlen_k, nheads_k, headdim), dtype=dtype, device=device)
        v = torch.empty((seqlen_k, nheads_k, headdim), dtype=dtype, device=device)
        q.normal_(mean=0.1, std=0.2)
        k.normal_(mean=0.4, std=0.2)
        v.normal_(mean=0.3, std=0.2)

        q_chunks.append(q)
        k_chunks.append(k)
        v_chunks.append(v)
        total_q += seqlen_q
        total_k += seqlen_k
        cu_seqlens_q.append(total_q)
        cu_seqlens_k.append(total_k)

    q = torch.cat(q_chunks, dim=0).requires_grad_()
    k = torch.cat(k_chunks, dim=0).requires_grad_()
    v = torch.cat(v_chunks, dim=0).requires_grad_()
    cu_q = torch.tensor(cu_seqlens_q, device=device, dtype=torch.int32)
    cu_k = torch.tensor(cu_seqlens_k, device=device, dtype=torch.int32)
    return q, k, v, cu_q, cu_k


def _torch_flash_fwd(
    q,
    k,
    v,
    softmax_scale,
    causal,
    dropout_p=0.0,
    return_debug_mask=False,
    **extra_kwargs,
):
    return torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[-3],
        k.shape[-3],
        dropout_p,
        causal,
        return_debug_mask,
        scale=softmax_scale,
        **extra_kwargs,
    )


def _torch_flash_varlen_fwd(
    q,
    k,
    v,
    cu_q,
    cu_k,
    max_q,
    max_k,
    softmax_scale,
    causal,
    dropout_p=0.0,
    return_debug_mask=False,
    **extra_kwargs,
):
    return torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        dropout_p,
        causal,
        return_debug_mask,
        scale=softmax_scale,
        **extra_kwargs,
    )


def _triton_flash_fwd(
    q,
    k,
    v,
    softmax_scale,
    causal,
    dropout_p=0.0,
    return_debug_mask=False,
    **extra_kwargs,
):
    return triton.ops._flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[-3],
        k.shape[-3],
        dropout_p,
        causal,
        return_debug_mask,
        scale=softmax_scale,
        **extra_kwargs,
    )


def _triton_flash_varlen_fwd(
    q,
    k,
    v,
    cu_q,
    cu_k,
    max_q,
    max_k,
    softmax_scale,
    causal,
    dropout_p=0.0,
    return_debug_mask=False,
    **extra_kwargs,
):
    return triton.ops._flash_attention_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        dropout_p,
        causal,
        return_debug_mask,
        scale=softmax_scale,
        **extra_kwargs,
    )


def _manual_log2_lse(q, k, softmax_scale, causal):
    batch_size, seqlen_q, nheads_q, _ = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert nheads_q % nheads_k == 0

    if nheads_q != nheads_k:
        group_size = nheads_q // nheads_k
        k = k.repeat_interleave(group_size, dim=2)

    q_ref = q.permute(0, 2, 1, 3).to(torch.float32)
    k_ref = k.permute(0, 2, 1, 3).to(torch.float32)
    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * softmax_scale

    if causal:
        mask = torch.ones((seqlen_q, seqlen_k), dtype=torch.bool, device=q.device)
        mask = torch.tril(mask, diagonal=seqlen_k - seqlen_q)
        scores = torch.where(mask[None, None, :, :], scores, torch.full_like(scores, float("-inf")))
        valid_rows = mask.any(dim=-1).view(1, 1, seqlen_q).expand(batch_size, nheads_q, seqlen_q)
    else:
        valid_rows = None

    lse = torch.logsumexp(scores, dim=-1) / math.log(2)
    if valid_rows is not None:
        lse = torch.where(valid_rows, lse, torch.zeros_like(lse))
    return lse


def _reference_attention(q, k, v, softmax_scale, causal, bias=None):
    _, seqlen_q, nheads_q, _ = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert nheads_q % nheads_k == 0

    if nheads_q != nheads_k:
        group_size = nheads_q // nheads_k
        k = k.repeat_interleave(group_size, dim=2)
        v = v.repeat_interleave(group_size, dim=2)

    q_ref = q.permute(0, 2, 1, 3).to(torch.float32)
    k_ref = k.permute(0, 2, 1, 3).to(torch.float32)
    v_ref = v.permute(0, 2, 1, 3).to(torch.float32)

    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * softmax_scale
    if bias is not None:
        scores = scores + bias

    if causal:
        mask = torch.ones((seqlen_q, seqlen_k), dtype=torch.bool, device=q.device)
        mask = torch.tril(mask, diagonal=seqlen_k - seqlen_q)
        scores = torch.where(mask[None, None, :, :], scores, torch.full_like(scores, float("-inf")))

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_ref)
    return out.permute(0, 2, 1, 3).to(q.dtype)


def _manual_alibi_out(q, k, v, alibi_slopes, softmax_scale, causal):
    _, seqlen_q, _, _ = q.shape
    _, seqlen_k, _, _ = k.shape

    q_pos = torch.arange(seqlen_q, device=q.device, dtype=torch.int32)
    k_pos = torch.arange(seqlen_k, device=q.device, dtype=torch.int32)
    rel = (q_pos[:, None] + seqlen_k - seqlen_q - k_pos[None, :]).abs().to(torch.float32)
    bias = (-alibi_slopes[:, :, None, None].to(torch.float32)) * rel[None, None, :, :]

    return _reference_attention(q, k, v, softmax_scale, causal, bias=bias)


def _assert_output_close(ref_out, tri_out, dtype):
    atol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(ref_out, tri_out, atol=atol, rtol=rtol, equal_nan=True)


def _assert_lse_close(ref_lse, tri_lse, dtype):
    atol = 5e-3 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(ref_lse, tri_lse, atol=atol, rtol=0.0, equal_nan=True)


@pytest.mark.parametrize(
    "batch_size,nheads_q,nheads_k,seqlen_q,seqlen_k,headdim",
    [
        # base cases
        (4, 8, 8, 1024, 1024, 64),
        (4, 8, 8, 1024, 1024, 128),
        (4, 8, 8, 2048, 2048, 64),
        (4, 8, 8, 2048, 2048, 128),
        (4, 8, 8, 4096, 4096, 64),
        (4, 8, 8, 4096, 4096, 128),
        # seqlen_q != seqlen_k
        (4, 8, 8, 2048, 256, 64),
        (4, 8, 8, 2048, 256, 128),
        # seqlen_q != seqlen_k OR seqlen_q % BM != 0 OR seqlen_k % BN != 0
        (4, 8, 8, 17, 1030, 64),
        (4, 8, 8, 17, 1030, 128),
        (2, 4, 4, 512, 612, 128),
        (2, 4, 4, 1024, 1034, 64),
        (2, 4, 4, 4001, 4096, 64),
        (2, 4, 4, 4096, 4000, 128),
        # headdim != 64 OR 128
        (2, 4, 4, 2048, 2048, 32),
        (2, 4, 4, 4096, 4096, 16),
        (2, 4, 4, 4001, 4001, 32),
        (1, 2, 2, 8192, 8202, 16),
        (1, 2, 2, 8192, 8192, 32),
        # mqa/gqa
        (2, 4, 2, 512, 612, 128),
        (2, 4, 1, 1024, 1034, 64),
        (2, 4, 2, 2048, 2048, 32),
        (2, 4, 1, 4096, 4096, 16),
        (2, 4, 2, 4001, 4001, 32),
        (2, 4, 1, 4001, 4096, 64),
        (2, 4, 2, 4096, 4000, 128),
        (1, 2, 1, 8192, 8202, 16),
        (1, 2, 1, 8192, 8192, 32),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_op_fwd(batch_size, nheads_q, nheads_k, seqlen_q, seqlen_k, headdim, dtype, causal):
    torch.manual_seed(42)
    device = "cuda"
    softmax_scale = 0.2

    q, k, v = _make_inputs(batch_size, nheads_q, nheads_k, seqlen_q, seqlen_k, headdim, dtype, device)

    tri_out, _, _, _, _ = _triton_flash_fwd(q, k, v, softmax_scale, causal)
    ref_out, _, _, _, _ = _torch_flash_fwd(q, k, v, softmax_scale, causal)

    _assert_output_close(ref_out, tri_out, dtype)


@pytest.mark.parametrize(
    "lengths_q,lengths_k,nheads_q,nheads_k,headdim",
    [
        ([128, 17, 96], [128, 103, 64], 4, 4, 64),
        ([96, 64, 33], [64, 96, 47], 4, 2, 32),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_op_varlen_fwd(lengths_q, lengths_k, nheads_q, nheads_k, headdim, dtype, causal):
    torch.manual_seed(42)
    device = "cuda"
    softmax_scale = 0.2
    max_q = max(lengths_q)
    max_k = max(lengths_k)

    q, k, v, cu_q, cu_k = _make_varlen_inputs(
        lengths_q,
        lengths_k,
        nheads_q,
        nheads_k,
        headdim,
        dtype,
        device,
    )

    tri_out, _, _, _, _ = _triton_flash_varlen_fwd(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale, causal)
    ref_out, _, _, _, _ = _torch_flash_varlen_fwd(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale, causal)

    _assert_output_close(ref_out, tri_out, dtype)


@pytest.mark.parametrize(
    "batch_size,nheads_q,nheads_k,seqlen_q,seqlen_k,headdim",
    [
        (1, 1, 1, 128, 128, 64),
        (1, 2, 2, 17, 103, 64),
        (1, 2, 1, 96, 64, 32),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_op_lse_matches_log2sumexp(batch_size, nheads_q, nheads_k, seqlen_q, seqlen_k, headdim, dtype, causal):
    torch.manual_seed(42)
    device = "cuda"
    softmax_scale = 0.2

    q, k, v = _make_inputs(batch_size, nheads_q, nheads_k, seqlen_q, seqlen_k, headdim, dtype, device)

    _, tri_lse, _, _, _ = _triton_flash_fwd(q, k, v, softmax_scale, causal)
    ref_lse = _manual_log2_lse(q, k, softmax_scale, causal)

    _assert_lse_close(ref_lse, tri_lse, dtype)


@pytest.mark.parametrize("batch_size,nheads_q,nheads_k,seqlen_q,seqlen_k,headdim", [
    (4, 2, 2, 1024, 1024, 64),
    (4, 2, 2, 1024, 1024, 128),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_op_fwd_dropout(batch_size, nheads_q, nheads_k, seqlen_q, seqlen_k, headdim, dtype, causal):
    torch.manual_seed(42)
    device = "cuda"
    softmax_scale = 0.2
    dropout_p = 0.2

    q, k, v = _make_inputs(batch_size, nheads_q, nheads_k, seqlen_q, seqlen_k, headdim, dtype, device)

    _, _, _, _, debug_attn_mask = _triton_flash_fwd(
        q,
        k,
        v,
        softmax_scale,
        causal,
        dropout_p=dropout_p,
        return_debug_mask=True,
    )

    assert debug_attn_mask is not None
    valid = debug_attn_mask != 0
    dropout_ratio = torch.sum(debug_attn_mask < 0).to(torch.float32) / torch.sum(valid).to(torch.float32)
    torch.testing.assert_close(
        dropout_ratio,
        torch.tensor(dropout_p, device=device),
        atol=0.05,
        rtol=0.0,
    )


@pytest.mark.parametrize("batch_size,nheads_q,nheads_k,seqlen_q,seqlen_k,headdim", [
    (4, 2, 2, 1024, 1024, 64),
    (4, 2, 2, 1024, 1024, 128),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_op_fwd_alibi(batch_size, nheads_q, nheads_k, seqlen_q, seqlen_k, headdim, dtype, causal):
    torch.manual_seed(42)
    device = "cuda"
    softmax_scale = 0.2

    q, k, v = _make_inputs(batch_size, nheads_q, nheads_k, seqlen_q, seqlen_k, headdim, dtype, device)
    alibi_slopes = torch.empty((batch_size, nheads_q), dtype=torch.float32, device=device).uniform_(0.01, 0.08)

    tri_out, _, _, _, _ = _triton_flash_fwd(
        q,
        k,
        v,
        softmax_scale,
        causal,
        alibi_slopes=alibi_slopes,
    )
    ref_out = _manual_alibi_out(q, k, v, alibi_slopes, softmax_scale, causal)

    _assert_output_close(ref_out, tri_out, dtype)


# ---------------------------------------------------------------------------
# Flex-attention homologous config: fp32, BLOCK 128x128, num_warps=16, which
# mirrors the Inductor flex templates.
# ---------------------------------------------------------------------------
_FLEX_HOMOLOGOUS_CONFIG = triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=16, num_stages=1)


def _jit_function(kernel):
    """Unwrap Autotuner/Heuristics down to the JITFunction holding compiled kernels."""
    while not hasattr(kernel, "device_caches"):
        kernel = kernel.fn
    return kernel


def _compiled_shared_bytes(kernel):
    """Shared-memory footprint of every kernel compiled from `kernel` so far."""
    return [
        compiled.metadata.shared
        for kernel_cache, *_ in _jit_function(kernel).device_caches.values()
        for compiled in kernel_cache.values()
    ]


def _pin_launch_config(config):
    """Pin both MR forward kernels to `config` and drop what they compiled before.
    """
    from triton.ops.flash_attn import fwd_prefill as fa_fwd

    fa_fwd.launch_configs_mr[:] = [config]
    for kernel in (fa_fwd.attn_fwd_mr, fa_fwd.attn_fwd_mr_fast):
        kernel.cache.clear()
        _jit_function(kernel).device_caches.clear()
    return fa_fwd


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("headdim,shared_budget", [(64, 98304), (128, 131072)])
@pytest.mark.parametrize(
    "batch_size,nheads_q,nheads_k,seqlen_q,seqlen_k,expected_kernel",
    [
        # seqlen_q == seqlen_k and both % 256 == 0 takes the fast path
        (1, 8, 8, 1024, 1024, "attn_fwd_mr_fast"),
        # seqlen_q != seqlen_k falls back to the two-scf.if causal kernel
        (1, 2, 2, 512, 640, "attn_fwd_mr"),
    ],
)
def test_op_fwd_flex_homologous_fp32_block128(
    batch_size,
    nheads_q,
    nheads_k,
    seqlen_q,
    seqlen_k,
    expected_kernel,
    headdim,
    shared_budget,
    causal,
):
    """Force the flex-homologous fp32 tile and hold its shared-memory budget."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32
    softmax_scale = headdim**-0.5

    q, k, v = _make_inputs(batch_size, nheads_q, nheads_k, seqlen_q, seqlen_k, headdim, dtype, device)

    fa_fwd = _pin_launch_config(_FLEX_HOMOLOGOUS_CONFIG)
    tri_out, _, _, _, _ = _triton_flash_fwd(q, k, v, softmax_scale, causal)
    shared = {name: _compiled_shared_bytes(getattr(fa_fwd, name)) for name in ("attn_fwd_mr", "attn_fwd_mr_fast")}

    # The dispatch heuristic decides which kernel the shared-memory rules above
    # are exercised on, so pin it down instead of trusting it to stay put.
    assert shared[expected_kernel], f"expected {expected_kernel}, compiled {shared}"
    for name, sizes in shared.items():
        if name != expected_kernel:
            assert not sizes, f"unexpected dispatch to {name}: {shared}"
    assert max(shared[expected_kernel]) <= shared_budget, shared

    ref_out = _reference_attention(q, k, v, softmax_scale, causal)
    torch.testing.assert_close(ref_out, tri_out, atol=1e-5, rtol=1e-5, equal_nan=False)
