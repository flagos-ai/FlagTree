"""Regression tests for packed rotary embedding on the Cambricon backend."""

import pytest
import torch


pytest.importorskip("torch_mlu")
if not torch.mlu.is_available():
    pytest.skip("Cambricon MLU is not available", allow_module_level=True)

from triton.ops.apply_rotary import apply_rotary


def _fold_reference(x, positions, cos, sin, cu_seqlens):
    out = torch.empty_like(x)
    rotary_dim = cos.shape[-1] * 2
    for batch in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[batch].item())
        end = int(cu_seqlens[batch + 1].item())
        pos = positions[batch, : end - start]
        values = x[start:end, :, :rotary_dim].float()
        even = values[..., : rotary_dim // 2]
        odd = values[..., rotary_dim // 2 :]
        c = cos.index_select(0, pos).view(-1, 1, rotary_dim // 2)
        s = sin.index_select(0, pos).view(-1, 1, rotary_dim // 2)
        rotated = torch.cat((even * c - odd * s, even * s + odd * c), dim=-1)
        out[start:end, :, :rotary_dim] = rotated.to(x.dtype)
    return out


def test_packed_fold_rotary_uses_sequence_stride():
    device = torch.device("mlu")
    dtype = torch.float16
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    positions = torch.tensor([[4, 7, 0], [1, 3, 9]], dtype=torch.int32, device=device)
    x = torch.randn((5, 2, 16), dtype=dtype, device=device)
    cos = torch.randn((12, 8), dtype=dtype, device=device)
    sin = torch.randn((12, 8), dtype=dtype, device=device)
    output = torch.empty_like(x)

    apply_rotary(
        output,
        x,
        cos,
        sin,
        BLOCK_M=2,
        token_offsets=positions,
        cu_seqlens=cu_seqlens,
        max_seqlen=3,
        interleaved=False,
    )

    torch.testing.assert_close(
        output,
        _fold_reference(x, positions, cos, sin, cu_seqlens),
        atol=2e-2,
        rtol=2e-2,
    )
