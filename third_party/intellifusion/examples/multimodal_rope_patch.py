"""E20: 2D/3D split-half RoPE followed by window-major vision patch layout."""
import sys

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_patch(X, COS, SIN, Y, H: tl.constexpr, W: tl.constexpr, D: tl.constexpr, AXES: tl.constexpr, WH: tl.constexpr,
                WW: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    patch = token % (H * W)
    time = token // (H * W)
    win = patch // (WH * WW)
    local = patch % (WH * WW)
    height = (win // (W // WW)) * WH + local // WW
    width = (win % (W // WW)) * WW + local % WW
    source = ((time * H + height) * W + width) * D
    d = tl.arange(0, BLOCK)
    mask = d < D
    axis_d: tl.constexpr = D // AXES
    half: tl.constexpr = axis_d // 2
    axis = d // axis_d
    local_d = d % axis_d
    partner = d + tl.where(local_d < half, half, -half)
    if AXES == 2:
        position = tl.where(axis == 0, height, width)
    else:
        position = tl.where(axis == 0, time, tl.where(axis == 1, height, width))
    x = tl.load(X + source + d, mask, other=0).to(tl.float32)
    xp = tl.load(X + source + partner, mask, other=0).to(tl.float32)
    cs_offset = position * half + local_d % half
    cosine = tl.load(COS + cs_offset, mask, other=0)
    sine = tl.load(SIN + cs_offset, mask, other=0)
    y = x * cosine + tl.where(local_d < half, -xp, xp) * sine
    tl.store(Y + token * D + d, y, mask)


def multimodal_rope_patch(x, cos, sin, *, axes=2, window=(2, 2)):
    """x[T,H,W,D] -> window-major [T*H*W,D], with 2D or 3D RoPE.

    Each axis receives D/axes channels and uses split-half rotation within
    those channels. cos/sin are FP32 [max(T,H,W), D/(2*axes)] tables on IFWA.
    T may be 1 in 2D mode. H/W must divide window dimensions. No mutation.
    """
    if x.ndim != 4 or axes not in (2, 3):
        raise ValueError('expected x[T,H,W,D] and axes=2 or 3')
    t, h, w, d = x.shape
    if min(x.shape) <= 0 or d % (2 * axes) or not 2 <= d // axes <= 128:
        raise ValueError('positive shape and even channels per axis required')
    if len(window) != 2 or min(window) <= 0 or h % window[0] or w % window[1]:
        raise ValueError('window dimensions must divide H,W')
    expected = (max(t, h, w), d // (2 * axes))
    if x.device.type != 'ifwa' or x.dtype not in (torch.float32,
                                                  torch.bfloat16) or not x.is_contiguous() or x.requires_grad:
        raise ValueError('x must be contiguous inference FP32/BF16 on IFWA')
    if any(v.shape != expected or v.device != x.device or v.dtype != torch.float32 or not v.is_contiguous()
           or v.requires_grad for v in (cos, sin)):
        raise ValueError(f'cos/sin must be FP32 IFWA tables of shape {expected}')
    out = torch.empty((t * h * w, d), device=x.device, dtype=x.dtype)
    _rope_patch[(t * h * w, )](x, cos, sin, out, h, w, d, axes, window[0], window[1], triton.next_power_of_2(d),
                               num_warps=4, enable_fp_fusion=False)
    return out


def reference(x, cos, sin, axes, window):
    t, h, w, d = x.shape
    axis_d = d // axes
    result = []
    for frame in range(t):
        for win_h in range(0, h, window[0]):
            for win_w in range(0, w, window[1]):
                for local_h in range(window[0]):
                    for local_w in range(window[1]):
                        row, col = win_h + local_h, win_w + local_w
                        positions = (row, col) if axes == 2 else (frame, row, col)
                        rotated = []
                        for axis, position in enumerate(positions):
                            values = x[frame, row, col, axis * axis_d:(axis + 1) * axis_d].float()
                            half = axis_d // 2
                            rotated.append(
                                torch.cat((values[:half] * cos[position] - values[half:] * sin[position],
                                           values[half:] * cos[position] + values[:half] * sin[position])))
                        result.append(torch.cat(rotated).to(x.dtype))
    return torch.stack(result)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('axes,shape', [(2, (1, 4, 4, 16)), (3, (2, 4, 4, 24))])
@pytest.mark.parametrize('window', [(2, 2), (1, 2)])
def test_multimodal_rope_patch(dtype, axes, shape, window):
    gen = torch.Generator().manual_seed(200)
    x = torch.randn(shape, generator=gen).to(dtype)
    max_pos = max(shape[:3])
    half = shape[-1] // (2 * axes)
    angles = torch.arange(max_pos)[:, None] * torch.pow(10000.0, -torch.arange(half).float()[None, :] / half)
    cos, sin = angles.cos(), angles.sin()
    expected = reference(x, cos, sin, axes, window)
    dev = [v.to('ifwa:0') for v in (x, cos, sin)]
    actual = multimodal_rope_patch(*dev, axes=axes, window=window).cpu()
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    for original, moved in zip((x, cos, sin), dev):
        torch.testing.assert_close(moved.cpu(), original, rtol=0, atol=0)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
