# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""CPU diagnostic for contiguous TF32 projection; WGMMA association is device-defined."""


def _tf32_rn(value, torch):
    value = value.float().contiguous()
    bits = value.view(torch.int32)
    rounded = ((bits + 0xfff + ((bits >> 13) & 1)) & -8192).view(torch.float32)
    return torch.where(torch.isfinite(value), rounded, value)


def contiguous_tf32_projection(streams, projection, splits, torch):
    tokens = streams.shape[0]
    flat = streams.reshape(tokens, -1).float()
    width = flat.shape[-1] // splits
    source, weight = _tf32_rn(flat, torch), _tf32_rn(projection, torch)
    products = torch.zeros((tokens, projection.shape[0]), dtype=torch.float32, device=flat.device)
    squares = torch.zeros((tokens,), dtype=torch.float32, device=flat.device)
    for split in range(splits):
        lo, hi = split * width, (split + 1) * width
        products = products + source[:, lo:hi] @ weight[:, lo:hi].t()
        partial = torch.zeros((tokens, 4), dtype=torch.float32, device=flat.device)
        for block in range(width // 64):
            for pair in range(8):
                start = lo + block * 64 + pair * 8
                first, second = flat[:, start:start + 4], flat[:, start + 4:start + 8]
                partial = partial + (first.square() + second.square())
        squares = squares + ((partial[:, 0] + partial[:, 2]) + (partial[:, 1] + partial[:, 3]))
    return products, squares
