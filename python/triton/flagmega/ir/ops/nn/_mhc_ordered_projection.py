# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reference for the explicit mHC 128-lane stream/FMA reduction."""


def _fma32(a, b, c, torch):
    """Round once to FP32, including FP64 sums exactly on an FP32 midpoint."""
    product = a.double() * b.double()
    addend = c.double()
    total = product + addend
    # TwoSum retains a tiny addend that FP64 addition can lose. Such an
    # addend matters if the FP64 result lands exactly on an FP32 midpoint.
    recovered = total - product
    error = (product - (total - recovered)) + (addend - recovered)
    rounded = total.float()
    wide = rounded.double()
    direction = torch.where(total > wide, float("inf"), float("-inf"))
    adjacent = torch.nextafter(rounded, direction.float())
    midpoint = (wide + adjacent.double()) * .5
    towards_adjacent = (adjacent.double() > wide) == (error > 0)
    return torch.where((total == midpoint) & (error != 0) & towards_adjacent, adjacent, rounded)


def ordered_stream_projection(streams, projection, splits, torch):
    tokens, hc, hidden = streams.shape
    count = projection.shape[0]
    width = hidden // splits
    lanes = torch.arange(128, device=streams.device)
    values = streams.float()
    weights = projection.float().reshape(count, hc, hidden)
    mixes = torch.zeros((tokens, count, splits, 128), device=streams.device, dtype=torch.float32)
    squares = torch.zeros((tokens, splits, 128), device=streams.device, dtype=torch.float32)
    for iteration in range(width // 128):
        indices = torch.arange(splits, device=streams.device)[:, None] * width + iteration * 128 + lanes
        for stream in range(hc):
            value = values[:, stream, indices]
            weight = weights[:, stream, indices]
            mixes = _fma32(weight[None], value[:, None], mixes, torch)
            squares = _fma32(value, value, squares, torch)

    def finish(partials):
        partials = partials.reshape(*partials.shape[:-1], 4, 32)
        for step in (16, 8, 4, 2, 1):
            partials = partials + partials[..., lanes[:32] ^ step]
        total = torch.zeros_like(partials[..., 0, 0])
        for warp in range(4):
            total = total + partials[..., warp, 0]
        result = torch.zeros_like(total[..., 0])
        for split in range(splits):
            result = result + total[..., split]
        return result

    return finish(mixes), finish(squares)
