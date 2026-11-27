# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent scalar FMA evaluation for tiled normalization statistics."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _cuda_square_fma():
    from torch.cuda.jiterator import _create_jit_fn

    return _create_jit_fn('''template<typename T> T flagmega_norm_square_fma(T value, T previous) {
        float result;
        asm("fma.rn.f32 %0, %1, %1, %2;" : "=f"(result) : "f"((float)value), "f"((float)previous));
        return (T)result;
    }''')


def tiled_squares(value, *, axis, group, block, torch, traversal="forward"):
    outer = value.shape[:axis]
    source = value.reshape(*outer, -1)
    size = source.shape[-1]
    source = torch.nn.functional.pad(source, (0, (-size) % block))
    source = source.reshape(*outer, -1, block // group, group)
    squares = torch.zeros_like(source[..., 0, :, :])
    totals = torch.zeros_like(squares)
    blocks = range(source.shape[-3])
    if traversal == "reverse":
        blocks = reversed(blocks)
    for index in blocks:
        tile = source[..., index, :, :]
        squares = (_cuda_square_fma()(tile, squares) if value.is_cuda
                   else (tile.double().square() + squares.double()).float())
        totals = totals + tile
    square = torch.zeros_like(squares[..., 0])
    total = torch.zeros_like(square)
    for column in range(group):
        square = square + squares[..., column]
        total = total + totals[..., column]
    return square, total


def grouped_squares(value, *, axis, group, seed, torch):
    """Seed one product, then accumulate the remaining columns with FP32 FMA."""
    source = value.reshape(*value.shape[:axis], -1, group)
    square = torch.zeros_like(source[..., 0])
    total = torch.zeros_like(square)
    for column in (seed, *(i for i in range(group) if i != seed)):
        element = source[..., column]
        square = (_cuda_square_fma()(element, square) if value.is_cuda
                  else (element.double().square() + square.double()).float())
        total = total + element
    return square, total
