# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""PyTorch pointwise reference for declared CUDA approximate L2 arithmetic.

This evaluates two scalar instructions through PyTorch's CUDA Jiterator. It
uses no FlagMega artifact, kernel template, memory plan or generated entry.
"""

from functools import lru_cache


@lru_cache(maxsize=2)
def _pointwise(reciprocal):
    from torch.cuda.jiterator import _create_jit_fn

    if reciprocal:
        return _create_jit_fn('''template<typename T> T flagmega_inverse_l2_denominator(T square_sum) {
            float denominator, inverse;
            asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(denominator) : "f"((float)square_sum));
            asm("div.full.f32 %0, %1, %2;" : "=f"(inverse) : "f"(1.f), "f"(denominator));
            return (T)inverse;
        }''')
    return _create_jit_fn('''template<typename T> T flagmega_divide_l2_denominator(T value, T square_sum) {
        float denominator, normalized;
        asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(denominator) : "f"((float)square_sum));
        asm("div.full.f32 %0, %1, %2;" : "=f"(normalized) : "f"((float)value), "f"(denominator));
        return (T)normalized;
    }''')


def normalize(value, square_sum, *, reciprocal):
    return value * _pointwise(True)(square_sum) if reciprocal else _pointwise(False)(value, square_sum)
