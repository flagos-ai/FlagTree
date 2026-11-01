# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent scalar CUDA evaluation of declared activation arithmetic."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _accurate_silu():
    from torch.cuda.jiterator import _create_jit_fn

    return _create_jit_fn('''template<typename T> T flagmega_accurate_silu(T value) {
        float wide = (float)value;
        float denominator = expf(-wide) + 1.f;
        float output;
        asm("div.full.f32 %0, %1, %2;" : "=f"(output) : "f"(wide), "f"(denominator));
        return (T)output;
    }''')


def accurate_silu(value):
    """Use the exp/full-division expression declared by CUDA accurate SiLU.

    Torch's activation kernel can round its division differently. The scalar
    evaluator uses Jiterator, independently of FlagMega compilation and fusion.
    """
    return _accurate_silu()(value.float()).to(value.dtype)


@lru_cache(maxsize=1)
def _fast_silu():
    from torch.cuda.jiterator import _create_jit_fn

    return _create_jit_fn('''template<typename T> T flagmega_fast_silu(T value) {
        float wide = (float)value;
        float argument = -wide * 1.4426950408889634f;
        float exponential, output;
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponential) : "f"(argument));
        float denominator = exponential + 1.f;
        asm("div.full.f32 %0, %1, %2;" : "=f"(output) : "f"(wide), "f"(denominator));
        return (T)output;
    }''')


def fast_silu(value):
    """Evaluate approximate CUDA exp and full division before the final cast.

    Torch division rounds differently at some BF16 store midpoints, even when
    both paths have the same denominator. Keep the declared scalar expression.
    """
    return _fast_silu()(value.float()).to(value.dtype)


@lru_cache(maxsize=1)
def _fast_sigmoid():
    from torch.cuda.jiterator import _create_jit_fn

    return _create_jit_fn('''template<typename T> T flagmega_fast_sigmoid(T value) {
        float argument = -(float)value * 1.4426950408889634f;
        float exponential, output;
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponential) : "f"(argument));
        float denominator = exponential + 1.f;
        asm("div.full.f32 %0, %1, %2;" : "=f"(output) : "f"(1.f), "f"(denominator));
        return (T)output;
    }''')


def fast_sigmoid(value):
    """Preserve CUDA approximate exp and full reciprocal division."""
    return _fast_sigmoid()(value.float()).to(value.dtype)
