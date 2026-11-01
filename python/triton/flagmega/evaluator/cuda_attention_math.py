# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent scalar CUDA evaluation of online attention arithmetic."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _probability():
    from torch.cuda.jiterator import _create_jit_fn

    return _create_jit_fn('''template<typename T> T flagmega_online_probability(T score, T maximum) {
        float scale = 1.4426950408889634f;
        float max_scaled = (float)maximum * scale;
        float argument, output;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(argument)
            : "f"((float)score), "f"(scale), "f"(-max_scaled));
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(output) : "f"(argument));
        return (T)output;
    }''')


@lru_cache(maxsize=1)
def _inverse():
    from torch.cuda.jiterator import _create_jit_fn

    return _create_jit_fn('''template<typename T> T flagmega_online_inverse(T denominator) {
        float value = (float)denominator;
        float output;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(output) : "f"(value));
        return (T)((value == 0.f || value != value) ? 0.f : output);
    }''')


def probability(score, maximum):
    """Keep the FMA before exp2 and the later unnormalized BF16 store."""
    return _probability()(score, maximum)


def normalize(accumulator, denominator):
    """Apply the online softmax's approximate reciprocal after PV."""
    return accumulator * _inverse()(denominator)
