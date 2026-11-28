# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""PyTorch pointwise reference for the declared CUDA approximate log2."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _fast_log2():
    from torch.cuda.jiterator import _create_jit_fn

    return _create_jit_fn('''template<typename T> T flagmega_fast_log2(T value) {
        float result;
        asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"((float)value));
        return (T)result;
    }''')


def log2(value, *, mode, torch):
    if mode == "fast" and value.is_cuda:
        return _fast_log2()(value)
    return torch.log2(value)
