# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""CPU construction of NVIDIA's finite-range FP32 rotary cache."""

from functools import lru_cache


def _fma(a, b, c):
    import ctypes
    import numpy as np

    a, b, c = np.broadcast_arrays(np.asarray(a, dtype=np.float32),
                                  np.asarray(b, dtype=np.float32), np.asarray(c, dtype=np.float32))
    tiny = np.finfo(np.float32).tiny
    a, b, c = (np.where(np.abs(x) < tiny, np.copysign(np.float32(0), x), x) for x in (a, b, c))
    wide = a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)
    result = wide.astype(np.float32)
    # Double rounding can affect FP32 only when the FP64 result lands on an
    # FP32 midpoint. Resolve those entries with the system's true FP32 FMA.
    direction = np.where(wide >= result.astype(np.float64), np.float32(np.inf), np.float32(-np.inf))
    adjacent = np.nextafter(result, direction)
    midpoint = (result.astype(np.float64) + adjacent.astype(np.float64)) * .5
    ties = (wide == midpoint) & (np.abs(wide) >= tiny)
    if ties.any():
        fn = _system_fmaf()
        flat = result.ravel()
        for i in np.flatnonzero(ties.ravel()):
            flat[i] = fn(float(a.ravel()[i]), float(b.ravel()[i]), float(c.ravel()[i]))
    return np.where(np.abs(result) < tiny, np.copysign(np.float32(0), result), result)


@lru_cache(maxsize=1)
def _system_fmaf():
    import ctypes
    import ctypes.util

    function = ctypes.CDLL(ctypes.util.find_library("m")).fmaf
    function.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    function.restype = ctypes.c_float
    return function


@lru_cache(maxsize=32)
def rotary_table(positions, frequencies, scaling, dtype):
    import numpy as np
    import torch

    def bits(value):
        return np.asarray(value, dtype=np.uint32).view(np.float32)

    angle = np.arange(positions, dtype=np.float32)[:, None] * np.asarray(frequencies, dtype=np.float32)
    quadrant = np.rint(angle * bits(0x3f22f983)).astype(np.int32)
    multiple = quadrant.astype(np.float32)
    reduced = _fma(multiple, bits(0xbfc90fda), angle)
    reduced = _fma(multiple, bits(0xb3a22168), reduced)
    reduced = _fma(multiple, bits(0xa7c234c5), reduced)
    square = reduced * reduced
    square = np.where(np.abs(square) < np.finfo(np.float32).tiny, np.float32(0), square)
    cosine = _fma(bits(0x37cbac00), square, bits(0xbab607ed))
    cosine = _fma(cosine, square, bits(0x3d2aaabb))
    cosine = _fma(cosine, square, bits(0xbeffffff))
    cosine = _fma(cosine, _fma(square, 1., 0.), 1.)
    sine = _fma(bits(0xb94d4153), square, bits(0x3c0885e4))
    sine = _fma(sine, square, bits(0xbe2aaaa8))
    sine = _fma(sine, _fma(square, reduced, 0.), reduced)
    cos_value = np.where((quadrant & 1) != 0, sine, cosine)
    sin_value = np.where((quadrant & 1) != 0, cosine, sine)
    cos_value = np.where(((quadrant + 1) & 2) != 0, _fma(cos_value, -1., 0.), cos_value)
    sin_value = np.where((quadrant & 2) != 0, _fma(sin_value, -1., 0.), sin_value)
    cos_value = np.concatenate((cos_value, cos_value), axis=-1) * np.float32(scaling)
    sin_value = np.concatenate((sin_value, sin_value), axis=-1) * np.float32(scaling)
    return torch.from_numpy(np.stack((cos_value, sin_value), axis=1)).to(getattr(torch, dtype))
