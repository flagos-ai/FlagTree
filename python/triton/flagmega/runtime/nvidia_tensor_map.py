# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""NVIDIA host tensor-map encoding independent of Triton driver API version."""

from __future__ import annotations

import ctypes
from functools import lru_cache
from typing import Sequence

from triton.flagmega.errors import RuntimeContractError


_TENSOR_MAP_BYTES = 128
_L2_PROMOTION_128B = 2
_INTERLEAVE_NONE = 0


def encode_tma_descriptor_for_driver(
    driver_utils,
    global_address: int,
    swizzle: int,
    item_size: int,
    host_dtype: int,
    block_shape: Sequence[int],
    shape: Sequence[int],
    strides: Sequence[int],
    padding: int,
) -> bytes:
    """Encode through either generation of FlagTree's descriptor API.

    Newer drivers expose a byte-returning helper.  Older drivers expose only
    ``fill_tma_descriptor``, whose opaque Python object is launchable but cannot
    form a device-resident table.  For that ABI, call the same public CUDA
    Driver API directly instead of depending on the object's CPython layout.
    """

    arguments = (
        int(global_address),
        int(swizzle),
        int(item_size),
        int(host_dtype),
        tuple(int(value) for value in block_shape),
        tuple(int(value) for value in shape),
        tuple(int(value) for value in strides),
        int(padding),
    )
    encoder = getattr(driver_utils, "encode_tma_descriptor", None)
    if encoder is not None:
        payload = encoder(*arguments)
    elif getattr(driver_utils, "fill_tma_descriptor", None) is not None:
        payload = encode_tma_descriptor(*arguments)
    else:
        raise RuntimeContractError(
            "The active Triton driver does not support tensor maps; a descriptor "
            "table cannot be materialized."
        )
    if not isinstance(payload, bytes) or len(payload) != _TENSOR_MAP_BYTES:
        raise RuntimeContractError(
            "The tensor-map encoder returned an invalid "
            f"{type(payload).__name__} payload of length "
            f"{len(payload) if hasattr(payload, '__len__') else 'unknown'}."
        )
    return payload


def encode_tma_descriptor(
    global_address: int,
    swizzle: int,
    item_size: int,
    host_dtype: int,
    block_shape: Sequence[int],
    shape: Sequence[int],
    strides: Sequence[int],
    padding: int,
) -> bytes:
    """Return the 128-byte CUDA tensor-map encoding for one tensor view."""

    block_shape = tuple(int(value) for value in block_shape)
    shape = tuple(int(value) for value in shape)
    strides = tuple(int(value) for value in strides)
    rank = len(shape)
    if not 1 <= rank <= 5:
        raise RuntimeContractError(
            f"CUDA tensor-map rank must be in [1, 5], got {rank}."
        )
    if len(block_shape) != rank or len(strides) != rank:
        raise RuntimeContractError(
            "CUDA tensor-map shape, stride, and block-shape ranks must match."
        )
    if any(value <= 0 for value in (*block_shape, *shape, *strides)):
        raise RuntimeContractError(
            "CUDA tensor-map extents and strides must be positive."
        )

    # CUDA's tensor-map dimensions are fastest-to-slowest, whereas FlagMega's
    # serialized ABI follows Python's slowest-to-fastest tensor convention.
    global_dimensions = (ctypes.c_uint64 * rank)(*reversed(shape))
    box_dimensions = (ctypes.c_uint32 * rank)(*reversed(block_shape))
    element_strides = (ctypes.c_uint32 * rank)(*([1] * rank))
    global_stride_values = [0] * rank
    for axis in range(rank - 1):
        global_stride_values[rank - axis - 2] = int(item_size) * strides[axis]
    global_stride_values[rank - 1] = int(global_dimensions[rank - 1]) * (
        int(item_size) if rank == 1 else global_stride_values[rank - 2]
    )
    global_strides = (ctypes.c_uint64 * rank)(*global_stride_values)

    # cuTensorMapEncodeTiled requires a 128-byte tensor map.  Keep the backing
    # allocation alive until its aligned slice has been copied into Python bytes.
    storage = (ctypes.c_ubyte * (_TENSOR_MAP_BYTES * 2 - 1))()
    output = (ctypes.addressof(storage) + _TENSOR_MAP_BYTES - 1) & ~(
        _TENSOR_MAP_BYTES - 1
    )
    result = _cuda_encode_tiled()(
        output,
        int(host_dtype),
        rank,
        int(global_address),
        global_dimensions,
        global_strides,
        box_dimensions,
        element_strides,
        _INTERLEAVE_NONE,
        int(swizzle),
        _L2_PROMOTION_128B,
        int(padding),
    )
    if result != 0:
        raise RuntimeContractError(
            "cuTensorMapEncodeTiled failed while materializing a tensor-map "
            f"table: {_cuda_error_text(int(result))}."
        )
    return ctypes.string_at(output, _TENSOR_MAP_BYTES)


@lru_cache(maxsize=1)
def _cuda_library():
    try:
        return ctypes.CDLL("libcuda.so.1")
    except OSError as error:
        raise RuntimeContractError(
            "The NVIDIA CUDA driver library is unavailable; a tensor-map table "
            "cannot be materialized."
        ) from error


@lru_cache(maxsize=1)
def _cuda_encode_tiled():
    try:
        function = _cuda_library().cuTensorMapEncodeTiled
    except AttributeError as error:
        raise RuntimeContractError(
            "The installed CUDA driver does not expose cuTensorMapEncodeTiled."
        ) from error
    function.restype = ctypes.c_int
    function.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    )
    return function


def _cuda_error_text(code: int) -> str:
    message = ctypes.c_char_p()
    try:
        result = _cuda_library().cuGetErrorString(int(code), ctypes.byref(message))
    except (AttributeError, TypeError):
        return f"CUDA error {code}"
    if result != 0 or not message.value:
        return f"CUDA error {code}"
    return message.value.decode(errors="replace")


__all__ = ["encode_tma_descriptor", "encode_tma_descriptor_for_driver"]
