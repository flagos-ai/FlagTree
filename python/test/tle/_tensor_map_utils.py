# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""CUDA driver descriptors for the TLE raw tensor-map tests."""

import ctypes

import torch


def make_tensor_map_table(tensors):
    """Encode independent 8x8 FP16 tensor maps without a FlagMega dependency."""
    encode = ctypes.CDLL("libcuda.so.1").cuTensorMapEncodeTiled
    encode.restype = ctypes.c_int
    encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    payload = bytearray()
    for tensor in tensors:
        assert tensor.shape == (8, 8) and tensor.dtype == torch.float16
        assert tensor.is_cuda and tensor.stride()[-1] == 1
        storage = ctypes.create_string_buffer(255)
        offset = (-ctypes.addressof(storage)) % 128
        dimensions = (ctypes.c_uint64 * 2)(8, 8)
        strides = (ctypes.c_uint64 * 1)(tensor.stride()[0] * tensor.element_size())
        box = (ctypes.c_uint32 * 2)(8, 8)
        element_strides = (ctypes.c_uint32 * 2)(1, 1)
        result = encode(
            ctypes.addressof(storage) + offset, 6, 2, tensor.data_ptr(), dimensions, strides, box, element_strides, 0,
            0, 0, 0)
        assert result == 0, f"cuTensorMapEncodeTiled failed with CUDA error {result}"
        payload.extend(storage.raw[offset:offset + 128])
    return torch.frombuffer(payload, dtype=torch.uint8).to(tensors[0].device)
