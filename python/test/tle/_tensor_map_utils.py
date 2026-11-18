# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""CUDA driver descriptors for the TLE raw tensor-map tests."""

import ctypes

import torch


def encode_tensor_map(tensor, block_shape, swizzle=0):
    """Encode a CUDA tensor map from a test tensor and its physical box shape."""
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
    rank = tensor.ndim
    assert tensor.is_cuda and tensor.stride()[-1] == 1
    assert 1 <= rank <= 5 and len(block_shape) == rank
    storage = ctypes.create_string_buffer(255)
    offset = (-ctypes.addressof(storage)) % 128
    dimensions = (ctypes.c_uint64 * rank)(*reversed(tensor.shape))
    strides = (ctypes.c_uint64 * (rank - 1))(*(stride * tensor.element_size()
                                               for stride in reversed(tensor.stride()[:-1])))
    box = (ctypes.c_uint32 * rank)(*reversed(block_shape))
    element_strides = (ctypes.c_uint32 * rank)(*([1] * rank))
    dtype = {torch.float16: 6, torch.bfloat16: 9}[tensor.dtype]
    result = encode(
        ctypes.addressof(storage) + offset, dtype, rank, tensor.data_ptr(), dimensions, strides, box, element_strides,
        0, swizzle, 0, 0)
    assert result == 0, f"cuTensorMapEncodeTiled failed with CUDA error {result}"
    return storage.raw[offset:offset + 128]


def make_tensor_map_table(tensors):
    """Encode independent 8x8 FP16 tensor maps without a FlagMega dependency."""
    payload = bytearray()
    for tensor in tensors:
        assert tensor.shape == (8, 8) and tensor.dtype == torch.float16
        payload.extend(encode_tensor_map(tensor, (8, 8)))
    return torch.frombuffer(payload, dtype=torch.uint8).to(tensors[0].device)
