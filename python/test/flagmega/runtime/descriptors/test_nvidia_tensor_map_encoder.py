# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import ctypes

import pytest

from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.runtime.nvidia_tensor_map import (
    encode_tma_descriptor,
    encode_tma_descriptor_for_driver,
)


def test_current_driver_bytes_encoder_is_preferred():
    calls = []

    class DriverUtils:
        fill_tma_descriptor = object()

        @staticmethod
        def encode_tma_descriptor(*args):
            calls.append(args)
            return bytes(range(128))

    arguments = (0x1000, 3, 1, 0, (8, 128), (16, 256), (256, 1), 0)
    assert encode_tma_descriptor_for_driver(DriverUtils(), *arguments) == bytes(range(128))
    assert calls == [arguments]


def test_legacy_object_driver_uses_official_cuda_encoder(monkeypatch):
    calls = []

    class DriverUtils:
        fill_tma_descriptor = object()

    def fake_encoder(*args):
        calls.append(args)
        return b"x" * 128

    monkeypatch.setattr(
        "triton.flagmega.runtime.nvidia_tensor_map.encode_tma_descriptor",
        fake_encoder,
    )
    arguments = (0x1000, 3, 1, 0, (8, 128), (16, 256), (256, 1), 0)
    assert encode_tma_descriptor_for_driver(DriverUtils(), *arguments) == b"x" * 128
    assert calls == [arguments]


def test_unsupported_driver_does_not_silently_change_descriptor_kind():
    with pytest.raises(RuntimeContractError, match="does not support tensor maps"):
        encode_tma_descriptor_for_driver(
            object(), 0x1000, 0, 1, 0, (8, 16), (8, 16), (16, 1), 0
        )


def test_cuda_encoder_matches_triton_dimension_and_stride_abi(monkeypatch):
    observed = {}

    def fake_encode(
        output,
        dtype,
        rank,
        address,
        shape,
        strides,
        block_shape,
        element_strides,
        interleave,
        swizzle,
        l2_promotion,
        padding,
    ):
        rank_value = int(rank)
        observed.update(
            dtype=int(dtype),
            rank=rank_value,
            address=int(address),
            shape=tuple(shape[index] for index in range(rank_value)),
            strides=tuple(strides[index] for index in range(rank_value)),
            block_shape=tuple(block_shape[index] for index in range(rank_value)),
            element_strides=tuple(
                element_strides[index] for index in range(rank_value)
            ),
            interleave=int(interleave),
            swizzle=int(swizzle),
            l2_promotion=int(l2_promotion),
            padding=int(padding),
        )
        ctypes.memset(output, 0xA5, 128)
        return 0

    monkeypatch.setattr(
        "triton.flagmega.runtime.nvidia_tensor_map._cuda_encode_tiled",
        lambda: fake_encode,
    )
    payload = encode_tma_descriptor(
        0x2000,
        3,
        2,
        6,
        (4, 8, 32),
        (7, 11, 64),
        (704, 64, 1),
        1,
    )

    assert payload == b"\xA5" * 128
    assert observed == {
        "dtype": 6,
        "rank": 3,
        "address": 0x2000,
        "shape": (64, 11, 7),
        "strides": (128, 1408, 9856),
        "block_shape": (32, 8, 4),
        "element_strides": (1, 1, 1),
        "interleave": 0,
        "swizzle": 3,
        "l2_promotion": 2,
        "padding": 1,
    }


def test_cuda_encoder_reports_driver_failure(monkeypatch):
    monkeypatch.setattr(
        "triton.flagmega.runtime.nvidia_tensor_map._cuda_encode_tiled",
        lambda: lambda *_args: 1,
    )
    monkeypatch.setattr(
        "triton.flagmega.runtime.nvidia_tensor_map._cuda_error_text",
        lambda code: f"CUDA error {code}",
    )

    with pytest.raises(RuntimeContractError, match="CUDA error 1"):
        encode_tma_descriptor(
            0x1000, 0, 2, 6, (8, 8), (16, 16), (16, 1), 0
        )
