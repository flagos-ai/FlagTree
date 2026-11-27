# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shared semantics for target-parameterized byte-preserving K-major layouts."""

from __future__ import annotations

from triton.flagmega.errors import IRSchemaError
import re


K_MAJOR_N_LANE = 8
K_MAJOR_K_LANE = 16
K_MAJOR_PAYLOAD_GROUPS = 2
K_MAJOR_PAYLOAD_WIDTH = 64

_LAYOUT = re.compile(
    r"^k_major_(?P<mesh>mesh_interleaved_)?n(?P<n>[1-9][0-9]*)_k(?P<k>[1-9][0-9]*)$"
)
_QKV_SPLIT_K_LAYOUT = re.compile(
    r"^qkv_split_k_k_major_n(?P<n>[1-9][0-9]*)_k(?P<k>[1-9][0-9]*)$"
)


def k_major_layout_name(n_lane: int, k_lane: int, *, mesh_interleaved: bool = False) -> str:
    if n_lane <= 0 or k_lane <= 0:
        raise IRSchemaError("K-major layout lanes must be positive.")
    mesh = "mesh_interleaved_" if mesh_interleaved else ""
    return f"k_major_{mesh}n{n_lane}_k{k_lane}"


def parse_k_major_layout(layout: str) -> tuple[int, int, bool]:
    match = _LAYOUT.fullmatch(str(layout))
    if match is None:
        raise IRSchemaError(f"Unsupported K-major layout {layout!r}.")
    return int(match.group("n")), int(match.group("k")), match.group("mesh") is not None


def qkv_split_k_layout_name(n_lane: int, k_lane: int) -> str:
    """Name the QKV mesh/split-K asset by its actual tensor layout.

    The layout is independent of the kernel instruction used to consume it;
    MMA/SIMT/TMA are implementation choices made later by TIR selection.
    """

    if n_lane <= 0 or k_lane <= 0:
        raise IRSchemaError("QKV split-K K-major layout lanes must be positive.")
    return f"qkv_split_k_k_major_n{n_lane}_k{k_lane}"


def parse_qkv_split_k_layout(layout: str) -> tuple[int, int]:
    match = _QKV_SPLIT_K_LAYOUT.fullmatch(str(layout))
    if match is None:
        raise IRSchemaError(f"Unsupported QKV split-K K-major layout {layout!r}.")
    return int(match.group("n")), int(match.group("k"))


def unpack_k_major_weight(packed, layout: str, logical_n: int | None = None):
    """Invert a K-major packing independent of the target's lane geometry."""

    n_lane, k_lane, mesh_interleaved = parse_k_major_layout(layout)
    if mesh_interleaved:
        if logical_n is None or logical_n <= 0 or logical_n % n_lane:
            raise IRSchemaError(
                "Mesh-interleaved K-major storage requires aligned logical N."
            )
        k_groups, local_n_groups, mesh_size, *payload = packed.shape
        if _product(payload) != n_lane * k_lane:
            raise IRSchemaError("K-major storage payload does not match its lanes.")
        logical_n_groups = logical_n // n_lane
        if logical_n_groups > local_n_groups * mesh_size:
            raise IRSchemaError("logical N exceeds mesh-interleaved physical storage.")
        packed = packed.reshape(k_groups, local_n_groups * mesh_size, n_lane, k_lane)
        packed = packed[:, :logical_n_groups]
    else:
        k_groups, n_groups, *payload = packed.shape
        if _product(payload) != n_lane * k_lane:
            raise IRSchemaError("K-major storage payload does not match its lanes.")
        packed = packed.reshape(k_groups, n_groups, n_lane, k_lane)
    return (
        packed.permute(1, 2, 0, 3)
        .reshape(packed.shape[1] * n_lane, packed.shape[0] * k_lane)
        .contiguous()
    )


def k_major_n8_k16_shape(n: int, k: int) -> tuple[int, int, int, int]:
    if n % K_MAJOR_N_LANE or k % K_MAJOR_K_LANE:
        raise IRSchemaError("k_major_n8_k16 requires N divisible by 8 and K divisible by 16.")
    return (
        k // K_MAJOR_K_LANE,
        n // K_MAJOR_N_LANE,
        K_MAJOR_PAYLOAD_GROUPS,
        K_MAJOR_PAYLOAD_WIDTH,
    )


def unpack_k_major_n8_k16_weight(packed):
    """Invert ``reshape(N/8,8,K/16,16) -> permute -> reshape``."""

    return unpack_k_major_weight(packed, "k_major_n8_k16")


def k_major_mesh_interleaved_n8_k16_shape(
    n: int,
    k: int,
    mesh_size: int,
) -> tuple[int, int, int, int, int]:
    k_groups, n_groups, payload_groups, payload_width = k_major_n8_k16_shape(n, k)
    local_n_groups = (n_groups + mesh_size - 1) // mesh_size
    return (k_groups, local_n_groups, mesh_size, payload_groups, payload_width)


def unpack_k_major_mesh_interleaved_n8_k16_weight(packed, logical_n: int):
    """Invert the padded ``[K/16,local-N/8,mesh,2,64]`` representation."""

    return unpack_k_major_weight(
        packed, "k_major_mesh_interleaved_n8_k16", logical_n
    )


def _product(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


__all__ = [
    "K_MAJOR_K_LANE",
    "K_MAJOR_N_LANE",
    "K_MAJOR_PAYLOAD_GROUPS",
    "K_MAJOR_PAYLOAD_WIDTH",
    "k_major_n8_k16_shape",
    "k_major_mesh_interleaved_n8_k16_shape",
    "k_major_layout_name",
    "parse_k_major_layout",
    "parse_qkv_split_k_layout",
    "qkv_split_k_layout_name",
    "unpack_k_major_weight",
    "unpack_k_major_n8_k16_weight",
    "unpack_k_major_mesh_interleaved_n8_k16_weight",
]
