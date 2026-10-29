# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""MetaX tensor layouts for ``tle.gpu.set_layout``."""

import triton.language.core as tl
from triton.experimental.tle.language.gpu.types import distributed_encoding

__all__ = ["MmaEncoding", "DotOperandEncoding"]


def _int_tuple(values, name, length):
    values = tl._unwrap_if_constexpr(values)
    if isinstance(values, tl.tuple):
        values = values.values
    values = tuple(tl._unwrap_if_constexpr(value) for value in values)
    if len(values) != length or any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"{name} must contain {length} non-negative integers")
    return values


class MmaEncoding(distributed_encoding):
    """Single-CTA ``#ttg.maca_mma`` layout for a dot result or accumulator.

    ``elements_mnk`` specifies per-thread elements along M, N and K.
    The current CommonIR path supports MMA v2, unit element strides and
    no LDS transpose. Tensor-shape and target constraints are checked during lowering.
    """

    def __init__(self, version, warps_per_cta, elements_mnk, col_major=False):
        self.version = _int_tuple(version, "version", 2)
        self.warps_per_cta = _int_tuple(warps_per_cta, "warps_per_cta", 2)
        self.elements_mnk = _int_tuple(elements_mnk, "elements_mnk", 3)
        col_major = tl._unwrap_if_constexpr(col_major)
        if self.version[0] != 2:
            raise ValueError("MetaX MmaEncoding currently requires MMA v2")
        for name in ("warps_per_cta", "elements_mnk"):
            if any(value == 0 or value & (value - 1) for value in getattr(self, name)):
                raise ValueError(f"{name} must contain positive powers of two")
        if type(col_major) not in (bool, int) or col_major not in (0, 1):
            raise ValueError("col_major must be a boolean or 0/1")
        self.col_major = bool(col_major)

    @property
    def rank(self):
        return 2

    def to_ir(self, builder):
        return builder.get_metax_mma_layout(
            self.version, self.warps_per_cta, self.elements_mnk, self.col_major)

    def __repr__(self):
        return (f"metax.MmaEncoding(version={self.version}, warps_per_cta={self.warps_per_cta}, "
                f"elements_mnk={self.elements_mnk}, col_major={self.col_major})")

    def __eq__(self, other):
        return (type(self) is type(other) and self.version == other.version
                and self.warps_per_cta == other.warps_per_cta and self.elements_mnk == other.elements_mnk
                and self.col_major == other.col_major)

    def __hash__(self):
        return hash((self.version, self.warps_per_cta, self.elements_mnk, self.col_major))


class DotOperandEncoding(distributed_encoding):
    """MetaX ``#ttg.dot_op`` layout: operand 0 is A, operand 1 is B.

    ``parent`` is a MetaX MmaEncoding; its elements_mnk determines packing,
    so the current CommonIR path requires ``k_width=0``.
    """

    def __init__(self, operand_index, parent, k_width=0):
        self.operand_index = tl._unwrap_if_constexpr(operand_index)
        self.parent = tl._unwrap_if_constexpr(parent)
        self.k_width = tl._unwrap_if_constexpr(k_width)
        if type(self.operand_index) is not int or self.operand_index not in (0, 1):
            raise ValueError("operand_index must be 0 or 1")
        if not isinstance(self.parent, MmaEncoding):
            raise ValueError("parent must be a MetaX MmaEncoding")
        if type(self.k_width) is not int or self.k_width != 0:
            raise ValueError("MetaX DotOperandEncoding currently requires k_width=0")

    @property
    def rank(self):
        return self.parent.rank

    def to_ir(self, builder):
        return builder.get_dot_operand_layout(self.operand_index, self.parent.to_ir(builder), self.k_width)

    def __repr__(self):
        return (f"metax.DotOperandEncoding(operand_index={self.operand_index}, "
                f"parent={self.parent!r}, k_width={self.k_width})")

    def __eq__(self, other):
        return (type(self) is type(other) and self.operand_index == other.operand_index
                and self.parent == other.parent and self.k_width == other.k_width)

    def __hash__(self):
        return hash((self.operand_index, self.parent, self.k_width))
