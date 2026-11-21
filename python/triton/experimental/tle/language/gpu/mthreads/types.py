# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""mthreads-specific TLE types."""

from dataclasses import dataclass

import triton.language.core as tl

from .._policy_base import DESCRIPTOR, LOAD, NativeCachePolicy, POINTER, STORE
from ..types import distributed_encoding

_MUSA_PH1_VERSION = [3, 1]


def _normalize_static_int_sequence(values, name, *, require_positive=False):
    values = tl._unwrap_if_constexpr(values)
    if isinstance(values, tl.tuple):
        values = tuple(values.values)
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"mthreads TLE {name} must be a static sequence")

    normalized = []
    for index, value in enumerate(values):
        value = tl._unwrap_if_constexpr(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"mthreads TLE {name}[{index}] must be a compile-time integer")
        if require_positive and value <= 0:
            raise ValueError(f"mthreads TLE {name}[{index}] must be positive")
        normalized.append(value)
    return normalized


def _is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


_TME_CACHE_POLICIES = ("none", "once", "normal", "persist")


@dataclass(frozen=True)
class TMECachePolicy(NativeCachePolicy):
    """Per-copy MThreads TME inner/outer cache persistence hints.

    Each field accepts ``"none"``, ``"once"``, ``"normal"``, or ``"persist"``,
    or their integer aliases 0, 1, 2, and 3 respectively.
    ``inner`` additionally accepts native SDK encodings 4 and 5 on mp_31;
    their replacement semantics are not assigned portable eviction names.
    ``prefetch_size`` accepts 0, 64, 128, or 256 bytes for descriptor loads.
    The default preserves normal persistence in both domains. Non-default
    hints are performance-sensitive and must be measured on the target kernel.
    This immutable value can be used as a ``tl.constexpr`` kernel or worker
    argument. It does not change copy completion or synchronization.
    """

    backend = "mthreads"
    kind = "tme"
    paths = frozenset({DESCRIPTOR})
    directions = frozenset({LOAD, STORE})

    inner: str | int = "normal"
    outer: str | int = "normal"
    prefetch_size: int = 0

    def __post_init__(self):
        for name in ("inner", "outer"):
            value = tl._unwrap_if_constexpr(getattr(self, name))
            if type(value) is int and value in range(4):
                value = _TME_CACHE_POLICIES[value]
            native_inner = name == "inner" and type(value) is int and value in (4, 5)
            named_policy = isinstance(value, str) and value in _TME_CACHE_POLICIES
            if not native_inner and not named_policy:
                suffix = "; inner also accepts native integers 4 or 5" if name == "inner" else ""
                raise ValueError(f"TMECachePolicy {name} must be none, once, normal, persist, or integer 0..3{suffix}")
            object.__setattr__(self, name, value)
        prefetch = tl._unwrap_if_constexpr(self.prefetch_size)
        if type(prefetch) is not int or prefetch not in (0, 64, 128, 256):
            raise ValueError("TMECachePolicy prefetch_size must be 0, 64, 128, or 256 bytes")
        object.__setattr__(self, "prefetch_size", prefetch)

    def check_access(self, path, direction):
        if direction == STORE and self.prefetch_size:
            raise ValueError("TMECachePolicy prefetch_size is only supported for descriptor loads")

    @property
    def inner_code(self):
        """SDK encoding of ``inner`` (names map to 0..3, native ints pass through)."""
        return self.inner if type(self.inner) is int else _TME_CACHE_POLICIES.index(self.inner)

    @property
    def outer_code(self):
        return _TME_CACHE_POLICIES.index(self.outer)

    def native_fields(self, is_load):
        return {"inner": self.inner_code, "outer": self.outer_code, "prefetch_size": self.prefetch_size}


@dataclass(frozen=True)
class LSUCachePolicy(NativeCachePolicy):
    """Native MUSA cache controls for pointer-based global/shared copies.

    Persistence fields use SDK integers 0..5 and ``l2_policy`` uses 0 or 1.
    These are MUSA encodings, not portable L1/L2 eviction names. ``None``
    selects the SDK default for the copy direction: inner/outer 4/2, L2 0
    for loads and 1 for stores. The SDK coherence control is kept at its
    direction-specific default and is not exposed as a performance hint.
    Ordinary loads and stores use the native controls on mp_31 and the
    default memory instruction on other MUSA targets. Explicit LSU copies
    with native controls require mp_31. Copy synchronization is unchanged.
    """

    backend = "mthreads"
    kind = "lsu"
    paths = frozenset({POINTER})
    directions = frozenset({LOAD, STORE})

    inner_persistence: int | None = None
    outer_persistence: int | None = None
    l2_policy: int | None = None

    def __post_init__(self):
        for name, maximum in (("inner_persistence", 5), ("outer_persistence", 5), ("l2_policy", 1)):
            value = tl._unwrap_if_constexpr(getattr(self, name))
            if value is not None and (type(value) is not int or not 0 <= value <= maximum):
                raise ValueError(f"LSUCachePolicy {name} must be None or a compile-time integer in [0, {maximum}]")
            object.__setattr__(self, name, value)

    def to_ir(self, is_load):
        from triton._C.libtriton import ir
        fields = self.native_fields(is_load)
        # Matches the named musa_iN_oN_lN cases in NativeCacheModifiers.td.
        value = (128 + fields["inner_persistence"] * 12 + fields["outer_persistence"] * 2 + fields["l2_policy"])
        return ir.CACHE_MODIFIER(value), ir.EVICTION_POLICY.NORMAL

    def native_fields(self, is_load):
        # ``None`` resolves to the SDK default for the direction here, so the
        # compiler only ever sees concrete encodings.
        return {
            "inner_persistence": 4 if self.inner_persistence is None else self.inner_persistence,
            "outer_persistence": 2 if self.outer_persistence is None else self.outer_persistence,
            "l2_policy": (0 if is_load else 1) if self.l2_policy is None else self.l2_policy,
        }


class _MusaMmaEncoding(distributed_encoding):
    """Shared Python contract for PH1 MUSA matrix instruction layouts."""

    _builder_method = None

    def __init__(self, version, warps_per_cta, instr_shape, cga_layout=None):
        super().__init__()
        prefix = self.__class__.__name__
        self.version = _normalize_static_int_sequence(version, f"{prefix} version", require_positive=True)
        self.warps_per_cta = _normalize_static_int_sequence(
            warps_per_cta,
            f"{prefix} warps_per_cta",
            require_positive=True,
        )
        self.instr_shape = _normalize_static_int_sequence(
            instr_shape,
            f"{prefix} instr_shape",
            require_positive=True,
        )

        if len(self.version) != 2:
            raise ValueError(f"{prefix} version must contain major and minor")
        if self.version != _MUSA_PH1_VERSION:
            raise ValueError(f"{prefix} currently supports only MUSA PH1 version [3, 1], got {self.version}")
        if len(self.warps_per_cta) not in (2, 3):
            raise ValueError(f"{prefix} warps_per_cta rank must be 2 or 3")
        if any(not _is_power_of_two(value) for value in self.warps_per_cta):
            raise ValueError(f"{prefix} warps_per_cta entries must be positive powers of two")
        if len(self.warps_per_cta) == 3 and self.warps_per_cta[2] != 1:
            raise ValueError(f"{prefix} rank-3 warps_per_cta must be [warps_m, warps_n, 1]")
        if len(self.instr_shape) != 3:
            raise ValueError(f"{prefix} instr_shape must contain logical (M, N, K)")

        self.cga_layout = [] if cga_layout is None else self._normalize_cga_layout(cga_layout)
        self._validate_instruction_shape()

    def _normalize_cga_layout(self, cga_layout):
        prefix = self.__class__.__name__
        cga_layout = tl._unwrap_if_constexpr(cga_layout)
        if isinstance(cga_layout, tl.tuple):
            cga_layout = tuple(cga_layout.values)
        if not isinstance(cga_layout, (list, tuple)):
            raise ValueError(f"{prefix} cga_layout must be a static sequence of basis vectors")

        normalized = []
        for index, basis in enumerate(cga_layout):
            basis = _normalize_static_int_sequence(basis, f"{prefix} cga_layout[{index}]")
            if len(basis) != self.rank:
                raise ValueError(f"{prefix} cga_layout[{index}] rank must match layout rank {self.rank}")
            if any(value < 0 for value in basis):
                raise ValueError(f"{prefix} cga_layout[{index}] entries must be non-negative")
            normalized.append(basis)
        return normalized

    @property
    def rank(self):
        return len(self.warps_per_cta)

    def _validate_instruction_shape(self):
        raise NotImplementedError

    def to_ir(self, builder):
        method = getattr(builder, self._builder_method)
        return method(self.version, self.warps_per_cta, self.cga_layout, self.instr_shape)

    def __repr__(self):
        return (f"{self.__class__.__name__}(version={self.version}, warps_per_cta={self.warps_per_cta}, "
                f"instr_shape={self.instr_shape}, cga_layout={self.cga_layout})")

    def __eq__(self, other):
        return (type(self) is type(other) and self.version == other.version
                and self.warps_per_cta == other.warps_per_cta and self.instr_shape == other.instr_shape
                and self.cga_layout == other.cga_layout)

    def __hash__(self):
        return hash((type(self), tuple(self.version), tuple(self.warps_per_cta), tuple(self.instr_shape),
                     tuple(tuple(basis) for basis in self.cga_layout)))


class MusaWmmaEncoding(_MusaMmaEncoding):
    """Explicit ``#ttg.musa_wmma`` encoding for a WMMA result or accumulator."""

    _builder_method = "get_musa_wmma_layout"

    def _validate_instruction_shape(self):
        m, n, k = self.instr_shape
        if m % 8 != 0:
            raise ValueError("MusaWmmaEncoding instr_shape M must be a non-zero multiple of 8")
        if n % 8 != 0:
            raise ValueError("MusaWmmaEncoding instr_shape N must be a non-zero multiple of 8")
        if k != 4 and k % 8 != 0:
            raise ValueError("MusaWmmaEncoding instr_shape K must be 4 or a non-zero multiple of 8")


class MusaSqmmaEncoding(_MusaMmaEncoding):
    """Explicit ``#ttg.musa_sqmma`` encoding for an SQMMA result or accumulator."""

    _builder_method = "get_musa_sqmma_layout"

    def __init__(self, version, warps_per_cta, instr_shape, cga_layout=None):
        super().__init__(version, warps_per_cta, instr_shape, cga_layout)
        if self.warps_per_cta[0] % 4 != 0:
            raise ValueError("MusaSqmmaEncoding warps_per_cta[0] must be a multiple of 4")

    def _validate_instruction_shape(self):
        m, n, k = self.instr_shape
        if m < 16 or m % 8 != 0:
            raise ValueError("MusaSqmmaEncoding instr_shape M must be at least 16 and a multiple of 8")
        if n % 8 != 0:
            raise ValueError("MusaSqmmaEncoding instr_shape N must be a non-zero multiple of 8")
        if k % 8 != 0:
            raise ValueError("MusaSqmmaEncoding instr_shape K must be a non-zero multiple of 8")


class MusaDotOperandEncoding(distributed_encoding):
    """Explicit ``#ttg.dot_op`` encoding for a MUSA WMMA register operand.

    PH1 SQMMA operands are consumed from shared memory and are staged by the
    backend.  ``MusaSqmmaEncoding`` is therefore only valid for an SQMMA
    accumulator/result, not as the parent of this register encoding.
    """

    def __init__(self, operand_index, parent, k_width=0):
        super().__init__()
        self.operand_index = tl._unwrap_if_constexpr(operand_index)
        self.parent = tl._unwrap_if_constexpr(parent)
        self.k_width = tl._unwrap_if_constexpr(k_width)

        if isinstance(self.operand_index, bool) or not isinstance(self.operand_index, int):
            raise ValueError("MusaDotOperandEncoding operand_index must be a compile-time integer")
        if self.operand_index not in (0, 1):
            raise ValueError("MusaDotOperandEncoding operand_index must be 0 or 1")
        if isinstance(self.parent, MusaSqmmaEncoding):
            raise ValueError("MusaDotOperandEncoding does not support MusaSqmmaEncoding: "
                             "PH1 SQMMA operands are staged through shared memory by the backend")
        if not isinstance(self.parent, MusaWmmaEncoding):
            raise ValueError("MusaDotOperandEncoding parent must be a MusaWmmaEncoding")
        if isinstance(self.k_width, bool) or not isinstance(self.k_width, int):
            raise ValueError("MusaDotOperandEncoding k_width must be a compile-time integer")
        if self.k_width != 0:
            raise ValueError("MusaDotOperandEncoding requires k_width=0")

    @property
    def rank(self):
        return self.parent.rank

    def to_ir(self, builder):
        return builder.get_dot_operand_layout(self.operand_index, self.parent.to_ir(builder), self.k_width)

    def __repr__(self):
        return (f"MusaDotOperandEncoding(operand_index={self.operand_index}, parent={self.parent!r}, "
                f"k_width={self.k_width})")

    def __eq__(self, other):
        return (type(self) is type(other) and self.operand_index == other.operand_index and self.parent == other.parent
                and self.k_width == other.k_width)

    def __hash__(self):
        return hash((self.operand_index, self.parent, self.k_width))


__all__ = ["MusaWmmaEncoding", "MusaSqmmaEncoding", "MusaDotOperandEncoding"]
