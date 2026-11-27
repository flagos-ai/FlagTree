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
"""Implementation of the NVIDIA cache hints exported by ``tle.gpu``.

Cache operators and single-level eviction priorities are distinct controls.
They are performance hints, not memory ordering or residency guarantees.
"""
from dataclasses import dataclass

from triton.language.core import _unwrap_if_constexpr, constexpr_type

from ._policy_base import DESCRIPTOR, LOAD, NativeCachePolicy, POINTER, STORE

_PRIORITIES = {"evict_normal", "evict_first", "evict_last", "evict_unchanged"}


@dataclass(frozen=True)
class FractionalEvictionPolicy:
    """Select a static L2 priority (fraction=1, secondary=evict_unchanged).

    Non-unit fractions are deliberately outside this API. The fraction never
    denotes a share of L2 capacity.
    """

    primary: str
    fraction: float = 1.0
    secondary: str = "evict_unchanged"

    def __post_init__(self):
        for field in ("primary", "fraction", "secondary"):
            object.__setattr__(self, field, _unwrap_if_constexpr(getattr(self, field)))
        if not isinstance(self.primary, str) or self.primary not in _PRIORITIES:
            raise ValueError("Unsupported NVIDIA L2 primary eviction priority")
        if (type(self.fraction) not in (int, float) or self.fraction != 1 or self.secondary != "evict_unchanged"):
            raise ValueError("Only fraction=1 and secondary=evict_unchanged are supported")
        object.__setattr__(self, "fraction", float(self.fraction))

    @property
    def type(self):
        return constexpr_type(self)


@dataclass(frozen=True)
class CachePolicy(NativeCachePolicy):
    """NVIDIA policy for ``tle.gpu.copy`` and ordinary global load/store.

    Pointer accesses support the cache operators for their direction, L1
    eviction or full-probability L2 eviction. Descriptor
    TMA copies support only L2 eviction. Unsupported combinations are errors.
    Independent L1/L2 combinations and L2 prefetch are rejected.
    ``None`` fields leave the corresponding instruction control unspecified.
    """

    backend = "nvidia"
    kind = "memory"
    paths = frozenset({POINTER, DESCRIPTOR})
    directions = frozenset({LOAD, STORE})
    modifiers = {
        LOAD: ("ca", "cg", "cs", "lu", "cv"),
        STORE: ("wb", "cg", "cs", "wt"),
    }

    cache_modifier: str | None = None
    l1: str | None = None
    l2: FractionalEvictionPolicy | None = None
    l2_prefetch_size: int | None = None

    def __post_init__(self):
        for field in ("cache_modifier", "l1", "l2", "l2_prefetch_size"):
            object.__setattr__(self, field, _unwrap_if_constexpr(getattr(self, field)))
        modifier = self.cache_modifier
        if modifier is not None:
            if not isinstance(modifier, str):
                raise ValueError("NVIDIA cache_modifier must be a compile-time string")
            modifier = modifier.removeprefix(".")
            if modifier not in ("ca", "cg", "cs", "lu", "cv", "wb", "wt"):
                raise ValueError(f"Unsupported NVIDIA cache modifier {modifier!r}")
            object.__setattr__(self, "cache_modifier", modifier)
        if self.l1 is not None and (not isinstance(self.l1, str) or self.l1 not in _PRIORITIES | {"no_allocate"}):
            raise ValueError("Unsupported NVIDIA L1 eviction priority")
        if self.l2 is not None and not isinstance(self.l2, FractionalEvictionPolicy):
            raise TypeError("NVIDIA l2 must be a compile-time FractionalEvictionPolicy")
        if self.l2_prefetch_size is not None:
            raise ValueError("NVIDIA L2 prefetch is not supported by this cache policy API")
        if self.l1 is not None and self.l2 is not None:
            raise ValueError("Independent L1 and L2 eviction policies cannot be combined")
        if self.cache_modifier is not None and self.l1 is not None:
            raise ValueError("NVIDIA cache_modifier and explicit L1 eviction cannot be combined")
        if all(value is None for value in (self.cache_modifier, self.l1, self.l2, self.l2_prefetch_size)):
            raise ValueError("CachePolicy needs at least one explicit hint; use None for the default")

    def check_access(self, path, direction):
        if path == DESCRIPTOR:
            if self.cache_modifier is not None or self.l1 is not None:
                raise ValueError("NVIDIA TMA copy supports only L2 eviction policy")

    def to_ir(self, is_load):
        from triton._C.libtriton import ir
        cache, evict = super().to_ir(is_load)
        if self.l1 is not None:
            evict = getattr(ir.EVICTION_POLICY, "L1_" + self.l1.upper())
        elif self.l2 is not None:
            evict = getattr(ir.EVICTION_POLICY, "L2_" + self.l2.primary.upper())
        return cache, evict

    def native_fields(self, is_load):
        fields = {}
        if self.l1 is not None:
            fields["l1"] = self.l1
        if isinstance(self.l2, FractionalEvictionPolicy):
            fields.update(l2_primary=self.l2.primary, l2_secondary=self.l2.secondary, l2_fraction=self.l2.fraction)
        return fields


__all__ = ["CachePolicy", "FractionalEvictionPolicy"]
