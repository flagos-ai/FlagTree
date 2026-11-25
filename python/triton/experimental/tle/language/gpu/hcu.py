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
"""HCU cache hints for the global-memory side of a pointer access."""

from dataclasses import dataclass

import triton.language.core as tl

from ._policy_base import LOAD, NativeCachePolicy, POINTER, STORE

_LOAD_MODIFIERS = ("", ".ca", ".cg", ".cs", ".cv")
_STORE_MODIFIERS = ("", ".wb", ".cg", ".cs")


@dataclass(frozen=True)
class CachePolicy(NativeCachePolicy):
    """Per-access HCU global-memory cache modifier.

    Loads accept ``.ca``, ``.cg``, ``.cs``, and ``.cv``; stores accept
    ``.wb``, ``.cg``, and ``.cs``. The empty string selects the default.
    Load ``.cg``/``.cs`` and store ``.cs`` select LLVM non-temporal memory
    operations. Load ``.cv`` additionally selects a volatile operation.
    Load ``.ca`` and store ``.wb``/``.cg`` are aliases for default caching.

    These modifiers use the HCU target's cache mapping. On gfx936 and gfx938, load
    ``.cg``/``.cs`` constrain the address to vector registers to preserve
    cache controls when LLVM would otherwise scalarize a uniform load.
    This can increase address register use or require register moves.
    They do not promise NVIDIA cache-level semantics. Descriptor copies,
    eviction priorities, and store ``.wt`` are unsupported. Copy
    synchronization is unchanged.
    """

    backend = "hcu"
    kind = "memory"
    paths = frozenset({POINTER})
    directions = frozenset({LOAD, STORE})
    modifiers = {LOAD: _LOAD_MODIFIERS, STORE: _STORE_MODIFIERS}

    cache_modifier: str = ""

    def __post_init__(self):
        modifier = tl._unwrap_if_constexpr(self.cache_modifier)
        if not isinstance(modifier, str) or modifier not in _LOAD_MODIFIERS + _STORE_MODIFIERS:
            raise ValueError("HCU CachePolicy cache_modifier must be a supported compile-time string")
        object.__setattr__(self, "cache_modifier", modifier)

    def native_fields(self, is_load):
        return {}


__all__ = ["CachePolicy"]
