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
"""MetaX SDK memory policies for the global side of a synchronous copy.

Names follow the installed MACA SDK intrinsics, whose semantics differ from
PTX. In particular ``cg`` loads are volatile, ``cs``/``lu`` loads carry both
nontemporal and L2 replacement metadata, and ``cv``/``ca`` are default aliases.
Stores use device metadata (``wb``), nontemporal (``cg``), nontemporal plus L2
replacement metadata (``cs``), or the default behavior (``wt``). These hints do
not promise cache residency or supply synchronization.
"""
from dataclasses import dataclass

from triton.language.core import _unwrap_if_constexpr

from ._policy_base import LOAD, NativeCachePolicy, POINTER, STORE


@dataclass(frozen=True)
class CachePolicy(NativeCachePolicy):
    backend = "metax"
    kind = "memory"
    paths = frozenset({POINTER})
    directions = frozenset({LOAD, STORE})
    modifiers = {
        LOAD: ("ca", "cg", "cs", "lu", "cv"),
        STORE: ("wb", "cg", "cs", "wt"),
    }

    #: Required. Unlike HCU and PPU, MetaX has no encoding for "native
    #: default": the MACA lowering emits an SDK helper named after the
    #: modifier, so an empty value has nothing to lower to.
    cache_modifier: str

    def __post_init__(self):
        modifier = _unwrap_if_constexpr(self.cache_modifier)
        if not isinstance(modifier, str):
            raise ValueError("MetaX cache_modifier must be a compile-time string")
        modifier = modifier.removeprefix(".")
        if modifier not in ("ca", "cg", "cs", "lu", "cv", "wb", "wt"):
            raise ValueError(f"Unsupported MetaX cache modifier {modifier!r}")
        object.__setattr__(self, "cache_modifier", modifier)

    def native_fields(self, is_load):
        return {}


__all__ = ["CachePolicy"]
