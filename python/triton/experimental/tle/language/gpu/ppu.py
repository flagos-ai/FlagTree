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
"""PPU cache hints for the global-memory side of TLE pointer and AIU copies."""

from dataclasses import dataclass

import triton.language.core as tl

from ._policy_base import DESCRIPTOR, LOAD, NativeCachePolicy, POINTER, STORE

_LOAD_MODIFIERS = ("", ".ca", ".cg", ".cs", ".cv", ".lu", ".nc", ".bl")
_STORE_MODIFIERS = ("", ".wb", ".cg", ".cs", ".wt", ".bl")
_KP_MODIFIERS = {1: ".nc", 2: ".cg", 4: ".cs", 5: ".lu", 6: ".bl"}
_KP7_VISIBILITY = "kp7 does not satisfy copy visibility on SDK 2.1"

#: Native AIU mode used when a PPU descriptor copy carries no explicit policy.
#: It matches the mode the SDK's default AIU load emits, so a policy-free copy
#: keeps the behavior it had before cache policies existed; it is spelled out
#: here rather than inside the lowering so the default is visible and testable.
DEFAULT_AIU_KP = 1


@dataclass(frozen=True)
class CachePolicy(NativeCachePolicy):
    """Per-copy PPU memory instruction cache modifier.

    Loads accept ``.ca``, ``.cg``, ``.cs``, ``.cv``, ``.lu``, ``.nc``,
    and ``.bl``. Stores accept ``.wb``, ``.cg``, ``.cs``, ``.wt``,
    and ``.bl``. The empty string preserves the default modifier.

    These are PPU SDK spellings, not promises of NVIDIA cache semantics.
    On SDK 2.1, ``.cg``/``.cv`` loads and ``.cg``/``.wt`` stores use the
    same native kp2 mode. ``.nc`` uses kp1, ``.cs`` kp4, ``.lu`` kp5,
    and ``.bl`` kp6. Integer aliases 1/2/4/5/6 normalize to .nc/.cg/.cs/.lu/.bl;
    stores only accept 2/4/6. Ordinary LSU kp3 is not an alias. These are
    whole-instruction modes, not separate L1/L2 controls. Their semantics depend
    on the PPU architecture. Hints do not change copy synchronization.

    Both load and store ``.ba`` (kp7) are rejected: SDK 2.1 did not satisfy
    ordinary copy visibility, including GPU-updated inputs and CPU readback.
    AIU tensor copies and NVIDIA-style eviction priorities are not supported
    by this policy. The SDK ignores the latter, so they are not exposed.
    """

    backend = "ppu"
    kind = "memory"
    paths = frozenset({POINTER})
    directions = frozenset({LOAD, STORE})
    modifiers = {LOAD: _LOAD_MODIFIERS, STORE: _STORE_MODIFIERS}
    rejected_modifiers = {".ba": _KP7_VISIBILITY}

    cache_modifier: str | int = ""

    def __post_init__(self):
        modifier = tl._unwrap_if_constexpr(self.cache_modifier)
        if type(modifier) is int:
            if modifier not in _KP_MODIFIERS:
                raise ValueError("PPU LSU kp must be one of 1, 2, 4, 5, or 6")
            modifier = _KP_MODIFIERS[modifier]
        if not isinstance(modifier, str) or modifier not in _LOAD_MODIFIERS + _STORE_MODIFIERS + (".ba", ):
            raise ValueError("PPU CachePolicy cache_modifier must be a supported compile-time string or LSU kp integer")
        object.__setattr__(self, "cache_modifier", modifier)

    @property
    def _label(self):
        return "PPU CachePolicy"

    def native_fields(self, is_load):
        return {}


@dataclass(frozen=True)
class AIUCachePolicy(NativeCachePolicy):
    """PPU0010 AIU cache mode for supported descriptor and block-pointer loads.

    The existing swizzled AIU load supplies zero padding and acp commit/wait.
    Tiles require 16..2048 rows and at least 32 contiguous bytes per row;
    elements must be 8, 16, or 32 bits. Tensor dimensions must fit 23 bits
    and nonnegative X/Z coordinates must fit the SDK's 24-bit encoding.
    Rank-2 host/device descriptors use packed row-major storage. Rank-3
    descriptors use native 2D AIU slices with packed innermost dimensions.
    Shapes must match the shared tile; only zero padding is supported.
    The original AIU source alignment requirements apply. Ordinary pointer
    tensors, stores, explicit TMA barriers and PPU0015 are rejected.

    kp1..kp6 are supported native modes; on SDK 2.1 kp3 lowers to kp2.
    Their numeric values are not NVIDIA eviction priorities. Native kp7
    is rejected because GPU-produced input can be stale on SDK 2.1;
    ordinary .ba stores also remain unsupported.

    This is a tensor-copy mode, not an LSU modifier, so it has no
    ``cache_modifier`` field.
    """

    backend = "ppu"
    kind = "aiu"
    copy_only = True
    paths = frozenset({POINTER, DESCRIPTOR})
    directions = frozenset({LOAD})

    kp: int = DEFAULT_AIU_KP

    def __post_init__(self):
        kp = tl._unwrap_if_constexpr(self.kp)
        if type(kp) is not int or kp not in range(1, 8):
            raise ValueError("PPU AIUCachePolicy kp must be an integer from 1 to 7")
        object.__setattr__(self, "kp", kp)

    @property
    def _label(self):
        return "PPU AIUCachePolicy"

    def check_access(self, path, direction):
        if self.kp == 7:
            raise ValueError("PPU AIU kp7 is unsupported: GPU-produced input can be stale on SDK 2.1")

    def to_ir(self, is_load):
        from triton._C.libtriton import ir
        self.validate_copy(False, is_load)
        return ir.CACHE_MODIFIER(31 + self.kp), ir.EVICTION_POLICY.NORMAL

    def native_fields(self, is_load):
        return {"kp": self.kp}


@tl.builtin
def _copy_descriptor(src, dst, shape, offsets, barrier, policy, _semantic=None):
    """Decompose a PPU descriptor copy into an asynchronous descriptor load.

    The backend retains a block pointer through descriptor expansion, then
    lowers it to native AIU with its commit/wait completion protocol.

    ``policy`` is the resolved :class:`AIUCachePolicy` or ``None``; ``None``
    selects :data:`DEFAULT_AIU_KP`.
    """
    from .core import local_ptr, _make_full_indices
    from . import types as tle

    offsets = tl._unwrap_if_constexpr(offsets)
    if not isinstance(src, tl.tensor_descriptor) or not isinstance(dst, tle.buffered_tensor):
        raise ValueError("PPU AIU descriptor copy supports global-to-shared loads only")
    if tl._unwrap_if_constexpr(barrier) is not None:
        raise ValueError("PPU AIU descriptor copy completes internally; explicit TMA barriers are unsupported")
    if isinstance(src, tle._logical_tensor_descriptor):
        raise ValueError("PPU AIU descriptor copy requires a physical power-of-two block shape")
    block = [int(tl._unwrap_if_constexpr(x)) for x in src.block_shape]
    extent = [int(tl._unwrap_if_constexpr(x)) for x in shape]
    if len(block) not in (2, 3) or extent != block or list(dst.shape) != block:
        raise ValueError(
            "PPU AIU descriptor copy requires matching rank-2 or rank-3 descriptor, copy and shared shapes")
    if src.dtype != dst.dtype:
        raise ValueError("PPU AIU descriptor and shared tile must have the same element type")
    if offsets is None or len(offsets) != len(block):
        raise ValueError("PPU AIU descriptor copy requires one offset per dimension")
    if policy is None:
        policy = AIUCachePolicy(kp=DEFAULT_AIU_KP)
    if not isinstance(policy, AIUCachePolicy):
        raise ValueError("PPU descriptor copy requires AIUCachePolicy")

    cache, evict = policy.to_ir(True)
    loaded = _semantic.descriptor_load(src, offsets, cache, evict)
    loaded.handle.set_attr("tt.load.async", _semantic.builder.get_bool_attr(True))
    pointers = local_ptr(dst, _make_full_indices(dst, _semantic), _semantic=_semantic)
    _semantic.store(pointers, loaded, None, (), "", "")


__all__ = ["AIUCachePolicy", "CachePolicy", "DEFAULT_AIU_KP"]
