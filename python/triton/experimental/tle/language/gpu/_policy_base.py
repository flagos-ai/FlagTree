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
"""Shared contract for native per-access cache policies.

Every backend policy declares which instruction family it belongs to and which
copy paths and directions it supports. The base class turns those declarations
into uniform validation, existing cache/evict enum controls and constexpr
specialization, so a backend only has to describe its own native fields.

A policy never implies portable semantics across backends and never changes
synchronization or visibility.
"""

from triton.language.core import _unwrap_if_constexpr, constexpr_type

#: Copy paths a policy can apply to.
POINTER = "pointer"
DESCRIPTOR = "descriptor"

#: Access directions a policy can apply to.
LOAD = "load"
STORE = "store"


class NativeCachePolicy:
    """Base class for the per-access policies exported under ``tle.gpu``.

    Subclasses are frozen dataclasses. They declare:

    ``backend``
        Compilation target that accepts this policy.
    ``kind``
        Native instruction family, used for diagnostics; the formal cache/evict
        values select the lowering.
    ``paths`` / ``directions``
        Which copy forms accept the policy at all.
    ``copy_only``
        True for tensor-copy modes that have no meaning on an ordinary
        ``tl.load``/``tl.store`` even though ``tle.gpu.copy`` lowers them
        through the same pointer path.
    ``modifiers``
        Optional per-direction tuple of accepted cache modifiers. When present,
        the base class validates ``self.cache_modifier`` against it.

    Subclasses implement :meth:`to_ir` for native multi-field controls and may
    override :meth:`check_access` for other access constraints.

    Default construction follows one rule: an all-default policy object is
    legal exactly when it still says something that omitting ``cache_policy``
    does not, namely "this access is explicitly governed and keeps the
    validated native memory path". HCU, PPU, MThreads LSU and TME accept it
    for that reason; their empty value selects the native default while
    pinning the path. Two backends reject it, and both say so in their own
    error: NVIDIA, because a hintless NVIDIA policy selects exactly what an
    omitted ``cache_policy`` already produces, and MetaX, because its SDK
    lowering is named after the modifier and has nothing to emit for an
    empty one.
    """

    # Declarations. Subclasses override.
    backend: str = ""
    kind: str = "memory"
    paths: frozenset = frozenset({POINTER})
    directions: frozenset = frozenset({LOAD, STORE})
    copy_only: bool = False
    modifiers: dict = {}
    #: Modifier -> reason, for spellings the SDK encodes but this API refuses.
    #: Checked before :attr:`modifiers` so the diagnostic states the real
    #: reason rather than "unsupported for this direction".
    rejected_modifiers: dict = {}

    #: Human-readable noun used in path/direction diagnostics.
    @property
    def _label(self):
        return type(self).__name__

    # ---------------------------------------------------------------- fields

    def native_fields(self, is_load):
        """Return this policy's native fields for one direction.

        The mapping must contain only compile-time scalars. It must not contain
        ``backend``, ``kind``, or ``cache_modifier``; those are supplied by the
        base class.
        """
        raise NotImplementedError

    def native_modifier(self):
        """Return the simple cache modifier, or ``None`` when unused.

        The modifier is stored in the operation's existing ``cache`` enum
        rather than in the policy's own field dictionary.
        """
        modifier = getattr(self, "cache_modifier", None)
        modifier = _unwrap_if_constexpr(modifier)
        return modifier or None

    # ------------------------------------------------------------ validation

    def check_access(self, path, direction):
        """Hook for constraints beyond path/direction/modifier declarations."""

    def validate_copy(self, is_descriptor, is_load):
        """Reject a policy the target copy form cannot represent.

        Raises ``ValueError`` with a message naming the policy and the
        unsupported form, so the frontend diagnostic is uniform across
        backends.
        """
        path = DESCRIPTOR if is_descriptor else POINTER
        direction = LOAD if is_load else STORE
        if path not in self.paths:
            # Name both the rejected form and the supported ones: callers
            # diagnose against either half depending on the backend.
            supported = " and ".join(f"{name} copies" for name in sorted(self.paths))
            raise ValueError(f"{self._label} does not support {path} copies; "
                             f"it is only supported for {supported}")
        if direction not in self.directions:
            supported = " and ".join(f"{name}s" for name in sorted(self.directions))
            raise ValueError(f"{self._label} supports global-to-shared {supported} only")
        modifier = self.native_modifier()
        if modifier is not None:
            reason = self.rejected_modifiers.get(modifier)
            if reason is not None:
                raise ValueError(f"{self.backend} {modifier} {direction} is unsupported: {reason}")
            allowed = self.modifiers.get(direction)
            if allowed is not None and modifier not in allowed:
                raise ValueError(f"{self.backend} {direction} does not support cache modifier {modifier!r}")
        self.check_access(path, direction)

    # ----------------------------------------------------------- compilation

    @property
    def type(self):
        return constexpr_type(self)

    # ``tl.load``/``tl.store`` accept a policy object without importing this
    # package: they only call ``resolve_memory_access`` and ``to_ir``. TLE knows
    # about core, so the core language stays free of experimental imports.

    def resolve_memory_access(self, pointer, is_load, cache_modifier, eviction_policy, semantic):
        """Validate this policy for an ordinary ``tl.load``/``tl.store``."""
        from ._cache_policy import resolve_memory_policy
        return resolve_memory_policy(self, pointer, is_load, cache_modifier, eviction_policy, semantic)

    def to_ir(self, is_load):
        """Normalize to the existing ODS cache/evict fields before building IR."""
        from triton._C.libtriton import ir
        modifier = self.native_modifier()
        if not modifier and self.backend == "ppu":
            return ir.CACHE_MODIFIER(11), ir.EVICTION_POLICY.NORMAL
        # An explicit HCU default normalizes to the direction's default alias.
        if not modifier and self.backend == "hcu":
            modifier = "ca" if is_load else "wb"
        cache = getattr(ir.CACHE_MODIFIER, modifier.lstrip(".").upper()) if modifier else ir.CACHE_MODIFIER.NONE
        return cache, ir.EVICTION_POLICY.NORMAL

    def to_attributes(self, is_load):
        """Describe the resolved native controls (not an MLIR attribute)."""
        fields = {"backend": self.backend, "kind": self.kind}
        if self.native_modifier() is not None:
            fields["cache_modifier"] = self.native_modifier()
        fields.update(self.native_fields(is_load))
        return fields


__all__ = ["NativeCachePolicy", "POINTER", "DESCRIPTOR", "LOAD", "STORE"]
