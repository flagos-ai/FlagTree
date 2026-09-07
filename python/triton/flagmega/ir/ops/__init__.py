# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Built-in FlagMega op definitions.

Definitions are imported explicitly so registration is deterministic and
reviewable. This intentionally avoids source generation, an IoC container,
entry points and filesystem scanning.
"""

from __future__ import annotations


_REGISTERED = False


def ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    # Set the guard before importing children because decorators call back into
    # the registry while this package is being initialized.
    _REGISTERED = True
    from triton.flagmega.ir.ops import builtin as _builtin
    from triton.flagmega.ir.ops import distributed as _distributed
    from triton.flagmega.ir.ops import math as _math
    from triton.flagmega.ir.ops import nn as _nn
    from triton.flagmega.ir.ops import ntt as _ntt
    from triton.flagmega.ir.ops import tensors as _tensors
    from triton.flagmega.ir.ops import tir as _tir

    # Keep explicit references so static analyzers see these as intentional
    # side-effect imports rather than removable discovery machinery.
    assert _builtin and _distributed and _math and _nn and _ntt and _tensors and _tir


ensure_registered()


__all__ = ["ensure_registered"]
