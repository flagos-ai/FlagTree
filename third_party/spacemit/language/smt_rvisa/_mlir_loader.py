# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""Runtime resolver for the vendored MLIR Python bindings (mlir_core).

The MLIR Python bindings used by llvm_direct.py / mixed_bridge.py are vendored
into the backend directory at build time (setup.py BackendInstaller copies
${LLVM_SYSPATH}/python_packages/mlir_core), next to the other backend payload
(bin/, lib/). They are located the same way env.py locates libspert — purely
relative to this file, no environment variable:

  - installed layout:  <triton_pkg>/backends/<name>/mlir_core
    (this file lives at <triton_pkg>/language/extra/smt_rvisa/, 3 levels up
    is the triton package root; the backend name is globbed so both
    spacemit and spine_triton builds work)
  - source-tree layout: <repo_root>/backend/mlir_core
    (this file lives at <repo_root>/language/smt_rvisa/, 2 levels up)

``mlir`` is a namespace package, so the directory that *contains* it must be
on ``sys.path`` — appended (never inserted at position 0) so it cannot shadow
anything the user installed deliberately. If ``mlir`` is already importable
(e.g. an explicit PYTHONPATH), it wins and nothing is injected.
"""
from __future__ import annotations

import importlib.util
import os

_injected: str | None = None


def _mlir_importable() -> bool:
    return importlib.util.find_spec("mlir") is not None


def _candidates() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    # installed layout: <triton_pkg>/backends/<backend>/mlir_core
    pkg_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    backends_dir = os.path.join(pkg_root, "backends")
    out = [os.path.join(backends_dir, name, "mlir_core")
           for name in sorted(os.listdir(backends_dir))] \
        if os.path.isdir(backends_dir) else []
    # source-tree layout: <repo_root>/backend/mlir_core
    out.append(os.path.abspath(os.path.join(here, "..", "..", "backend", "mlir_core")))
    return out


def ensure_mlir_on_path() -> str | None:
    """Make the vendored MLIR bindings importable; return the injected dir.

    Returns None when nothing needed to be done (``mlir`` already importable)
    or when no vendored copy could be located — importing ``mlir`` afterwards
    raises the caller's ImportError with actionable text either way.
    """
    global _injected
    if _mlir_importable():
        return _injected
    for cand in _candidates():
        if os.path.isdir(os.path.join(cand, "mlir")):
            import sys
            if cand not in sys.path:
                sys.path.append(cand)
            _injected = cand
            return cand
    return None
