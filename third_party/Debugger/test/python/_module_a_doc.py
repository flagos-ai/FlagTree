# SPDX-License-Identifier: MIT
"""Load module A checklist for test module docstrings (path-safe; no package import)."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def extend_doc(short_doc: str | None) -> str:
    base = short_doc or ""
    p = Path(__file__).resolve().parent / "module_a_contract_checklist.py"
    spec = importlib.util.spec_from_file_location("_mac_list", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return base + "\n\n" + mod.MODULE_A_CHECKLIST
