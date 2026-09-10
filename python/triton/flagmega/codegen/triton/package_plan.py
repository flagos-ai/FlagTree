# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Serialized target-owned package-plan contract."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

from triton.flagmega.errors import CodegenError


PACKAGE_PLAN_SCHEMA = "flagmega.triton-package-plan/v1"


def require_package_plan(module, kind: str) -> Mapping[str, object]:
    plan = module.metadata.get("codegen_package_plan")
    if not isinstance(plan, Mapping):
        raise CodegenError(
            "Selected TIR has no target-owned codegen package plan.",
            stage=module.stage,
        )
    if plan.get("schema") != PACKAGE_PLAN_SCHEMA or plan.get("kind") != kind:
        raise CodegenError(
            f"Codegen package plan is {plan.get('schema')!r}/{plan.get('kind')!r}; "
            f"expected {PACKAGE_PLAN_SCHEMA!r}/{kind!r}.",
            stage=module.stage,
        )
    return plan


def plain_package_value(value):
    """Convert frozen checkpoint values into JSON-safe descriptor data."""

    from triton.flagmega.ir.fusion import Fusion
    if isinstance(value, Fusion):
        return {"$fusion": plain_package_value(value.to_data())}
    if isinstance(value, Mapping):
        return {
            str(key): plain_package_value(item) for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [plain_package_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [plain_package_value(item) for item in sorted(value, key=str)]
    to_data = getattr(value, "to_data", None)
    if callable(to_data):
        return plain_package_value(to_data())
    if isinstance(value, Enum):
        return plain_package_value(value.value)
    return value


__all__ = ["PACKAGE_PLAN_SCHEMA", "plain_package_value", "require_package_plan"]
