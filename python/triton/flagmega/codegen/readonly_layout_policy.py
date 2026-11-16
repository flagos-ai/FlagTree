# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit opt-in policies for readonly runtime byte layouts."""

from dataclasses import dataclass

from triton.flagmega.errors import CodegenError


@dataclass(frozen=True)
class ReadonlyLayoutPolicy:
    """Select a proved device layout without changing canonical artifact bytes."""

    kind: str = "owner_major_k_walk"
    families: tuple[str, ...] = ("dense_matmul_glu",)

    def __post_init__(self):
        if (self.kind != "owner_major_k_walk" or type(self.families) is not tuple
                or self.families != ("dense_matmul_glu",)):
            raise CodegenError("Readonly layout policy requires the supported GLU owner-major K walk.")

    def to_data(self):
        return {"schema": "flagmega.readonly-layout-policy/v1", "kind": self.kind,
                "families": list(self.families)}


def validate_option(policy, *, target, codegen_backend, emit_executable):
    if (not isinstance(policy, ReadonlyLayoutPolicy) or target != "nvidia-sm90"
            or codegen_backend != "triton" or not emit_executable):
        raise CodegenError("Readonly layout policy requires executable NVIDIA SM90 Triton emission.")


__all__ = ["ReadonlyLayoutPolicy"]
