# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""First-class bufferized orchestration function."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRNode, tir_node
from triton.flagmega.ir.tir.sequential import Sequential


@tir_node("execution_function")
@dataclass(frozen=True)
class ExecutionFunction(TIRNode):
    """Executable call order for one graph function after Bufferize.

    Graph ``Function`` remains available for semantic edit/resume, while this
    object is the target-independent, physical orchestration IR consumed by
    synchronization, transfer-pipeline lowering, and codegen.

    Ownership of ``num_warps`` and co-residency
    -------------------------------------------
    Warp count is a **TIR/Kernel** decision, not a distribution decision.  It
    selects physical execution resources inside one task -- tile/warp roles,
    the per-thread register budget and pipeline depth -- so it belongs with the
    implementation, next to tile shape.  AutoDistribution owns mesh, placement
    and participants: how many logical tasks exist and who owns which data.

    Distribution does, however, **constrain** the choice rather than make it,
    because warp count leaks into a distribution-visible contract: role splits
    inside ``warp_specialize`` and the participant count of any two-role
    rendezvous are functions of it.  Changing a participant count is an
    arrival-protocol change and must be reviewed as one.  The relation is TIR
    proposes, Distribution vetoes.

    Ordering therefore matters: a per-program warp count requires the program
    partition to be settled first, after which each program may carry its own
    count.  ``codegen/triton/kernel_programs.py`` already enforces this shape,
    validating a per-program ``num_warps`` against the target resource
    contract.

    Two resource facts that make this ownership load-bearing, both measured on
    this target rather than assumed:

    * Register pressure is a property of the *program*, not of the operators.
      A whole-model megakernel measured 153 registers/thread (58,752 of 65,536
      per SM, 89.6%) and was pinned to one CTA per SM, while a single-operator
      component of the same model measured 74 and the driver reported two.  A
      bound derived inside one program must be re-derived after partitioning.
    * Widening a CTA and co-residing two CTAs cost the same register budget
      (both need 153 -> 85 at 24 warps / 2 CTAs per SM), but they are not
      interchangeable: extra warps in one CTA leave the number of CTAs, and
      hence the per-SM write volume each release fence must flush, unchanged.
    """

    name: str
    parameters: tuple[str, ...]
    results: tuple[str, ...]
    body: Sequential
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "attrs", MappingProxyType(dict(sorted(self.attrs.items()))))
        if not self.name:
            raise IRSchemaError("ExecutionFunction requires a non-empty name.")
        if len(set(self.parameters)) != len(self.parameters):
            raise IRSchemaError(f"ExecutionFunction @{self.name} has duplicate physical parameters.")
        if len(set(self.results)) != len(self.results):
            raise IRSchemaError(f"ExecutionFunction @{self.name} has duplicate physical results.")
        if not isinstance(self.body, Sequential):
            raise IRSchemaError("ExecutionFunction body must be Sequential.")


__all__ = ["ExecutionFunction"]
