# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-injected implementation catalog for generic Triton candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.tir.microkernel_selection import TIRMicroKernelSelection
from triton.flagmega.ir.tir.shared_workspace_descriptor import (
    TIRSharedWorkspaceDescriptor,
)
from triton.flagmega.ir.tir.transfer_pipeline_contract import (
    TIRTransferPipelineContract,
)


@dataclass(frozen=True)
class TritonImplementation:
    """One target-owned implementation of a semantic TIR kernel family.

    ``contract`` describes the semantic situations in which a generic
    candidate provider may use the implementation. ``parameters`` and
    ``requires`` describe the concrete implementation and machine features.
    Providers therefore do not need to know target-specific candidate ids,
    tile suffixes, or resource geometry.
    """

    id: str
    family: str
    variant: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    contract: Mapping[str, object] = field(default_factory=dict)
    requires: Sequence[str] = field(default_factory=tuple)
    facts: Mapping[str, object] = field(default_factory=dict)
    shared_workspaces: Sequence[TIRSharedWorkspaceDescriptor] = field(
        default_factory=tuple
    )
    transfer_pipeline: TIRTransferPipelineContract | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.family or not self.variant:
            raise CodegenError(
                "A Triton implementation requires non-empty id, family, and variant."
            )
        parameters = MappingProxyType(dict(self.parameters))
        contract = MappingProxyType(dict(self.contract))
        requires = tuple(str(value) for value in self.requires)
        facts = MappingProxyType(dict(self.facts))
        shared_workspaces = tuple(self.shared_workspaces)
        if len(set(requires)) != len(requires) or any(not value for value in requires):
            raise CodegenError(
                f"Triton implementation {self.id!r} requirements must be non-empty and unique."
            )
        overlap = set(parameters) & {"family", "variant"}
        if overlap:
            raise CodegenError(
                f"Triton implementation {self.id!r} parameters contain reserved keys {overlap}."
            )
        if "requires" in facts:
            raise CodegenError(
                f"Triton implementation {self.id!r} must use requires, not a facts entry."
            )
        _plain(parameters)
        _plain(contract)
        _plain(requires)
        _plain(facts)
        if any(
            not isinstance(value, TIRSharedWorkspaceDescriptor)
            for value in shared_workspaces
        ):
            raise CodegenError(
                f"Triton implementation {self.id!r} shared workspaces must be "
                "typed descriptors."
            )
        if self.transfer_pipeline is not None and not isinstance(
            self.transfer_pipeline, TIRTransferPipelineContract
        ):
            raise CodegenError(
                f"Triton implementation {self.id!r} transfer pipeline must be typed."
            )
        try:
            TIRMicroKernelSelection(
                self.id,
                self.family,
                self.variant,
                parameters,
                facts,
                requires,
                shared_workspaces,
                self.transfer_pipeline,
            )
        except Exception as error:
            raise CodegenError(
                f"Triton implementation {self.id!r} has an invalid target resource "
                f"contract: {error}"
            ) from error
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "contract", contract)
        object.__setattr__(self, "requires", requires)
        object.__setattr__(self, "facts", facts)
        object.__setattr__(self, "shared_workspaces", shared_workspaces)


@dataclass(frozen=True)
class TritonImplementationModel:
    """Concrete implementations separated from semantic candidate providers.

    This is the Python equivalent of nncase's target-owned
    ``ITIRMicroKernelSelector`` catalog boundary. Generic providers enumerate
    implementations by family and semantic contract; the target owns concrete
    ids, template variants, tile/pipeline geometry, requirements, and ordering.
    """

    implementations: Sequence[TritonImplementation] = field(default_factory=tuple)
    preferences: Mapping[str, Sequence[str]] = field(default_factory=dict)
    name: str = "unspecified"

    def __post_init__(self) -> None:
        if not self.name:
            raise CodegenError("A Triton implementation model requires a non-empty name.")
        implementations = tuple(self.implementations)
        if any(not isinstance(value, TritonImplementation) for value in implementations):
            raise CodegenError(
                "Triton implementation models require TritonImplementation entries."
            )
        by_id = {value.id: value for value in implementations}
        if len(by_id) != len(implementations):
            duplicates = tuple(sorted({
                value.id for value in implementations
                if sum(other.id == value.id for other in implementations) > 1
            }))
            raise CodegenError(
                f"Triton implementation candidate ids must be unique: {duplicates}."
            )
        preferences = {
            str(family): tuple(str(candidate_id) for candidate_id in candidate_ids)
            for family, candidate_ids in self.preferences.items()
        }
        for family, candidate_ids in preferences.items():
            if not family or not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
                raise CodegenError(
                    f"Triton implementation preference {family!r} must be non-empty and unique."
                )
            missing = tuple(value for value in candidate_ids if value not in by_id)
            if missing:
                raise CodegenError(
                    f"Triton implementation preference {family!r} references missing {missing}."
                )
            wrong_family = tuple(
                value for value in candidate_ids if by_id[value].family != family
            )
            if wrong_family:
                raise CodegenError(
                    f"Triton implementation preference {family!r} contains entries from "
                    f"another family: {wrong_family}."
                )
        object.__setattr__(self, "implementations", implementations)
        object.__setattr__(self, "preferences", MappingProxyType(preferences))
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))

    @property
    def variants(self) -> Mapping[str, Mapping[str, object]]:
        """Physical parameter view retained for verifier and inspection APIs."""

        return MappingProxyType({
            implementation.id: implementation.parameters
            for implementation in self.implementations
        })

    def implementation(self, candidate_id: str) -> TritonImplementation | None:
        return self._by_id.get(candidate_id)

    def parameters(self, candidate_id: str) -> Mapping[str, object] | None:
        """Return physical parameters, or ``None`` when target has no implementation."""

        implementation = self.implementation(candidate_id)
        return None if implementation is None else implementation.parameters

    def find(
        self,
        family: str,
        **contract: object,
    ) -> tuple[TritonImplementation, ...]:
        """Enumerate implementations whose target-owned contract matches exactly."""

        return tuple(
            implementation
            for implementation in self.implementations
            if implementation.family == family
            and all(implementation.contract.get(key) == value for key, value in contract.items())
        )

    def snapshot(self) -> Mapping[str, object]:
        """Return a stable identity for editable selected-TIR checkpoints."""

        payload = {
            "name": self.name,
            "implementations": [
                {
                    "id": implementation.id,
                    "family": implementation.family,
                    "variant": implementation.variant,
                    "parameters": _plain(implementation.parameters),
                    "contract": _plain(implementation.contract),
                    "requires": list(implementation.requires),
                    "facts": _plain(implementation.facts),
                    "shared_workspaces": [
                        value.to_data() for value in implementation.shared_workspaces
                    ],
                    "transfer_pipeline": (
                        None
                        if implementation.transfer_pipeline is None
                        else implementation.transfer_pipeline.to_data()
                    ),
                }
                for implementation in sorted(self.implementations, key=lambda value: value.id)
            ],
            "preferences": {
                family: list(candidate_ids)
                for family, candidate_ids in sorted(self.preferences.items())
            },
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return MappingProxyType({
            "schema": "flagmega.triton-implementation-model/v3",
            "name": self.name,
            "sha256": hashlib.sha256(encoded).hexdigest(),
        })

    def choose_default(
        self,
        family: str,
        candidate_ids: Sequence[str],
        *,
        portable_fallback: str | None = None,
    ) -> str:
        available = tuple(candidate_ids)
        if not available:
            raise CodegenError(f"Triton family {family!r} has no applicable candidate.")
        for candidate_id in self.preferences.get(family, ()):
            if candidate_id in available:
                return candidate_id
        if portable_fallback in available:
            return str(portable_fallback)
        raise CodegenError(
            f"Implementation model {self.name!r} has no default for Triton family "
            f"{family!r}; applicable candidates={available}."
        )


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise CodegenError(
        f"Triton implementation model contains non-serializable value {value!r}."
    )


__all__ = ["TritonImplementation", "TritonImplementationModel"]
