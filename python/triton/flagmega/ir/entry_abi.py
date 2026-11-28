# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Published entry ABI contracts and their node-id redirects.

A model importer may publish a structural ABI contract in module metadata so the
runtime can carry state without reconstructing the model.  The contract names
entry parameter and output node ids, so any pass that redirects an entry
boundary -- inserting a reshard, restoring a raw parameter, resolving an alias --
must redirect the contract with it.  Keeping that rule in one place means a new
boundary-rewriting pass cannot silently leave the contract pointing at a node
that no longer appears in the entry signature.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from triton.flagmega.errors import IRVerificationError

# Contract keys whose values are entry node ids (a single id or a tuple of ids).
# Structural keys carry no node reference and are copied through.
_VALUE_KEYS: frozenset[str] = frozenset({"input_id", "logits_output", "token_output", "output", "input"})
_STRUCTURAL_KEYS: frozenset[str] = frozenset({"schema", "state_id"})

_ABI_CONTRACTS: tuple[str, ...] = ("composite_abi", )


def _redirect_value(value, mapping: Mapping[str, str]):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, (tuple, list)):
        return tuple(_redirect_value(item, mapping) for item in value)
    return value


def _redirect_contract(contract: Mapping, mapping: Mapping[str, str]) -> dict:
    redirected: dict = {}
    for key, value in contract.items():
        if key in _STRUCTURAL_KEYS:
            redirected[key] = value
        elif isinstance(value, Mapping):
            redirected[key] = _redirect_contract(value, mapping)
        elif isinstance(value, (tuple, list)) and any(isinstance(item, Mapping) for item in value):
            redirected[key] = tuple(
                _redirect_contract(item, mapping) if isinstance(item, Mapping) else _redirect_value(item, mapping)
                for item in value)
        elif key in _VALUE_KEYS:
            redirected[key] = _redirect_value(value, mapping)
        else:
            redirected[key] = value
    return redirected


def entry_abi_values(contract: Mapping) -> tuple[str, ...]:
    """Every entry node id named by one ABI contract."""

    values: list[str] = []

    def walk(node) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in _STRUCTURAL_KEYS:
                    continue
                if isinstance(value, (Mapping, tuple, list)):
                    walk(value)
                elif isinstance(value, str) and key in _VALUE_KEYS:
                    values.append(value)
        elif isinstance(node, (tuple, list)):
            for item in node:
                walk(item)

    walk(contract)
    return tuple(values)


def redirect_entry_abi(
    metadata: Mapping,
    mapping: Mapping[str, str],
    *,
    known_values: Iterable[str] | None = None,
    stage: str | None = None,
) -> Mapping:
    """Apply an entry-boundary node-id redirect to every published ABI contract.

    ``metadata`` is returned unchanged when it publishes no contract or when no
    named id is redirected, so callers can apply this unconditionally.  Passing
    ``known_values`` checks the result against the surviving entry signature,
    turning a missed redirect into a verification failure at the pass that caused
    it instead of an artifact that only fails at runtime load.
    """

    present = tuple(name for name in _ABI_CONTRACTS if isinstance(metadata.get(name), Mapping))
    if not present:
        return metadata
    updated = dict(metadata)
    for name in present:
        updated[name] = _redirect_contract(metadata[name], mapping)
    if known_values is not None:
        allowed = set(known_values)
        for name in present:
            missing = sorted(set(entry_abi_values(updated[name])) - allowed)
            if missing:
                raise IRVerificationError(
                    f"{name} references unknown entry value(s) {missing} after redirect; "
                    "an entry boundary was rewritten without redirecting its ABI contract.",
                    stage=stage,
                )
    if all(updated[name] == metadata[name] for name in present):
        return metadata
    return updated


__all__ = ["entry_abi_values", "redirect_entry_abi"]
