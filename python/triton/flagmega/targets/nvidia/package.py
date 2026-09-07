# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Model-independent SM90 package planning for selected TIR call graphs."""

from __future__ import annotations

from collections.abc import Mapping
from math import prod

from triton.flagmega.codegen.triton.distributed_abi import kernel_execution_kind
from triton.flagmega.codegen.triton.package_plan import PACKAGE_PLAN_SCHEMA
from triton.flagmega.errors import IRVerificationError


def sm90_codegen_package_plan(
    module,
    kernel_nodes,
    capability,
    options,
) -> dict[str, object]:
    """Freeze the selected kernel-call closure without interpreting a model.

    Importers own model structure. The target owns implementation selection
    and launch resources. Code generation receives only selected
    ``PrimFunction`` calls, so package planning must never reconstruct a model
    from architecture names, graph node names, or weight roles.
    """

    del capability, options
    if not kernel_nodes:
        raise IRVerificationError(
            "SM90 package planning requires at least one selected TIR call.",
            stage=module.stage,
        )

    calls: list[dict[str, object]] = []
    templates: list[dict[str, str]] = []
    seen_templates: set[tuple[str, str]] = set()
    for node in kernel_nodes:
        parameters = node.attrs.get("parameters")
        facts = node.attrs.get("facts", {})
        if not isinstance(parameters, Mapping) or not isinstance(facts, Mapping):
            raise IRVerificationError(
                f"Selected TIR call {node.id!r} has no implementation contract.",
                stage=module.stage,
                node_id=node.id,
            )
        family = str(parameters.get("family", ""))
        variant = str(parameters.get("variant", ""))
        if not family or not variant:
            raise IRVerificationError(
                f"Selected TIR call {node.id!r} has no family/variant identity.",
                stage=module.stage,
                node_id=node.id,
            )
        # ``lower_to_tir`` asks for a provisional plan before semantic
        # candidates are materialized as PrimFunctions. Later package stages
        # replace this empty value with the real callee symbol.
        callee = str(node.attrs.get("callee", ""))
        semantic_op = str(node.attrs.get("semantic_op", ""))
        calls.append({
            "call": node.id,
            "callee": callee,
            "semantic_op": semantic_op,
            "implementation": str(node.attrs.get("candidate", "")),
            "family": family,
            "variant": variant,
            "execution_kind": kernel_execution_kind(
                semantic_op, facts
            ).value,
            "requires": tuple(str(value) for value in facts.get("requires", ())),
        })
        identity = (family, variant)
        if identity not in seen_templates:
            seen_templates.add(identity)
            templates.append({"kernel": family, "variant": variant})

    launch = module.metadata.get("launch_contract")
    if not isinstance(launch, Mapping):
        raise IRVerificationError(
            "SM90 package planning requires a target launch contract.",
            stage=module.stage,
        )
    grid_mesh = launch.get("grid_mesh")
    hierarchy: tuple[int, ...] = ()
    if isinstance(grid_mesh, Mapping):
        raw_hierarchy = grid_mesh.get("hierarchy", ())
        if isinstance(raw_hierarchy, (tuple, list)):
            hierarchy = tuple(int(value) for value in raw_hierarchy)
    cooperative = bool(launch.get("cooperative_grid", False))
    if cooperative and (not hierarchy or any(value <= 0 for value in hierarchy)):
        raise IRVerificationError(
            "A cooperative SM90 package requires a positive physical mesh.",
            stage=module.stage,
        )
    owner_count = prod(hierarchy, start=1) if cooperative else 1
    num_warps = launch.get("num_warps")
    if isinstance(num_warps, bool) or not isinstance(num_warps, int) or num_warps <= 0:
        raise IRVerificationError(
            f"SM90 package launch has invalid num_warps {num_warps!r}.",
            stage=module.stage,
        )

    return {
        "schema": PACKAGE_PLAN_SCHEMA,
        "kind": "tir_call_graph",
        "profile": "nvidia_sm90",
        "calls": tuple(calls),
        "kernel_templates": tuple(templates),
        "launch": {
            "grid": (owner_count, 1, 1),
            "num_warps": num_warps,
            "cooperative_grid": cooperative,
            "mesh_hierarchy": hierarchy,
        },
    }


__all__ = ["PACKAGE_PLAN_SCHEMA", "sm90_codegen_package_plan"]
