# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Module-owned communication byte ranges appended to one symmetric window."""

from collections.abc import Mapping

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.program_order import program_ancestors


SCHEMA = "flagmega.persistent-window-resources/v1"


def _require(condition, message):
    if not condition:
        raise IRVerificationError(message)


def plan_window_resources(request, pools, programs, window_storage):
    """Allocate disjoint persistent ranges after the complete scratch arena.

    Each resource is one mutable channel, so its declared invocations must be
    ordered by the actual dependency DAG. Stream assignment alone does not
    establish matching cross-rank channel generations.
    """
    _require(isinstance(request, Mapping) and set(request) == {"storage", "resources"},
             "Persistent window request requires storage and resources.")
    storage = request["storage"]
    _require(storage == window_storage and storage is not None,
             "Persistent resources must use the existing symmetric window.")
    pool = next((pool for pool in pools if pool["storage"] == storage), None)
    _require(pool is not None and pool.get("scope_count", 1) == 1 and storage != "rdata",
             "Persistent window resources require one writable chip-scoped pool.")
    scratch = pool["nbytes"]
    _require(type(scratch) is int and scratch > 0, "Persistent window scratch size must be positive.")
    ancestors = program_ancestors(programs)
    _require(ancestors is not None, "Persistent channels require an explicit program dependency DAG.")
    resources = request["resources"]
    _require(isinstance(resources, (tuple, list)) and bool(resources),
             "Persistent window resource list must be nonempty.")
    cursor, names, result = scratch, set(), []
    for resource in resources:
        _require(isinstance(resource, Mapping) and set(resource) == {
            "name", "nbytes", "alignment", "initialization", "programs"
        }, "Persistent resource declaration has unknown or missing fields.")
        name, size, alignment = resource["name"], resource["nbytes"], resource["alignment"]
        _require(isinstance(name, str) and bool(name) and name not in names,
                 "Persistent resource names must be nonempty and unique.")
        _require(type(size) is int and size > 0 and type(alignment) is int
                 and alignment > 0 and alignment <= 4096 and alignment & (alignment - 1) == 0,
                 "Persistent resource byte geometry must be positive and power-of-two aligned.")
        _require(resource["initialization"] == "zero", "Persistent resources require explicit zero initialization.")
        actors = resource["programs"]
        _require(isinstance(actors, (tuple, list)) and bool(actors)
                 and all(isinstance(actor, str) and actor in ancestors for actor in actors)
                 and len(set(actors)) == len(actors), "Persistent resource actors must name unique execution programs.")
        _require(all(left in ancestors[right] for index, right in enumerate(actors) for left in actors[:index]),
                 "Persistent channel invocations must be ordered by dependencies.")
        cursor = -(-cursor // alignment) * alignment
        result.append({**resource, "programs": list(actors), "offset": cursor})
        cursor += size
        names.add(name)
    return dict(schema=SCHEMA, storage=storage, scratch_nbytes=scratch,
                allocation_nbytes=-(-cursor // 4096) * 4096, resources=result)


def validate_window_resources(contract, pools):
    """Validate a serialized plan before any CUDA allocation or registration."""
    _require(isinstance(contract, Mapping) and set(contract) == {
        "schema", "storage", "scratch_nbytes", "allocation_nbytes", "resources"
    } and contract["schema"] == SCHEMA, "Malformed persistent window resource plan.")
    pool = next((pool for pool in pools if pool["storage"] == contract["storage"]), None)
    scratch, total = contract["scratch_nbytes"], contract["allocation_nbytes"]
    _require(pool is not None and pool.get("scope_count", 1) == 1 and contract["storage"] != "rdata"
             and type(scratch) is int and scratch == pool["nbytes"]
             and type(total) is int and 0 < scratch < total,
             "Persistent window resource capacity disagrees with its scratch pool.")
    resources = contract["resources"]
    _require(isinstance(resources, (tuple, list)) and bool(resources), "Persistent resource plan must be nonempty.")
    end, names = scratch, set()
    for resource in resources:
        _require(isinstance(resource, Mapping) and set(resource) == {
            "name", "offset", "nbytes", "alignment", "initialization", "programs"
        }, "Malformed persistent resource range.")
        name, offset, size, alignment = (resource[key] for key in ("name", "offset", "nbytes", "alignment"))
        _require(isinstance(name, str) and bool(name) and name not in names
                 and type(offset) is int and type(size) is int and size > 0
                 and type(alignment) is int and 0 < alignment <= 4096
                 and alignment & (alignment - 1) == 0 and offset % alignment == 0
                 and end <= offset < offset + size <= total and resource["initialization"] == "zero",
                 "Persistent resource overlaps scratch, another channel, or window capacity.")
        _require(isinstance(resource["programs"], (tuple, list)) and bool(resource["programs"]),
                 "Persistent resource must declare execution programs.")
        end = offset + size
        names.add(name)
    return contract
