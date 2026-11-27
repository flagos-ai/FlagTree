# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Validate and own the immutable data for compact kernel-function calls."""

from collections.abc import Mapping

from triton.flagmega.errors import ArtifactError


def validate_function_offset_tables(raw, *, existing_names):
    if not isinstance(raw, (tuple, list)):
        raise ArtifactError("Function offset tables must be a sequence.")
    names = set(existing_names)
    specs = []
    for spec in raw:
        if not isinstance(spec, Mapping):
            raise ArtifactError("Function offset table specs must be mappings.")
        name = spec.get("name")
        values = spec.get("values")
        if (not isinstance(name, str) or not name.isidentifier() or name in names
                or spec.get("dtype") != "int64" or not isinstance(values, (tuple, list))
                or not values or any(type(value) is not int or not -(2**63) <= value < 2**63
                                     for value in values)):
            raise ArtifactError("Invalid function offset table name, dtype or signed-i64 payload.")
        names.add(name)
        specs.append({"name": name, "dtype": "int64", "values": tuple(values)})
    return tuple(specs)


def materialize_function_offset_tables(torch, specs, *, device):
    """Create tables before registering any FlagCX symmetric window.

    These tensors are immutable static launch arguments, owned by the loaded
    module and its prepared kernel. They require no checkpoint repacking.
    """
    return tuple(torch.tensor(spec["values"], dtype=torch.int64, device=device) for spec in specs)


def verify_function_offset_source(kernel, specs, *, existing_names):
    """Bind launch metadata to the literal in the verified generated source.

    The artifact's existing source digest covers this literal. Comparing the
    validated payload also detects manifest changes without hashing weights
    or introducing another integrity mechanism.
    """
    function = getattr(kernel, "fn", kernel)
    raw = getattr(function, "__globals__", {}).get("FLAGMEGA_FUNCTION_OFFSET_TABLES", ())
    source_specs = validate_function_offset_tables(raw, existing_names=existing_names)
    if source_specs != specs:
        raise ArtifactError("Function offset tables do not match the verified generated source.")
