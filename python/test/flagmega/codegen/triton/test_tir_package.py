# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.tir_package import (
    _kernel_source_event,
    _verify_package_call_closure,
)
from triton.flagmega.errors import CodegenError


def _binding(call: str, family: str, variant: str) -> dict[str, object]:
    return {
        "call_abi": {
            "kernel_calls": ({
                "call": call,
                "family": family,
                "variant": variant,
            },),
        },
    }


def test_package_closure_is_independent_of_function_storage_order():
    package = {
        "calls": (
            {"call": "worker_call", "family": "worker", "variant": "v1"},
            {"call": "entry_call", "family": "entry", "variant": "v2"},
        ),
    }
    bindings = {
        "main": _binding("entry_call", "entry", "v2"),
        "worker": _binding("worker_call", "worker", "v1"),
    }

    _verify_package_call_closure(package, bindings)


def test_package_closure_rejects_a_different_selected_implementation():
    package = {
        "calls": ({
            "call": "call",
            "family": "elementwise",
            "variant": "add",
        },),
    }
    bindings = {"main": _binding("call", "elementwise", "mul")}

    with pytest.raises(CodegenError, match="call closure differs"):
        _verify_package_call_closure(package, bindings)


def test_package_closure_rejects_duplicate_call_ids():
    package = {
        "calls": (
            {"call": "call", "family": "elementwise", "variant": "add"},
            {"call": "call", "family": "elementwise", "variant": "add"},
        ),
    }
    bindings = {"main": _binding("call", "elementwise", "add")}

    with pytest.raises(CodegenError, match="Duplicate TIR call id 'call'"):
        _verify_package_call_closure(package, bindings)


def _source_event(execution_kind: str, scope: str | None):
    dependencies = () if scope is None else ({"scope": scope},)
    return _kernel_source_event(
        {
            "call": "call",
            "family": "elementwise",
            "variant": "add",
            "execution_kind": execution_kind,
            "arguments": (),
            "symbol": "_call",
        },
        {},
        {},
        [],
        ("call",),
        memory_dependencies=dependencies,
    )


def test_package_event_preserves_block_memory_scope():
    event = _source_event("local_shard", "block")

    assert event["barrier_before"] is True
    assert event["barrier_scope"] == "block"


def test_collective_package_event_requires_grid_scope_without_dependency():
    event = _source_event("collective", None)

    assert event["barrier_before"] is True
    assert event["barrier_scope"] == "grid"
