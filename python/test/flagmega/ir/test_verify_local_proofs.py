# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reuse only immutable local proofs; module contracts are checked afresh."""

import gc
import weakref
from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import verify as verifier
from triton.flagmega.ir.ops.math.silu import Silu


def _module():
    x = fm.Node("x", "builtin.var", (), fm.tensor_type("float32", (8,)), attrs={"name": "x"})
    y = fm.Node("y", "math.silu", (x.id,), x.type)
    return fm.IRModule("high_level", "imported", (x, y), (fm.Function("main", (x.id,), (y.id,)),), "main")


def test_unchanged_nodes_keep_local_type_proofs_across_stage_snapshots(monkeypatch):
    module = _module()
    calls = []
    original = verifier._verify_type

    def measure(value, stage, node_id):
        calls.append(node_id)
        return original(value, stage, node_id)

    monkeypatch.setattr(verifier, "_verify_type", measure)
    for stage in ("imported", "decomposed", "packed"):
        fm.verify_module(replace(module, stage=stage))
    assert calls == ["x", "y"]


def test_unchanged_node_sequence_reuses_its_local_proof_without_dispatch(monkeypatch):
    module = _module()
    original = verifier._verify_local_node
    calls = []

    def record(node, *args):
        calls.append(node.id)
        return original(node, *args)

    monkeypatch.setattr(verifier, "_verify_local_node", record)
    first = fm.verify_module(module)
    second = fm.verify_module(replace(first, stage="packed"))
    assert calls == ["x", "y"]
    assert second.node_map is first.node_map


def test_changing_a_local_input_invalidates_its_users_proof():
    module = fm.verify_module(_module())
    x, y = module.nodes
    bad = replace(x, type=fm.tensor_type("float32", (16,)))
    with pytest.raises(IRVerificationError):
        fm.verify_module(replace(module, nodes=(bad, y)))


def test_unchanged_nodes_do_not_hide_broken_topology_or_function_contracts():
    module = fm.verify_module(_module())
    with pytest.raises(IRVerificationError, match="non-topological"):
        fm.verify_module(replace(module, nodes=tuple(reversed(module.nodes))))
    with pytest.raises(IRVerificationError, match="does not exist"):
        fm.verify_module(replace(module, functions=(fm.Function("main", ("x",), ("missing",)),)))


def test_changed_verifier_and_ambient_verifiers_are_not_cached(monkeypatch):
    module = fm.verify_module(_module())
    original = Silu.verify

    def verify(cls, node, view):
        if view.stage == "invalid":
            raise IRVerificationError("ambient check", stage=view.stage)
        return original(node, view)

    monkeypatch.setattr(Silu, "verify", classmethod(verify))
    fm.verify_module(replace(module, stage="packed"))
    with pytest.raises(IRVerificationError, match="ambient check"):
        fm.verify_module(replace(module, stage="invalid"))


def test_changed_inference_cannot_inherit_an_earlier_local_proof(monkeypatch):
    module = fm.verify_module(_module())
    monkeypatch.setattr(Silu, "infer_type", classmethod(lambda cls, inputs, attrs: fm.tensor_type("float32", (16,))))
    with pytest.raises(IRVerificationError):
        fm.verify_module(replace(module, stage="packed"))


def test_local_proofs_do_not_retain_nodes():
    module = fm.verify_module(_module())
    references = [weakref.ref(node) for node in module.nodes]
    del module
    gc.collect()
    assert all(reference() is None for reference in references)
