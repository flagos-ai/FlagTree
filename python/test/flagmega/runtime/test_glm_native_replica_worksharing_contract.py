# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""CTA tasks must publish through shared storage before private placement."""

from importlib import import_module
from pathlib import Path

import pytest
from triton.flagmega.targets import NvidiaSm90Target


@pytest.mark.parametrize("family", ("mhc_post", "dsa_paged_attention"))
def test_native_replica_tasks_declare_shared_results(monkeypatch, family):
    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    profile = import_module("local_optimizations.native_tp4")
    implementations = tuple(value for value in profile.native_implementation_model().implementations
                            if value.family == family)
    assert implementations
    assert all(value.facts.get("result_memory_scope", "inferred") == "chip" for value in implementations)
    ordinary = tuple(value for value in NvidiaSm90Target().triton_implementation_model.implementations
                     if value.family == family)
    assert all(value.facts.get("result_memory_scope", "inferred") == "inferred" for value in ordinary)


def test_native_peer_gather_catalog_preserves_the_owner_local_alternative(monkeypatch):
    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    model = import_module("local_optimizations.native_tp4").native_implementation_model()
    peers = tuple(value for value in model.implementations if value.id.endswith(".cooperative_peer_gather"))
    assert len(peers) == 2
    assert all(value.contract["peer_gather"] and value.facts["result_memory_scope"] == "chip" for value in peers)
    ordinary = model.implementation("tir.distributed_boxing.gather_reduce_scatter")
    assert ordinary.facts.get("result_memory_scope", "inferred") == "inferred"


def test_native_sum_rows_keep_the_owner_local_storage_contract(monkeypatch):
    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    model = import_module("local_optimizations.native_tp4").native_implementation_model()
    selected = model.implementation(model.preferences["reduce_sum"][0])
    assert selected.variant == "tiled_rows"
    assert selected.facts.get("result_memory_scope", "inferred") == "inferred"
    assert model.implementation("tir.reduce_sum.local") is not None
