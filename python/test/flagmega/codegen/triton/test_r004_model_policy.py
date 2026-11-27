# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Graph contract checks, including the public IR constructor/reload path."""
from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.rules.ntt.r004_model_policy import (
    REQUEST_KEY, ColumnDistributionPolicy, ColumnEffectProvider,
    configure_model_phases, model_request, model_distribution_target,
    prepare_model_column_distribution,
)
from triton.flagmega.rules.ntt.r004_column_phase_bindings import column_phase_plan
from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import configure_standalone_phases
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidateContext, DistributedCandidateProviderRegistry,
)
from triton.flagmega.targets import NvidiaSm120Target
from .test_r004_phase_bindings import boundary


def repeated_effect_fixture():
    """48 cell invocations in a small graph, for policy tests only."""
    m = boundary(requested=False)
    cell = replace(m.functions[0], name="cell")
    result_type = m.node_map[cell.outputs[0]].type
    calls = tuple(fm.Node(f"cell_call_{i}", "builtin.call", cell.parameters, result_type,
                          effect=fm.effect("read_write", "gated_delta_net_state"),
                          attrs={"callee": "cell"}) for i in range(48))
    main = fm.Function("main", cell.parameters, tuple(n.id for n in calls))
    return replace(m, nodes=(*m.nodes, *calls), functions=(cell, main),
                   metadata={"num_hidden_layers": 64,
                             "imported_layer_indices": tuple(range(64))})


def test_default_has_no_model_policy_or_phase():
    m = boundary(requested=False)
    t = NvidiaSm120Target()
    assert model_request(m) is None
    assert configure_model_phases(m) is m
    assert model_distribution_target(t, m) is t
    assert configure_standalone_phases(m) is m


def test_complete_effect_and_public_constructor_reload(tmp_path):
    m = repeated_effect_fixture()
    fm.verify_module(m)
    configured = prepare_model_column_distribution(m)
    for before, after in zip(m.nodes, configured.nodes, strict=True):
        if before.op == "nn.gdn_recurrent_core":
            assert before.inputs == after.inputs
            assert before.type == after.type and before.effect == after.effect
            assert after.attrs["phase_plan"]["module"] == "r004_T_full_p5_v4"
        else:
            assert before == after
    path = tmp_path / "configured.py"
    fm.emit_module(configured, path)
    loaded = fm.load_module(path)
    assert loaded.semantic_hash == configured.semantic_hash
    assert prepare_model_column_distribution(loaded).semantic_hash == configured.semantic_hash
    fresh = model_distribution_target(NvidiaSm120Target(), loaded)
    assert isinstance(fresh.distribution_policy, ColumnDistributionPolicy)
    assert fresh.distribution_policy.base is not fresh.distribution_policy


def test_device_global_domain_and_neighbor_registry():
    m = prepare_model_column_distribution(repeated_effect_fixture())
    base = NvidiaSm120Target()
    target = model_distribution_target(base, m)
    original = DistributedCandidateProviderRegistry()
    registry = DistributedCandidateProviderRegistry()
    base.distribution_policy.register_candidate_providers(original)
    target.distribution_policy.register_candidate_providers(registry)
    assert registry.split_candidate_provider is original.split_candidate_provider
    assert isinstance(registry.try_get("nn.gdn_recurrent_core"), ColumnEffectProvider)
    for op in original.op_names:
        if op != "nn.gdn_recurrent_core":
            assert type(registry.try_get(op).base) is type(original.try_get(op))
            assert registry.try_get(op).is_exhaustive == original.try_get(op).is_exhaustive
    node = m.node_map["recurrent"]
    inputs = tuple((m.node_map[i].type,) for i in node.inputs)
    context = DistributedCandidateContext(m, node, fm.Placement((1,), "b", "b"), inputs)
    candidate, = registry.try_get(node.op).get_candidates(context)
    from triton.flagmega.ir.distributed_inference import placement_of
    assert placement_of(candidate.return_type.fields[0]).physical_level_size("b") == 1
    with pytest.raises(IRVerificationError, match="device-global"):
        registry.try_get(node.op).get_candidates(replace(context, placement=fm.Placement((32,), "b", "b")))


@pytest.mark.parametrize("cause", ["stage", "layers", "invocations", "identity", "standalone", "plan"])
def test_model_policy_rejects_stale_or_wrong_contract(cause):
    m = prepare_model_column_distribution(repeated_effect_fixture())
    if cause == "stage":
        m = replace(m, stage="distributed")
    elif cause == "layers":
        m = replace(m, metadata={**m.metadata, "num_hidden_layers": 63})
    elif cause == "invocations":
        main = m.function_map["main"]
        m = replace(m, functions=(m.function_map["cell"], replace(main, outputs=main.outputs[:47])))
    elif cause == "identity":
        m = replace(m, metadata={**m.metadata, REQUEST_KEY: {**m.metadata[REQUEST_KEY], "candidate": "wrong"}})
    elif cause == "standalone":
        m = replace(m, metadata={**m.metadata, "r004_standalone_request": {
            "schema": "r004.gdn.standalone/v4", "module": "r004_T_full_p5_v4", "readonly_layer": 0}})
    else:
        plan = column_phase_plan()
        plan["entries"][1]["grid"] = [32, 1, 1]
        m = replace(m, nodes=tuple(replace(n, attrs={**n.attrs, "phase_plan": plan})
                                  if n.op == "nn.gdn_recurrent_core" else n for n in m.nodes))
    from triton.flagmega.errors import IRSchemaError
    with pytest.raises((IRVerificationError, IRSchemaError)):
        prepare_model_column_distribution(m)
