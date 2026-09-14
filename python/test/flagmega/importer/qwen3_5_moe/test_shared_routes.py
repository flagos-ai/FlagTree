# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.importer import import_model
from triton.flagmega.ir.ops.nn.sparse_experts import SparseExperts
from triton.flagmega.passes.target_independent import decompose_complex_ops
from python.test.flagmega.importer.qwen3_5_moe.helpers import checkpoint, configuration


@pytest.mark.parametrize("phase,tokens", [("decode", 1), ("prefill", 3)])
@pytest.mark.parametrize("kind", ["linear", "attention"])
@pytest.mark.parametrize("intermediate", [8, 12])
def test_shared_expert_is_an_independent_always_active_group(phase, tokens, kind, intermediate):
    config = configuration()
    config["text_config"]["shared_expert_intermediate_size"] = intermediate
    module = import_model(checkpoint(config), mode="decode-1" if phase == "decode" else "prefill", num_tokens=tokens)
    output = module.node_map[f"{phase}_{kind}_moe_output"]
    routed, shared = (module.node_map[n] for n in output.inputs)
    assert shared.op == SparseExperts.op_name
    ids = module.node_map[SparseExperts.router_expert_ids.read(shared.inputs)]
    coefficient = module.node_map[SparseExperts.router_expert_weights.read(shared.inputs)]
    assert ids.op == "builtin.splat_const" and ids.attrs["value"] == 0
    assert tuple(d.fixed_value for d in ids.type.shape) == (tokens, 1)
    assert coefficient.op == "math.sigmoid"
    shared_weight = module.node_map[SparseExperts.gate_weight.read(shared.inputs)]
    routed_weight = module.node_map[SparseExperts.gate_weight.read(routed.inputs)]
    assert shared_weight.type.shape[0].fixed_value == 1
    assert shared_weight.type.shape[1].fixed_value == intermediate
    assert routed_weight.type.shape[1].fixed_value == 8
    decomposed = decompose_complex_ops(module)
    for branch in (routed, shared):
        assert decomposed.node_map[branch.id + ".dispatch"].op == "nn.sparse_experts_dispatch"
        assert decomposed.node_map[branch.id + ".down"].op == "nn.sparse_experts_down"
        assert decomposed.node_map[branch.id].op == "nn.sparse_experts_combine"


@pytest.mark.parametrize("profile", ["nncase", "vllm-ae10e855a-inductor-level3"])
def test_shared_group_preserves_the_dense_branch_numerical_contract(profile):
    import torch
    from triton.flagmega.evaluator import TorchEvaluator, CheckpointWeightResolver
    from triton.flagmega.importer import Qwen35MoeImporter, apply_numerical_profile
    from triton.flagmega.ir.ops.nn._gdn_state import create_gdn_state
    from triton.flagmega.ir.ops.nn._paged_attention_state import create_paged_attention_state

    source = checkpoint(with_values=True)
    importer = Qwen35MoeImporter(source)
    module = apply_numerical_profile(importer.import_module(), profile)
    _, trace = TorchEvaluator(CheckpointWeightResolver(source)).run_with_trace(module, {
        "input_ids": torch.tensor([7], dtype=torch.int32),
        "gated_delta_net_state": create_gdn_state(importer.gdn_config),
        "paged_attention_state": create_paged_attention_state(importer.paged_config),
    })
    for kind in ("linear", "attention"):
        node = module.node_map[f"decode_{kind}_shared_experts"]
        values = {p.name: trace[p.read(node.inputs)] for p in SparseExperts.input_parameters}
        q = values["q"]
        gate = torch.nn.functional.linear(q, values["gate_weight"][0])
        up = torch.nn.functional.linear(q, values["up_weight"][0])
        active = torch.nn.functional.silu(gate if profile == "nncase" else gate.float())
        hidden = (active * up).to(q.dtype)
        projected = torch.nn.functional.linear(hidden, values["down_weight"][0])
        expected = projected * values["router_expert_weights"]
        torch.testing.assert_close(trace[node.id], expected, rtol=0, atol=0)
