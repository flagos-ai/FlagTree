# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A TP arithmetic profile must not expand row projection into compute adapters."""

import importlib.util
from pathlib import Path

from triton.flagmega import ir as fm


def test_native_bf16_row_projection_is_one_semantic_operation():
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 256)))
            w = self.input("w", fm.tensor_type("bfloat16", (64, 256)))
            y = fm.F.math.matmul(x, w, transpose_b=True,
                                 name="decode_linear_attention_self_attn_o_proj_weight_apply")
            self.function("main", (x, w), (y,))

    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    spec = importlib.util.spec_from_file_location("glm_partitioned_projection_profile", tutorial / "local_optimizations/native_profile.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    result = profile._apply_projection_profile(Graph(dialect="high_level", stage="imported", entry="main").build())
    forbidden = {"tensors.slice", "tensors.cast", "tensors.concat", "tensors.pack", "tensors.unpack"}
    assert not [(node.id, node.op) for node in result.nodes if node.op in forbidden]
    projection, = (node for node in result.nodes if node.op == "nn.partitioned_linear")
    assert projection.attrs["partitions"] == 4
    assert projection.type == fm.tensor_type("bfloat16", (1, 64))


def test_tp4_partition_candidate_retains_complete_local_k(monkeypatch):
    from triton.flagmega.passes.auto_distributed import (
        DistributedCandidateContext,
        DistributedCandidateProviderRegistry,
    )
    from triton.flagmega.passes.auto_distributed.policy import NttDistributionPolicy
    from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider

    from python.test.flagmega.ir.ops.nn.test_partitioned_linear import graph

    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    from local_optimizations.partitioned_projection_distribution import owns_one_partition

    module = graph(k=8192, n=4096, lanes=(2, 4), input_lanes=(2, 4))
    node = module.node_map["projection"]
    placement = fm.Placement((4, 8, 16), "cyx", "cbb")
    split = PyNttDistributedSplitCandidateProvider(128)
    context = DistributedCandidateContext(module, node, placement,
        tuple((module.node_map[name].type,) for name in node.inputs), split)
    registry = DistributedCandidateProviderRegistry()
    NttDistributionPolicy((placement,), split).register_candidate_providers(registry)
    candidates = registry.try_get(node.op).get_candidates(context)
    selected, = (candidate for candidate in candidates if owns_one_partition(candidate.return_type))
    assert tuple(d.fixed_value for d in fm.local_shape(selected.input_types[0])) == (1, 256)
    assert tuple(d.fixed_value for d in fm.local_shape(selected.input_types[1])) == (4096, 2048)


def test_offline_mla_row_projection_keeps_part_and_subdot_rounding_in_one_owner(monkeypatch):
    import torch
    from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 1024)), id="x")
            w = self.input("w", fm.tensor_type("float8_e4m3fn", (64, 1024)), id="w")
            scale = self.input("scale", fm.tensor_type("float32", (1, 8)), id="scale")
            out = fm.F.math.block_scaled_matmul(x, w, scale, weight_block_n=128, weight_block_k=128,
                name="decode_mla_attention_sparse_self_attn_o_proj_weight_apply")
            self.function("main", (x, w, scale), (out,))

    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    from local_optimizations import native_profile

    source = Graph(dialect="high_level", stage="imported", entry="main").build()
    legacy = native_profile._apply_projection_profile(source)
    coarse = native_profile._apply_projection_profile(source, coarse_mla_rows=True)
    projection, = (node for node in coarse.nodes if node.op == "nn.partitioned_linear")
    assert projection.attrs["partitions"] == 4
    assert projection.attrs["accumulation_splits"] == 2
    assert projection.type == legacy.node_map[legacy.function_map["main"].outputs[0]].type
    weight = coarse.node_map[projection.inputs[1]]
    assert set(weight.metadata["packed_from"]) == {"w", "scale"}
    recipe, pending = set(), [weight.id]
    while pending:
        value = pending.pop()
        if value in recipe:
            continue
        recipe.add(value)
        pending.extend(coarse.node_map[value].inputs)
    assert "x" not in recipe
    assert not any(node.op in {"tensors.slice", "tensors.concat"} and node.id not in recipe
                   for node in coarse.nodes)
    generator = torch.Generator().manual_seed(20261019)
    inputs = {"x": (torch.randint(-7, 8, (1, 1024), generator=generator).float() / 128).bfloat16(),
              "w": torch.randint(-7, 8, (64, 1024), generator=generator).to(torch.float8_e4m3fn),
              "scale": torch.full((1, 8), .5)}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    a, = evaluator.run(legacy, inputs)
    b, = evaluator.run(coarse, inputs)
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
