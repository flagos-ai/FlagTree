# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The native FP8 profile keeps rounded K partitions inside one projection."""

import importlib.util
from pathlib import Path

from triton.flagmega import ir as fm


def test_native_fp8_row_projection_has_one_partitioned_producer():
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 512)))
            w = self.input("w", fm.tensor_type("float8_e4m3fn", (64, 512)))
            s = self.input("scale", fm.tensor_type("float32", (1, 4)))
            y = fm.F.math.block_scaled_matmul(x, w, s, weight_block_n=128, weight_block_k=128,
                name="decode_linear_attention_dense_dense_mlp_down")
            self.function("main", (x, w, s), (y,))

    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    spec = importlib.util.spec_from_file_location("glm_partitioned_fp8_profile", tutorial / "local_optimizations/native_profile.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    original = Graph(dialect="high_level", stage="imported", entry="main").build()
    result = profile._apply_projection_profile(original)
    producers = [node for node in result.nodes if node.op == "nn.partitioned_block_scaled_linear"]
    assert len(producers) == 1, [(node.id, node.op) for node in result.nodes]
    assert producers[0].attrs["partitions"] == 4
    assert producers[0].type == fm.tensor_type("bfloat16", (4, 1, 64))
    assert not any(node.op in {"tensors.slice", "tensors.concat"} for node in result.nodes)
    assert result.node_map[result.function_map["main"].outputs[0]].type == fm.tensor_type("bfloat16", (1, 64))


def test_tp4_candidate_owns_one_material_partition(monkeypatch):
    from triton.flagmega.passes.auto_distributed import DistributedCandidateContext, DistributedCandidateProviderRegistry
    from triton.flagmega.passes.auto_distributed.policy import NttDistributionPolicy
    from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider

    from python.test.flagmega.ir.ops.nn.test_partitioned_block_scaled_linear import Graph

    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    from local_optimizations.partitioned_projection_distribution import owns_material_partition

    module = Graph(tokens=1, n=4096, k=8192, input_lanes=(2, 4), output_lanes=(2, 4)).build()
    node = module.node_map["projection"]
    placement = fm.Placement((4, 8, 16), "cyx", "cbb")
    split = PyNttDistributedSplitCandidateProvider(128)
    context = DistributedCandidateContext(module, node, placement,
        tuple((module.node_map[name].type,) for name in node.inputs), split)
    registry = DistributedCandidateProviderRegistry()
    NttDistributionPolicy((placement,), split).register_candidate_providers(registry)
    candidates = registry.try_get(node.op).get_candidates(context)
    selected, = (candidate for candidate in candidates if owns_material_partition(candidate.return_type))
    assert tuple(dim.fixed_value for dim in fm.local_shape(selected.input_types[0])) == (1, 256)
    assert tuple(dim.fixed_value for dim in fm.local_shape(selected.input_types[1])) == (4096, 2048)
    assert tuple(dim.fixed_value for dim in fm.local_shape(selected.return_type)) == (1, 1, 512)
    assert selected.return_type.partial is None
