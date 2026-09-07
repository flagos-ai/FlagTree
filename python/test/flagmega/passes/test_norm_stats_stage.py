# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.stages import get_stage, next_stage
from triton.flagmega.targets import NvidiaSm90Target


class _Module(fm.Module):
    def __init__(self):
        super().__init__(dialect="nn", stage="distributed", entry="main")

    def forward(self):
        value = self.input("value", fm.tensor_type("bfloat16", [1, 16]))
        weight = self.input("weight", fm.tensor_type("bfloat16", [16]))
        bias = fm.F.builtin.splat_const(weight.type, 0.0, name="bias")
        stats = fm.F.nn.norm_stats(value, axis=-1, use_mean=False, name="stats")
        output = fm.F.nn.norm_apply(
            value, stats, weight, bias,
            axis=-1, epsilon=1e-6, use_mean=False, round_before_scale=True, name="output")
        self.function("main", (value, weight), (output,))


def test_post_distribution_norm_fusion_is_a_named_dumpable_stage():
    stage = next_stage("distributed")
    assert stage is not None and stage.name == "fold-materialized-packed-qkv-combine"
    folded = get_stage(stage.name).run(_Module().build(), NvidiaSm90Target())
    assert folded.stage == "qkv_combine_folded"
    lowered = get_stage(next_stage(folded.stage).name).run(
        folded, NvidiaSm90Target())
    assert lowered.stage == "qkv_combine_lowered"
    sunk = get_stage(next_stage(lowered.stage).name).run(
        lowered, NvidiaSm90Target())
    assert sunk.stage == "norm_stats_boxing_sunk"
    propagated = get_stage(next_stage(sunk.stage).name).run(
        sunk, NvidiaSm90Target())
    assert propagated.stage == "distributed_boundary_layout_propagated"
    finalized = get_stage(next_stage(propagated.stage).name).run(
        propagated, NvidiaSm90Target())
    assert finalized.stage == "norm_bindings_finalized"
    finalized_sunk = get_stage(next_stage(finalized.stage).name).run(
        finalized, NvidiaSm90Target())
    assert finalized_sunk.stage == "finalized_norm_stats_boxing_sunk"
    contracted = get_stage(next_stage(finalized_sunk.stage).name).run(
        finalized_sunk, NvidiaSm90Target())
    assert contracted.stage == "matmul_norm_stats_lowered"
    contracted = get_stage(next_stage(contracted.stage).name).run(
        contracted, NvidiaSm90Target())
    assert contracted.stage == "vector_contracts_lowered"
    result = get_stage(next_stage(contracted.stage).name).run(
        contracted, NvidiaSm90Target())
    assert result.stage == "fused_norm"
    assert result.node_map["output"].op == "nn.rms_norm"
    assert next_stage(result.stage).name == "constant-cse"
