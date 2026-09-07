# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm
from triton.flagmega.ir.ops.nn.qkv_parallel_linear import QKVParallelLinear
from triton.flagmega.rules.ntt.packing import NttPackingPolicy
from triton.flagmega.targets import NvidiaSm90Target


class QKVPatternModule(fm.Module):
    def __init__(self):
        super().__init__(dialect="high_level", stage="imported", entry="main")

    def forward(self):
        value = self.input("value", fm.tensor_type("bfloat16", (1, 32)))
        weights = tuple(
            self.weight(
                f"{role}_weight",
                fm.tensor_type("bfloat16", (32, size)),
                source="memory",
                key=f"{role}_weight",
            )
            for role, size in (("q", 64), ("k", 32), ("v", 32))
        )
        none = fm.F.builtin.none(name="none")
        qkv = fm.F.nn.qkv_parallel_linear(
            value,
            *weights,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
            num_heads=8,
            num_kv_heads=4,
            output_data_type="bfloat16",
            name="qkv",
        )
        self.function("main", (value,), (qkv,))


def test_qkv_pattern_uses_named_parameter_info_before_and_after_packing():
    logical = QKVPatternModule().build()
    q_weight = pm.wildcard("q_weight")
    logical_pattern = pm.F.nn.is_qkv_parallel_linear(
        q_weight=q_weight,
        num_heads=8,
        num_kv_heads=4,
        output_data_type="bfloat16",
        call_name="qkv",
    )
    logical_match = pm.try_match_root(logical.node_map["qkv"], logical_pattern, logical)
    assert logical_match is not None
    assert logical_match["q_weight"].id == logical.node_map["qkv"].inputs[1]
    assert logical_pattern[QKVParallelLinear.q_weight] is q_weight

    target = NvidiaSm90Target()
    policy = NttPackingPolicy(vector_bytes=16, k_pack=2)
    packed = policy.apply(policy.propose(logical, target), target)
    combine = packed.node_map["qkv.packed_combine"]
    projection_pattern = pm.F.ntt.is_packed_qkv_parallel_linear(
        num_heads=8,
        num_kv_heads=4,
        rhs_layout="k_major",
        call_name="projection",
    )
    combine_pattern = pm.F.ntt.is_packed_qkv_parallel_linear_combine(
        projection_pattern,
        output_type=combine.type,
        call_name="combine",
    )
    packed_match = pm.try_match_root(combine, combine_pattern, packed)
    assert packed_match is not None
    assert packed_match["projection"].id == "qkv.packed_projection"
    assert packed_match["combine"].id == combine.id


def test_none_and_variadic_tuple_patterns_are_first_class():
    module = QKVPatternModule().build()
    none = module.node_map["none"]
    assert pm.try_match_root(none, pm.F.builtin.is_none(call_name="none"), module)
    assert pm.F.builtin.is_tuple(pm.wildcard(), pm.wildcard()).target.op_name == (
        "builtin.tuple"
    )
