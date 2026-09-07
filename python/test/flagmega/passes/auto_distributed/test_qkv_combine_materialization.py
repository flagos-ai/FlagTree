# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import AutoDistributedPass
from triton.flagmega.rules.ntt.packing import NttPackingPolicy
from triton.flagmega.targets import NvidiaSm90Target


class RaggedQKVModule(fm.Module):
    """Q/K/V extents exercise independent ragged block-cyclic policies."""

    def __init__(self):
        super().__init__(dialect="high_level", stage="imported", entry="main")

    def forward(self):
        value = self.input("value", fm.tensor_type("bfloat16", (1, 2048)))
        q_weight = self.weight(
            "q_weight", fm.tensor_type("bfloat16", (2048, 96)),
            source="memory", key="q_weight"
        )
        k_weight = self.weight(
            "k_weight", fm.tensor_type("bfloat16", (2048, 48)),
            source="memory", key="k_weight"
        )
        v_weight = self.weight(
            "v_weight", fm.tensor_type("bfloat16", (2048, 48)),
            source="memory", key="v_weight"
        )
        none = fm.F.builtin.none(name="none")
        qkv = fm.F.nn.qkv_parallel_linear(
            value,
            q_weight,
            k_weight,
            v_weight,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
            none,
            num_heads=6,
            num_kv_heads=3,
            output_data_type="bfloat16",
            name="qkv",
        )
        self.function("main", (value,), (qkv,))


def test_cp_sat_preserves_coupled_direct_qkv_before_three_unpacks(tmp_path):
    target = NvidiaSm90Target()
    policy = NttPackingPolicy(vector_bytes=16, k_pack=2)
    packed = policy.apply(policy.propose(RaggedQKVModule().build(), target), target)

    result = AutoDistributedPass.run(packed, target)

    projection = result.node_map["qkv.packed_projection"]
    combine = result.node_map["qkv.packed_combine"]
    assert all(field.partial == fm.SBP.partial((0,)) for field in projection.type.fields)
    assert all(
        isinstance(field.axis_policies[-1], fm.SBPSplit)
        for field in projection.type.fields
    )
    assert combine.inputs == (projection.id,)
    assert all(field.partial is None for field in combine.type.fields)
    assert all(
        left.tensor == right.tensor
        and left.axis_policies == right.axis_policies
        for left, right in zip(projection.type.fields, combine.type.fields)
    )
    assert combine.attrs["output_type"] == combine.type
    assert not any(
        node.op == "distributed.boxing" and node.inputs == (projection.id,)
        for node in result.nodes
    )
    assert all(
        result.node_map[f"qkv.{role}.packed"].inputs == (combine.id,)
        for role in ("q", "k", "v")
    )

    path = fm.emit_module(result, tmp_path / "auto_distributed_qkv.py")
    assert fm.load_module(path) == result
