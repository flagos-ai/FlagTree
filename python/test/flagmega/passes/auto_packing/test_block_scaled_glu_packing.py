# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules.ntt.packing import NttPackingPolicy
from triton.flagmega.targets import NvidiaSm90Target


def _module(*, k: int = 128) -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="vectorized")
    value_type = fm.tensor_type("bfloat16", (2, k))
    weight_type = fm.tensor_type("float8_e4m3fn", (64, k))
    scale_type = fm.tensor_type("float32", (1, 1))
    value = builder.var("value", value_type, id="value")
    gate = builder.weight(
        "gate", weight_type, source="memory", key="gate", id="gate"
    )
    up = builder.weight(
        "up", weight_type, source="memory", key="up", id="up"
    )
    gate_scale = builder.weight(
        "gate_scale",
        scale_type,
        source="memory",
        key="gate_scale",
        id="gate_scale",
    )
    up_scale = builder.weight(
        "up_scale",
        scale_type,
        source="memory",
        key="up_scale",
        id="up_scale",
    )
    output = builder.call(
        "nn.matmul_glu",
        (value, gate, up, gate_scale, up_scale),
        fm.tensor_type("bfloat16", (2, 64)),
        id="output",
        attrs={
            "activation": "silu",
            "weight_block_n": 128,
            "weight_block_k": 128,
        },
    )
    builder.function("main", (value,), (output,))
    return fm.verify_module(builder.build(entry="main"))


def _pack(module: fm.IRModule) -> fm.IRModule:
    policy = NttPackingPolicy(vector_bytes=16, k_pack=2)
    target = NvidiaSm90Target()
    return fm.verify_module(policy.apply(policy.propose(module, target), target))


def test_block_scaled_glu_packs_both_k_axes_and_preserves_evaluation():
    original = _module()
    packed = _pack(original)
    output = packed.node_map["output"]

    assert output.op == "nn.packed_matmul_glu"
    assert packed.selection_map["packing.output"].candidate_id == (
        "packing.n_major_k_packed"
    )
    for packed_id, source_id in zip(output.inputs[1:3], ("gate", "up")):
        weight = packed.node_map[packed_id]
        assert weight.op == "tensors.pack"
        assert weight.inputs == (source_id,)
        assert weight.type == fm.tensor_type(
            fm.vector_type("float8_e4m3fn", (2, 16)), (64, 4)
        )

    generator = torch.Generator().manual_seed(20260904)
    value = torch.randn((2, 128), generator=generator).to(torch.bfloat16)
    weights = {
        "gate": torch.randn((64, 128), generator=generator).to(
            torch.float8_e4m3fn
        ),
        "up": torch.randn((64, 128), generator=generator).to(
            torch.float8_e4m3fn
        ),
        "gate_scale": torch.ones((1, 1), dtype=torch.float32),
        "up_scale": torch.ones((1, 1), dtype=torch.float32),
    }
    evaluator = TorchEvaluator(DictWeightResolver(weights))
    torch.testing.assert_close(
        evaluator.run(packed, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
        rtol=0,
        atol=0,
    )


def test_block_scaled_glu_rejects_nonintegral_k_pack_without_partial_rewrite():
    module = _module(k=120)
    policy = NttPackingPolicy(vector_bytes=16, k_pack=2)

    assert policy.candidate_points(module) == ()
