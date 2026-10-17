# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Per-head projection contracts, packing order and editable IR."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


def graph(direction="absorb_query", lanes=(), *, input_lanes=(), fp8=False, tokens=2, placement=None, **changes):
    qk, value_dim, latent, heads = 16, 8, 32, 2
    width = qk if direction == "absorb_query" else latent
    attrs = {
        "direction": direction,
        "qk_head_dim": qk,
        "value_head_dim": value_dim,
        "latent_dim": latent,
        "output_lanes": lanes,
        "weight_block_n": 8,
        "weight_block_k": 16,
    }
    attrs.update(changes)

    class Graph(fm.Module):
        def forward(self):
            kind = fm.vector_type("bfloat16", input_lanes) if input_lanes else "bfloat16"
            factor = 1
            for lane in input_lanes:
                factor *= lane
            x = self.input("x", fm.tensor_type(kind, (tokens, heads, width // factor)), id="x")
            w = self.input("w", fm.tensor_type("float8_e4m3fn" if fp8 else "bfloat16", (48, latent)), id="w")
            scale = self.input("scale", fm.tensor_type("float32", (6, 2)), id="scale") if fp8 else fm.F.builtin.none()
            result = fm.F.nn.mla_projection(x, w, scale, **attrs)
            parameters = [x, w, *((scale,) if fp8 else ())]
            outputs = [result]
            if placement is not None:
                kind = fm.tensor_type("float32", (128,))
                sentinel = self.input("sentinel", kind, id="sentinel")
                split = fm.DistributedType(kind, (fm.SBP.split_contiguous((0, 1), 1),), placement)
                local = fm.F.distributed.force_boxing(sentinel, split)
                computed = fm.F.math.add(local, local)
                outputs.append(fm.F.distributed.force_boxing(computed, kind))
                parameters.append(sentinel)
            self.function("main", parameters, outputs)

    return Graph(
        dialect="distributed" if placement is not None else "high_level",
        stage="frozen_constants" if placement is not None else "imported",
        entry="main",
        metadata={} if placement is None else {"auto_distribution": {"placement": placement.to_data()}},
    ).build()


@pytest.mark.parametrize("direction", ["absorb_query", "expand_value"])
@pytest.mark.parametrize("lanes", [(), (8,), (2, 4)])
@pytest.mark.parametrize("fp8", [False, True])
def test_projection_scalar_order_and_resume(tmp_path, direction, lanes, fp8):
    module = graph(direction, lanes, input_lanes=(2, 4), fp8=fp8)
    restored = fm.load_module(fm.emit_module(module, tmp_path / "projection.py"))
    assert restored.semantic_hash == module.semantic_hash
    width = 16 if direction == "absorb_query" else 32
    x = ((torch.arange(4 * width) % 13) - 6).float().mul_(0.125).bfloat16().reshape(2, 2, width)
    w = ((torch.arange(48 * 32) % 17) - 8).float().mul_(0.125).bfloat16().reshape(48, 32)
    values = {"x": x.reshape(2, 2, width // 8, 2, 4), "w": w}
    if fp8:
        values.update(w=w.to(torch.float8_e4m3fn), scale=torch.full((6, 2), 0.5))
        w = (values["w"].float() * 0.5).bfloat16()
    selected = w.reshape(2, 24, 32)
    selected = selected[:, :16] if direction == "absorb_query" else selected[:, 16:].transpose(1, 2)
    expected = torch.bmm(x.transpose(0, 1).float(), selected.float()).bfloat16().transpose(0, 1)
    (actual,) = TorchEvaluator(DictWeightResolver({})).run(restored, values)
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"direction": "wrong"},
        {"lanes": (3,)},
        {"latent_dim": 16},
        {"qk_head_dim": 8},
        {"weight_block_n": 0},
        {"lanes": (True,)},
    ],
)
def test_invalid_projection_geometry_is_rejected(kwargs):
    with pytest.raises(IRSchemaError, match="MLA projection"):
        graph(**kwargs)


def test_named_projection_pattern_preserves_direction_and_operand_identity():
    from triton.flagmega import pattern_match as pm
    from triton.flagmega.ir.ops.nn.mla_projection import MlaProjection

    source = graph(lanes=(2, 4))
    node = next(n for n in source.nodes if n.op == "nn.mla_projection")
    weight = pm.wildcard("weight")
    pattern = pm.F.nn.is_mla_projection(weight=weight, direction="absorb_query", output_lanes=(2, 4))
    matched = pm.try_match_root(node, pattern, source)
    assert matched is not None and matched["weight"].id == node.inputs[1]
    assert pattern[MlaProjection.weight] is weight
    assert pm.try_match_root(node, pm.F.nn.is_mla_projection(direction="expand_value"), source) is None


@pytest.mark.parametrize("tokens, backend, message", [(2, "nvidia", "single-token"), (1, "iluvatar", "SM90")])
def test_candidate_rejects_unimplemented_device_contracts(tokens, backend, message):
    from triton.flagmega.codegen.triton.candidates.core import TritonCandidateContext
    from triton.flagmega.errors import CodegenError
    from triton.flagmega.targets import NvidiaSm90Target
    from triton.flagmega.targets.iluvatar_bi_v150 import IluvatarBiV150Target
    from triton.flagmega.targets.nvidia.mla import NvidiaMlaProjectionSemanticTIRCandidateProvider

    source = graph(tokens=tokens)
    target = NvidiaSm90Target() if backend == "nvidia" else IluvatarBiV150Target()
    context = TritonCandidateContext(source, target, {}, {}, {}, frozenset(), True, target.triton_implementation_model)
    node = next(n for n in source.nodes if n.op == "nn.mla_projection")
    with pytest.raises(CodegenError, match=message):
        NvidiaMlaProjectionSemanticTIRCandidateProvider().propose(node, context)
