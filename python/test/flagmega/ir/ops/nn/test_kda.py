# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""KDA type contracts, including the distributed broadcast boundary."""

from pathlib import Path

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.kda import KdaConvolution, KdaRecurrentCore
from triton.flagmega.ir.ops.nn.staged_routing import StagedRouting


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def _broadcast(tensor, placement):
    return fm.DistributedType(tensor, (fm.SBP.broadcast(),) * tensor.rank, placement)


def test_kda_convolution_preserves_one_common_broadcast_placement():
    placement = fm.Placement((4,), "x", "b")
    projection = _broadcast(fm.tensor_type("bfloat16", (1, 2, 2)), placement)
    types = (
        projection, projection, projection,
        _broadcast(fm.tensor_type("bfloat16", (3, 2, 2, 3)), placement),
        _broadcast(fm.tensor_type("bfloat16", (3, 2, 2, 4)), placement),
    )
    result = KdaConvolution.infer_type(tuple(_node(str(i), value) for i, value in enumerate(types)),
                                       {"kernel_size": 4, "activation": "silu"})

    assert len(result.fields) == 4
    assert all(isinstance(field, fm.DistributedType) for field in result.fields)
    assert all(field.placement == placement for field in result.fields)


def test_kda_convolution_evaluator_matches_causal_reference():
    class Graph(fm.Module):
        def forward(self):
            parts = tuple(self.input(name, fm.tensor_type("bfloat16", (1, 2, 2)), id=name)
                          for name in ("q", "k", "v"))
            state = self.input("state", fm.tensor_type("bfloat16", (3, 2, 2, 3)), id="state")
            weight = self.input("weight", fm.tensor_type("bfloat16", (3, 2, 2, 4)), id="weight")
            values = fm.F.nn.kda_convolution(*parts, state, weight, kernel_size=4)
            items = fm.F.tensors.get_items(values, 0, 1, 2, 3, name_prefix="conv")
            self.function("main", (*parts, state, weight), items)

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    generator = torch.Generator().manual_seed(4)
    parts = {name: torch.randn((1, 2, 2), generator=generator).bfloat16() for name in ("q", "k", "v")}
    state = torch.randn((3, 2, 2, 3), generator=generator).bfloat16()
    weight = torch.randn((3, 2, 2, 4), generator=generator).bfloat16()
    outputs = TorchEvaluator(DictWeightResolver({})).run(
        module, {**parts, "state": state, "weight": weight})

    current = torch.stack([parts[name].float().squeeze(0) for name in ("q", "k", "v")])
    expected = torch.nn.functional.silu(
        (state.float() * weight[..., :3].float()).sum(dim=-1) + weight[..., 3].float() * current
    ).bfloat16()
    for part in range(3):
        torch.testing.assert_close(outputs[part], expected[part].unsqueeze(0))
    torch.testing.assert_close(
        outputs[3], torch.cat((state[..., 1:], current.unsqueeze(-1).bfloat16()), dim=-1))


def test_kda_convolution_rejects_a_flat_interleaved_channel_axis():
    """The flat ``[3*heads*head_dim, kernel]`` layout is what broke head splits."""

    projection = fm.tensor_type("bfloat16", (1, 12))
    types = (projection, projection, projection,
             fm.tensor_type("bfloat16", (12, 3)), fm.tensor_type("bfloat16", (12, 4)))
    with pytest.raises(IRSchemaError, match="rank-3"):
        KdaConvolution.infer_type(tuple(_node(str(i), value) for i, value in enumerate(types)),
                                  {"kernel_size": 4, "activation": "silu"})


def test_kda_recurrent_evaluator_matches_delta_reference():
    class Graph(fm.Module):
        def forward(self):
            q = self.input("q", fm.tensor_type("bfloat16", (1, 2, 4)), id="q")
            k = self.input("k", fm.tensor_type("bfloat16", (1, 2, 4)), id="k")
            value = self.input("value", fm.tensor_type("bfloat16", (1, 2, 4)), id="value")
            decay = self.input("decay", fm.tensor_type("float32", (1, 2, 4)), id="decay")
            beta = self.input("beta", fm.tensor_type("float32", (1, 2)), id="beta")
            state = self.input("state", fm.tensor_type("float32", (1, 2, 4, 4)), id="state")
            pair = fm.F.nn.kda_recurrent_core(q, k, value, decay, beta, state)
            output, new_state = fm.F.tensors.get_items(pair, 0, 1, name_prefix="recurrent")
            self.function("main", (q, k, value, decay, beta, state), (output, new_state))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    generator = torch.Generator().manual_seed(5)
    q = torch.randn((1, 2, 4), generator=generator).bfloat16()
    k = torch.randn((1, 2, 4), generator=generator).bfloat16()
    value = torch.randn((1, 2, 4), generator=generator).bfloat16()
    decay = torch.randn((1, 2, 4), generator=generator)
    beta = torch.sigmoid(torch.randn((1, 2), generator=generator))
    state = torch.randn((1, 2, 4, 4), generator=generator)
    actual, actual_state = TorchEvaluator(DictWeightResolver({})).run(
        module, {"q": q, "k": k, "value": value, "decay": decay, "beta": beta, "state": state})
    qf, kf = q.float(), k.float()
    qf = qf / torch.sqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
    kf = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    expected_state = state * decay.exp().unsqueeze(-1)
    correction = (expected_state * kf.unsqueeze(-1)).sum(-2)
    update = (value.float() - correction) * beta.unsqueeze(-1)
    expected_state = expected_state + kf.unsqueeze(-1) * update.unsqueeze(-2)
    expected = (expected_state * qf.unsqueeze(-1)).sum(-2).bfloat16()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state, expected_state)


def test_kda_recurrent_evaluator_honors_disabled_l2_normalization():
    class Graph(fm.Module):
        def forward(self):
            q = self.input("q", fm.tensor_type("bfloat16", (1, 1, 2)), id="q")
            k = self.input("k", fm.tensor_type("bfloat16", (1, 1, 2)), id="k")
            value = self.input("value", fm.tensor_type("bfloat16", (1, 1, 2)), id="value")
            decay = self.input("decay", fm.tensor_type("float32", (1, 1, 2)), id="decay")
            beta = self.input("beta", fm.tensor_type("float32", (1, 1)), id="beta")
            state = self.input("state", fm.tensor_type("float32", (1, 1, 2, 2)), id="state")
            pair = fm.F.nn.kda_recurrent_core(q, k, value, decay, beta, state,
                                              l2norm_in_kernel=False)
            output, new_state = fm.F.tensors.get_items(pair, 0, 1, name_prefix="recurrent")
            self.function("main", (q, k, value, decay, beta, state), (output, new_state))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    q = torch.tensor([[[2.0, 0.5]]], dtype=torch.bfloat16)
    k = torch.tensor([[[0.25, 1.5]]], dtype=torch.bfloat16)
    value = torch.tensor([[[1.0, -2.0]]], dtype=torch.bfloat16)
    decay = torch.zeros((1, 1, 2))
    beta = torch.ones((1, 1))
    state = torch.zeros((1, 1, 2, 2))
    actual, actual_state = TorchEvaluator(DictWeightResolver({})).run(
        module, {"q": q, "k": k, "value": value, "decay": decay, "beta": beta, "state": state})

    expected_state = k.float().unsqueeze(-1) * value.float().unsqueeze(-2)
    expected = (expected_state * q.float().unsqueeze(-1)).sum(-2).bfloat16()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state, expected_state)


def test_kda_recurrent_rejects_mixed_or_mismatched_distributed_inputs():
    placement = fm.Placement((4,), "x", "b")
    other = fm.Placement((2,), "x", "b")
    q = fm.tensor_type("bfloat16", (1, 2, 4))
    beta = fm.tensor_type("float32", (1, 2))
    state = fm.tensor_type("float32", (1, 2, 4, 3))
    plain = (q, q, q, fm.tensor_type("float32", (1, 2, 4)), beta, state)
    with pytest.raises(IRSchemaError, match="all plain or all distributed"):
        KdaRecurrentCore.infer_type(tuple(_node(str(i), value) for i, value in enumerate(
            (*plain[:1], _broadcast(plain[1], placement), *plain[2:]))),
                                    {"epsilon": 1e-6, "l2norm_in_kernel": True})

    mismatched = tuple(_broadcast(value, other if index == 1 else placement)
                       for index, value in enumerate(plain))
    with pytest.raises(IRSchemaError, match="one placement"):
        KdaRecurrentCore.infer_type(tuple(_node(str(i), value) for i, value in enumerate(mismatched)),
                                    {"epsilon": 1e-6, "l2norm_in_kernel": True})


def test_staged_routing_rejects_distributed_bias_for_plain_logits():
    placement = fm.Placement((4,), "x", "b")
    logits = _node("logits", fm.tensor_type("float32", (1, 8)))
    bias = _node("bias", _broadcast(fm.tensor_type("float32", (8,)), placement))
    with pytest.raises(IRSchemaError, match="same distribution contract"):
        StagedRouting.infer_type((logits, bias), {
            "k": 2,
            "axis": -1,
            "index_dtype": "int32",
            "scoring": "sigmoid",
            "normalize": True,
            "scaling": 1.0,
        })


def test_kda_broadcast_codegen_does_not_write_partial_head_or_channel_slices():
    """Broadcast outputs require one canonical writer for every element."""
    root = Path(__file__).parents[5] / "triton" / "flagmega" / "codegen" / "triton" / "kernels"
    convolution = (root / "kda_convolution" / "local.py.jinja").read_text()
    recurrent = (root / "kda_recurrent_core" / "local.py.jinja").read_text()

    assert "shard_index" not in convolution
    assert "shard_index" not in recurrent
    assert "part.result_writer_active" in convolution
    assert "out_writer_active" in recurrent
    assert "if call.l2norm_in_kernel" in recurrent
