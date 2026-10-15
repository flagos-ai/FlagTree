# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The mHC pre/post stages match the serving-runtime reference contract.

``_reference_pre`` and ``_reference_post`` transcribe vLLM's
``model_executor/kernels/mhc/torch.py`` (``mhc_pre_torch`` / ``mhc_post_torch``),
which is the numerical contract the tilelang and triton kernels are validated
against there.  They are kept here verbatim rather than paraphrased so a drift
in either implementation shows up as a test failure instead of a judgement call.
"""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.mhc_post import MhcPost
from triton.flagmega.ir.ops.nn.mhc_pre import MhcPre

B = fm.SBP.broadcast()
MESH = fm.Placement((2, 2), "yx", "bb")

TOKENS, HC, HIDDEN = 3, 4, 16
COEFFICIENTS = (2 + HC) * HC
RMS_EPS, HC_EPS, POST_MULT, SINKHORN = 1e-5, 1e-6, 2.0, 20


def _reference_pre(residual, fn, hc_scale, hc_base, iterations=SINKHORN):
    """vLLM ``mhc_pre_torch``, minus the shape bookkeeping around it.

    ``iterations`` is exposed only so a caller can compare at a count where the
    Sinkhorn sweeps have not yet converged; vLLM's own value is ``SINKHORN``.
    """
    hc_mult, hidden = residual.shape[-2], residual.shape[-1]
    x = residual.view(-1, hc_mult * hidden).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden) + RMS_EPS)
    pre_mix = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + HC_EPS
    post_mix = torch.sigmoid(mixes[:, hc_mult:2 * hc_mult] * hc_scale[1]
                             + hc_base[hc_mult:2 * hc_mult]) * POST_MULT
    comb_logits = mixes[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult) * hc_scale[2] \
        + hc_base[2 * hc_mult:].view(1, hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_logits, dim=-1) + HC_EPS
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + HC_EPS)
    for _ in range(iterations - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + HC_EPS)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + HC_EPS)
    layer_input = torch.sum(pre_mix.unsqueeze(-1) * residual.to(torch.float32),
                            dim=1).to(residual.dtype)
    return post_mix, comb_mix, layer_input


def _reference_post(x, residual, post_layer_mix, comb_res_mix):
    """vLLM ``mhc_post_torch``; ``post_layer_mix`` arrives as ``(T, hc, 1)``."""
    mixed_residual = torch.einsum("...ij,...ih->...jh", comb_res_mix.to(torch.float32),
                                 residual.to(torch.float32))
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return (mixed_residual + post_term).to(residual.dtype)


def _operands(generator=None):
    generator = generator or torch.Generator().manual_seed(7)

    def normal(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=generator).to(dtype)

    return {"streams": normal(TOKENS, HC, HIDDEN, dtype=torch.bfloat16),
            "projection": normal(COEFFICIENTS, HC * HIDDEN),
            "base": normal(COEFFICIENTS),
            "scale": normal(3),
            "value": normal(TOKENS, HIDDEN, dtype=torch.bfloat16)}


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def _pre_types(streams=None):
    return (_node("streams", streams or fm.tensor_type("bfloat16", (TOKENS, HC, HIDDEN))),
            _node("projection", fm.tensor_type("float32", (COEFFICIENTS, HC * HIDDEN))),
            _node("base", fm.tensor_type("float32", (COEFFICIENTS, ))),
            _node("scale", fm.tensor_type("float32", (3, ))))


def _pre_attrs(**overrides):
    return dict({"epsilon": RMS_EPS, "sinkhorn_iterations": SINKHORN,
                 "sinkhorn_epsilon": HC_EPS, "post_multiplier": POST_MULT}, **overrides)


class _PreModule(fm.Module):
    def __init__(self, output_lanes=(), stream_dtype="bfloat16"):
        super().__init__(dialect="nn", stage="imported", entry="main")
        self.output_lanes, self.stream_dtype = output_lanes, stream_dtype

    def forward(self):
        streams = self.input("streams", fm.tensor_type(self.stream_dtype, (TOKENS, HC, HIDDEN)))
        projection = self.input("projection", fm.tensor_type("float32", (COEFFICIENTS, HC * HIDDEN)))
        base = self.input("base", fm.tensor_type("float32", (COEFFICIENTS, )))
        scale = self.input("scale", fm.tensor_type("float32", (3, )))
        stage = fm.F.nn.mhc_pre(streams, projection, base, scale, name="pre",
                               output_lanes=self.output_lanes, **_pre_attrs())
        items = fm.F.tensors.get_items(stage, 0, 1, 2, name_prefix="pre")
        self.function("main", (streams, projection, base, scale), items)


class _RoundTripModule(fm.Module):
    """Pre feeds Post, the way a decoder block wires the two stages."""

    def __init__(self):
        super().__init__(dialect="nn", stage="imported", entry="main")

    def forward(self):
        streams = self.input("streams", fm.tensor_type("bfloat16", (TOKENS, HC, HIDDEN)))
        projection = self.input("projection", fm.tensor_type("float32", (COEFFICIENTS, HC * HIDDEN)))
        base = self.input("base", fm.tensor_type("float32", (COEFFICIENTS, )))
        scale = self.input("scale", fm.tensor_type("float32", (3, )))
        value = self.input("value", fm.tensor_type("bfloat16", (TOKENS, HIDDEN)))
        stage = fm.F.nn.mhc_pre(streams, projection, base, scale, name="pre", **_pre_attrs())
        post, comb, collapsed = fm.F.tensors.get_items(stage, 0, 1, 2, name_prefix="pre")
        output = fm.F.nn.mhc_post(value, streams, post, comb, name="post")
        self.function("main", (streams, projection, base, scale, value), (collapsed, output))


def test_pre_emits_the_coefficient_and_collapsed_stream_contract():
    result = MhcPre.infer_type(_pre_types(), _pre_attrs())
    assert result == fm.TupleType((fm.tensor_type("float32", (TOKENS, HC)),
                                   fm.tensor_type("float32", (TOKENS, HC, HC)),
                                   fm.tensor_type("bfloat16", (TOKENS, HIDDEN))))


@pytest.mark.parametrize("lanes", [(8,), (2, 4), (4, 2)])
def test_pre_infers_packing_only_for_the_collapsed_result(lanes):
    result = MhcPre.infer_type(_pre_types(), _pre_attrs(output_lanes=lanes))
    assert result.fields[:2] == MhcPre.infer_type(_pre_types(), _pre_attrs()).fields[:2]
    assert result.fields[2] == fm.tensor_type(fm.vector_type("bfloat16", lanes), (TOKENS, HIDDEN // 8))


@pytest.mark.parametrize("lanes", [(8,), (2, 4), (4, 2), (1,)])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_pre_packed_evaluation_preserves_scalar_order_and_rounding(lanes, dtype):
    values = {name: value for name, value in _operands().items() if name != "value"}
    values["streams"] = values["streams"].to(getattr(torch, dtype))
    evaluator = TorchEvaluator(DictWeightResolver({}))
    scalar = evaluator.run(_PreModule(stream_dtype=dtype).build(), values)
    packed = evaluator.run(_PreModule(lanes, stream_dtype=dtype).build(), values)
    for actual, expected in zip(packed, scalar):
        torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=0, atol=0)


@pytest.mark.parametrize("lanes", [(0,), (-2,), (True,), (2.0,), None, 8])
def test_pre_rejects_invalid_output_lanes(lanes):
    with pytest.raises(IRSchemaError, match="output_lanes.*positive integers"):
        MhcPre.normalize_attrs(_pre_attrs(output_lanes=lanes))


def test_pre_rejects_output_lanes_that_do_not_divide_hidden():
    with pytest.raises(IRSchemaError, match="hidden must be divisible"):
        MhcPre.prepare(_pre_types(), _pre_attrs(output_lanes=(3,)))


@pytest.mark.parametrize("index", range(4))
def test_pre_does_not_claim_packed_input_support(index):
    inputs = list(_pre_types())
    operand = inputs[index]
    inputs[index] = _node(operand.id, fm.tensor_type(fm.vector_type(operand.type.dtype, (2,)), operand.type.shape))
    with pytest.raises(IRSchemaError, match=f"{operand.id} must use a scalar element type"):
        MhcPre.prepare(inputs, _pre_attrs(output_lanes=(8,)))


@pytest.mark.parametrize("lanes", [(), (8,), (2, 4)])
def test_pre_output_layout_round_trips_and_keeps_the_old_default_attrs(tmp_path, lanes):
    from triton.flagmega.ir.python_ir import emit_module, load_module

    module = _PreModule(lanes).build()
    path = emit_module(module, tmp_path / "pre.py")
    assert load_module(path).semantic_hash == module.semantic_hash
    pre = module.node_map["pre"]
    assert pre.attrs.get("output_lanes", ()) == lanes
    if not lanes:
        assert dict(pre.attrs) == _pre_attrs()
    else:
        assert MhcPre.cost(pre).bytes_written == MhcPre.cost(_PreModule().build().node_map["pre"]).bytes_written


def test_pre_packed_distributed_output_keeps_coefficients_and_broadcast_policies():
    inputs = tuple(_node(node.id, fm.DistributedType(node.type, tuple(B for _ in node.type.shape), MESH))
                   for node in _pre_types())
    result = MhcPre.prepare(inputs, _pre_attrs(output_lanes=(2, 4))).result_type
    expected = MhcPre.prepare(_pre_types(), _pre_attrs(output_lanes=(2, 4))).result_type
    for field, tensor in zip(result.fields, expected.fields):
        assert field == fm.DistributedType(tensor, tuple(B for _ in tensor.shape), MESH)


@pytest.mark.parametrize("attrs, message", [
    ({"sinkhorn_iterations": 0}, "positive integer"),
    ({"sinkhorn_iterations": 2.0}, "positive integer"),
    ({"epsilon": 0.0}, "must be positive"),
    ({"sinkhorn_epsilon": -1e-6}, "must be positive"),
])
def test_pre_rejects_out_of_contract_attributes(attrs, message):
    with pytest.raises(IRSchemaError, match=message):
        MhcPre.normalize_attrs(_pre_attrs(**attrs))


def test_pre_rejects_a_stream_split_it_cannot_honour():
    """The RMS term reduces over all streams, so a split stream axis needs Boxing."""
    streams = fm.DistributedType(fm.tensor_type("bfloat16", (TOKENS, HC, HIDDEN)),
                                (B, fm.SBP.split_contiguous((0, )), B), MESH)
    inputs = list(_pre_types(streams))
    for index in range(1, 4):
        tensor = inputs[index].type
        inputs[index] = _node(inputs[index].id,
                              fm.DistributedType(tensor, tuple(B for _ in tensor.shape), MESH))
    with pytest.raises(IRSchemaError, match="broadcast streams"):
        MhcPre.infer_type(tuple(inputs), _pre_attrs())


def test_pre_matches_the_serving_runtime_reference():
    operands = _operands()
    actual = TorchEvaluator(DictWeightResolver({})).run(
        _PreModule().build(), {name: operands[name]
                               for name in ("streams", "projection", "base", "scale")})
    expected = _reference_pre(operands["streams"], operands["projection"],
                              operands["scale"], operands["base"])
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


class _Fp32PreModule(_PreModule):
    """``_PreModule`` with FP32 streams, so the collapse keeps its own precision.

    ``test_pre_matches_the_serving_runtime_reference`` runs bf16 streams, whose
    8-bit mantissa rounds away any coefficient error below ~4e-3 -- it passed
    while ``pre_mix`` was adding the wrong epsilon. FP32 streams make the
    comparison load-bearing.
    """

    def forward(self):
        streams = self.input("streams", fm.tensor_type("float32", (TOKENS, HC, HIDDEN)))
        projection = self.input("projection", fm.tensor_type("float32", (COEFFICIENTS, HC * HIDDEN)))
        base = self.input("base", fm.tensor_type("float32", (COEFFICIENTS, )))
        scale = self.input("scale", fm.tensor_type("float32", (3, )))
        stage = fm.F.nn.mhc_pre(streams, projection, base, scale, name="pre", **_pre_attrs())
        items = fm.F.tensors.get_items(stage, 0, 1, 2, name_prefix="pre")
        self.function("main", (streams, projection, base, scale), items)


def test_pre_adds_the_hc_epsilon_to_pre_mix_not_the_rms_epsilon():
    """vLLM adds ``hc_eps`` to ``pre_mix``; the RMS ``epsilon`` is a different knob.

    GLM-5.3's published config sets ``hc_eps=1e-6`` while the RMS epsilon is
    ``1e-5``, so confusing the two shifts every stream coefficient by 9e-6 and
    the collapsed residual by ~4e-5.
    """
    operands = _operands()
    streams = operands["streams"].float()
    actual = TorchEvaluator(DictWeightResolver({})).run(
        _Fp32PreModule().build(),
        {"streams": streams, **{name: operands[name]
                                for name in ("projection", "base", "scale")}})
    expected = _reference_pre(streams, operands["projection"], operands["scale"],
                              operands["base"])
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_pre_comb_is_doubly_stochastic_in_the_direction_normalized_last():
    operands = _operands()
    _, comb, _ = TorchEvaluator(DictWeightResolver({})).run(
        _PreModule().build(), {name: operands[name]
                               for name in ("streams", "projection", "base", "scale")})
    # Sinkhorn ends on the column direction, so columns are the tight one.
    torch.testing.assert_close(comb.sum(dim=-2), torch.ones(TOKENS, HC), rtol=0, atol=1e-5)
    torch.testing.assert_close(comb.sum(dim=-1), torch.ones(TOKENS, HC), rtol=0, atol=5e-2)


def test_post_requires_float32_coefficients_from_the_pre_stage():
    inputs = (_node("value", fm.tensor_type("bfloat16", (TOKENS, HIDDEN))),
              _node("residual", fm.tensor_type("bfloat16", (TOKENS, HC, HIDDEN))),
              _node("post", fm.tensor_type("bfloat16", (TOKENS, HC))),
              _node("comb", fm.tensor_type("float32", (TOKENS, HC, HC))))
    with pytest.raises(IRSchemaError, match="post must be FP32"):
        MhcPost.infer_type(inputs, {})


def test_pre_and_post_round_trip_matches_the_serving_runtime_reference():
    operands = _operands()
    collapsed, output = TorchEvaluator(DictWeightResolver({})).run(
        _RoundTripModule().build(), operands)
    post, comb, reference_collapsed = _reference_pre(
        operands["streams"], operands["projection"], operands["scale"], operands["base"])
    torch.testing.assert_close(collapsed, reference_collapsed, rtol=0, atol=0)
    torch.testing.assert_close(
        output,
        _reference_post(operands["value"], operands["streams"], post.unsqueeze(-1), comb),
        rtol=0, atol=0)
