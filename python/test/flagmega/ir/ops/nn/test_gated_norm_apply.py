# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""``nn.gated_norm_apply`` matches the serving-runtime ``FusedRMSNormGated``.

``_reference`` transcribes vLLM's
``third_party/flash_linear_attention/ops/kda.py::FusedRMSNormGated.forward_native``,
which is the contract its triton ``rms_norm_gated`` kernel is validated against.
It is kept verbatim rather than paraphrased so drift on either side fails a test
instead of becoming a judgement call.

The point of the fused op is the FP32 interior: an expansion into
``norm_apply`` + ``cast`` + ``sigmoid`` + ``mul`` + ``cast`` rounds the gate's
sigmoid to the gate's own dtype, because every materialized IR node publishes a
dtype.  ``test_the_gate_activation_does_not_round_to_the_gate_dtype`` pins that
difference with a gate value chosen so BF16 rounding is observable.
"""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.gated_norm_apply import GatedNormApply

B = fm.SBP.broadcast()
MESH = fm.Placement((2, 2), "yx", "bb")

TOKENS, HEADS, DIM = 2, 4, 8
EPS = 1e-5


def _reference(x, g, weight, activation="sigmoid"):
    """vLLM ``FusedRMSNormGated.forward_native``, no-residual branch."""
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + EPS)
    if weight is not None:
        x_normed = x_normed * weight.float()
    g_float = g.float()
    if activation in ("swish", "silu"):
        out = x_normed * g_float * torch.sigmoid(g_float)
    else:  # sigmoid
        out = x_normed * torch.sigmoid(g_float)
    return out.to(x.dtype)


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def _attrs(**overrides):
    return dict({"axis": -1, "epsilon": EPS, "use_mean": False, "activation": "sigmoid"},
                **overrides)


def _types(value=None, gate=None, stats=None):
    return (
        _node("value", value or fm.tensor_type("float32", (TOKENS, HEADS, DIM))),
        _node("stats", stats or fm.tensor_type("float32", (1, TOKENS, HEADS, 1))),
        _node("scale", fm.tensor_type("float32", (DIM, ))),
        _node("bias", fm.tensor_type("float32", (DIM, ))),
        _node("gate", gate or fm.tensor_type("bfloat16", (TOKENS, HEADS, DIM))),
    )


def _distributed(value_type, policies):
    return fm.DistributedType(value_type, policies, MESH)


class _Module(fm.Module):
    """``norm_stats`` then the fused gated apply, as KDA's ``o_norm`` wires it."""

    def __init__(self, *, activation="sigmoid", output_dtype="bfloat16", core_dtype="bfloat16",
                 gate_dtype="bfloat16"):
        super().__init__(dialect="nn", stage="imported", entry="main")
        self._activation = activation
        self._output_dtype = output_dtype
        self._core_dtype = core_dtype
        self._gate_dtype = gate_dtype

    def forward(self):
        core = self.input("core", fm.tensor_type(self._core_dtype, (TOKENS, HEADS, DIM)))
        weight = self.input("weight", fm.tensor_type("float32", (DIM, )))
        gate = self.input("gate", fm.tensor_type(self._gate_dtype, (TOKENS, HEADS, DIM)))
        stats = fm.F.nn.norm_stats(core, axis=-1, use_mean=False, name="stats")
        zero = fm.F.builtin.splat_const(weight.type, 0.0)
        output = fm.F.nn.gated_norm_apply(
            core, stats, weight, zero, gate, axis=-1, epsilon=EPS, use_mean=False,
            activation=self._activation, output_dtype=self._output_dtype, name="gated")
        self.function("main", (core, weight, gate), (output, ))


class _ExpandedModule(fm.Module):
    """The pre-fusion expansion: norm_apply, then cast/sigmoid/mul/cast."""

    def __init__(self, *, core_dtype="bfloat16", gate_dtype="bfloat16"):
        super().__init__(dialect="nn", stage="imported", entry="main")
        self._core_dtype = core_dtype
        self._gate_dtype = gate_dtype

    def forward(self):
        core = self.input("core", fm.tensor_type(self._core_dtype, (TOKENS, HEADS, DIM)))
        weight = self.input("weight", fm.tensor_type("float32", (DIM, )))
        gate = self.input("gate", fm.tensor_type(self._gate_dtype, (TOKENS, HEADS, DIM)))
        stats = fm.F.nn.norm_stats(core, axis=-1, use_mean=False, name="stats")
        zero = fm.F.builtin.splat_const(weight.type, 0.0)
        normalized = fm.F.nn.norm_apply(core, stats, weight, zero, axis=-1, epsilon=EPS,
                                        use_mean=False, output_dtype="float32", name="o_norm")
        # The gate's sigmoid rounds to the gate dtype here: that is the defect.
        activated = fm.F.math.sigmoid(gate, name="out_gate")
        wide = fm.F.tensors.cast(activated, "float32", name="out_gate_wide")
        gated = fm.F.math.mul(normalized, wide, name="out_gated")
        output = fm.F.tensors.cast(gated, "bfloat16", name="out_wide")
        self.function("main", (core, weight, gate), (output, ))


def _operands(*, core_dtype=torch.bfloat16, gate_dtype=torch.bfloat16, seed=11):
    generator = torch.Generator().manual_seed(seed)
    return {
        "core": torch.randn(TOKENS, HEADS, DIM, generator=generator).to(core_dtype),
        "weight": torch.randn(DIM, generator=generator),
        "gate": torch.randn(TOKENS, HEADS, DIM, generator=generator).to(gate_dtype),
    }


def _run(module, operands):
    return TorchEvaluator(DictWeightResolver({})).run(module.build(), operands)[0]


def _reference_bf16(operands, activation="sigmoid"):
    """The reference, rounded once at the declared BF16 output boundary."""
    return _reference(operands["core"], operands["gate"], operands["weight"],
                      activation=activation).to(torch.bfloat16)


def test_the_result_keeps_the_input_shape_and_declared_output_dtype():
    result = GatedNormApply.infer_type(_types(), _attrs(output_dtype="bfloat16"))
    assert result == fm.tensor_type("bfloat16", (TOKENS, HEADS, DIM))


@pytest.mark.parametrize("attrs, message", [
    ({"epsilon": 0.0}, "must be positive"),
    ({"activation": "gelu"}, "sigmoid, silu or swish"),
    ({"axis": 1.5}, "must be an integer"),
    ({"output_dtype": "float8_e4m3fn"}, "bfloat16 or float32"),
])
def test_out_of_contract_attributes_are_rejected(attrs, message):
    with pytest.raises(IRSchemaError, match=message):
        GatedNormApply.normalize_attrs(_attrs(**attrs))


def test_a_gate_that_does_not_cover_the_input_is_rejected():
    gate = fm.tensor_type("bfloat16", (TOKENS, HEADS, 1))
    with pytest.raises(IRSchemaError, match="gate shape must equal"):
        GatedNormApply.infer_type(_types(gate=gate), _attrs())


def test_a_gate_distributed_unlike_the_input_needs_boxing():
    value = fm.tensor_type("float32", (TOKENS, HEADS, DIM))
    inputs = list(_types())
    split = (B, fm.SBP.split_contiguous((0, )), B)
    inputs[0] = _node("value", _distributed(value, split))
    inputs[1] = _node("stats", _distributed(
        fm.tensor_type("float32", (1, TOKENS, HEADS, 1)), (B, ) + split))
    for index in (2, 3):
        inputs[index] = _node(inputs[index].id, _distributed(inputs[index].type, (B, )))
    # The gate stays fully replicated while the value is head-split.
    inputs[4] = _node("gate", _distributed(
        fm.tensor_type("bfloat16", (TOKENS, HEADS, DIM)), (B, B, B)))
    with pytest.raises(IRSchemaError, match="gate policies must match"):
        GatedNormApply.infer_type(tuple(inputs), _attrs())


def test_a_head_split_value_keeps_its_owner_map():
    value = fm.tensor_type("float32", (TOKENS, HEADS, DIM))
    split = (B, fm.SBP.split_contiguous((0, )), B)
    inputs = list(_types())
    inputs[0] = _node("value", _distributed(value, split))
    inputs[1] = _node("stats", _distributed(
        fm.tensor_type("float32", (1, TOKENS, HEADS, 1)), (B, ) + split))
    for index in (2, 3):
        inputs[index] = _node(inputs[index].id, _distributed(inputs[index].type, (B, )))
    inputs[4] = _node("gate", _distributed(
        fm.tensor_type("bfloat16", (TOKENS, HEADS, DIM)), split))
    result = GatedNormApply.infer_type(tuple(inputs), _attrs(output_dtype="bfloat16"))
    assert result == _distributed(fm.tensor_type("bfloat16", (TOKENS, HEADS, DIM)), split)


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
def test_it_matches_the_serving_runtime_reference(activation):
    operands = _operands()
    actual = _run(_Module(activation=activation), operands)
    expected = _reference(operands["core"], operands["gate"], operands["weight"],
                          activation=activation)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_the_gate_activation_does_not_round_to_the_gate_dtype():
    """First defect the fusion removes: sigmoid on a BF16 node rounds its result.

    ``sigmoid(x)`` for these gate values lands strictly between adjacent BF16
    values, so the expansion's BF16-typed ``math.sigmoid`` node loses bits the
    reference keeps.  Isolated here with an FP32 core, so the only BF16 boundary
    in play is the gate's.
    """
    operands = _operands(core_dtype=torch.float32)
    reference = _reference_bf16(operands)
    fused = _run(_Module(core_dtype="float32"), operands)
    expanded = _run(_ExpandedModule(core_dtype="float32"), operands)
    torch.testing.assert_close(fused, reference, rtol=0, atol=0)
    assert not torch.equal(expanded, reference), (
        "the expansion is expected to drift; if it stopped drifting this test "
        "no longer pins the FP32 gate interior")


def test_the_normalized_value_does_not_round_to_the_input_dtype():
    """Second defect: ``norm_apply`` publishes in the input dtype first.

    ``norm_apply_value`` rounds to the input dtype before honouring
    ``output_dtype`` — a deliberate boundary for a standalone norm, but the
    reference's gated norm keeps ``x_normed`` in FP32 all the way to the
    multiply.  Isolated here with an FP32 gate, so the only BF16 boundary in
    play is the core's.
    """
    operands = _operands(gate_dtype=torch.float32)
    reference = _reference_bf16(operands)
    fused = _run(_Module(gate_dtype="float32"), operands)
    expanded = _run(_ExpandedModule(gate_dtype="float32"), operands)
    torch.testing.assert_close(fused, reference, rtol=0, atol=0)
    assert not torch.equal(expanded, reference), (
        "the expansion is expected to drift; if it stopped drifting this test "
        "no longer pins the FP32 normalized interior")


def test_an_all_fp32_boundary_leaves_nothing_for_the_fusion_to_recover():
    """Scope check: the drift is the two BF16 boundaries, not the fusion itself."""
    operands = _operands(core_dtype=torch.float32, gate_dtype=torch.float32)
    fused = _run(_Module(core_dtype="float32", gate_dtype="float32"), operands)
    expanded = _run(_ExpandedModule(core_dtype="float32", gate_dtype="float32"), operands)
    torch.testing.assert_close(fused, expanded, rtol=0, atol=0)
