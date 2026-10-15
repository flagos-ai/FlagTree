# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The KDA forget gate belongs inside the recurrence, not in the graph.

``Glm5NextTextForgetGate.forward`` is ``lower_bound * sigmoid(exp(A_log) * (g +
dt_bias))`` with an entirely FP32 interior, and ``beta`` is
``sigmoid(self.b_proj(h))``.  Expressed with ``nn.kda_recurrent_core``, which
takes a ready log-decay and a ready beta, that needs a cast, an add, a multiply
and two sigmoids — five materialized nodes per layer, each of which publishes a
dtype, plus a FP32 ``[1, H, D]`` intermediate.  ``nn.kda_gated_recurrent_core``
consumes the raw projections and the two weight maps instead, so the whole gate
stays in registers.  These tests pin both the node-count contract and the
bit-exact agreement with the published arithmetic.
"""

import pytest
import torch

from triton.flagmega.evaluator import CheckpointWeightResolver, TorchEvaluator
from triton.flagmega.importer.glm5_moe import Glm5MoeModelImporter
from triton.flagmega.ir.ops.nn._paged_attention_state import create_paged_attention_state

from python.test.flagmega.importer.glm5_moe.helpers import REDUCED, checkpoint, configuration

PREFIX = "decode_linear_attention_dense"
LOWER_BOUND = -5.0


def _imported(*, with_values=False):
    scale = dict(REDUCED)
    source = checkpoint(scale, with_values=with_values, full_model=True,
                        expert_layout="separate")
    importer = Glm5MoeModelImporter(source, block_size=8, num_blocks=3)
    return source, importer, importer.import_module(), configuration(scale)["text_config"]


def _region_nodes(module, prefix=PREFIX):
    return [node for node in module.nodes if node.id.startswith(prefix)]


def _trace(source, importer, module, text):
    inputs = {"input_ids": torch.tensor([11], dtype=torch.int32),
              "paged_attention_state": create_paged_attention_state(importer.paged_config)}
    heads, dim = text["kda_num_heads"], text["kda_head_dim"]
    generator = torch.Generator().manual_seed(23)
    for layer in importer.kda_layers:
        inputs[f"kda_conv_state_{layer}"] = torch.zeros(
            (3, heads, dim, text["kda_conv_kernel_size"] - 1), dtype=torch.bfloat16)
        # A non-zero incoming state so the decay actually scales something.
        inputs[f"kda_recurrent_state_{layer}"] = torch.randn(
            (1, heads, dim, dim), generator=generator)
    return TorchEvaluator(CheckpointWeightResolver(source)).run_with_trace(module, inputs)[1]


def test_the_recurrence_owns_the_gate():
    _, _, module, _ = _imported()
    fused = [node for node in module.nodes if node.op == "nn.kda_gated_recurrent_core"]

    # One per KDA region (dense-MLP and sparse-MLP), and the unfused operator,
    # whose contract forces the five-node expansion, is gone.
    assert len(fused) == 2
    assert not [node for node in module.nodes if node.op == "nn.kda_recurrent_core"]
    for node in fused:
        assert node.attrs["lower_bound"] == LOWER_BOUND
        assert node.attrs["l2norm_in_kernel"] is True


def test_the_expansion_leaves_no_arithmetic_between_the_projection_and_the_gate():
    """No cast/add/mul/sigmoid survives on the gate or the beta path."""
    _, _, module, _ = _imported()
    gate_path = {f"{PREFIX}_gate_heads", f"{PREFIX}_self_attn_b_proj_weight_apply"}
    consumers = [node for node in _region_nodes(module) if gate_path & set(node.inputs)]

    # Both raw projections have exactly one consumer, and it is the recurrence.
    assert [node.op for node in consumers] == ["nn.kda_gated_recurrent_core"]
    # The raw projections reach the operator in the model dtype, matching
    # upstream's BF16 boundary plus an in-kernel FP32 widen.  The graph no
    # longer requests output_dtype="float32" on either projection.
    for name in (f"{PREFIX}_gate_heads", f"{PREFIX}_self_attn_b_proj_weight_apply"):
        node = next(item for item in module.nodes if item.id == name)
        assert node.type.dtype.name.lower() == "bfloat16"
    # The FP32 weight maps come from the constant island, not runtime nodes.
    for name in ("decay_rate_map", "dt_bias_map"):
        node = next(item for item in module.nodes if item.id == f"{PREFIX}_self_attn_{name}")
        assert node.op == "builtin.var"
        assert node.type.dtype.name.lower() == "float32"


def test_no_materialized_gate_intermediate_remains():
    """The FP32 [1, H, D] gate buffer the expansion needed is gone."""
    _, _, module, text = _imported()
    heads, dim = text["kda_num_heads"], text["kda_head_dim"]
    wide_activations = [
        node.id for node in _region_nodes(module)
        if node.op in {"math.add", "math.mul", "math.sigmoid", "tensors.cast"}
        and getattr(node.type, "shape", None) is not None
        and tuple(getattr(extent, "fixed_value", None) for extent in node.type.shape)
           == (1, heads, dim)
    ]

    assert wide_activations == []


def test_the_fused_gate_matches_the_published_forget_gate_bit_for_bit():
    source, importer, module, text = _imported(with_values=True)
    trace = _trace(source, importer, module, text)
    gate_raw = trace[f"{PREFIX}_gate_heads"]
    beta_raw = trace[f"{PREFIX}_self_attn_b_proj_weight_apply"]
    decay_rate = trace[f"{PREFIX}_self_attn_decay_rate_map"]
    dt_bias = trace[f"{PREFIX}_self_attn_dt_bias_map"]
    # q/k/v are the convolution's own head-major results now, not slices of a
    # flat channel axis.
    query, key = trace[f"{PREFIX}_conv_0"], trace[f"{PREFIX}_conv_1"]
    value, state = trace[f"{PREFIX}_conv_2"], trace[f"{PREFIX}_state_1"]
    observed, observed_state = trace[f"{PREFIX}_recurrent_0"], trace[f"{PREFIX}_recurrent_1"]

    # Glm5NextTextForgetGate.forward, then sigmoid(b_proj(h)), then the naive
    # KDA recurrence with L2-normalized q/k.
    gate = LOWER_BOUND * torch.sigmoid(decay_rate * (gate_raw.float() + dt_bias))
    beta = torch.sigmoid(beta_raw.float())
    normalized = tuple(
        tensor.float() / torch.sqrt((tensor.float() ** 2).sum(-1, keepdim=True) + 1e-6)
        for tensor in (query, key))
    expected_state = state * gate.exp().unsqueeze(-1)
    delta = (value.float() - (expected_state * normalized[1].unsqueeze(-1)).sum(-2)) \
        * beta.unsqueeze(-1)
    expected_state = expected_state + normalized[1].unsqueeze(-1) * delta.unsqueeze(-2)
    expected = (expected_state * normalized[0].unsqueeze(-1)).sum(-2).to(torch.bfloat16)

    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    torch.testing.assert_close(observed_state, expected_state, rtol=0, atol=0)

    # The rejected expansion, which had to publish each step's dtype: prove the
    # fixture separates the two policies so the assertions above are not vacuous.
    rounded_gate = (gate_raw.float() + dt_bias).to(torch.bfloat16).float()
    rounded = LOWER_BOUND * torch.sigmoid(
        (decay_rate * rounded_gate).to(torch.bfloat16).float())
    assert torch.count_nonzero(gate != rounded), (
        "This fixture no longer distinguishes the FP32 in-kernel gate from the "
        "graph expansion, so the bit-exact assertions above are vacuous.")


@pytest.mark.parametrize("heads", [2, 8])
def test_the_gate_view_keeps_the_head_axis(heads):
    scale = dict(REDUCED, num_attention_heads=heads)
    source = checkpoint(scale, full_model=True, expert_layout="separate")
    module = Glm5MoeModelImporter(source, block_size=8, num_blocks=3).import_module()
    text = configuration(scale)["text_config"]
    view = next(node for node in module.nodes if node.id == f"{PREFIX}_gate_heads")

    assert tuple(extent.fixed_value for extent in view.type.shape) == (
        1, text["kda_num_heads"], text["kda_head_dim"])
