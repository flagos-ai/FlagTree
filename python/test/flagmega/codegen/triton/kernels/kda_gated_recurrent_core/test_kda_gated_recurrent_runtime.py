# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The generated kernel must build the forget gate in FP32 registers.

``nn.kda_recurrent_core`` needs the gate pre-built in the graph, so the raw
``g`` sum, the ``exp(A_log)`` product and the sigmoid each cross a buffer whose
dtype the graph picks.  The fused kernel loads the raw projections and keeps the
whole gate plus both sigmoids in registers, which is what
``fused_recurrent_kda(use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
lower_bound=...)`` does.  These tests pin that against the published
``Glm5NextTextForgetGate`` arithmetic.
"""

import pathlib

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

HEADS, DIM = 2, 4
LOWER_BOUND = -5.0
NAMES = ("q", "k", "v", "gate", "decay_rate", "gate_bias", "beta", "state")


def _module(*, output_lanes=(), consumer_lanes=(), heads=HEADS, dim=DIM, gated=True,
            placement=None, head_policy=None, query_scale=1.0, l2norm=True, request_phase=False, **gated_attrs):
    activation = fm.tensor_type("bfloat16", (1, heads, dim))
    weight_map = fm.tensor_type("float32", (1, heads, dim))
    state_type = fm.tensor_type("float32", (1, heads, dim, dim))
    types = ((activation, activation, activation, activation, weight_map, weight_map,
              fm.tensor_type("bfloat16", (1, heads)), state_type) if gated else
             (activation, activation, activation, weight_map, fm.tensor_type("float32", (1, heads)), state_type))
    names = NAMES if gated else ("q", "k", "v", "decay", "beta", "state")
    if placement is not None:
        types = tuple(fm.DistributedType(kind, (fm.SBP.broadcast(), head_policy,
                                               *(fm.SBP.broadcast() for _ in range(kind.rank - 2))), placement)
                      for kind in types)

    if request_phase:
        phase_type = fm.tensor_type("bool", ())
        if placement is not None:
            phase_type = fm.DistributedType(phase_type, (), placement)
        types = (*types, phase_type)
        names = (*names, "prefill")

    class Graph(fm.Module):
        def forward(self):
            operands = tuple(self.input(name, value_type, id=name)
                             for name, value_type in zip(names, types))
            operation = fm.F.nn.kda_gated_step if request_phase else fm.F.nn.kda_gated_recurrent_core
            pair = (operation(
                *operands, lower_bound=LOWER_BOUND, epsilon=1e-6, output_lanes=output_lanes,
                query_scale=query_scale, l2norm_in_kernel=l2norm, name="rec", **gated_attrs) if gated else
                fm.F.nn.kda_recurrent_core(*operands, epsilon=1e-6, output_lanes=output_lanes,
                    query_scale=query_scale, l2norm_in_kernel=l2norm, name="rec"))
            output, new_state = fm.F.tensors.get_items(pair, 0, 1, name_prefix="rec")
            if consumer_lanes:
                output = fm.F.tensors.pack(output, consumer_lanes, axes=(-1,) * len(consumer_lanes))
            self.function("main", operands, (output, new_state))

    return Graph(dialect="high_level", stage="frozen_constants" if placement else "imported", entry="main",
                 metadata={"auto_distribution": {"placement": placement.to_data()}} if placement else {}).build()


def _artifact(tmp_path: pathlib.Path):
    return write_artifact(Compiler().compile(_module()).module, tmp_path / "kda-gated",
                          target="nvidia-sm90", emit_executable=True)


def _values(torch, *, device="cpu", heads=HEADS, dim=DIM):
    generator = torch.Generator().manual_seed(11)

    def normal(*shape):
        return torch.randn(shape, generator=generator)

    values = {
        "q": normal(1, heads, dim).bfloat16(),
        "k": normal(1, heads, dim).bfloat16(),
        "v": normal(1, heads, dim).bfloat16(),
        "gate": normal(1, heads, dim).bfloat16(),
        # The constant island stores exp(A_log) already broadcast per channel.
        "decay_rate": normal(1, heads, 1).exp().expand(1, heads, dim).contiguous(),
        "gate_bias": normal(1, heads, dim),
        "beta": normal(1, heads).bfloat16(),
        "state": normal(1, heads, dim, dim),
    }
    return {name: value.to(device) for name, value in values.items()}


def _reference(torch, values, *, round_gate=False, gated=True):
    """Glm5NextTextForgetGate.forward + sigmoid(b_proj) + the naive recurrence.

    ``round_gate`` reproduces the graph expansion this operator replaces, where
    the gate sum and the sigmoid argument land in BF16 buffers.
    """

    def maybe_round(tensor):
        return tensor.bfloat16().float() if round_gate else tensor

    if gated:
        gate = maybe_round(values["gate"].float() + values["gate_bias"])
        decay = LOWER_BOUND * torch.sigmoid(maybe_round(values["decay_rate"] * gate))
        beta = torch.sigmoid(maybe_round(values["beta"].float()))
    else:
        decay, beta = values["decay"], values["beta"]
    query, key = values["q"].float(), values["k"].float()
    query = query / torch.sqrt((query * query).sum(-1, keepdim=True) + 1e-6)
    key = key / torch.sqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    state = values["state"] * decay.exp().unsqueeze(-1)
    delta = (values["v"].float() - (state * key.unsqueeze(-1)).sum(-2)) * beta.unsqueeze(-1)
    state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
    return (state * query.unsqueeze(-1)).sum(-2).bfloat16(), state


def test_generated_source_computes_the_gate_in_registers(tmp_path):
    source = (_artifact(tmp_path) / "generated_kernels.py").read_text("utf-8")

    assert "# flagmega-kernel: kda_gated_recurrent_core/local platform=generic" in source
    # The lower bound is a compile-time literal, and the whole gate is one FP32
    # expression: no intermediate is written back or re-rounded.
    assert f"_fm_decay = {LOWER_BOUND!r} * tl.sigmoid(_fm_decay_rate * (_fm_gate + _fm_gate_bias))" in source
    assert "_fm_beta = tl.sigmoid(tl.load(" in source
    assert "_fm_decay_exp = tl.exp(_fm_decay)" in source
    # A pre-built decay/beta pair would show up as the unfused operator instead.
    assert "kda_recurrent_core/local" not in source


def test_generated_kernel_matches_the_published_forget_gate(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    runtime = load(_artifact(tmp_path), device="cuda:0")
    values = _values(torch, device="cuda")
    outputs = {
        "rec_0": torch.empty((1, HEADS, DIM), dtype=torch.bfloat16, device="cuda"),
        "rec_1": torch.empty((1, HEADS, DIM, DIM), dtype=torch.float32, device="cuda"),
    }
    arguments = tuple(outputs[spec["value"]] if spec["role"] == "result" else values[spec["value"]]
                      for spec in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    GeneratedTirCallGraphModule.run_into(runtime, *arguments)
    torch.cuda.synchronize()

    expected, expected_state = _reference(torch, values)
    torch.testing.assert_close(outputs["rec_0"], expected, rtol=0, atol=0)
    # tl.exp/tl.sigmoid differ from torch by at most one FP32 ULP on the decay,
    # which the rank-1 state update carries through linearly.
    torch.testing.assert_close(outputs["rec_1"], expected_state, rtol=1e-6, atol=1e-7)


def test_rounding_the_gate_would_be_observable(tmp_path):
    """Guard against a vacuous pass: the two gate policies must actually differ."""
    torch = pytest.importorskip("torch")
    values = _values(torch)
    fused, fused_state = _reference(torch, values)
    rounded, rounded_state = _reference(torch, values, round_gate=True)

    assert torch.count_nonzero(fused != rounded) or torch.count_nonzero(
        fused_state != rounded_state), (
            "This fixture no longer separates the FP32 in-kernel gate from the "
            "rounded graph expansion, so the assertions above are vacuous.")


@pytest.mark.parametrize("lanes", [(8,), (2, 4), (1,)])
@pytest.mark.parametrize("gated", [True, False], ids=["gated", "regular"])
def test_packed_kda_writes_all_heads_and_preserves_state(tmp_path, lanes, gated):
    from math import prod

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    heads, dim = 3, 8
    host = _values(torch, heads=heads, dim=dim)
    if not gated:
        decay = LOWER_BOUND * torch.sigmoid(host["decay_rate"] * (host["gate"].float() + host["gate_bias"]))
        host = {**{name: host[name] for name in ("q", "k", "v", "state")},
                "decay": decay, "beta": torch.sigmoid(host["beta"].float())}
    results = []
    for name, kwargs in (("scalar", {}), ("demand", {"consumer_lanes": lanes}), ("direct", {"output_lanes": lanes})):
        module = Compiler().compile(_module(heads=heads, dim=dim, gated=gated, **kwargs)).module
        path = write_artifact(module, tmp_path / name, target="nvidia-sm90", emit_executable=True)
        source = (path / "generated_kernels.py").read_text()
        family = "kda_gated_recurrent_core" if gated else "kda_recurrent_core"
        assert source.count(f"# flagmega-kernel: {family}/local") == 1
        assert "# flagmega-kernel: pack/" not in source
        runtime = load(path, device="cuda:0")
        # Each variant starts from its own state; in-place state reuse is legal.
        values = {key: value.cuda().clone() for key, value in host.items()}
        shape = (1, heads, dim) if name == "scalar" else (1, heads, dim // prod(lanes), *lanes)
        outputs = (torch.empty(shape, dtype=torch.bfloat16, device="cuda"),
                   torch.empty((1, heads, dim, dim), dtype=torch.float32, device="cuda"))
        result_specs = [spec for spec in runtime.external_arguments if spec["role"] == "result"]
        result_map = {spec["value"]: output for spec, output in zip(result_specs, outputs)}
        arguments = tuple(result_map[spec["value"]] if spec["role"] == "result" else values[spec["value"]]
                          for spec in runtime.external_arguments)
        GeneratedTirCallGraphModule.prepare(runtime, *arguments)
        GeneratedTirCallGraphModule.run_into(runtime, *arguments)
        torch.cuda.synchronize()
        results.append((outputs[0].cpu().reshape(1, heads, dim), outputs[1].cpu()))
    torch.testing.assert_close(results[1], results[0], rtol=0, atol=0)
    torch.testing.assert_close(results[2], results[0], rtol=0, atol=0)
    expected = _reference(torch, host, gated=gated)
    torch.testing.assert_close(results[2][0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(results[2][1], expected[1], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("gated", [True, False], ids=["gated", "regular"])
@pytest.mark.parametrize("heads,head_policy", [(4, fm.SBP.split_contiguous((0,))),
                                               (4, fm.SBP.split_block_cyclic((0,), 1)),
                                               (5, fm.SBP.split_block_cyclic((0,), 1))])
def test_packed_kda_preserves_distributed_head_addresses(tmp_path, gated, head_policy, heads):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    placement = fm.Placement((2,), "x", "b")
    dim = 8
    host = _values(torch, heads=heads, dim=dim)
    if not gated:
        decay = LOWER_BOUND * torch.sigmoid(host["decay_rate"] * (host["gate"].float() + host["gate_bias"]))
        host = {**{name: host[name] for name in ("q", "k", "v", "state")},
                "decay": decay, "beta": torch.sigmoid(host["beta"].float())}
    module = _module(gated=gated, heads=heads, dim=dim, output_lanes=(2, 4), placement=placement,
                     head_policy=head_policy)
    path = write_artifact(Compiler().compile(module).module, tmp_path / "head-owners",
                          target="nvidia-sm90", emit_executable=True)
    runtime = load(path, device="cuda:0")
    values = {name: value.cuda().clone() for name, value in host.items()}
    outputs = (torch.empty((1, heads, 1, 2, 4), dtype=torch.bfloat16, device="cuda"),
               torch.empty((1, heads, dim, dim), dtype=torch.float32, device="cuda"))
    result_specs = [spec for spec in runtime.external_arguments if spec["role"] == "result"]
    result_map = {spec["value"]: output for spec, output in zip(result_specs, outputs)}
    arguments = tuple(result_map[spec["value"]] if spec["role"] == "result" else values[spec["value"]]
                      for spec in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    GeneratedTirCallGraphModule.run_into(runtime, *arguments)
    torch.cuda.synchronize()
    expected, expected_state = _reference(torch, host, gated=gated)
    torch.testing.assert_close(outputs[0].cpu().reshape(expected.shape), expected, rtol=0, atol=0)
    torch.testing.assert_close(outputs[1].cpu(), expected_state, rtol=1e-6, atol=1e-7)
