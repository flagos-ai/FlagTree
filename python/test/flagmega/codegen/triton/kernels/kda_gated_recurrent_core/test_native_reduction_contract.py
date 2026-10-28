# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A selected KDA schedule must retain the native warp's FP32 reduction order."""

from dataclasses import replace

import pytest
import torch
import triton
import triton.language as tl
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError
from triton.flagmega.runtime import load
from triton.flagmega.targets import NvidiaSm90Target

from .test_kda_gated_recurrent_runtime import _module
from python.test.flagmega.devices import is_nvidia_sm90


@triton.jit
def _one_warp_reference(Q, K, V, Gate, Beta, LogRate, Bias, Initial, Out, State, Rate):
    head = tl.program_id(1)
    keys = tl.arange(0, 128)
    values = tl.program_id(0) * 8 + tl.arange(0, 8)
    q = tl.load(Q + head * 128 + keys).to(tl.float32)
    k = tl.load(K + head * 128 + keys).to(tl.float32)
    v = tl.load(V + head * 128 + values).to(tl.float32)
    q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
    k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
    q = q * (128**-0.5)
    rate = tl.exp(tl.load(LogRate + head))
    gate = tl.load(Gate + head * 128 + keys).to(tl.float32) + tl.load(Bias + head * 128 + keys)
    gate = -5.0 / (1.0 + tl.exp(-(rate * gate)))
    # Reference storage is value-major, as in the native recurrent kernel.
    state = tl.load(Initial + head * 16384 + values[:, None] * 128 + keys[None, :])
    state = state * tl.exp(gate[None, :])
    v = (v - tl.sum(state * k[None, :], 1)) * tl.sigmoid(tl.load(Beta + head).to(tl.float32))
    state = state + v[:, None] * k[None, :]
    out = tl.sum(state * q[None, :], 1)
    tl.store(Out + head * 128 + values, out.to(tl.bfloat16))
    tl.store(State + head * 16384 + keys[None, :] * 128 + values[:, None], state)
    if tl.program_id(0) == 0:
        tl.store(Rate + head * 128 + keys, rate)


def _compile(tmp_path, lanes, *, decay_input=None, layout="warp_k", dim=128, request_phase=True):
    compiler = Compiler()
    base = NvidiaSm90Target().triton_implementation_model
    catalog = replace(
        base,
        implementations=tuple(
            replace(value, parameters={**value.parameters, "compute_num_warps": 8, "reduction_layout": layout})
            if value.family == "kda_gated_recurrent_core"
            else value
            for value in base.implementations
        ),
    )
    compiler.target = NvidiaSm90Target(triton_implementation_model=catalog)
    source = _module(
        request_phase=request_phase,
        heads=2,
        dim=dim,
        query_scale=128**-0.5,
        output_lanes=lanes,
        **({} if decay_input is None else {"decay_input": decay_input}),
    )
    module = compiler.compile(source).module
    restored = fm.load_module(fm.emit_module(module, tmp_path / "selected.py"))
    assert restored.semantic_hash == module.semantic_hash
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    return artifact


@pytest.mark.parametrize("lanes", [(), (8,)])
@pytest.mark.parametrize("decay_input", ["rate", "log_rate"])
def test_warp_k_schedule_matches_independent_one_warp_state(tmp_path, lanes, decay_input):
    if not is_nvidia_sm90():
        pytest.skip("requires SM90")
    generator = torch.Generator().manual_seed(117)
    values = {name: torch.randn((1, 2, 128), generator=generator).bfloat16().cuda() for name in ("q", "k", "v", "gate")}
    values.update(
        beta=torch.randn((1, 2), generator=generator).bfloat16().cuda(),
        gate_bias=torch.randn((1, 2, 128), generator=generator).cuda(),
        state=(torch.randn((1, 2, 128, 128), generator=generator) * 0.01).cuda(),
        prefill=False,
    )
    a_log = torch.randn((2,), generator=generator).cuda()
    rate = torch.empty((1, 2, 128), device="cuda")
    expected = torch.empty_like(values["q"])
    expected_state = torch.empty_like(values["state"])
    _one_warp_reference[(16, 2)](
        values["q"],
        values["k"],
        values["v"],
        values["gate"],
        values["beta"],
        a_log,
        values["gate_bias"],
        values["state"].transpose(-1, -2).contiguous(),
        expected,
        expected_state,
        rate,
        num_warps=1,
        num_stages=3,
    )
    values["decay_rate"] = rate if decay_input == "rate" else a_log.reshape(1, 2, 1).expand(1, 2, 128).contiguous()
    shape = (1, 2, 128 // (8 if lanes else 1), *lanes)
    outputs = (torch.empty(shape, dtype=torch.bfloat16, device="cuda"), torch.empty_like(expected_state))
    runtime = load(_compile(tmp_path, lanes, decay_input=decay_input), device="cuda:0")
    try:
        specs = [s for s in runtime.external_arguments if s["role"] == "result"]
        values.update({spec["value"]: output for spec, output in zip(specs, outputs, strict=True)})
        args = tuple(values[spec["value"]] for spec in runtime.external_arguments)
        runtime.prepare(*args)
        runtime.run_into(*args)
        torch.testing.assert_close(outputs[0].reshape_as(expected), expected, rtol=0, atol=0)
        torch.testing.assert_close(outputs[1], expected_state, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("layout,dim", [("unknown", 128), ("warp_k", 64)])
def test_unsupported_reduction_schedule_is_rejected(tmp_path, layout, dim):
    with pytest.raises(CodegenError, match="KDA.*(reduction_layout|warp_k)"):
        _compile(tmp_path, (), layout=layout, dim=dim, request_phase=False)


def test_resumed_warp_k_schedule_requires_sm90(tmp_path, monkeypatch):
    import triton.flagmega.codegen.triton.tir_package as package

    monkeypatch.setattr(package, "module_template_target", lambda module: ("iluvatar", "bi-v150"))
    with pytest.raises(CodegenError, match="KDA warp_k requires the NVIDIA SM90 target"):
        _compile(tmp_path, ())
