# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""GDN recurrent must own M=1..5 tokens in one kernel, bitwise.

The recurrent state is sequential in the token, so a bucketed call replays the
decode-1 update once per row. This drives the real rendered template with a
frozen distribution, so the multi-row kernel and the looped one-token kernel
are the same program: any difference is the row contract itself.
"""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.nn._gdn_state import create_gdn_state
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from python.test.flagmega.gdn.recurrent_helpers import recurrent_case

PER_TOKEN = ("qkv", "z", "projection_input")


def _torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch


def _target(torch):
    return "nvidia-sm120" if torch.cuda.get_device_capability() == (12, 0) else "nvidia-sm90"


def _runner(tmp_path, torch, config, attrs, types, values, tokens, target, name):
    """Build one recurrent call owning ``tokens`` owner-local rows."""
    placement = fm.Placement((2, 4), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    split = fm.SBP.split_contiguous((0, 1))
    tensors = {
        name_: fm.tensor_type(values[name_].dtype.__str__().removeprefix("torch."),
                              (tokens, *values[name_].shape[1:]))
        if name_ in PER_TOKEN else types[name_]
        for name_ in types if name_ != "state"
    }
    distributed = {
        name_: fm.DistributedType(value, (broadcast, split) if name_ == "z" else (broadcast,) * value.rank,
                                  placement)
        for name_, value in tensors.items()
    }

    class Graph(fm.Module):
        def forward(self):
            inputs = {
                name_: self.input(
                    name_, config.ref_type if name_ == "state" else
                    (tensors[name_] if name_ == "z" else distributed[name_]), id=name_)
                for name_ in types
            }
            parameters = tuple(inputs.values())
            inputs["z"] = fm.F.distributed.force_boxing(inputs["z"], distributed["z"])
            recurrent = fm.F.nn.gated_delta_net_recurrent_core(**inputs, **attrs, name="recurrent")
            output = fm.F.tensors.get_item(recurrent, 0)
            output = fm.F.distributed.force_boxing(output, output.type.tensor)
            self.function("main", parameters, (output, inputs["state"]))

    module = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    namespace = {}
    exec(fm.module_source(module), namespace)
    compiled = Compiler(CompileOptions(target=target, codegen_backend="triton")).compile(
        namespace["MODULE"]).module
    artifact = write_artifact(compiled, tmp_path / name, target=target, emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    state = create_gdn_state(config, device="cuda:0")
    state.convolution.copy_(values["state"].convolution)
    state.recurrent.copy_(values["state"].recurrent)
    data = {
        name_: (values[name_][:tokens] if name_ in PER_TOKEN else values[name_]).cuda()
        for name_ in values if name_ != "state"
    }
    output = torch.empty_like(data["z"])
    state_buffers = dict(runtime.buffer_plan.function_map["main"].parameters)["state"]
    state_fields = dict(zip(state_buffers, (state.convolution, state.recurrent)))
    arguments = tuple(output if spec["role"] == "result" else (
        state_fields[spec["buffer"]] if spec["value"] == "state" else data[spec["value"]])
        for spec in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)

    def run(offset=0):
        for name_ in PER_TOKEN:
            data[name_].copy_(values[name_][offset:offset + tokens])
        output.fill_(float("nan"))
        GeneratedTirCallGraphModule.run_into(runtime, *arguments)
        torch.cuda.synchronize()
        return output.cpu().clone()

    return run, state


@pytest.mark.parametrize("tokens", [2, 3, 5])
def test_multirow_recurrent_is_bitwise_equal_to_sequential_tokens(tmp_path, tokens):
    torch = _torch()
    target = _target(torch)
    config, attrs, types, values = recurrent_case(tokens=tokens, dimension=8, qk_norm_mode="clamp",
                                                 qk_norm_epsilon=1e-6)
    run_multi, state_multi = _runner(tmp_path, torch, config, attrs, types, values, tokens,
                                     target, f"multi{tokens}")
    run_single, state_single = _runner(tmp_path, torch, config, attrs, types, values, 1,
                                       target, f"single{tokens}")
    actual = run_multi(0)
    expected = torch.cat([run_single(index) for index in range(tokens)], 0)
    assert torch.equal(actual, expected), (actual.float() - expected.float()).abs().max().item()
    assert torch.equal(state_multi.recurrent, state_single.recurrent)


def test_multirow_recurrent_rejects_more_than_five_rows(tmp_path):
    from triton.flagmega.errors import CodegenError

    torch = _torch()
    config, attrs, types, values = recurrent_case(tokens=6, dimension=8, qk_norm_mode="clamp",
                                                 qk_norm_epsilon=1e-6)
    with pytest.raises(CodegenError, match="at most 5 owner-local rows"):
        _runner(tmp_path, torch, config, attrs, types, values, 6, _target(torch), "multi6")
