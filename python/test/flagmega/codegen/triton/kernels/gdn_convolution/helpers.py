# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compile and invoke a single convolution through the normal stateful ABI."""

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig, create_gdn_state
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule


def execute_convolution(tmp_path, torch, inputs, weight, history, *, tokens_per_call=1, **attrs):
    channels, kernel = weight.shape
    config = GatedDeltaNetStateConfig(1, 2, 4, 8, 8, kernel, 32)
    assert channels == config.conv_dim
    placement = fm.Placement((2, 4), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    split = fm.SBP.split_contiguous((0, 1))
    assert inputs.shape[0] % tokens_per_call == 0
    qkv_type = fm.DistributedType(fm.tensor_type("bfloat16", (tokens_per_call, channels)), (broadcast, split),
                                  placement)
    weight_type = fm.DistributedType(fm.tensor_type("bfloat16", (channels, kernel)), (split, broadcast), placement)

    class Graph(fm.Module):

        def forward(self):
            qkv = self.input("qkv", qkv_type, id="qkv")
            coefficients = self.input("weight", weight_type, id="weight")
            state = self.input("state", config.ref_type, id="state")
            conv = fm.F.nn.gated_delta_net_convolution(qkv, state, coefficients, conv_kernel_size=kernel, name="conv",
                                                       **attrs)
            self.function("main", (qkv, coefficients, state), (fm.F.tensors.get_item(conv, 0), state))

    module = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    # Attributes and the state reference survive a real Python edit/resume.
    namespace = {}
    exec(fm.module_source(module), namespace)
    resumed = namespace["MODULE"]
    assert resumed.semantic_hash == module.semantic_hash
    compiled = Compiler().compile(resumed).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    state = create_gdn_state(config, device="cuda:0")
    state.update_convolution_layer(history.cuda())
    state.recurrent.fill_(17)
    qkv = inputs[:tokens_per_call].cuda()
    output = torch.empty_like(qkv)
    values = {"qkv": qkv, "weight": weight.cuda()}
    state_buffers = dict(runtime.buffer_plan.function_map["main"].parameters)["state"]
    state_fields = dict(zip(state_buffers, (state.convolution, state.recurrent)))
    arguments = tuple(output if spec["role"] == "result" else (
        state_fields[spec["buffer"]] if spec["value"] == "state" else values[spec["value"]])
                      for spec in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    outputs, states = [], []
    for start in range(0, inputs.shape[0], tokens_per_call):
        qkv.copy_(inputs[start:start + tokens_per_call])
        output.fill_(float("nan"))
        GeneratedTirCallGraphModule.run_into(runtime, *arguments)
        outputs.append(output.cpu().clone())
        states.append(state.convolution_layer().cpu().clone())
        assert torch.all(state.recurrent == 17)
    return torch.cat(outputs), states
