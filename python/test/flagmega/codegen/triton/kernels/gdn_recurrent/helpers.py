# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Execute the real recurrent kernel with distributed operands and shared state."""

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.nn._gdn_state import create_gdn_state
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule


def execute_recurrent(tmp_path, torch, config, attrs, types, values, *, local_z=True, local_result=True, placement=None,
                      split_axes=(0, 1)):
    placement = fm.Placement((2, 4), "yx", "bb") if placement is None else placement
    broadcast = fm.SBP.broadcast()
    split = fm.SBP.split_contiguous(split_axes) if split_axes else broadcast
    per_token = {"qkv", "z", "projection_input"}
    tensors = {
        name: fm.tensor_type(value.dtype, (1, *value.shape[1:])) if name in per_token else value
        for name, value in types.items()
        if name != "state"
    }
    distributed = {
        name: fm.DistributedType(value, (broadcast, split) if name == "z" else (broadcast, ) * value.rank, placement)
        for name, value in tensors.items()
    }

    class Graph(fm.Module):

        def forward(self):
            inputs = {
                name:
                self.input(
                    name, config.ref_type if name == "state" else
                    (tensors[name] if name == "z" and local_z else distributed[name]), id=name)
                for name in types
            }
            parameters = tuple(inputs.values())
            if local_z:
                inputs["z"] = fm.F.distributed.force_boxing(inputs["z"], distributed["z"])
            recurrent = fm.F.nn.gated_delta_net_recurrent_core(**inputs, **attrs, name="recurrent")
            output = fm.F.tensors.get_item(recurrent, 0)
            if local_result:
                output = fm.F.distributed.force_boxing(output, output.type.tensor)
            self.function("main", parameters, (output, inputs["state"]))

    module = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    namespace = {}
    exec(fm.module_source(module), namespace)
    assert namespace["MODULE"].semantic_hash == module.semantic_hash
    compiled = Compiler().compile(namespace["MODULE"]).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    state = create_gdn_state(config, device="cuda:0")
    state.convolution.copy_(values["state"].convolution)
    state.recurrent.copy_(values["state"].recurrent)
    data = {
        name: (value[:1] if name in per_token else value).cuda()
        for name, value in values.items()
        if name != "state"
    }
    output = torch.empty_like(data["z"])
    state_buffers = dict(runtime.buffer_plan.function_map["main"].parameters)["state"]
    state_fields = dict(zip(state_buffers, (state.convolution, state.recurrent)))
    arguments = tuple(output if spec["role"] == "result" else (
        state_fields[spec["buffer"]] if spec["value"] == "state" else data[spec["value"]])
                      for spec in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    outputs, states = [], []
    for index in range(values["qkv"].shape[0]):
        for name in per_token:
            data[name].copy_(values[name][index:index + 1])
        output.fill_(float("nan"))
        GeneratedTirCallGraphModule.run_into(runtime, *arguments)
        outputs.append(output.cpu().clone())
        states.append(state.recurrent_layer().cpu().clone())
        torch.testing.assert_close(state.convolution.cpu(), values["state"].convolution, rtol=0, atol=0)
    return torch.cat(outputs), states
