# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import json

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


def execute_block_update(tmp_path, torch, values, state_type, state_values, *, head_axes=(), local_inputs=False,
                         attrs=None, compose=False):
    attrs = {} if attrs is None else attrs
    placement = fm.Placement((2, 2), "yx", "bb")
    b = fm.SBP.broadcast()
    head = fm.SBP.split_contiguous(head_axes) if head_axes else b
    names = ("query", "key", "value", "beta", "alpha") if compose else ("query", "key", "value", "coefficients",
                                                                        "log_prefix")
    tensors = {
        name: fm.tensor_type(str(value.dtype).removeprefix("torch."), value.shape)
        for name, value in zip(names, values)
    }
    distributed = {
        name: fm.DistributedType(value, (b, head, *(b
                                                    for _ in range(value.rank - 2))), placement)
        for name, value in tensors.items()
    }

    class Graph(fm.Module):

        def forward(self):
            inputs = {
                name: self.input(name, tensors[name] if local_inputs else distributed[name], id=name)
                for name in names
            }
            state = self.input("state", state_type, id="state")
            parameters = (*inputs.values(), state)
            if local_inputs:
                inputs = {
                    name: fm.F.distributed.force_boxing(value, distributed[name])
                    for name, value in inputs.items()
                }
            if compose:
                inputs["coefficients"] = fm.F.nn.delta_rule_coefficients(inputs["key"], inputs.pop("beta"))
                inputs["log_prefix"] = fm.F.nn.delta_rule_log_prefix(inputs.pop("alpha"))
            update = fm.F.nn.delta_rule_block_update(**inputs, state=state, **attrs)
            output = fm.F.tensors.get_item(update, 0)
            output = fm.F.distributed.force_boxing(output, output.type.tensor)
            self.function("main", parameters, (output, fm.F.tensors.get_item(update, 1)))

    module = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    namespace = {}
    exec(fm.module_source(module), namespace)
    compiled = Compiler().compile(namespace["MODULE"]).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    tensors = {name: value.cuda() for name, value in zip(names, values)}
    state_buffers = dict(runtime.buffer_plan.function_map["main"].parameters)["state"]
    state = {name: value.cuda().clone() for name, value in state_values.items()}
    state_fields = dict(zip(state_buffers, (state[name] for name, _ in state_type.fields)))
    output = torch.full_like(tensors["value"], float("nan"))
    arguments = tuple(output if spec["role"] == "result" else (
        state_fields[spec["buffer"]] if spec["value"] == "state" else tensors[spec["value"]])
                      for spec in runtime.external_arguments)
    runtime.prepare(*arguments)
    runtime.run_into(*arguments)
    (tmp_path / "resources.json").write_text(json.dumps(runtime.resource_report, indent=2))
    return output.cpu(), {name: value.cpu() for name, value in state.items()}, runtime.resource_report
