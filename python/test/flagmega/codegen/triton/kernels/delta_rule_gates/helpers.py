# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.ir.ops.nn.delta_rule_gates import DeltaRuleGates
from triton.flagmega.runtime import load


def execute_gates(tmp_path, torch, values, *, token_policy=None, head_policy=None,
                  local_inputs=(), local_results=(False, False), **attrs):
    placement = fm.Placement((2, 2), "xy", "bb")
    broadcast = fm.SBP.broadcast()
    token = broadcast if token_policy is None else token_policy
    head = broadcast if head_policy is None else head_policy
    tensors = {name: fm.tensor_type(str(value.dtype).removeprefix("torch."), value.shape)
               for name, value in values.items()}
    distributed = {name: fm.DistributedType(value, (token, head) if value.rank == 2 else (head,), placement)
                   for name, value in tensors.items()}

    class Graph(fm.Module):
        def forward(self):
            parameters = {parameter.name: self.input(parameter.name,
                tensors[parameter.name] if parameter.name in local_inputs else distributed[parameter.name],
                id=parameter.name) for parameter in DeltaRuleGates.input_parameters}
            inputs = {name: fm.F.distributed.force_boxing(value, distributed[name]) if name in local_inputs else value
                      for name, value in parameters.items()}
            gates = fm.F.nn.delta_rule_gates(**inputs, **attrs)
            outputs = list(fm.F.tensors.get_items(gates, 0, 1))
            for index, local in enumerate(local_results):
                if local:
                    outputs[index] = fm.F.distributed.force_boxing(outputs[index], outputs[index].type.tensor)
            self.function("main", tuple(parameters.values()), outputs)

    module = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    namespace = {}
    exec(fm.module_source(module), namespace)
    assert namespace["MODULE"].semantic_hash == module.semantic_hash
    compiled = Compiler().compile(namespace["MODULE"]).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    data = {name: value.cuda() for name, value in values.items()}
    outputs = tuple(torch.full_like(data["a"], float("nan"), dtype=torch.float32) for _ in range(2))
    function = runtime.buffer_plan.function_map["main"]
    bindings = {buffers[0]: data[value] for value, buffers in function.parameters}
    bindings.update({buffers[0]: output for (_, buffers), output in zip(function.outputs, outputs, strict=True)})
    arguments = tuple(bindings[spec["buffer"]] for spec in runtime.external_arguments)
    runtime.prepare(*arguments)
    runtime.run_into(*arguments)
    actual = tuple(output.cpu().clone() for output in outputs)
    for name, value in data.items():
        torch.testing.assert_close(value.cpu(), values[name], rtol=0, atol=0)
    return actual
