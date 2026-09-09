# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Device fixtures for expert stages, independent of a checkpoint or model."""

from math import prod

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.runtime import load
from python.test.flagmega.sparse_experts.helpers import operand_types


def stage_module(definition, *, dtype="bfloat16", packed=False, distribution=None, hidden=72, intermediate=40, tokens=4,
                 **attrs):
    types = operand_types(dtype=dtype, tokens=tokens, hidden=hidden, intermediate=intermediate, routes=3, experts=5)
    activation = definition.input_parameters[0].name
    if packed:
        tensor = types[activation]
        types[activation] = fm.tensor_type(fm.vector_type(dtype, (2, 2)), (*tensor.shape[:-1], tensor.shape[-1] // 4))
        attrs["output_dtype"] = fm.vector_type(dtype, (2, 2))
    metadata = {}
    if distribution is not None:
        placement = fm.Placement((2, 2), "yx", "bb")
        metadata = {"auto_distribution": {"placement": placement.to_data()}}
        broadcast = fm.SBP.broadcast()
        token = fm.SBP.split_block_cyclic((0, ), 1) if distribution == "token_output" else broadcast
        feature_axis = 1 if distribution == "token_output" else 0
        feature = fm.SBP.split_block_cyclic((feature_axis, ), 2 if packed else 4)
        scalar_feature = fm.scale_split_units(feature, 4, 1) if packed else feature
        is_down = definition.op_name == "nn.sparse_experts_down"
        intermediate_policy = feature if distribution == "split_k" or not is_down else broadcast
        scalar_intermediate = (fm.scale_split_units(intermediate_policy, 4, 1)
                               if packed and intermediate_policy != broadcast else intermediate_policy)
        output_policy = broadcast if distribution == "split_k" else scalar_feature
        policies = {
            "q": (token, broadcast),
            "activations": (token, broadcast, intermediate_policy),
            "router_expert_ids": (token, broadcast),
            "router_expert_weights": (token, broadcast),
            "gate_weight": (broadcast, scalar_intermediate, broadcast),
            "up_weight": (broadcast, scalar_intermediate, broadcast),
            "down_weight": (broadcast, output_policy, scalar_intermediate),
        }
        types = {
            parameter.name:
            fm.DistributedType(types[parameter.name], policies.get(parameter.name, (broadcast, broadcast)), placement)
            for parameter in definition.input_parameters
        }

    class Graph(fm.Module):

        def forward(self):
            inputs = tuple(
                self.input(parameter.name, types[parameter.name], id=parameter.name)
                for parameter in definition.input_parameters)
            result = definition.construct(*inputs, **attrs, name="expert_stage")
            if distribution is not None:
                result = fm.F.distributed.boxing(result, result.type.tensor)
            self.function("main", inputs, (result, ))

    return Graph(dialect="nn", stage="frozen_constants", entry="main", metadata=metadata).build()


def execute_and_reference(module, tmp_path, torch, *, checkpoint=None):
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True,
                              checkpoint=checkpoint)
    runtime = load(artifact, device="cuda:0")
    generator = torch.Generator().manual_seed(1793)
    values = {}
    for node_id in module.function_map[module.entry].parameters:
        node = module.node_map[node_id]
        tensor = fm.logical_type(node.type)
        shape = tuple(dim.fixed_value for dim in tensor.shape)
        if node_id == "router_expert_ids":
            value = (torch.arange(prod(shape)).reshape(shape) * 3 + 1).remainder(5).int()
        elif node_id.endswith("_scale"):
            value = (2.**(torch.arange(shape[0]).remainder(3) - 1).float()).reshape(shape)
            if node_id.endswith("proj_scale"):
                value = value.flip(0).contiguous()
        elif node_id == "router_expert_weights":
            value = (torch.arange(prod(shape)).reshape(shape).remainder(3) + 1).float() / 8
        else:
            lanes = tensor.dtype.lanes if isinstance(tensor.dtype, fm.VectorType) else ()
            dtype = tensor.dtype.elem_type if lanes else tensor.dtype
            value = (torch.randint(-16, 17,
                                   (*shape, *lanes), generator=generator).float() / 16).to(getattr(torch, dtype.value))
        values[node_id] = value
    expected = TorchEvaluator(DictWeightResolver({})).run(module, values)[0]
    arguments = tuple(values[node_id].cuda() for node_id in module.function_map[module.entry].parameters)
    runtime.prepare(*arguments)
    output = runtime.run(*arguments)
    torch.cuda.synchronize()
    return output.cpu(), expected, runtime
