# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Small deterministic expert fixtures, independent of any model/checkpoint."""

import torch

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn.sparse_experts import SparseExperts


def operand_types(*, dtype="bfloat16", tokens=2, hidden=16, intermediate=12, experts=4, routes=2):
    shapes = {
        "q": (dtype, (tokens, hidden)),
        "activations": (dtype, (tokens, routes, intermediate)),
        "router_expert_ids": ("int32", (tokens, routes)),
        "router_expert_weights": ("float32", (tokens, routes)),
        "gate_weight": (dtype, (experts, intermediate, hidden)),
        "up_weight": (dtype, (experts, intermediate, hidden)),
        "down_weight": (dtype, (experts, hidden, intermediate)),
    }
    for stage in ("gate", "up", "down"):
        for kind in ("input", "proj"):
            shapes[f"{stage}_{kind}_scale"] = ("float32", (experts, 1))
    return {name: fm.tensor_type(dtype, shape) for name, (dtype, shape) in shapes.items()}


def operands(definition, types=None):
    types = operand_types() if types is None else types
    return tuple(
        fm.Node(parameter.name, "builtin.var", (), types[parameter.name], attrs={"name": parameter.name})
        for parameter in definition.input_parameters)


def build_module(definition=SparseExperts, *, types=None, attrs=None, duplicate_use=False):
    types = operand_types() if types is None else types

    class Experts(fm.Module):

        def forward(self):
            inputs = tuple(
                self.input(parameter.name, types[parameter.name]) for parameter in definition.input_parameters)
            result = definition.construct(*inputs, **(attrs or {}), name="experts", metadata={"test_tag": "kept"})
            if duplicate_use:
                output = fm.F.math.add(result, result, name="output")
                self.function("main", inputs, (output, result))
            else:
                self.function("main", inputs, (result, ))

    return Experts(dialect="nn", stage="imported", entry="main").build()


def values_for(module):
    generator = torch.Generator().manual_seed(784)
    values = {}
    for node in module.nodes:
        if node.op != "builtin.var":
            continue
        name = node.attrs["name"]
        shape = tuple(dim.fixed_value for dim in node.type.shape)
        if name == "router_expert_ids":
            values[name] = torch.tensor([[3, 1], [0, 3]], dtype=torch.int32)
        elif name.endswith("_scale"):
            values[name] = 0.5 + torch.rand(shape, generator=generator)
        else:
            dtype = getattr(torch, node.type.dtype.value)
            values[name] = torch.randn(shape, generator=generator).to(dtype)
    if "router_expert_weights" in values:
        values["router_expert_weights"] = values["router_expert_weights"].softmax(-1)
    return values
