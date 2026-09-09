# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


def primitive_module(definition, types, **attrs):

    class Graph(fm.Module):

        def forward(self):
            inputs = tuple(
                self.input(parameter.name, value_type)
                for parameter, value_type in zip(definition.input_parameters, types))
            output = definition.construct(*inputs, **attrs, name="output")
            self.function("main", inputs, (output, ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def evaluate(definition, values, **attrs):
    types = tuple(fm.tensor_type(str(value.dtype).removeprefix("torch."), value.shape) for value in values)
    module = primitive_module(definition, types, **attrs)
    fm.verify_module(module)
    arguments = {parameter.name: value for parameter, value in zip(definition.input_parameters, values)}
    return TorchEvaluator(DictWeightResolver({})).run(module, arguments)[0]
