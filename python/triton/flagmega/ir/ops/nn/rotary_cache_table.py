# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A precomputed cache retaining NVIDIA's declared FP32 rounding contract."""

from math import isfinite

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, op_definition


@op_definition("nn.rotary_cache_table", namespace="nn", functional_name="rotary_cache_table")
class RotaryCacheTable(OpDefinition):
    constant_source = True
    positions = attribute_parameter()
    inverse_frequencies = attribute_parameter()
    attention_scaling = attribute_parameter(default=1.0)
    output_dtype = attribute_parameter(default="bfloat16")

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        positions = attrs["positions"]
        if type(positions) is not int or positions <= 0:
            raise IRSchemaError("RotaryCacheTable positions must be a positive integer.")
        frequencies = tuple(float(x) for x in attrs["inverse_frequencies"])
        if not frequencies or any(not isfinite(x) or x <= 0 for x in frequencies):
            raise IRSchemaError("RotaryCacheTable requires positive finite inverse frequencies.")
        if positions * max(frequencies) >= 105615.0:
            raise IRSchemaError("RotaryCacheTable finite-angle contract requires angles below 105615.")
        scaling = float(attrs["attention_scaling"])
        if not isfinite(scaling) or scaling <= 0:
            raise IRSchemaError("RotaryCacheTable scaling must be positive and finite.")
        dtype = str(attrs["output_dtype"])
        if dtype not in ("float32", "bfloat16"):
            raise IRSchemaError("RotaryCacheTable output dtype must be FP32 or BF16.")
        return {"positions": positions, "inverse_frequencies": frequencies,
                "attention_scaling": scaling, "output_dtype": dtype}

    @classmethod
    def infer_type(cls, inputs, attrs):
        return tensor_type(attrs["output_dtype"], (attrs["positions"], 2, 2 * len(attrs["inverse_frequencies"])))

    @classmethod
    def evaluate(cls, node, arguments, context):
        from triton.flagmega.evaluator.nvidia_rotary_cache import rotary_table
        return rotary_table(node.attrs["positions"], node.attrs["inverse_frequencies"],
                            node.attrs["attention_scaling"], node.attrs["output_dtype"])

    @classmethod
    def cost(cls, node):
        return OpCost(notes=("precomputed-rotary-cache",))
