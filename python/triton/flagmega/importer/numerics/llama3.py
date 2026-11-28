# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Llama 3.x import contract for the pinned vLLM/Inductor fusion boundaries.

Same frontend semantics as the Qwen3 contract (FP32 pointwise chains inside an
attention-delimited segment, BF16 matmul/attention/carry rounding boundaries),
minus the QK-norm outlets: Llama has no query/key RMSNorm, so only the input and
post-attention normalizations are rewritten.
"""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DType, TupleType, tensor_type, verify_module
from triton.flagmega.ir.region import is_region
from triton.flagmega.rules.neutral._utility import make_node

NUMERICAL_CONTRACT = "vllm-493bd8323-inductor-level3"


def apply_llama3_vllm_profile(module):
    if module.stage != "imported" or "decode_layer" not in module.function_map:
        raise IRSchemaError("vLLM numerical specialization requires imported reusable decode_layer IR")
    if module.metadata.get("numerical_contract") == NUMERICAL_CONTRACT:
        return verify_module(module)
    original = module.node_map
    decode = module.function_map["decode_layer"]
    required = {
        "decode_layer_input_norm", "decode_layer_after_attention", "decode_layer_output", "decode_layer_mlp_gate_up"
    }
    if (not required.issubset(original) or original["decode_layer_input_norm"].op != "nn.rms_norm"
            or original["decode_layer_mlp_gate_up"].op != "nn.dense_matmul_glu"):
        raise IRSchemaError("Unexpected decode-layer topology for the pinned vLLM numerical contract")
    # Llama must not carry QK-norm outlets: those would be rewritten with a
    # widened output that has no consumer.
    if "decode_layer_query_norm" in original or "decode_layer_key_norm" in original:
        raise IRSchemaError("Llama profile requires a QK-norm-free decoder layer")
    hidden_id = decode.parameters[0]
    hidden_type = original[hidden_id].type
    wide_hidden = tensor_type(DType.FLOAT32, hidden_type.shape)
    wide_carries = {}
    for region in module.functions:
        if not is_region(region):
            continue
        count = int(region.attrs["iter_count"])
        positions = tuple(index for index, value in enumerate(region.parameters[1:1 + count])
                          if original[value].type == hidden_type)
        if positions:
            wide_carries[region.name] = positions
    wide_parameters = {
        module.function_map[name].parameters[1 + index]
        for name, positions in wide_carries.items()
        for index in positions
    }
    nodes, mapped = [], {}

    def emit(op, name, inputs=(), attrs=None):
        value = make_node(op, name, inputs, attrs or {}, {"numerical_contract": "vllm-inductor-fp32-fusion"})
        nodes.append(value)
        return value

    def cast(value, dtype, name, result_type=None):
        if value.type.dtype == dtype and result_type is None:
            return value
        node = emit("tensors.cast", name, (value, ), {"dtype": dtype.value, "rounding_contract": "preserve"})
        if result_type is not None and node.type != result_type:
            # The ABI node keeps its original packed lane configuration; the
            # cast only changes the scalar element type.
            node = replace(node, type=result_type)
            nodes[-1] = node
        return node

    def norm(source, inputs, *, wide_output):
        value, weight = inputs
        value = cast(value, DType.FLOAT32, source.id + ".wide_input")
        stats = emit("nn.norm_stats", source.id + ".stats", (value, ), {"axis": -1, "use_mean": False})
        bias = emit("builtin.splat_const", source.id + ".zero", (), {"result_type": weight.type, "value": 0.0})
        if source.attrs["weight_bias"] != 0:
            raise IRSchemaError("Pinned Llama normalization requires weight_bias=0")
        result = emit("nn.norm_apply", source.id if wide_output else source.id + ".wide", (value, stats, weight, bias),
                      {"axis": -1, "epsilon": source.attrs["epsilon"], "use_mean": False, "round_before_scale": False})
        return result if wide_output else cast(result, DType.BFLOAT16, source.id)

    for source in module.nodes:
        inputs = tuple(mapped[value] for value in source.inputs)
        if source.id == hidden_id or source.id in wide_parameters:
            result = replace(source, type=wide_hidden)
        elif source.op == "control.for":
            positions = wide_carries.get(str(source.attrs["callee"]), ())
            count = int(source.attrs["iter_count"])
            arguments = [value.id for value in inputs]
            for index in positions:
                initial = inputs[3 + index]
                arguments[3 + index] = cast(initial, DType.FLOAT32, source.id + f".carry_{index}.wide").id
            fields = tuple(wide_hidden if index in positions else field
                           for index, field in enumerate(source.type.fields[:count]))
            result = replace(source, inputs=tuple(arguments), type=TupleType(fields))
        elif source.op == "nn.rms_norm":
            result = norm(source, inputs, wide_output=False)
        elif source.op == "nn.rope":
            rotated = emit("nn.rope", source.id + ".wide", inputs)
            result = cast(rotated, DType.BFLOAT16, source.id)
        elif source.op == "nn.dense_matmul_glu":
            value, gate_weight, up_weight = inputs
            gate = emit("math.matmul", source.id + ".gate", (value, gate_weight), {"transpose_b": True})
            up = emit("math.matmul", source.id + ".up", (value, up_weight), {"transpose_b": True})
            activation = emit("math.silu", source.id + ".silu", (cast(gate, DType.FLOAT32, source.id + ".gate_wide"), ))
            product = emit("math.mul", source.id + ".product",
                           (activation, cast(up, DType.FLOAT32, source.id + ".up_wide")))
            result = cast(product, DType.BFLOAT16, source.id)
        elif source.id == "decode_layer_after_attention":
            residual = cast(cast(inputs[0], DType.BFLOAT16, source.id + ".residual_bf16"), DType.FLOAT32,
                            source.id + ".residual_wide")
            result = emit("math.add", source.id,
                          (residual, cast(inputs[1], DType.FLOAT32, source.id + ".projection_wide")))
        elif source.id == "decode_layer_output":
            result = emit("math.add", source.id,
                          (inputs[0], cast(inputs[1], DType.FLOAT32, source.id + ".projection_wide")))
        elif source.op == "builtin.call" and source.attrs.get("callee") == decode.name:
            argument = cast(inputs[0], DType.FLOAT32, source.id + ".wide_input")
            result = replace(source, inputs=(argument.id, *(value.id for value in inputs[1:])), type=TupleType(
                (wide_hidden, source.type.fields[1])))
        elif source.op == "builtin.get_item":
            result = replace(source, inputs=tuple(value.id for value in inputs),
                             type=inputs[0].type.fields[source.attrs["index"]])
        elif source.op == "tensors.slice":
            result = replace(source, inputs=tuple(value.id for value in inputs),
                             type=tensor_type(inputs[0].type.dtype, source.type.shape))
        else:
            result = replace(source, inputs=tuple(value.id for value in inputs))
        if not nodes or nodes[-1] is not result:
            nodes.append(result)
        mapped[source.id] = result
    specialized = replace(module, nodes=tuple(nodes),
                          metadata={**module.metadata, "numerical_contract": NUMERICAL_CONTRACT})
    return verify_module(specialized)


__all__ = ["apply_llama3_vllm_profile", "NUMERICAL_CONTRACT"]
