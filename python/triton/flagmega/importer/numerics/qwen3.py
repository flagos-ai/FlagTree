# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Qwen3 import contract for pinned vLLM/Inductor fusion boundaries.

Inductor keeps pointwise intermediates in FP32 inside each attention-delimited
segment. BF16 stores (matmul outputs, attention inputs, carried residual) remain
real rounding boundaries. This is frontend semantics, not a codegen model
special case or a target-specific optimization. Generic passes consume only the
resulting casts, types and operation attributes, independently of this profile.

Reference: vLLM 493bd8323, Qwen3, Inductor level 3, BF16 decode. Other runtime
versions/configurations must validate their own profile, not inherit this claim.
"""

from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DType, TupleType, tensor_type, verify_module
from triton.flagmega.rules.neutral._utility import make_node


def apply_qwen3_vllm_profile(module):
    if module.stage != "imported" or "decode_layer" not in module.function_map:
        raise IRSchemaError("vLLM numerical specialization requires imported reusable decode_layer IR")
    if module.metadata.get("numerical_contract") == "vllm-493bd8323-inductor-level3":
        return verify_module(module)
    original = module.node_map
    decode = module.function_map["decode_layer"]
    required = {"decode_layer_input_norm", "decode_layer_after_attention", "decode_layer_output",
                "decode_layer_mlp_gate_up", "decode_layer_query_norm", "decode_layer_key_norm"}
    if (not required.issubset(original) or original["decode_layer_input_norm"].op != "nn.rms_norm"
            or original["decode_layer_mlp_gate_up"].op != "nn.dense_matmul_glu"):
        raise IRSchemaError("Unexpected decode-layer topology for the pinned vLLM numerical contract")
    hidden_id = decode.parameters[0]
    hidden_type = original[hidden_id].type
    wide_hidden = tensor_type(DType.FLOAT32, hidden_type.shape)
    nodes, mapped = [], {}

    def emit(op, name, inputs=(), attrs=None):
        value = make_node(op, name, inputs, attrs or {}, {"numerical_contract": "vllm-inductor-fp32-fusion"})
        nodes.append(value)
        return value

    def cast(value, dtype, name):
        if value.type.dtype == dtype:
            return value
        return emit("tensors.cast", name, (value,), {"dtype": dtype.value})

    def norm(source, inputs, *, wide_output):
        value, weight = inputs
        value = cast(value, DType.FLOAT32, source.id + ".wide_input")
        stats = emit("nn.norm_stats", source.id + ".stats", (value,), {"axis": -1, "use_mean": False})
        bias = emit("builtin.splat_const", source.id + ".zero", (), {"result_type": weight.type, "value": 0.0})
        if source.attrs["weight_bias"] != 0:
            raise IRSchemaError("Pinned Qwen3 normalization requires weight_bias=0")
        result = emit("nn.norm_apply", source.id if wide_output else source.id + ".wide",
                      (value, stats, weight, bias), {"axis": -1, "epsilon": source.attrs["epsilon"],
                                                   "use_mean": False, "round_before_scale": False})
        return result if wide_output else cast(result, DType.BFLOAT16, source.id)

    # Importer nodes are topological. Keep function identities and per-layer
    # weight-table metadata: only the numerical boundaries change.
    for source in module.nodes:
        inputs = tuple(mapped[value] for value in source.inputs)
        if source.id == hidden_id:
            result = replace(source, type=wide_hidden)
        elif source.op == "nn.rms_norm":
            result = norm(source, inputs, wide_output=source.id in {"decode_layer_query_norm", "decode_layer_key_norm"})
        elif source.op == "nn.rope":
            value, cosine, sine = inputs
            tables = tuple(cast(cast(table, DType.BFLOAT16, source.id + f".table_{index}.bf16"),
                                DType.FLOAT32, source.id + f".table_{index}.f32")
                           for index, table in enumerate((cosine, sine)))
            rotated = emit("nn.rope", source.id + ".wide", (value, *tables))
            result = cast(rotated, DType.BFLOAT16, source.id)
        elif source.op == "nn.dense_matmul_glu":
            value, gate_weight, up_weight = inputs
            gate = emit("math.matmul", source.id + ".gate", (value, gate_weight), {"transpose_b": True})
            up = emit("math.matmul", source.id + ".up", (value, up_weight), {"transpose_b": True})
            activation = emit("math.silu", source.id + ".silu", (cast(gate, DType.FLOAT32, source.id + ".gate_wide"),))
            product = emit("math.mul", source.id + ".product", (activation, cast(up, DType.FLOAT32, source.id + ".up_wide")))
            result = cast(product, DType.BFLOAT16, source.id)
        elif source.id == "decode_layer_after_attention":
            # The residual is materialized at the attention boundary. The
            # unrounded input remains available to the preceding RMSNorm.
            residual = cast(cast(inputs[0], DType.BFLOAT16, source.id + ".residual_bf16"),
                            DType.FLOAT32, source.id + ".residual_wide")
            result = emit("math.add", source.id, (residual, cast(inputs[1], DType.FLOAT32, source.id + ".projection_wide")))
        elif source.id == "decode_layer_output":
            result = emit("math.add", source.id, (inputs[0], cast(inputs[1], DType.FLOAT32, source.id + ".projection_wide")))
        elif source.op == "builtin.call" and source.attrs.get("callee") == decode.name:
            argument = cast(inputs[0], DType.FLOAT32, source.id + ".wide_input")
            result = replace(source, inputs=(argument.id, *(value.id for value in inputs[1:])),
                             type=TupleType((wide_hidden, source.type.fields[1])))
        elif source.op == "builtin.get_item":
            result = replace(source, inputs=tuple(value.id for value in inputs), type=inputs[0].type.fields[source.attrs["index"]])
        else:
            result = replace(source, inputs=tuple(value.id for value in inputs))
        if not nodes or nodes[-1] is not result:
            nodes.append(result)
        mapped[source.id] = result
    specialized = replace(module, nodes=tuple(nodes), metadata={**module.metadata,
                          "numerical_contract": "vllm-493bd8323-inductor-level3"})
    return verify_module(specialized)
