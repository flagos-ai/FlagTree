# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Qwen3 BF16 boundaries for vLLM 0.24.0 + FlagGems eager execution.

FlagGems fused_add_rms_norm normalizes the FP32 residual sum and publishes
its BF16 copy for the next residual addition. Standalone RMSNorm rounds
normalized values to BF16 before multiplying by the weight. Q/K RMSNorm
also publishes BF16 before RoPE. This profile belongs to this tutorial's reference runtime.
"""

from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DType, TupleType, tensor_type, verify_module
from triton.flagmega.rules.neutral._utility import make_node


def apply_flagos_profile(module):
    if module.stage != "imported" or "decode_layer" not in module.function_map:
        raise IRSchemaError("vLLM numerical specialization requires imported reusable decode_layer IR")
    if module.metadata.get("numerical_contract") == "vllm-flagos-eager-bf16":
        if (module.metadata.get("floating_point_contract") != "strict"
                or module.metadata.get("flagos_numerical_revision") != 3):
            raise IRSchemaError("FlagOS profile requires strict rounding; rebuild from the original checkpoint.")
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
        value = make_node(op, name, inputs, attrs or {}, {"numerical_contract": "vllm-flagos-fp32-norm"})
        nodes.append(value)
        return value

    def cast(value, dtype, name):
        if value.type.dtype == dtype:
            return value
        return emit("tensors.cast", name, (value,), {"dtype": dtype.value})

    def norm(source, inputs, *, wide_output):
        value, weight = inputs
        standalone = source.id in {"decode_layer_query_norm", "decode_layer_key_norm"}
        if not standalone:
            value = cast(value, DType.FLOAT32, source.id + ".wide_input")
        stats = emit("nn.norm_stats", source.id + ".stats", (value,), {"axis": -1, "use_mean": False})
        bias = emit("builtin.splat_const", source.id + ".zero", (), {"result_type": weight.type, "value": 0.0})
        if source.attrs["weight_bias"] != 0:
            raise IRSchemaError("Pinned Qwen3 normalization requires weight_bias=0")
        result = emit("nn.norm_apply", source.id if standalone or wide_output else source.id + ".wide",
                      (value, stats, weight, bias), {"axis": -1, "epsilon": source.attrs["epsilon"],
                                                   "use_mean": False, "round_before_scale": standalone})
        return result if wide_output else cast(result, DType.BFLOAT16, source.id)

    # Importer nodes are topological. Keep function identities and per-layer
    # weight-table metadata: only the numerical boundaries change.
    for source in module.nodes:
        inputs = tuple(mapped[value] for value in source.inputs)
        if source.id == hidden_id:
            result = replace(source, type=wide_hidden)
        elif source.op == "nn.qkv_parallel_linear":
            # Split-K partials must stay FP32 until the complete projection
            # is materialized. Each observable Q/K/V is still stored as BF16.
            result = emit(source.op, source.id, inputs,
                          {**source.attrs, "output_data_type": "float32"})
        elif source.op == "nn.rms_norm":
            result = norm(source, inputs, wide_output=False)
        elif source.op == "nn.rope":
            value, cosine, sine = inputs
            tables = tuple(cast(table, DType.BFLOAT16, source.id + f".table_{index}.bf16")
                           for index, table in enumerate((cosine, sine)))
            # RoPE computes in FP32 internally; its BF16 input records the
            # materialized Q/K RMSNorm result before rotation.
            result = emit("nn.rope", source.id, (value, *tables))
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
            # The post-attention fused norm publishes its residual in BF16.
            residual = cast(cast(inputs[0], DType.BFLOAT16, source.id + ".residual_bf16"),
                            DType.FLOAT32, source.id + ".residual_wide")
            result = emit("math.add", source.id, (residual, cast(inputs[1], DType.FLOAT32, source.id + ".projection_wide")))
        elif source.op == "builtin.call" and source.attrs.get("callee") == decode.name:
            argument = cast(inputs[0], DType.FLOAT32, source.id + ".wide_input")
            result = replace(source, inputs=(argument.id, *(value.id for value in inputs[1:])),
                             type=TupleType((wide_hidden, source.type.fields[1])))
        elif source.op == "builtin.get_item":
            if original[source.inputs[0]].op == "nn.qkv_parallel_linear":
                wide = emit(source.op, source.id + ".fp32", inputs, source.attrs)
                result = cast(wide, DType.BFLOAT16, source.id)
            else:
                result = replace(source, inputs=tuple(value.id for value in inputs),
                                 type=inputs[0].type.fields[source.attrs["index"]])
        else:
            result = replace(source, inputs=tuple(value.id for value in inputs))
        if not nodes or nodes[-1] is not result:
            nodes.append(result)
        mapped[source.id] = result
    specialized = replace(module, nodes=tuple(nodes), metadata={**module.metadata,
                          "numerical_contract": "vllm-flagos-eager-bf16",
                          "floating_point_contract": "strict",
                          "flagos_numerical_revision": 3})
    return verify_module(_specialize_first_input_norm(specialized))


def _specialize_first_input_norm(module):
    """Only the first layer uses standalone RMSNorm; later layers fuse a sum."""
    from triton.flagmega.passes.functions.graph import function_nodes

    function = module.function_map["decode_layer"]
    body = function_nodes(module, function)
    suffix = ".flagos_first"
    mapping = {node.id: node.id + suffix for node in body}
    nodes = []
    by_id = dict(module.node_map)
    for source in body:
        clone = replace(source, id=mapping[source.id],
                        inputs=tuple(mapping.get(value, value) for value in source.inputs))
        if source.id == "decode_layer_input_norm.wide":
            value = by_id[clone.inputs[0]]
            rounded = make_node("tensors.cast", clone.id + ".input_bf16", (value,), {"dtype": "bfloat16"}, clone.metadata)
            nodes.append(rounded)
            by_id[rounded.id] = rounded
            clone = make_node("nn.norm_apply", clone.id,
                              (rounded, *(by_id[name] for name in clone.inputs[1:])),
                              {**clone.attrs, "round_before_scale": True, "output_dtype": "bfloat16"},
                              clone.metadata)
        nodes.append(clone)
        by_id[clone.id] = clone
    first = replace(function, name="decode_layer_flagos_first",
                    parameters=tuple(mapping[x] for x in function.parameters),
                    outputs=tuple(mapping[x] for x in function.outputs),
                    attrs={**function.attrs, "specialized_from": function.name})
    updated = tuple(replace(node, attrs={**node.attrs, "callee": first.name})
                    if node.id == "layer_0_decode_layer_call" else node for node in module.nodes)
    return partition_first_layer_readonly_groups(
        replace(module, nodes=(*updated, *nodes), functions=(*module.functions, first)))


def partition_first_layer_readonly_groups(module):
    """Keep each reusable function's parameter groups complete after specialization.

    The first function may pack a parameter differently from the remaining
    layers. Group indices organize physical readonly storage, not model layer
    IDs; original checkpoint keys and call arguments stay unchanged.
    """
    layer_count = int(module.metadata["num_layers"])
    if layer_count <= 1:
        return module

    def update(node):
        group = node.metadata.get("rdata_group")
        if group is None or int(group["count"]) != layer_count:
            return node
        index = int(group["index"])
        first = index == 0
        group = {**group, "name": group["name"] + (".first" if first else ".remaining"),
                 "count": 1 if first else layer_count - 1, "index": 0 if first else index - 1}
        return replace(node, metadata={**node.metadata, "rdata_group": group})

    return replace(module, nodes=tuple(update(node) for node in module.nodes),
                   constant_recipes=tuple(replace(recipe, nodes=tuple(update(node) for node in recipe.nodes))
                                          for recipe in module.constant_recipes))
