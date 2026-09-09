# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pinned vLLM prefill/decode boundaries for the Qwen3.5 hybrid MoE frontend.

The source contract is vLLM ae10e855a, BF16, TP1, Inductor level 3,
custom_ops=['none'] and text-only fused QK norm/RoPE. GDN decode uses the
native recurrent path; prefill exposes the FlashInfer CP block stages for
chunks up to 512 tokens. A profile declares source semantics; full-model numerical acceptance remains a separate
device test, not a consequence of this metadata.
"""

from dataclasses import replace

from triton.flagmega.errors import ImporterError
from triton.flagmega.importer.numerics import VLLM_AE10_INDUCTOR_LEVEL3
from triton.flagmega.importer.numerics.gdn_prefill import emit_gdn_prefill
from triton.flagmega.ir import DType, TupleType, tensor_type, verify_module
from triton.flagmega.ir.ops.nn.gdn_recurrent_core import GatedDeltaNetRecurrentCore
from triton.flagmega.rules.neutral._utility import make_node


def apply_qwen35_moe_vllm_profile(module):
    if module.stage != "imported":
        raise ImporterError("The Qwen3.5 vLLM numerical contract requires imported IR.")
    phase = module.metadata.get("execution_phase", "decode")
    if phase not in ("decode", "prefill"):
        raise ImporterError("The pinned numerical contract requires an explicit decode or prefill phase.")
    if phase == "prefill" and module.metadata["tokens_per_call"] > 512:
        raise ImporterError("The pinned CP prefill contract requires chunks of at most 512 tokens.")
    if module.metadata.get("numerical_contract") == VLLM_AE10_INDUCTOR_LEVEL3:
        return verify_module(module)
    if module.metadata.get("numerical_contract", "nncase") != "nncase":
        raise ImporterError("Select the Qwen3.5 numerical contract from a fresh nncase import.")
    decoder_names = set(module.function_map) & {phase + "_linear", phase + "_attention"}
    if not decoder_names:
        raise ImporterError("The Qwen3.5 numerical contract requires reusable decoder functions.")
    original = module.node_map
    hidden_parameters = {module.function_map[name].parameters[0] for name in decoder_names}
    # These are the reusable attention function's cos/sin parameters. Their
    # storage dtype carries the source's table rounding across the call ABI.
    rotary_parameters = {name for node in module.nodes if node.op == "nn.rope"
                         for name in node.inputs[1:] if original[name].op == "builtin.var"}
    shared_gate_ids = set()
    for name in decoder_names:
        for suffix, op in (("_input_norm", "nn.norm_apply"), ("_attention_residual", "math.add"),
                           ("_moe_output", "math.add"), ("_output", "math.add")):
            if name + suffix not in original or original[name + suffix].op != op:
                raise ImporterError(f"Unexpected Qwen3.5 decoder topology at {name + suffix!r}.")
        scaled = original[original[name + "_moe_output"].inputs[1]]
        broadcast = original[scaled.inputs[1]] if scaled.op == "math.mul" else None
        gate = original[
            broadcast.inputs[0]] if broadcast is not None and broadcast.op == "tensors.broadcast_to" else None
        if gate is None or gate.op != "math.sigmoid":
            raise ImporterError(f"Unexpected Qwen3.5 shared-expert gate topology in {name!r}.")
        shared_gate_ids.add(gate.id)
    nodes, mapped = [], {}

    def emit(op, name, inputs=(), attrs=None, metadata=None):
        value = make_node(op, name, inputs, attrs or {},
                          {**(metadata or {}), "numerical_contract": VLLM_AE10_INDUCTOR_LEVEL3})
        nodes.append(value)
        return value

    def cast(value, dtype, name):
        return value if value.type.dtype == dtype else emit("tensors.cast", name, (value, ), {"dtype": dtype.value})

    def wide(value, name):
        return cast(value, DType.FLOAT32, name)

    for source in module.nodes:
        inputs = tuple(mapped[name] for name in source.inputs)
        if source.id in hidden_parameters:
            result = replace(source, type=tensor_type(DType.FLOAT32, source.type.shape))
        elif source.id in rotary_parameters:
            result = replace(source, type=tensor_type(DType.BFLOAT16, source.type.shape))
        elif source.op == "builtin.call" and source.attrs.get("callee") in decoder_names:
            hidden = wide(inputs[0], source.id + ".wide_hidden")
            result = replace(source, inputs=(hidden.id, *(value.id for value in inputs[1:])), type=TupleType(
                (hidden.type, source.type.fields[1])))
        elif source.op == "nn.gdn_convolution":
            result = emit(
                source.op, source.id, inputs, {
                    **source.attrs, "round_products": True, "round_before_activation": False, "accumulation_order":
                    "chronological"
                }, source.metadata)
        elif source.op == "nn.gdn_recurrent_core":
            if phase == "prefill":
                result = emit_gdn_prefill(source, inputs, emit)
                mapped[source.id] = result
                continue
            operands = list(inputs)
            parameter = GatedDeltaNetRecurrentCore.norm_weight
            operands[parameter.input_index] = cast(parameter.read(inputs), DType.BFLOAT16,
                                                   source.id + ".norm_weight_bf16")
            result = emit(
                source.op, source.id, tuple(operands), {
                    **source.attrs, "qk_norm_mode": "add", "qk_norm_epsilon": 1e-6, "round_normalized_qk": False,
                    "round_beta": True, "round_core": True
                }, source.metadata)
        elif source.op == "nn.rotary_embedding":
            result = emit(source.op, source.id, inputs,
                          {**source.attrs, "output_dtype": "bfloat16"}, source.metadata)
        elif source.op == "nn.dense_matmul_glu":
            gate = emit("math.matmul", source.id + ".gate", inputs[:2], {"transpose_b": True})
            up = emit("math.matmul", source.id + ".up", (inputs[0], inputs[2]), {"transpose_b": True})
            activated = emit("math.silu", source.id + ".silu", (wide(gate, source.id + ".gate_wide"), ))
            product = emit("math.mul", source.id + ".product", (activated, wide(up, source.id + ".up_wide")))
            result = cast(product, DType.BFLOAT16, source.id)
        elif source.op == "nn.sparse_experts":
            result = emit(
                source.op, source.id, inputs, {
                    **source.attrs, "round_projections": True, "round_activation": True, "round_down_projection": False,
                    "round_weighted_output": True, "intermediate_dtype": "bfloat16", "output_dtype": "bfloat16"
                }, source.metadata)
        elif source.id in {name + "_attention_residual" for name in decoder_names}:
            residual = wide(cast(inputs[0], DType.BFLOAT16, source.id + ".residual_bf16"), source.id + ".residual_wide")
            result = emit("math.add", source.id, (residual, wide(inputs[1], source.id + ".projection_wide")))
        elif source.id in {name + suffix for name in decoder_names for suffix in ("_moe_output", "_output")}:
            result = emit(
                "math.add", source.id,
                tuple(wide(value, source.id + f".operand_{index}_wide") for index, value in enumerate(inputs)))
        elif source.op == "math.sigmoid" and source.id not in shared_gate_ids:
            # Only the attention gate is inside the surrounding Inductor
            # graph. The shared-expert gate runs eagerly inside the opaque
            # MoE custom op and must retain its BF16 sigmoid result store.
            result = emit(source.op, source.id, (wide(inputs[0], source.id + ".input_wide"), ))
        elif source.op == "math.mul" and any(value.type.dtype == DType.FLOAT32 for value in inputs):
            product = emit(
                source.op, source.id + ".wide",
                tuple(wide(value, source.id + f".operand_{index}_wide") for index, value in enumerate(inputs)),
                source.attrs)
            result = cast(product, source.type.dtype, source.id)
        elif not source.inputs:
            result = source
        else:
            result = emit(source.op, source.id, inputs, source.attrs, source.metadata)
        if not nodes or nodes[-1] is not result:
            nodes.append(result)
        mapped[source.id] = result
    return verify_module(
        replace(
            module, nodes=tuple(nodes), metadata={
                **module.metadata, "numerical_contract": VLLM_AE10_INDUCTOR_LEVEL3, "numerical_profile_reference": {
                    "vllm": "ae10e855abf4ff5e24e2088aef16029ee1cb7de8", "phase": phase, "compile_mode": 3,
                    "custom_ops": ("none", ), "tensor_parallel_size": 1
                }
            }))
