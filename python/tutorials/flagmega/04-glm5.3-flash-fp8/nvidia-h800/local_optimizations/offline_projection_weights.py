# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Offline BF16 weight decoding for the native MLA projection contract.

This explicit experiment changes immutable storage, keeping FP32 scale
multiplication, BF16 weight rounding, and the native ordered K partitions.
It must run after native MLA absorption and before vectorization/packing.
"""

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.functions.graph import function_nodes
from triton.flagmega.passes.functions.lift_parameter_transforms import lift_parameter_constant_transforms
from triton.flagmega.passes.tir.fuse_bank_qkv_weights import fuse_bank_qkv_weights
from triton.flagmega.rules.neutral._utility import make_node
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.targets.nvidia.dense_matmul import packed_mma_accumulator_implementations

IMPLEMENTATION = "tir.dense_matmul.packed_k_major_mma_accumulator"
SPLIT_IMPLEMENTATION = IMPLEMENTATION + "_split2"
POLICY = "glm53-offline-mla-weight-decode/v1"
MARKER = "offline_projection_weight_decode"
_PREFIX = "decode_mla_attention_sparse_self_attn_"
NATIVE_SPLITS = {
    **{_PREFIX + f"o_proj_weight_apply.partition{p}.dot": 2 for p in range(4)},
    _PREFIX + "kv_a_proj_with_mqa_weight_apply": 2,
    _PREFIX + "q_a_proj_weight_apply": 2,
    _PREFIX + "q_b_proj_weight_apply": 1,
}

COARSE_MARKER = "coarse_native_row_projection"
COARSE_ORIGINAL = _PREFIX + "o_proj_weight_apply"
COARSE_REPLACED = tuple(COARSE_ORIGINAL + f".partition{p}.dot" for p in range(4))


def expected_native_splits(module):
    from .mla_input_projection import replaced_projections
    removed = set(replaced_projections(module))
    report=module.metadata.get(COARSE_MARKER)
    if report is None:
        return {name:splits for name,splits in NATIVE_SPLITS.items() if name not in removed}
    if (tuple(report["projections"])!=(COARSE_ORIGINAL,) or report["partitions"]!=4
            or report["accumulation_splits"]!=2):
        raise ValueError(f"Incomplete coarse native row projection contract: {report}")
    return {name:splits for name,splits in NATIVE_SPLITS.items() if name not in (*COARSE_REPLACED,*removed)}


def decode_bf16_weight(emit, weight, scale, attrs, prefix):
    """The shared pure recipe preserves FP32 scale multiplication and BF16 rounding."""
    n, k = (dimension.fixed_value for dimension in weight.type.shape)
    bn, bk = (int(attrs[key]) for key in ("weight_block_n", "weight_block_k"))
    ng, kg = ((n + bn - 1) // bn, (k + bk - 1) // bk)
    wide = emit("tensors.cast", prefix + ".wide", (weight,), {"dtype": "float32"})
    if n % bn or k % bk:
        wide = emit("tensors.pad", prefix + ".pad", (wide,), {"pad_end": (ng * bn - n, kg * bk - k)})
    blocked = emit("tensors.reshape", prefix + ".blocks", (wide,), {"shape": (ng, bn, kg, bk)})
    scales = emit("tensors.reshape", prefix + ".scales", (scale,), {"shape": (ng, 1, kg, 1)})
    scales = emit("tensors.broadcast_to", prefix + ".broadcast", (scales,), {"shape": (ng, bn, kg, bk)})
    decoded = emit("math.mul", prefix + ".decoded", (blocked, scales), {})
    rounded = emit("tensors.cast", prefix + ".rounded", (decoded,), {"dtype": "bfloat16"})
    decoded = emit("tensors.reshape", prefix + ".matrix", (rounded,), {"shape": (ng * bn, kg * bk)})
    if n % bn or k % bk:
        decoded = emit("tensors.slice", prefix + ".trim", (decoded,),
                       {"starts": (0, 0), "ends": (n, k), "axes": (0, 1)})
    return decoded


def apply(module):
    if module.stage != "imported":
        raise ValueError("Offline projection weights require the imported numerical profile.")
    owner_parameters = {
        node.id: set(function.parameters)
        for function in module.functions for node in function_nodes(module, function)
    }
    nodes, rewritten = [], []

    def emit(op, identity, inputs, attrs, metadata=None):
        value = make_node(op, identity, inputs, attrs, metadata or {})
        nodes.append(value)
        return value

    def roots(value, parameters):
        if value.id in parameters:
            return (value.id,)
        if not value.effect.is_pure:
            raise ValueError(f"Weight decode cannot cross an effect: {value.id}")
        return tuple(dict.fromkeys(root for item in value.inputs
                                   for root in roots(module.node_map[item], parameters)))

    for node in module.nodes:
        if (node.op != "math.block_scaled_matmul" or node.attrs.get("activation_quantization") != "none"
                or node.attrs.get("weight_dequant_dtype") != "bfloat16"):
            nodes.append(node)
            continue
        activation, weight, scale = (module.node_map[value] for value in node.inputs)
        if activation.type.dtype != fm.DType.BFLOAT16 or node.attrs.get("output_lanes", ()):
            raise ValueError(f"Unsupported offline projection input/output representation: {node.id}")
        decoded = decode_bf16_weight(emit, weight, scale, node.attrs, node.id + ".offline")
        parameters = owner_parameters.get(node.id, set())
        sources = tuple(dict.fromkeys((*roots(weight, parameters), *roots(scale, parameters))))
        if sources:
            decoded = replace(decoded, metadata={**decoded.metadata, "packed_from": sources})
            nodes[-1] = decoded
        # The FP32 matmul result is explicit: a BF16 matmul followed by a
        # widening cast would round before the consumer's numerical boundary.
        accumulator = emit("math.matmul", node.id + ".accumulator", (activation, decoded),
            {"transpose_b": True, "output_data_type": "float32"},
            {**node.metadata, MARKER: {"original": node.id, "block_k": int(node.attrs["weight_block_k"])}})
        final = emit("tensors.cast", node.id, (accumulator,), {"dtype": node.type.dtype.value}, node.metadata)
        if final.type != node.type:
            raise ValueError(f"Offline projection changed the result type: {node.id}")
        rewritten.append(node.id)
    if not rewritten:
        raise ValueError("No BF16-decoded FP8 projections matched the offline-weight experiment.")
    result = fm.verify_module(replace(module, nodes=tuple(nodes), metadata={**module.metadata,
        MARKER: {"policy": POLICY, "projections": tuple(rewritten)}}))
    result = lift_parameter_constant_transforms(result)
    return fm.verify_module(fuse_bank_qkv_weights(result))


def implementation_model(base):
    implementation, = packed_mma_accumulator_implementations()
    split = replace(implementation, id=SPLIT_IMPLEMENTATION,
                    parameters={**implementation.parameters, "accumulation_splits": 2})
    implementations = list(base.implementations)
    for value in (implementation, split):
        existing = base.implementation(value.id)
        if existing is None:
            implementations.append(value)
        elif existing != value:
            raise ValueError(f"Offline projection implementation has a conflicting contract: {value.id}")
    return replace(base, implementations=tuple(implementations))


def select(module):
    choices, originals = [], set()
    for point in module.selection_points:
        if point.kind != "tir" or point.owner not in module.node_map:
            continue
        node = module.node_map[point.owner]
        if MARKER not in node.metadata:
            continue
        original = node.metadata[MARKER]["original"]
        originals.add(original)
        implementation = SPLIT_IMPLEMENTATION if NATIVE_SPLITS.get(original, 1) == 2 else IMPLEMENTATION
        choices.append((point.id, implementation))
    if module.metadata.get("num_hidden_layers") == 45 and originals != set(expected_native_splits(module)):
        raise ValueError(f"Offline native projection set changed: {sorted(originals)}")
    return apply_plan(module, override_plan(module, choices, policy=POLICY,
        rationale="Offline BF16 weight decoding; retain a wide accumulator before explicit output rounding."))


def verify_native_schedules(module):
    from .mla_input_projection import verify_schedules
    verify_schedules(module)
    definitions = {definition.name: definition for definition in module.kernel_definitions}
    found = {}
    for node in module.nodes:
        if node.op != "tir.call" or MARKER not in node.metadata:
            continue
        original = node.metadata[MARKER]["original"]
        micro = definitions[str(node.attrs["callee"])].dispatch.microkernel
        if (micro.family != "dense_matmul" or micro.variant != "packed_k_major_mma_accumulator"
                or micro.parameters["accumulation_splits"] != NATIVE_SPLITS[original]):
            raise ValueError(f"Offline projection lost its ordered native accumulation: {node.id}")
        found[original] = micro.parameters["accumulation_splits"]
    expected=expected_native_splits(module)
    if module.metadata.get(COARSE_MARKER):
        coarse={}
        for node in module.nodes:
            if node.op!="tir.call" or COARSE_MARKER not in node.metadata:
                continue
            data=node.metadata[COARSE_MARKER]
            dispatch=definitions[str(node.attrs["callee"])].dispatch
            attrs=dispatch.semantic_attrs
            if (dispatch.semantic_op!="nn.partitioned_linear" or attrs["partitions"]!=4
                    or attrs.get("accumulation_splits",1)!=2
                    or dispatch.microkernel.family!="partitioned_linear"):
                raise ValueError(f"Coarse row projection lost ordered rounding: {node.id}")
            coarse[data["original"]]=(attrs["partitions"],attrs["accumulation_splits"])
        if coarse!={COARSE_ORIGINAL:(4,2)}:
            raise ValueError(f"Incomplete coarse native row projection schedules: {coarse}")
    if found != expected:
        raise ValueError(f"Incomplete offline native projection schedules: {found}")
    return tuple(module.kernel_definitions)
