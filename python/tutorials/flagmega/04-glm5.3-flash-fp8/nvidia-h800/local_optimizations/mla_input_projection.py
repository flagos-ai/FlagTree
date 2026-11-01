"""Explicit existing-op lowering for the measured complete-chip MLA KV_A row.

This is a workload-local IR choice, before distribution and packing. It keeps
FP32 scale multiply, BF16 weight rounding, two ordered FP32 sub-dots and the
original BF16 result. Both Q_A and KV_A use the independently validated parallel-split candidate.
"""
from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.functions.graph import function_nodes
from triton.flagmega.rules.neutral._utility import make_node
from .offline_projection_weights import decode_bf16_weight

MARKER = "coarse_native_input_projection"
ORIGINALS = ("decode_mla_attention_sparse_self_attn_q_a_proj_weight_apply",
             "decode_mla_attention_sparse_self_attn_kv_a_proj_with_mqa_weight_apply")
ROWS = dict(zip(ORIGINALS,(1536,512),strict=True))


def replaced_projections(module):
    report = module.metadata.get(MARKER)
    if report is None:
        return ()
    if (tuple(report["projections"]) != ORIGINALS or report["partitions"] != 1
            or report["accumulation_splits"] != 2):
        raise ValueError("Incomplete measured coarse input projection contract.")
    return ORIGINALS


def select_broadcast(graph,node):
    if (node.metadata.get(MARKER,{}).get("original") not in replaced_projections(graph.module)
            or node.attrs.get("accumulation_splits",1) != 2):
        raise ValueError("Only the explicitly measured complete input row is selected here.")
    def broadcast(kind):
        return (isinstance(kind,fm.DistributedType) and kind.partial is None and kind.exclusive is None
                and all(p == fm.SBP.broadcast() for p in kind.axis_policies))
    matches = [c for c in graph.bucket_map[node.id].candidates
               if broadcast(c.return_type) and all(broadcast(t) for t in c.input_types)]
    if len(matches) != 1:
        raise ValueError(f"Complete KV_A requires one full-chip broadcast owner contract: {[c.id for c in matches]}")
    return matches[0]


def verify_schedules(module):
    expected = set(replaced_projections(module))
    definitions = {d.name:d for d in module.kernel_definitions}
    found = set()
    for node in module.nodes:
        if node.op != "tir.call" or MARKER not in node.metadata:
            continue
        original = node.metadata[MARKER]["original"]
        dispatch = definitions[str(node.attrs["callee"])].dispatch
        if (dispatch.semantic_op != "nn.partitioned_linear" or dispatch.semantic_attrs["partitions"] != 1
                or dispatch.semantic_attrs.get("accumulation_splits",1) != 2
                or dispatch.microkernel.family != "partitioned_linear"
                or dispatch.microkernel.variant != "wgmma_local"
                or dispatch.microkernel.parameters.get("parallel_accumulation_splits") is not True
                or dispatch.microkernel.facts.get("internal_grid_barriers") != 1):
            raise ValueError("Complete KV_A lost the measured two-sub-dot TMA/WGMMA contract.")
        found.add(original)
    if found != expected:
        raise ValueError(f"Incomplete coarse input projection schedules: {found}")


def apply(module):
    if module.stage != "imported" or MARKER in module.metadata:
        raise ValueError("MLA input rows require a fresh imported native numerical profile.")
    owner_parameters = {n.id:set(f.parameters) for f in module.functions for n in function_nodes(module,f)}
    def roots(value,parameters):
        if value.id in parameters:
            return (value.id,)
        if not value.effect.is_pure:
            raise ValueError("Offline input projection decode cannot cross mutable effects.")
        return tuple(dict.fromkeys(root for identity in value.inputs for root in roots(module.node_map[identity],parameters)))
    nodes = []
    def emit(op,name,inputs,attrs,metadata=None):
        result = make_node(op,name,inputs,attrs,metadata or {})
        nodes.append(result)
        return result
    for source in module.nodes:
        if source.id not in ROWS:
            nodes.append(source)
            continue
        if (source.op != "math.block_scaled_matmul" or len(source.inputs) != 3
                or source.attrs.get("activation_quantization") != "none"
                or source.attrs.get("weight_dequant_dtype") != "bfloat16"
                or source.attrs.get("weight_encoding") is not None
                or source.attrs.get("output_lanes", ())):
            raise ValueError("Measured MLA input requires unquantized BF16 activations and decoded BF16 weights.")
        activation,weight,scale = (module.node_map[value] for value in source.inputs)
        n = ROWS[source.id]
        if (activation.type != fm.tensor_type("bfloat16",(1,4096))
                or weight.type.shape != fm.tensor_type("float32",(n,4096)).shape
                or source.type != fm.tensor_type("bfloat16",(1,n))):
            raise ValueError("The measured complete input row contract has changed.")
        parameters = owner_parameters[source.id]
        decoded = decode_bf16_weight(emit,weight,scale,source.attrs,source.id+".coarse_input_offline")
        provenance = tuple(dict.fromkeys((*roots(weight,parameters),*roots(scale,parameters))))
        if provenance:
            decoded = replace(decoded,metadata={**decoded.metadata,"packed_from":provenance})
            nodes[-1] = decoded
        result = emit("nn.partitioned_linear",source.id,(activation,decoded),
            {"partitions":1,"accumulation_splits":2},
            {**source.metadata,MARKER:{"original":source.id,"partitions":1,"accumulation_splits":2}})
        if result.type != source.type:
            raise ValueError("Complete input row changed the public KV_A result type.")
    return fm.verify_module(replace(module,nodes=tuple(nodes),metadata={**module.metadata,
        MARKER:{"projections":ORIGINALS,"partitions":1,"accumulation_splits":2}}))
