"""Exact GDN projection staging, with a new typed Shared contract.

This extension composes the complete existing scalar-head-gates contract with
one additional bounded workspace and a 20-chunk publication/reuse protocol.
The base ABI validator receives an explicit view of the unchanged base
subcontract; the actual new implementation and its 41 barriers remain in IR.
"""
from dataclasses import replace

from triton.flagmega.codegen.triton.candidates import TritonCandidateProviderRegistry
from triton.flagmega.codegen.triton.candidates.gdn_head_gates import IMPLEMENTATION as BASE
from triton.flagmega.codegen.triton.implementation import TritonImplementationModel
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import T, tensor_type
from triton.flagmega.targets import get_target

IMPLEMENTATION = "tir.gdn_recurrent.persistent_scalar_head_gates_chunk256_v1"
CHUNK = 256
CHUNKS = 20
CTA_BARRIERS = 2 * CHUNKS + 1


def _base_view(raw):
    """Project only the existing 16B gate contract for its existing validator."""
    return {**raw, "implementation": BASE,
            "shared_workspaces": [w for w in raw.get("shared_workspaces", ())
                                  if w.get("name") == "head_gates"],
            "facts": {**raw.get("facts", {}), "internal_cta_barriers": 1}}


def _workspace_fields(raw):
    ws = raw.get("shared_workspaces", ())
    if len(ws) != 2 or {w.get("name") for w in ws} != {"head_gates", "projection_dequant"}:
        raise CodegenError("Exact chunk gates require exactly the two planned Shared workspaces")
    dq = next(w for w in ws if w["name"] == "projection_dequant")
    offset = dq.get("offset_bytes")
    if (dq.get("dtype") != "float32" or tuple(dq.get("shape", ())) != (2, 2, CHUNK)
            or tuple(dq.get("strides", ())) != (2 * CHUNK, CHUNK, 1)
            or dq.get("nbytes") != 4096 or dq.get("alignment_bytes") != 16
            or type(offset) is not int or offset < 0 or offset % 16
            or not dq.get("physical_buffer") or dq.get("matrix_compatible", False)):
        raise CodegenError("Exact chunk gates require planned F32[2,2,256], 4096B/alignment16")
    gate = next(w for w in ws if w["name"] == "head_gates")
    # Separate logical workspaces may share the planner arena, but their live
    # spans overlap in time and therefore must be disjoint in byte coordinates.
    gate_offset = gate.get("offset_bytes")
    if type(gate_offset) is not int or not (offset + 4096 <= gate_offset or gate_offset + 16 <= offset):
        raise CodegenError("Gate and projection workspaces must have disjoint live spans")
    return {"local_head_gate_chunk": True, "head_dequant_offset_bytes": offset,
            "head_dequant_arena_bytes": max(offset + 4096, gate_offset + 16),
            "head_dequant_physical_buffer": dq["physical_buffer"]}


def validate_call(call, descriptor):
    if call.get("implementation") != IMPLEMENTATION:
        raise CodegenError("Requested exact chunk GDN implementation was not selected")
    from triton.flagmega.codegen.triton.gdn_head_gates import (
        prepare_fields as prepare_base, validate_call as validate_base,
    )
    base = _base_view(call)
    # Recheck typed base storage at the final package boundary as well as at
    # initial encoding; resumed or externally supplied descriptors can differ.
    expected_base = prepare_base(base, call)
    if any(call.get(k) != value for k, value in expected_base.items()):
        raise CodegenError("Exact chunk gate fields disagree with the 16B base workspace")
    base["head_gate_arena_bytes"] = call["head_gate_offset_bytes"] + 16
    validate_base(base, descriptor)
    expected = _workspace_fields(call)
    if any(call.get(k) != value for k, value in expected.items()):
        raise CodegenError("Exact chunk GDN fields disagree with planned Shared storage")
    params, facts = call.get("parameters", {}), call.get("facts", {})
    if (params.get("projection_chunk") != CHUNK
            or params.get("projection_staging") != "weights_f32_only"
            or params.get("projection_order") != "ascending_scalar_C1"
            or params.get("projection_accumulators") != "one_F32_per_row"
            or params.get("dequant_epoch") != "per_chunk_publish_consume"
            or facts.get("internal_cta_barriers") != CTA_BARRIERS
            or call.get("internal_grid_barriers") != 1
            or call["head_dequant_arena_bytes"] > descriptor["launch_resource_limits"]["max_shared_memory_bytes"]):
        raise CodegenError("Exact chunk GDN staging/order/barrier/resource contract changed")


def validate_selection(descriptor):
    calls = [c for c in descriptor["render_calls"] if c.get("family") == "gdn_recurrent"]
    if not calls:
        raise CodegenError("Requested exact chunk gates but module has no recurrent GDN call")
    for call in calls:
        validate_call(call, descriptor)
    return [{"symbol": c["symbol"], "implementation": c["implementation"],
             "shared_workspaces": c["shared_workspaces"],
             "arena_bytes": c["head_dequant_arena_bytes"],
             "cta_barriers": CTA_BARRIERS} for c in calls]


class ChunkCandidateProvider:
    def __init__(self, original):
        self.original, self.op_names = original, original.op_names

    def propose(self, node, context):
        proposal = self.original.propose(node, context)
        if node.op != "nn.gdn_recurrent_core":
            return proposal
        if proposal is None or not {BASE, IMPLEMENTATION} <= {c.id for c in proposal.candidates}:
            raise CodegenError("Exact chunk gates require the unchanged base admission guards")
        return replace(proposal, default_candidate=IMPLEMENTATION)


def enable():
    from .head_gates import HeadGateCandidateProvider
    from triton.flagmega.codegen.triton import kernel_call_renderers as encoders
    target = get_target("nvidia-sm120")
    model = target.triton_implementation_model
    base = model.implementation(BASE)
    if base is None or model.implementation(IMPLEMENTATION) is not None:
        raise CodegenError("Exact chunk implementation requires one unmodified base model")
    extension = replace(base, id=IMPLEMENTATION,
        parameters={**base.parameters, "projection_chunk": CHUNK,
                    "projection_staging": "weights_f32_only",
                    "projection_accumulators": "one_F32_per_row",
                    "dequant_epoch": "per_chunk_publish_consume"},
        facts={**base.facts, "internal_cta_barriers": CTA_BARRIERS},
        shared_workspaces=(*base.shared_workspaces,
            T.shared_workspace_descriptor("projection_dequant", tensor_type("float32", (2, 2, CHUNK)), 16)))
    policy = target.tir_selection_policy
    registry = TritonCandidateProviderRegistry()
    count = 0
    for provider in policy.registry.providers:
        if isinstance(provider, HeadGateCandidateProvider):
            provider = ChunkCandidateProvider(provider)
            count += 1
        registry.add(provider)
    if count != 1:
        raise CodegenError("Exact chunk gates require head_gate_cache=True and its scoped provider")
    target.triton_implementation_model = TritonImplementationModel(
        (*model.implementations, extension), model.preferences, model.name + "+local-exact-chunk256-v1")
    target.tir_selection_policy = TritonTirSelectionPolicy(policy.workspace_annotator, registry)
    original = encoders._FAMILY_ENCODERS["gdn_recurrent"]

    def encode(raw):
        if raw.get("implementation") != IMPLEMENTATION:
            return original(raw)
        # The ordinary encoder validates unchanged operand/alias/head geometry
        # through the base contract, then attaches actual new workspace fields.
        result = original(_base_view(raw))
        result.update(_workspace_fields(raw))
        return result

    encoders._FAMILY_ENCODERS["gdn_recurrent"] = encode
