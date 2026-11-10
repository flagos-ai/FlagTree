# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit CPP SM120/48-CTA full-head candidate with parent warp state.

This provider is selected directly by the recipe instead of the parent GDN
provider. Its descriptor declares32B scalar statistics plus512B scalar head core; the normal
workspace annotator declares chip-global dual-Q8 storage. Main recipe
integration must finalize the combined arena after both r8 configurators and
then call the Vector-owned validate_selection; this module does not allocate
hidden Shared storage or install an old identity for the new arithmetic.
"""
from dataclasses import replace

from triton.flagmega.codegen.triton.candidates import TritonCandidateProviderRegistry
from triton.flagmega.codegen.triton.candidates import TritonCandidateProposal
from triton.flagmega.codegen.triton.candidates.gdn_head_gates import (
    _broadcast, _dense, _ref_derived, _shape, dense_value_geometry,
)
from triton.flagmega.ir import DType, RefType, VectorType
from triton.flagmega.ir.execution_mesh import effective_execution_mesh, ExecutionMeshKind
from triton.flagmega.codegen.triton.candidates.simple import GdnCandidateProvider
from triton.flagmega.codegen.triton.implementation import TritonImplementation, TritonImplementationModel
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import T, tensor_type
from triton.flagmega.targets import get_target

from .gdn_head_globalq8_qkregs_fullhead48_abi import IMPLEMENTATION, prepare_fields, validate_call, validate_selection


def implementation():
    return TritonImplementation(
        IMPLEMENTATION, "gdn_recurrent", "persistent_warp_state_head_globalq8_qkregs_fullhead48",
        {"tile_state": (128, 4), "projection_tile": 128, "num_warps": 8,
         "head_statistics_cache": True, "head_cache_slots": 2,
         "projection_order": "q8_residual2_f32scale_roundf", "qk_norm_order": "ascending_scalar_C1",
         "cache_contents": ("beta", "decay", "query_norm", "key_norm"),
         "cache_epoch": "per_GDN_invocation", "state_reduction_order": "warp32_xor_16_8_4_2_1",
         "state_rows_per_warp": 1, "state_values_per_lane": 4,
         "normalized_qk_math": "f32_division", "fullhead_shared_rms": True,
         "rms_order": "ascending_scalar_v0_to127", "q8_producer_map": "first4warps_first40ctas"},
        # A distinct provider proves the new geometry without altering old admission.
        {"fullhead48_original_order_rms": True, "required_codegen_backend": "cpp"},
        ("cooperative_grid", "grid_sync"),
        {"internal_grid_barriers": 1, "internal_cta_barriers": 3,
         "required_codegen_backend": "cpp", "state_reassociation": True,
         "state_bitwise_equivalent": False, "projection_bitwise_equivalent": False},
        (T.shared_workspace_descriptor("head_statistics", tensor_type("float32", (2, 4)), 16),
         T.shared_workspace_descriptor("head_core", tensor_type("float32", (128,)), 16)),
    )


class GlobalQ8QKRegsFullHead48Provider:
    def __init__(self, original):
        self.original, self.op_names = original, original.op_names

    def propose(self, node, context):
        if node.op != "nn.gdn_recurrent_core":
            return self.original.propose(node, context)
        if not fullhead48_applicable(node, context):
            raise CodegenError("fullhead48 requires actual CPP SM120/fixed4x12/8warps and unchanged dense nncase input boundaries")
        choices = [impl for impl in context.implementations("gdn_recurrent") if impl.id == IMPLEMENTATION]
        if len(choices) != 1:
            raise CodegenError("fullhead48 requires its explicitly registered implementation")
        candidate = context.configure_implementation(choices[0])
        return TritonCandidateProposal((candidate,), candidate.id)


def fullhead48_applicable(node, context):
    """Exact new launch admission, retaining parent tensor/numeric predicates."""
    if (node.op != "nn.gdn_recurrent_core" or context.codegen_backend != "cpp"
            or not context.cooperative_grid
            or getattr(context.target, "codegen_platform", None) != "nvidia"
            or getattr(context.target, "codegen_architecture", None) != "sm120"
            or node.metadata.get("kernel_num_warps") != 8):
        return False
    attrs = node.attrs
    flags = {"round_beta": True, "round_core": False, "round_normalized_qk": False,
             "qk_norm_mode": "clamp", "qk_norm_epsilon": 1e-12,
             "gate_math_mode": "accurate", "norm_math_mode": "fast"}
    if (attrs.get("b_weight_encoding") != "IQ4_XS" or attrs.get("a_weight_encoding") != "IQ4_XS"
            or any(attrs.get(k, v) != v for k, v in flags.items())
            or attrs.get("key_head_dim") != 128 or attrs.get("value_head_dim") != 128
            or attrs.get("num_key_heads") != 16 or attrs.get("num_value_heads") != 48
            or attrs.get("value_head_order", "grouped") != "tiled"):
        return False
    out = node.type.fields[0]
    geometry = dense_value_geometry(out, 48, 128)
    if geometry is None or geometry[0] != 128 or out.placement.hierarchy != (4, 12):
        return False
    mesh = effective_execution_mesh(context.module, node)
    if (mesh is None or mesh.kind != ExecutionMeshKind.FIXED
            or mesh.shape != (4, 12) or mesh.mapping != "row_major"):
        return False
    operands = [context.module.node_map[i].type for i in node.inputs]
    if operands[2] != out:
        return False
    for index, shape, dtype in (
            (1, (1, 10240), DType.BFLOAT16), (3, (1, 5120), DType.BFLOAT16),
            (4, (48, 2720), DType.UINT8), (5, (48, 2720), DType.UINT8),
            (6, (48,), DType.FLOAT32), (7, (48,), DType.FLOAT32), (8, (128,), DType.FLOAT32)):
        if not _broadcast(operands[index], out.placement, shape, dtype):
            return False
    if any(_ref_derived(i, context.module) for i in node.inputs[1:]):
        return False
    state = operands[0]
    if not isinstance(state, RefType):
        return False
    recurrent = dict(state.fields).get("recurrent")
    return (recurrent is not None and recurrent.dtype == VectorType(DType.FLOAT32, (4,))
            and _shape(recurrent) == (1, 48, 128, 32) and _dense(recurrent))


def enable():
    from triton.flagmega.codegen.triton import kernel_call_renderers as encoders
    from triton.flagmega.codegen.triton.physical_access import emit_logical_coordinate
    target = get_target("nvidia-sm120")
    model = target.triton_implementation_model
    if model.implementation(IMPLEMENTATION) is not None:
        raise CodegenError("Dual-Q8 fullhead48 implementation already installed")
    policy = target.tir_selection_policy
    if not isinstance(policy, TritonTirSelectionPolicy):
        raise CodegenError("Dual-Q8 fullhead48 requires the normal TIR selection policy")
    registry = TritonCandidateProviderRegistry()
    changed = 0
    for provider in policy.registry.providers:
        if isinstance(provider, GdnCandidateProvider):
            provider = GlobalQ8QKRegsFullHead48Provider(provider)
            changed += 1
        registry.add(provider)
    if changed != 1:
        raise CodegenError("Dual-Q8 fullhead48 must be selected directly, without the old head/cache selector")
    target.triton_implementation_model = TritonImplementationModel(
        (*model.implementations, implementation()), model.preferences, model.name + "+local-gdn-head-globalq8-qkregs-fullhead48-v1")
    def attach_global_q8(node, candidates, module, *, mesh_hierarchy):
        annotated = policy.workspace_annotator(node, candidates, module, mesh_hierarchy=mesh_hierarchy)
        result = []
        for candidate in annotated:
            if candidate.id == IMPLEMENTATION:
                workspaces = tuple(candidate.parameters.get("workspaces", ()))
                if tuple(w["name"] for w in workspaces) != ("core_scratch",):
                    raise CodegenError("Global-Q8 requires the unchanged normal core scratch requirement")
                requirement = {"name": "head_projection_q8_global",
                               "type": tensor_type("uint8", (2, 160, 36)),
                               "memory_space": "workspace", "alignment": 16, "lifetime": "invocation"}
                candidate = replace(candidate, parameters={**dict(candidate.parameters),
                                    "workspaces": (*workspaces, requirement)})
            result.append(candidate)
        return tuple(result)
    target.tir_selection_policy = TritonTirSelectionPolicy(attach_global_q8, registry)
    original = encoders._FAMILY_ENCODERS["gdn_recurrent"]

    def encode(raw):
        result = original(raw)
        if raw.get("implementation") != IMPLEMENTATION:
            return result
        # The core family encoder omits generic workspace metadata. Retain the
        # complete planner entries as well as the frozen ABI convenience fields.
        result["shared_workspaces"] = raw.get("shared_workspaces", ())
        result.update(prepare_fields(raw, result))
        out = next(x for x in raw["outputs"] if x["formal"] == "result_0")["buffers"][0]
        result["head_statistics_first_value"] = emit_logical_coordinate(out["abi"], 1, ("0", "0"))
        return result

    encoders._FAMILY_ENCODERS["gdn_recurrent"] = encode

    # CPP's shared descriptor builder resolves a semantic Triton closure before
    # replacing it with its CPP closure. Register this distinct identity only;
    # attempted Triton rendering is explicitly rejected by the metadata file.
    from pathlib import Path
    from jinja2 import ChoiceLoader, FileSystemLoader
    from triton.flagmega.codegen.triton import tir_package
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
    local_root = Path(__file__).parent / "tir_metadata"
    class GlobalQ8MetadataRegistry(TritonTemplateRegistry):
        def __init__(self):
            super().__init__()
            self.environment.loader = ChoiceLoader([FileSystemLoader(str(local_root)), self.environment.loader])
        def resolve(self, spec):
            if (spec.kernel, spec.variant, spec.platform, spec.architecture) == (
                    "gdn_recurrent", "persistent_warp_state_head_globalq8_qkregs_fullhead48", "nvidia", "sm120"):
                name = "kernels/gdn_recurrent/persistent_warp_state_head_globalq8_qkregs_fullhead48.py.jinja"
                if not (local_root / name).is_file():
                    raise CodegenError("Missing explicit CPP-only warp metadata template")
                return name
            return super().resolve(spec)
    tir_package.TritonTemplateRegistry = GlobalQ8MetadataRegistry
