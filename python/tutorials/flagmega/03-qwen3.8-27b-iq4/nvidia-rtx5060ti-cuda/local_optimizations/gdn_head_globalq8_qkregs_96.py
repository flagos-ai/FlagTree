# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit CPP SM120/96-CTA dual-Q8 head candidate with parent warp state.

This provider is selected directly by the recipe instead of the parent GDN
provider. Its descriptor retains only planned32B Shared head statistics; the normal
workspace annotator declares chip-global dual-Q8 storage. Main recipe
integration must finalize the combined arena after both r8 configurators and
then call the Vector-owned validate_selection; this module does not allocate
hidden Shared storage or install an old identity for the new arithmetic.
"""
from dataclasses import replace

from triton.flagmega.codegen.triton.candidates import TritonCandidateProviderRegistry
from triton.flagmega.codegen.triton.candidates.gdn_head_gates import applicable as head_input_applicable
from triton.flagmega.codegen.triton.candidates.simple import GdnCandidateProvider
from triton.flagmega.codegen.triton.implementation import TritonImplementation, TritonImplementationModel
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import T, tensor_type
from triton.flagmega.targets import get_target

from .gdn_head_globalq8_qkregs_96_abi import IMPLEMENTATION, prepare_fields, validate_call, validate_selection


def implementation():
    return TritonImplementation(
        IMPLEMENTATION, "gdn_recurrent", "persistent_warp_state_head_globalq8_qkregs_96",
        {"tile_state": (128, 4), "projection_tile": 128, "num_warps": 4,
         "head_statistics_cache": True, "head_cache_slots": 2,
         "projection_order": "q8_residual2_f32scale_roundf", "qk_norm_order": "ascending_scalar_C1",
         "cache_contents": ("beta", "decay", "query_norm", "key_norm"),
         "cache_epoch": "per_GDN_invocation", "state_reduction_order": "warp32_xor_16_8_4_2_1",
         "state_rows_per_warp": 1, "state_values_per_lane": 4,
         "normalized_qk_math": "f32_division"},
        # This flag reuses only the common input/type/owner admission predicate.
        # The new identity declares its approximate head arithmetic explicitly.
        {"scalar_head_gates": True, "required_codegen_backend": "cpp"},
        ("cooperative_grid", "grid_sync"),
        {"internal_grid_barriers": 2, "internal_cta_barriers": 1,
         "required_codegen_backend": "cpp", "state_reassociation": True,
         "state_bitwise_equivalent": False, "projection_bitwise_equivalent": False},
        (T.shared_workspace_descriptor("head_statistics", tensor_type("float32", (2, 4)), 16),),
    )


class GlobalQ8QKRegsHead96Provider:
    def __init__(self, original):
        self.original, self.op_names = original, original.op_names

    def propose(self, node, context):
        proposal = self.original.propose(node, context)
        if node.op != "nn.gdn_recurrent_core":
            return proposal
        # No synthetic implementation or context is sent into old guards.
        out = node.type.fields[0]
        hierarchy = getattr(getattr(out, "placement", None), "hierarchy", None)
        if (not head_input_applicable(node, context) or hierarchy != (4, 24)
                or proposal is None or IMPLEMENTATION not in {c.id for c in proposal.candidates}):
            raise CodegenError("dual_q8_global_qkregs requires CPP SM120/block128, exact4x24 dense inputs and nncase boundaries")
        return replace(proposal, default_candidate=IMPLEMENTATION)


def enable():
    from triton.flagmega.codegen.triton import kernel_call_renderers as encoders
    from triton.flagmega.codegen.triton.physical_access import emit_logical_coordinate
    target = get_target("nvidia-sm120")
    model = target.triton_implementation_model
    if model.implementation(IMPLEMENTATION) is not None:
        raise CodegenError("Dual-Q8 head96 implementation already installed")
    policy = target.tir_selection_policy
    if not isinstance(policy, TritonTirSelectionPolicy):
        raise CodegenError("Dual-Q8 head96 requires the normal TIR selection policy")
    registry = TritonCandidateProviderRegistry()
    changed = 0
    for provider in policy.registry.providers:
        if isinstance(provider, GdnCandidateProvider):
            provider = GlobalQ8QKRegsHead96Provider(provider)
            changed += 1
        registry.add(provider)
    if changed != 1:
        raise CodegenError("Dual-Q8 head96 must be selected directly, without the old head/cache selector")
    target.triton_implementation_model = TritonImplementationModel(
        (*model.implementations, implementation()), model.preferences, model.name + "+local-gdn-head-globalq8-qkregs-96-v1")
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
                    "gdn_recurrent", "persistent_warp_state_head_globalq8_qkregs_96", "nvidia", "sm120"):
                name = "kernels/gdn_recurrent/persistent_warp_state_head_globalq8_qkregs_96.py.jinja"
                if not (local_root / name).is_file():
                    raise CodegenError("Missing explicit CPP-only warp metadata template")
                return name
            return super().resolve(spec)
    tir_package.TritonTemplateRegistry = GlobalQ8MetadataRegistry
