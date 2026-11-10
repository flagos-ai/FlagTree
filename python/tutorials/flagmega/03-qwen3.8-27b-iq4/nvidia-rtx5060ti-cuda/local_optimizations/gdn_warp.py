"""Explicit SM120 warp-per-value GDN candidate with reassociated state sums."""
from dataclasses import replace

from triton.flagmega.codegen.triton.candidates import TritonCandidateProviderRegistry
from triton.flagmega.codegen.triton.candidates.gdn_head_gates import applicable as head_input_applicable
from triton.flagmega.codegen.triton.candidates.simple import GdnCandidateProvider
from triton.flagmega.codegen.triton.implementation import TritonImplementation, TritonImplementationModel
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import T, tensor_type
from triton.flagmega.targets import get_target

from .gdn_warp_abi import IMPLEMENTATION, prepare_fields, validate_call


def implementation():
    return TritonImplementation(
        IMPLEMENTATION, "gdn_recurrent", "persistent_warp_state",
        {"tile_state": (128, 4), "projection_tile": 128, "num_warps": 4,
         "head_statistics_cache": True, "head_cache_slots": 2,
         "projection_order": "ascending_scalar_C1", "qk_norm_order": "ascending_scalar_C1",
         "cache_contents": ("beta", "decay", "query_norm", "key_norm"),
         "cache_epoch": "per_GDN_invocation", "state_reduction_order": "warp32_xor_16_8_4_2_1",
         "state_rows_per_warp": 1, "state_values_per_lane": 4,
         "normalized_qk_math": "f32_division"},
        # This existing flag selects only the common input/type/owner admission
        # predicate; the new identity owns a separate32B ABI and numeric facts.
        {"scalar_head_gates": True, "required_codegen_backend": "cpp"},
        ("cooperative_grid", "grid_sync"),
        {"internal_grid_barriers": 1, "internal_cta_barriers": 1,
         "required_codegen_backend": "cpp", "state_reassociation": True,
         "state_bitwise_equivalent": False},
        (T.shared_workspace_descriptor("head_statistics", tensor_type("float32", (2, 4)), 16),),
    )


class WarpStateProvider:
    def __init__(self, original):
        self.original, self.op_names = original, original.op_names

    def propose(self, node, context):
        proposal = self.original.propose(node, context)
        if node.op != "nn.gdn_recurrent_core":
            return proposal
        # No synthetic implementation or context is sent into old guards.
        out = node.type.fields[0]
        hierarchy = getattr(getattr(out, "placement", None), "hierarchy", None)
        if (not head_input_applicable(node, context) or hierarchy not in ((4, 8), (4, 24))
                or proposal is None or IMPLEMENTATION not in {c.id for c in proposal.candidates}):
            raise CodegenError("gdn_warp_state requires CPP SM120/block128, certified4x8/4x24 dense inputs and nncase boundaries")
        return replace(proposal, default_candidate=IMPLEMENTATION)


def validate_selection(descriptor):
    calls = [c for c in descriptor["render_calls"] if c.get("family") == "gdn_recurrent"]
    if not calls or any(c.get("implementation") != IMPLEMENTATION for c in calls):
        raise CodegenError("gdn_warp_state requires its explicit selected identity for every recurrent call")
    for call in calls:
        # Recheck physical storage at final emission, including resumed input.
        expected = prepare_fields(call, call)
        if any(call.get(k) != v for k, v in expected.items()):
            raise CodegenError("Warp-state descriptor fields differ from actual planned32B statistics")
        validate_call(call, descriptor)
        if call["head_statistics_arena_bytes"] != 32:
            raise CodegenError("First warp-state resource profile requires exactly32B planned Shared")
    return [{"symbol": c["symbol"], "implementation": c["implementation"],
             "shared_workspaces": c["shared_workspaces"],
             "state_reassociation": True, "state_bitwise_equivalent": False} for c in calls]


def enable():
    from triton.flagmega.codegen.triton import kernel_call_renderers as encoders
    from triton.flagmega.codegen.triton.physical_access import emit_logical_coordinate
    target = get_target("nvidia-sm120")
    model = target.triton_implementation_model
    if model.implementation(IMPLEMENTATION) is not None:
        raise CodegenError("Warp-state implementation already installed")
    policy = target.tir_selection_policy
    if not isinstance(policy, TritonTirSelectionPolicy):
        raise CodegenError("Warp-state requires the normal TIR selection policy")
    registry = TritonCandidateProviderRegistry()
    changed = 0
    for provider in policy.registry.providers:
        if isinstance(provider, GdnCandidateProvider):
            provider = WarpStateProvider(provider)
            changed += 1
        registry.add(provider)
    if changed != 1:
        raise CodegenError("Warp-state must be selected directly, without the old head/cache selector")
    target.triton_implementation_model = TritonImplementationModel(
        (*model.implementations, implementation()), model.preferences, model.name + "+local-gdn-warp-state-v1")
    target.tir_selection_policy = TritonTirSelectionPolicy(policy.workspace_annotator, registry)
    original = encoders._FAMILY_ENCODERS["gdn_recurrent"]

    def encode(raw):
        result = original(raw)
        if raw.get("implementation") != IMPLEMENTATION:
            return result
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
    class WarpMetadataRegistry(TritonTemplateRegistry):
        def __init__(self):
            super().__init__()
            self.environment.loader = ChoiceLoader([FileSystemLoader(str(local_root)), self.environment.loader])
        def resolve(self, spec):
            if (spec.kernel, spec.variant, spec.platform, spec.architecture) == (
                    "gdn_recurrent", "persistent_warp_state", "nvidia", "sm120"):
                name = "kernels/gdn_recurrent/persistent_warp_state.py.jinja"
                if not (local_root / name).is_file():
                    raise CodegenError("Missing explicit CPP-only warp metadata template")
                return name
            return super().resolve(spec)
    tir_package.TritonTemplateRegistry = WarpMetadataRegistry
