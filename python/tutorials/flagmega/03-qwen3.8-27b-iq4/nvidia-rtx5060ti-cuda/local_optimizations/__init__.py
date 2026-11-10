"""Model-local native IQ4 MMVQ port for the NVIDIA C++ backend.

Install before compilation and artifact loading; generated source is never edited.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jinja2 import ChoiceLoader, FileSystemLoader

ROOT = Path(__file__).resolve().parent
_CONFIG = None
_ORIGINAL_DESCRIBE = None


def _source_identity():
    files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(ROOT.rglob("*"))
             if p.suffix in (".py", ".jinja", ".h", ".json", ".txt") and "__pycache__" not in p.parts}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def install(*, mode="iq4_mmvq", precision="dual_glu", head_gate_cache=False, interleaved="off", async_weights="off", async_interleaved=False, head_gate_chunk=False, gdn_warp_state=False, gdn_head_projection="off", u4_scope=None, u4_fixture=False, conv_channel128_fixed96=False, input_norm_q8=False, output_norm_owner=False):
    """Port llama.cpp IQ4 MMVQ to the selected ordinary CPP native calls.

    precision=single uses one Q8 component, dual_glu adds residual Q8 for GLU,
    and dual_all adds it for every replaced matrix. Scale storage is F32.
    interleaved=single/dual explicitly enables that Q8 approximation for
    single-token K5120 interleaved projections; off preserves core behavior.
    head_gate_cache=True explicitly selects the core scalar-ordered CTA head
    cache after its existing applicability guards. Other families and Q4_K
    retain their normal core implementation.
    gdn_warp_state=True selects the separate32B warp-state contract with
    explicit state-sum reassociation. It can compose with uniform128; the
    former head/cache flags remain mutually exclusive with this new mode.
    gdn_head_projection=dual_q8_warp is an explicit approximate head experiment
    scoped to the r8/all recipe. Its distinct provider replaces the old warp
    provider and validates the combined Shared arena after r8 configuration.
    """
    global _CONFIG, _ORIGINAL_DESCRIBE
    if mode != "iq4_mmvq" or precision not in ("single", "dual_glu", "dual_all"):
        raise ValueError("Expected mode=iq4_mmvq and precision=single/dual_glu/dual_all")
    if type(head_gate_cache) is not bool:
        raise ValueError("head_gate_cache must be a JSON boolean")
    if interleaved not in ("off", "single", "dual"):
        raise ValueError("Expected interleaved=off/single/dual")
    if type(head_gate_chunk) is not bool or (head_gate_chunk and not head_gate_cache):
        raise ValueError("head_gate_chunk requires head_gate_cache=True and a JSON boolean")
    if type(gdn_warp_state) is not bool or (gdn_warp_state and (head_gate_cache or head_gate_chunk)):
        raise ValueError("gdn_warp_state must be a JSON boolean and requires both old head/cache flags disabled")
    if head_gate_chunk and async_weights != "off":
        raise ValueError("head_gate_chunk=True cannot compose with async_weights: static Shared exceeds the reviewed limit")
    if async_weights not in ("off", "uniform128", "uniform128_r8"):
        raise ValueError("Expected async_weights=off/uniform128/uniform128_r8")
    if async_weights != "off" and (precision != "dual_glu" or not (head_gate_cache or gdn_warp_state) or interleaved != "dual"):
        raise ValueError("Uniform128 requires dual_glu, one explicit head_gate_cache/gdn_warp_state mode, interleaved=dual")
    if type(async_interleaved) is not bool or async_interleaved and async_weights not in ("uniform128", "uniform128_r8"):
        raise ValueError("async_interleaved must be boolean and requires uniform128 or uniform128_r8")
    if gdn_head_projection not in ("off", "dual_q8_warp", "dual_q8_global", "dual_q8_global_qkregs", "gdn_head_globalq8_qkregs_fullhead48"):
        raise ValueError("Expected gdn_head_projection=off/dual_q8_warp/dual_q8_global/dual_q8_global_qkregs/gdn_head_globalq8_qkregs_fullhead48")
    if gdn_head_projection in ("dual_q8_warp", "dual_q8_global", "dual_q8_global_qkregs", "gdn_head_globalq8_qkregs_fullhead48") and (
            not gdn_warp_state or head_gate_cache or head_gate_chunk
            or precision != "dual_glu" or interleaved != "dual"
            or async_weights != "uniform128_r8" or not async_interleaved):
        raise ValueError("dual_q8_warp requires gdn_warp_state, dual_glu, dual interleaved and uniform128_r8/all; old head selectors disabled")
    if type(u4_fixture) is not bool:
        raise ValueError("u4_fixture must be a boolean")
    if u4_scope not in (None, "motif", "full12"):
        raise ValueError("u4_scope must be motif/full12 or None")
    if u4_scope is not None and gdn_head_projection not in ("dual_q8_warp", "dual_q8_global", "dual_q8_global_qkregs", "gdn_head_globalq8_qkregs_fullhead48"):
        raise ValueError("U4 preserves the t9 dual-Q8 head recipe")
    if gdn_head_projection == "gdn_head_globalq8_qkregs_fullhead48" and (u4_scope != "full12" or u4_fixture):
        raise ValueError("fullhead48 requires explicit full12 normal distribution planning")
    if type(conv_channel128_fixed96) is not bool or (conv_channel128_fixed96 and u4_scope != "full12"):
        raise ValueError("conv_channel128_fixed96 requires full12 and a boolean")
    if type(input_norm_q8) is not bool or (input_norm_q8 and u4_scope != "full12"):
        raise ValueError("input_norm_q8 requires full12 and a boolean")
    if type(output_norm_owner) is not bool or (output_norm_owner and (
            not input_norm_q8 or u4_scope != "full12" or u4_fixture
            or gdn_head_projection != "gdn_head_globalq8_qkregs_fullhead48")):
        raise ValueError("output_norm_owner requires the exact complete inherited common-Q8 recipe")
    config = {"mode": mode, "precision": precision}
    if output_norm_owner:
        config["output_norm_owner"] = True
    if input_norm_q8:
        config["input_norm_q8"] = True
    if conv_channel128_fixed96:
        config["conv_channel128_fixed96"] = True
    if u4_fixture:
        config["u4_fixture"] = True
    if u4_scope is not None:
        config["u4_scope"] = u4_scope
    if gdn_head_projection != "off":
        config["gdn_head_projection"] = gdn_head_projection
    if async_interleaved:
        config["async_interleaved"] = True
    if async_weights != "off":
        config["async_weights"] = async_weights
    if interleaved != "off":
        config["interleaved"] = interleaved
    if head_gate_cache:
        config["head_gate_cache"] = True
    if head_gate_chunk:
        config["head_gate_chunk"] = True
    if gdn_warp_state:
        config["gdn_warp_state"] = True
    if output_norm_owner:
        from .output_norm_owner import validate_recipe
        validate_recipe(config)
    if _CONFIG is not None:
        if _CONFIG != config:
            raise ValueError("Cannot change an installed recipe; use a fresh process")
        return
    from triton.flagmega.codegen.cpp import renderer
    from triton.flagmega.codegen.cpp.templates import CppTemplateRegistry
    from triton.flagmega.errors import CodegenError

    class IQ4Registry(CppTemplateRegistry):
        def __init__(self):
            super().__init__()
            self.environment.loader = ChoiceLoader([
                FileSystemLoader(str(ROOT / "templates")), self.environment.loader])

        def available(self, spec):
            return any((ROOT / "templates" / path).is_file() for path in self.candidates(spec)) or super().available(spec)

        def resolve(self, spec):
            for path in self.candidates(spec):
                if (ROOT / "templates" / path).is_file():
                    return path
            return super().resolve(spec)

    if head_gate_cache:
        from .head_gates import enable_head_gate_cache
        enable_head_gate_cache()
    if head_gate_chunk:
        from .gdn_chunk import enable as enable_chunk
        enable_chunk()

    if gdn_head_projection == "dual_q8_warp":
        from .gdn_head_dualq8_96 import enable as enable_head_dualq8
        enable_head_dualq8()
    elif gdn_head_projection == "gdn_head_globalq8_qkregs_fullhead48":
        from .gdn_head_globalq8_qkregs_fullhead48 import enable as enable_fullhead48
        enable_fullhead48()
    elif gdn_head_projection == "dual_q8_global_qkregs":
        from .gdn_head_globalq8_qkregs_96 import enable as enable_head_globalq8_qkregs
        enable_head_globalq8_qkregs()
    elif gdn_head_projection == "dual_q8_global":
        from .gdn_head_globalq8_96 import enable as enable_head_globalq8
        enable_head_globalq8()
    elif gdn_warp_state:
        from .gdn_warp import enable as enable_warp
        enable_warp()

    if async_weights != "off":
        if async_weights == "uniform128_r8":
            from .r8_async_pipeline import enable
            from .r8_async_interleaved import enable as enable_interleaved
        else:
            from .async_pipeline import enable
            from .async_interleaved import enable as enable_interleaved
        enable()
        if async_interleaved:
            enable_interleaved()

    original = renderer.describe_cpp_package

    def configure_entry(result, *, typed_owner=None):
        if result["platform"] != "nvidia" or result["target_architecture"] != "sm120":
            raise CodegenError("The local IQ4 MMVQ recipe is scoped to NVIDIA SM120")
        if result["warp_size"] != 32 or result["block_threads"] % 32:
            raise CodegenError("IQ4 MMVQ requires complete 32-lane warps")
        if input_norm_q8:
            from .u4_norm_q8_kernel import validate_context as validate_norm_q8_context
            result["norm_q8_owner_contracts"] = validate_norm_q8_context(result)
        if conv_channel128_fixed96:
            from .conv_channel128 import validate_context
            result["conv_channel_owner_contracts"] = validate_context(result)
        scratch_bytes = 0
        sites = []
        for call in result["render_calls"]:
            is_interleaved = call.get("family") == "interleaved_projection"
            selected = call.get("family") in ("block_fp8", "matmul_glu") or (is_interleaved and interleaved != "off")
            if not selected or call.get("weight_encoding") != "IQ4_XS":
                continue
            if call.get("variant") != "simt":
                raise CodegenError("The local IQ4 MMVQ recipe requires the normal simt native call")
            k = int(call["k"])
            if k <= 0 or k % 256:
                raise CodegenError("IQ4 MMVQ requires positive complete 256-element blocks")
            if is_interleaved and (call.get("local_m") != 1 or k != 5120):
                raise CodegenError("The local interleaved port requires single-token K5120 decode")
            dual = (interleaved == "dual") if is_interleaved else (precision == "dual_all" or (precision == "dual_glu" and call["family"] == "matmul_glu"))
            call["local_iq4_dual"] = dual
            call["local_iq4_mmvq"] = True
            nbytes = (k // 32) * 36 * (2 if dual else 1)
            scratch_bytes = max(scratch_bytes, nbytes)
            sites.append({"symbol": call["symbol"], "family": call["family"], "k": k,
                          "local_n_capacity": call["local_n_capacity"], "dual": dual,
                          "scratch_bytes": nbytes})
        if not u4_fixture and u4_scope is None and interleaved != "off" and not any(site["family"] == "interleaved_projection" for site in sites):
            raise CodegenError("The explicit interleaved recipe requires an eligible IQ4 projection site")
        if not sites and u4_scope is None:
            raise CodegenError("The selected module has no compatible IQ4 matrix sites")
        if u4_scope is not None or u4_fixture:
            # Splitting GLU out of a fixed entry removes the old dominating
            # arena size. Cover this entry's real declared legacy spans; the
            # unchanged family validators below still check every span.
            for call in result["render_calls"]:
                if not call.get("local_iq4_mmvq"):
                    continue
                for span in call.get("shared_workspaces", ()):
                    start, size = span.get("offset_bytes"), span.get("nbytes")
                    if type(start) is not int or type(size) is not int or start < 0 or size <= 0:
                        raise CodegenError("Legacy entry has an unproved Shared span")
                    scratch_bytes = max(scratch_bytes, start + size)
        if scratch_bytes > result["launch_resource_limits"]["max_shared_memory_bytes"]:
            raise CodegenError("Local IQ4 activation scratch exceeds the per-CTA shared limit")
        result["local_iq4_shared_bytes"] = scratch_bytes
        result["local_iq4_recipe"] = {
            **config, "source_sha256": _source_identity(),
            "upstream_revision": "5ce06fc71f5e938301f7dc51697a59ae308bd6b7",
            "sites": sites,
            "scratch": {"name": "_local_iq4_q8", "scope": "CTA-private shared", "nbytes": scratch_bytes,
                        "alignment": 16, "lifetime": "one synchronous matrix call",
                        "publication": "CTA barrier after quantize and before reuse",
                        "aliases": "one arena reused only by sequential IQ4 matrix calls",
                        "resource_admission": "CUDA compiled static shared and actual occupancy"},
        }
        if gdn_warp_state and gdn_head_projection == "off":
            from .gdn_warp import validate_selection
            result["local_iq4_recipe"]["gdn_warp_state_sites"] = validate_selection(result)
        elif head_gate_chunk:
            from .gdn_chunk import validate_selection
            result["local_iq4_recipe"]["head_gate_chunk_sites"] = validate_selection(result)
        elif head_gate_cache:
            from .head_gates import validate_head_gate_selection
            result["local_iq4_recipe"]["head_gate_sites"] = validate_head_gate_selection(result)
        needs_legacy_arena = bool(sites) or any(c.get("family") == "gdn_recurrent" for c in result["render_calls"])
        helpers = ("kernels/iq4_mmvq_helpers.cpp.jinja",) if needs_legacy_arena else ()
        if async_weights != "off":
            if async_weights == "uniform128_r8":
                from .r8_async_pipeline import configure_descriptor
                from .r8_async_interleaved import configure_descriptor as configure_interleaved
                helper = "kernels/iq4_async_r8_helpers.cpp.jinja"
            else:
                from .async_pipeline import configure_descriptor
                from .async_interleaved import configure_descriptor as configure_interleaved
                helper = "kernels/iq4_async_helpers.cpp.jinja"
            if any(site["family"] in ("block_fp8", "matmul_glu") for site in sites):
                configure_descriptor(result)
            if async_interleaved and any(site["family"] == "interleaved_projection" for site in sites):
                configure_interleaved(result)
            if needs_legacy_arena:
                helpers += (helper,)
        if gdn_head_projection in ("dual_q8_warp", "dual_q8_global", "dual_q8_global_qkregs", "gdn_head_globalq8_qkregs_fullhead48") and any(c.get("family") == "gdn_recurrent" for c in result["render_calls"]):
            if gdn_head_projection in ("dual_q8_global", "dual_q8_global_qkregs", "gdn_head_globalq8_qkregs_fullhead48"):
                from .gdn_head_globalq8_arena import configure_descriptor as configure_head_arena
            else:
                from .gdn_head_arena import configure_descriptor as configure_head_arena
            arena_context = result
            if typed_owner is not None:
                # A split entry directly executes its owning function's leaves.
                # Preserve every event and planned span; interpret the old
                # validator's root in that verified function context.
                arena_context = {**result, "function_schedule": {
                    **result["function_schedule"], "function": typed_owner}}
            configure_head_arena(arena_context)
            if arena_context is not result:
                result["local_iq4_shared_bytes"] = arena_context["local_iq4_shared_bytes"]
                result["local_iq4_recipe"] = arena_context["local_iq4_recipe"]
                result["local_iq4_recipe"]["scoped_root_binding"] = {
                    "source_root": result["function_schedule"]["function"],
                    "owning_function": typed_owner,
                    "program_symbol": result["symbol"],
                    "actual_calls": [c["call"] for c in result["render_calls"]]}
        if head_gate_chunk:
            helpers = ("kernels/gdn_exact_chunk_helpers.cpp.jinja", *helpers)
        result["kernel_templates"] = (*helpers, *result["kernel_templates"])
        return result

    def describe(module):
        if input_norm_q8:
            from .u4_norm_q8 import CONTRACT_ID
            if module.metadata.get("u4_input_norm_q8") != CONTRACT_ID:
                # A56 is a distinct actual three-result producer. Preserve the
                # original A49 branch and require the complete new physical
                # contract instead of relabelling metadata as the old producer.
                from .u4_norm_q8_c1_c2 import CONTRACT_ID as COMMON_Q8_CONTRACT_ID
                if module.metadata.get("u4_input_norm_q8") != COMMON_Q8_CONTRACT_ID:
                    raise CodegenError("A49 recipe requires the matching complete input norm/Q8 rewrite")
                from triton.flagmega.ir import verify_buffer_plan
                from .u4_common_q8_distribution import validate_common_q8_physical_contract
                validate_common_q8_physical_contract(
                    module, verify_buffer_plan(module),
                    records=module.metadata.get("u4_input_norm_q8_records", ()))
        if u4_scope is not None:
            from .u4_encoders import register_gather_context
            with register_gather_context(module):
                result = original(module)
        else:
            result = original(module)
        if output_norm_owner:
            from .output_norm_owner import validate_and_annotate
            validate_and_annotate(module, result)
        if u4_scope is None and not u4_fixture:
            return configure_entry(result)
        # The complete model contract is checked from typed semantic call IDs;
        # each generated entry gets only its own legacy Shared arena.
        from .u4_regions import SITES, MOTIF
        records = module.metadata.get("u4_site_records", ())
        expected = MOTIF if u4_scope == "motif" else frozenset(SITES)
        if u4_fixture:
            expected = frozenset(("decode_linear_0_mlp_gate_up", "decode_linear_0_mlp_down")) if u4_scope else frozenset()
            if module.metadata.get("u4_fixture") != "complete-glu-down" or tuple(module.metadata.get("fixture_outputs", ())) != ("X", "Y", "Z", "O", "norm_stats"):
                raise CodegenError("Fixture registration requires complete GLU/down outputs")
        if (u4_scope is not None and module.metadata.get("u4_rewrite_scope") != u4_scope) or {r["original"] for r in records} != expected:
            raise CodegenError("U4 registration requires the matching complete rewrite census")
        contexts = result.get("native_program_contexts")
        if not contexts and u4_fixture and u4_scope is None:
            return configure_entry(result)
        if not contexts:
            raise CodegenError("U4 requires ordinary compiler native program contexts")
        partition = {p["symbol"]: p for p in module.metadata.get("kernel_program_partition", ())}
        scoped_validation_contexts = []
        for context in contexts:
            program = partition.get(context["symbol"])
            owner = context.get("owning_function", context["function_schedule"]["function"])
            program_owner = None if program is None else program.get("function")
            if program is not None and program_owner is None and set(module.function_map) == {module.entry}:
                program_owner = module.entry
            if program is None or owner != program_owner or set(program["calls"]) != {c["call"] for c in context["render_calls"]}:
                raise CodegenError("Local arena context differs from typed program ownership/call closure")
            configure_entry(context, typed_owner=owner)
            scoped_validation_contexts.append({**context, "function_schedule": {
                **context["function_schedule"], "function": owner}})
        result["local_iq4_recipe"] = {**config, "source_sha256": _source_identity(),
            "upstream_revision": "5ce06fc71f5e938301f7dc51697a59ae308bd6b7",
            "per_entry_arenas": [{"entry_id": c["entry_id"],
                "nbytes": c["local_iq4_shared_bytes"], "recipe": c["local_iq4_recipe"]}
                for c in contexts]}
        if gdn_head_projection == "gdn_head_globalq8_qkregs_fullhead48":
            from .gdn_head_globalq8_qkregs_fullhead48_abi import validate_selection
        elif gdn_head_projection == "dual_q8_global_qkregs":
            from .gdn_head_globalq8_qkregs_96_abi import validate_selection
        elif gdn_head_projection == "dual_q8_global":
            from .gdn_head_globalq8_96_abi import validate_selection
        else:
            from .gdn_head_dualq8_96_abi import validate_selection
        if not u4_fixture:
            result["local_iq4_recipe"]["gdn_head_projection_sites"] = validate_selection({
                **result, "native_program_contexts": scoped_validation_contexts})
        return result

    renderer.CppTemplateRegistry = IQ4Registry
    renderer.describe_cpp_package = describe
    _ORIGINAL_DESCRIBE = original
    if u4_scope is not None:
        from .u4_execution_plan import enable_distribution
        from .u4_mmvq import enable as enable_u4
        enable_distribution()
        enable_u4()
    if input_norm_q8:
        from .u4_norm_q8_kernel import enable as enable_norm_q8
        enable_norm_q8()
    if conv_channel128_fixed96:
        from .conv_channel128 import enable as enable_conv
        enable_conv()
    _CONFIG = config
