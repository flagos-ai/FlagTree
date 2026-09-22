"""CommonIR layout candidate composition, injection and compilation."""

import hashlib
import logging

from triton._C.libtriton import ir, metax, passes
from triton._common_ir import ENABLED
from triton.runtime.cache import get_cache_manager


def compose_kernel_candidates(domains):
    """Return dot_id -> plan mappings, with the first shared profile first.

    Shared profiles follow the first dot's order, using the first matching plan
    per dot. With no shared profile, use each dot's first plan. No dots produce
    one empty assignment. Selected plans are reused; inputs are not modified.
    """
    fallback = {}
    profiles_by_dot = {}
    for domain in domains:
        dot_id, plans = domain["dot_id"], domain["plans"]
        if dot_id in profiles_by_dot:
            raise ValueError(f"duplicate dot_id: {dot_id}")
        if not plans:
            raise ValueError(f"no layout plans for dot: {dot_id}")
        fallback[dot_id] = plans[0]
        by_profile = {}
        for plan in plans:
            by_profile.setdefault(plan["profile_id"], plan)
        profiles_by_dot[dot_id] = by_profile

    if not profiles_by_dot:
        return [fallback]
    candidates = []
    for profile_id in next(iter(profiles_by_dot.values())):
        if all(profile_id in profiles for profiles in profiles_by_dot.values()):
            candidate = {dot_id: profiles[profile_id] for dot_id, profiles in profiles_by_dot.items()}
            candidates.append(candidate)
    return candidates or [fallback]


def apply_dot_layout_candidate(module, candidate, capability, num_warps):
    """Apply one dot_id -> plan assignment to TTGIR in the same context."""
    layouts = {dot_id: [plan[key] for key in ("mma", "operand_a", "operand_b")] for dot_id, plan in candidate.items()}
    pm = ir.pass_manager(module.context)
    pm.enable_debug()
    passes.commonir.add_inject_dot_plan(pm, capability, num_warps, layouts)
    pm.run(module, "apply_dot_layout_candidate")
    return module


def prepare_dot_layouts(module, options, capability):
    pm = ir.pass_manager(module.context)
    passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", options.num_warps, 64, options.num_ctas)
    passes.ttgpuir.add_coalesce(pm)
    passes.ttgpuir.add_f32_dot_tc(pm, False)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_thread_locality(pm)
    pm.run(module, "commonir_prepare_dot_layouts")
    return module


def finalize_dot_layouts(module, capability):
    pm = ir.pass_manager(module.context)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.ttgpuir.add_reduce_data_duplication(pm)
    passes.ttgpuir.add_reorder_instructions(pm)
    pm.run(module, "commonir_finalize_dot_layouts")
    return module


def make_ttgir_candidates(module, metadata, options, capability, backend):
    if not ENABLED:
        raise ValueError("autolayout requires a FLAGTREE_COMMON_IR build")
    dots = []
    module.walk(lambda op: dots.append(op) if op.get_name() == "tt.dot" else None)
    if not dots:
        return backend.make_ttgir(module, metadata, options, capability)

    prepared = prepare_dot_layouts(module, options, capability)
    domains = metax.autolayout.enumerate_dot_layout_plans(prepared, capability, options.num_warps)
    assignments = compose_kernel_candidates(domains)
    cache = get_cache_manager(metadata["hash"])
    variants, failures = [], []
    fallback = None
    for index, assignment in enumerate(assignments):
        selection = {
            dot_id: {key: plan[key]
                     for key in ("plan_id", "profile_id")}
            for dot_id, plan in assignment.items()
        }
        try:
            variant = prepared.clone()
            apply_dot_layout_candidate(variant, assignment, capability, options.num_warps)
            finalize_dot_layouts(variant, capability)
        except Exception as error:
            if index == 0:
                raise
            failures.append({"plans": selection, "error": str(error)})
            logging.getLogger(__name__).warning("CommonIR layout candidate %d failed: %s", index, error)
            continue
        source = variant.str()
        digest = hashlib.sha256(source.encode()).hexdigest()
        filename = f"commonir-{digest}.ttgir"
        cache.put(source, filename, binary=False)
        variants.append({"id": digest, "file": filename, "plans": selection})
        if index == 0:
            fallback = variant

    entry = fallback.get_function(fallback.get_entry_func_name())
    metadata["commonir_layout_candidates"] = {
        "version": 1,
        "signature": list(fallback.get_function_signature(entry)),
        "variants": variants,
        "failures": failures,
    }
    return fallback
