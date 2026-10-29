"""CommonIR layout candidate composition, injection and compilation."""

import hashlib
import logging
from dataclasses import replace

from triton._C.libtriton import metax
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


def insert_dot_layout_candidate(module, candidate, capability, num_warps):
    """Insert candidate A/B/result anchors in unencoded TTIR."""
    layouts = {dot_id: [plan[key] for key in ("mma", "operand_a", "operand_b")]
               for dot_id, plan in candidate.items()}
    metax.autolayout.insert_dot_layout_hints(module, capability, num_warps, layouts)
    return module


def make_ttgir_candidates(module, metadata, options, capability, backend):
    if not ENABLED:
        raise ValueError("autolayout requires a FLAGTREE_COMMON_IR build")
    dots = []
    module.walk(lambda op: dots.append(op) if op.get_name() == "tt.dot" else None)
    normal_options = replace(options, autolayout=False)
    if not dots:
        return backend.make_ttgir(module, metadata, normal_options, capability)

    domains = metax.autolayout.enumerate_dot_layout_plans(module, capability, options.num_warps)
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
            variant = module.clone()
            insert_dot_layout_candidate(variant, assignment, capability, options.num_warps)
            variant_metadata = dict(metadata)
            variant = backend.make_ttgir(variant, variant_metadata, normal_options, capability)
        except Exception as error:
            failures.append({"plans": selection, "error": str(error)})
            logging.getLogger(__name__).warning("CommonIR layout candidate %d failed: %s", index, error)
            continue
        source = variant.str()
        digest = hashlib.sha256(source.encode()).hexdigest()
        filename = f"commonir-{digest}.ttgir"
        cache.put(source, filename, binary=False)
        variants.append({"id": digest, "file": filename, "plans": selection})
        if fallback is None:
            fallback = variant
            metadata.update(variant_metadata)

    if fallback is None:
        details = "; ".join(item["error"] for item in failures)
        raise RuntimeError(f"CommonIR autolayout: all layout candidates failed. {details}")

    entry = fallback.get_function(fallback.get_entry_func_name())
    metadata["commonir_layout_candidates"] = {
        "version": 1,
        "signature": list(fallback.get_function_signature(entry)),
        "variants": variants,
        "failures": failures,
    }
    return fallback
