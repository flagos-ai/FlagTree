"""Prepare layout candidates without requiring an executable kernel."""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from triton._C.libtriton import metax
from triton._common_ir import ENABLED
from triton.runtime.cache import get_cache_key, get_cache_manager


def compose_kernel_candidates(domains):
    """Compose candidates with matching accumulator profiles across dots.

    Shared profiles follow the first dot's order, using the first matching plan
    per dot. With no shared profile, use each dot's first plan. No dots produce
    one empty assignment.
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
    layouts = {dot_id: [plan[key] for key in ("mma", "operand_a", "operand_b")] for dot_id, plan in candidate.items()}
    metax.autolayout.insert_dot_layout_hints(module, capability, num_warps, layouts)
    return module


@dataclass
class LayoutCandidates:
    hash: str
    src: object
    target: object
    options: dict
    env_vars: dict
    manifest: dict


def prepare_layout_candidates(src, backend, options, env_vars):
    """Cache candidate TTIR modules; compilation and benchmarking happen later."""
    from triton import knobs
    from triton._C.libtriton import ir
    from flagtree import _flagprism

    if not ENABLED:
        raise ValueError("autolayout requires a FLAGTREE_COMMON_IR build")
    key = get_cache_key(src, backend, options, env_vars)
    identity = hashlib.sha256(("commonir-candidates-v2:" + key).encode()).hexdigest()
    cache = get_cache_manager(identity)
    manifest_path = cache.get_file("commonir-candidates.json")
    manifest = None
    if manifest_path and not knobs.compilation.always_compile:
        try:
            cached = json.loads(Path(manifest_path).read_text())
            if cached["version"] == 2 and cached["variants"] and all(
                (path := cache.get_file(v["file"])) is not None
                    and hashlib.sha256(Path(path).read_bytes()).hexdigest() == v["id"] for v in cached["variants"]):
                manifest = cached
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if manifest is None:
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
        _flagprism.load_dialects(context)
        module = backend.make_ir(src, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                                 context)
        metadata = {"hash": identity, "target": backend.target, **options.__dict__, **env_vars}
        stages = {}
        backend.add_stages(stages, options, src.language)
        module = stages["ttir"](module, metadata)
        manifest = _make_candidates(module, options, backend._parse_arch(options.arch), cache)
        cache.put(json.dumps(manifest), "commonir-candidates.json", binary=False)
    return LayoutCandidates(identity, src, backend.target, options.__dict__, env_vars, manifest)


def _make_candidates(module, options, capability, cache):
    dots = []
    module.walk(lambda op: dots.append(op) if op.get_name() == "tt.dot" else None)
    domains = metax.autolayout.enumerate_dot_layout_plans(module, capability, options.num_warps) if dots else []

    assignments = compose_kernel_candidates(domains)
    variants, failures = [], []
    for index, assignment in enumerate(assignments):
        selection = {
            dot_id: {key: plan[key]
                     for key in ("plan_id", "profile_id")}
            for dot_id, plan in assignment.items()
        }
        try:
            variant = module.clone()
            if assignment:
                insert_dot_layout_candidate(variant, assignment, capability, options.num_warps)
        except Exception as error:
            failures.append({"plans": selection, "error": str(error)})
            logging.getLogger(__name__).warning("CommonIR layout candidate %d failed: %s", index, error)
            continue
        source = variant.str()
        digest = hashlib.sha256(source.encode()).hexdigest()
        filename = f"commonir-{digest}.ttir"
        cache.put(source, filename, binary=False)
        variants.append({"id": digest, "file": filename, "plans": selection})

    if not variants:
        details = "; ".join(item["error"] for item in failures)
        raise RuntimeError(f"CommonIR autolayout: all layout candidates failed. {details}")

    entry = module.get_function(module.get_entry_func_name())
    return {
        "version": 2,
        "signature": list(module.get_function_signature(entry)),
        "variants": variants,
        "failures": failures,
    }
