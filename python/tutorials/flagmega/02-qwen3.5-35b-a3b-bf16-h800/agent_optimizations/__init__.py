"""Workload-local IR, providers and contract-compatible kernel overrides."""

from dataclasses import replace
from triton.flagmega.targets import NvidiaSm90Target
from .sigmoid_product import create_target as base_target, install

IMPLEMENTATION = "tir.paged_attention_partial.decode_t32"


def create_target(*, bufferize_opt_level="optimized"):
    base = base_target()
    model = base.triton_implementation_model
    order = (IMPLEMENTATION, *(v for v in model.preferences["paged_attention_partial"] if v != IMPLEMENTATION))
    target = NvidiaSm90Target(triton_implementation_model=replace(
        model, preferences={**model.preferences, "paged_attention_partial": order}))
    target.tir_selection_policy.registry = base.tir_selection_policy.registry
    return target.with_bufferize_opt_level(bufferize_opt_level)


__all__ = ["IMPLEMENTATION", "create_target", "install"]
