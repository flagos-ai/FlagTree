"""Scoped catalog/provider extension with local-capacity tile selection."""

from dataclasses import replace

from triton.flagmega.codegen.triton.candidates import (
    ElementwiseCandidateProvider, TritonCandidateProviderRegistry,
)
from triton.flagmega.targets import NvidiaSm90Target
from agent_optimizations.routing import install
from agent_optimizations.routing import create_target as original_target
from .candidates import LocalExtentElementwiseProvider


def create_target(*, max_elements=2048):
    provider = LocalExtentElementwiseProvider(max_elements)
    base = original_target(tuned=True)
    model = base.triton_implementation_model
    extra = []
    for implementation in model.implementations:
        if implementation.family != "elementwise":
            continue
        key = "vector_groups" if "vector_groups" in implementation.parameters else "elements_per_program"
        for extent in (64, 128, 256, 512, 1024, 2048):
            if extent <= implementation.parameters[key]:
                continue
            extra.append(replace(
                implementation, id=f"{implementation.id}.{key}_{extent}",
                parameters={**implementation.parameters, key: extent},
            ))
    target = NvidiaSm90Target(triton_implementation_model=replace(
        model, implementations=(*model.implementations, *extra)))
    registry = TritonCandidateProviderRegistry()
    for original in base.tir_selection_policy.registry.providers:
        registry.add(provider if isinstance(original, ElementwiseCandidateProvider) else original)
    target.tir_selection_policy.registry = registry
    return target


__all__ = ["create_target", "install"]
