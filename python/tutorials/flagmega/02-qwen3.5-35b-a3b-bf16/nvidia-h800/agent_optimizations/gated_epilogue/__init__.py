"""Workload-local rounding-aware epilogue fusion."""

from dataclasses import replace
from pathlib import Path
from jinja2 import ChoiceLoader, FileSystemLoader
from triton.flagmega.codegen.triton import kernel_call_renderers, tir_package
from triton.flagmega.codegen.triton.candidates import TritonCandidateProposal
from triton.flagmega.codegen.triton.implementation import TritonImplementation
from triton.flagmega.targets import NvidiaSm90Target
from agent_optimizations.scalar_scale import ScalarScaleRegistry, create_target as base_target, install as install_base
from .op import GatedResidualNormStats, gated_residual_norm_stats
from .render import gated_epilogue_call


class EpilogueProvider:
    op_names = frozenset({GatedResidualNormStats.op_name})

    def propose(self, node, context):
        candidates = tuple(context.configure_implementation(i) for i in context.implementations(
            "gated_residual_norm_stats", indexing="local"))
        return TritonCandidateProposal(candidates, candidates[0].id)


class EpilogueRegistry(ScalarScaleRegistry):
    def __init__(self):
        super().__init__()
        self.epilogue_root = Path(__file__).parent / "templates"
        self.environment.loader = ChoiceLoader([FileSystemLoader(str(self.epilogue_root)), self.environment.loader])

    def resolve(self, spec):
        for relative in self.candidates(spec):
            if (self.epilogue_root / relative).is_file():
                return relative
        return super().resolve(spec)


def install():
    install_base()
    kernel_call_renderers._FAMILY_ENCODERS["gated_residual_norm_stats"] = gated_epilogue_call
    tir_package.TritonTemplateRegistry = EpilogueRegistry


def create_target():
    base = base_target()
    implementation = TritonImplementation(
        "tir.gated_residual_norm_stats.local", "gated_residual_norm_stats", "local",
        {"block_size": 128, "compute_num_warps": 8}, {"indexing": "local"},
        facts={"portable_triton": True})
    model = base.triton_implementation_model
    target = NvidiaSm90Target(triton_implementation_model=replace(
        model, implementations=(*model.implementations, implementation)))
    registry = base.tir_selection_policy.registry
    registry.add(EpilogueProvider())
    target.tir_selection_policy.registry = registry
    return target


__all__ = ["GatedResidualNormStats", "gated_residual_norm_stats", "create_target", "install"]
