"""Local IR/kernel extension layered over the measured elementwise strategy."""

from dataclasses import replace
from pathlib import Path
from jinja2 import ChoiceLoader, FileSystemLoader
from triton.flagmega.codegen.triton import kernel_call_renderers, tir_package
from triton.flagmega.codegen.triton.candidates import TritonCandidateProposal
from triton.flagmega.codegen.triton.implementation import TritonImplementation
from triton.flagmega.targets import NvidiaSm90Target
from agent_optimizations.elementwise import create_target as base_target
from agent_optimizations.elementwise.candidates import scalar_capacity
from agent_optimizations.routing import RoutingRegistry as PairedProjectionRegistry, install as install_base
from .op import ScalarScale, scalar_scale
from .render import scalar_scale_call


class ScalarScaleProvider:
    op_names = frozenset({ScalarScale.op_name})

    def propose(self, node, context):
        candidates = tuple(context.configure_implementation(i)
                           for i in context.implementations("scalar_scale", indexing="local"))
        capacity = scalar_capacity(node.type)
        desired = min(2048, 1 << (max(1, capacity or 1) - 1).bit_length())
        default = min(candidates, key=lambda c: abs(int(c.parameters["elements_per_program"]) - desired))
        return TritonCandidateProposal(candidates, default.id)


class ScalarScaleRegistry(PairedProjectionRegistry):
    def __init__(self):
        super().__init__()
        self.scalar_root = Path(__file__).parent / "templates"
        self.environment.loader = ChoiceLoader([FileSystemLoader(str(self.scalar_root)), self.environment.loader])

    def resolve(self, spec):
        for relative in self.candidates(spec):
            if (self.scalar_root / relative).is_file():
                return relative
        return super().resolve(spec)


def install():
    install_base()
    kernel_call_renderers._FAMILY_ENCODERS["scalar_scale"] = scalar_scale_call
    tir_package.TritonTemplateRegistry = ScalarScaleRegistry


def create_target():
    base = base_target()
    model = base.triton_implementation_model
    extra = tuple(TritonImplementation(f"tir.scalar_scale.local.tile_{tile}", "scalar_scale", "local",
                                      {"elements_per_program": tile}, {"indexing": "local"},
                                      facts={"portable_triton": True}) for tile in (128, 256, 512, 1024, 2048))
    target = NvidiaSm90Target(triton_implementation_model=replace(model, implementations=(*model.implementations, *extra)))
    registry = base.tir_selection_policy.registry
    registry.add(ScalarScaleProvider())
    target.tir_selection_policy.registry = registry
    return target


__all__ = ["ScalarScale", "scalar_scale", "create_target", "install"]
