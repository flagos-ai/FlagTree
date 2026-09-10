"""Workload-local typed gate fusion and kernel provider."""

from dataclasses import replace
from pathlib import Path
from jinja2 import ChoiceLoader, FileSystemLoader
from triton.flagmega.codegen.triton import kernel_call_renderers, tir_package
from triton.flagmega.codegen.triton.candidates import TritonCandidateProposal
from triton.flagmega.codegen.triton.implementation import TritonImplementation
from triton.flagmega.targets import NvidiaSm90Target
from agent_optimizations.gated_epilogue import EpilogueRegistry, create_target as base_target, install as install_base
from .op import SigmoidProduct, sigmoid_product
from .render import sigmoid_product_call


class SigmoidProductProvider:
    op_names = frozenset({SigmoidProduct.op_name})

    def propose(self, node, context):
        candidates = tuple(
            context.configure_implementation(i) for i in context.implementations("sigmoid_product", indexing="local"))
        return TritonCandidateProposal(candidates, candidates[0].id)


class SigmoidProductRegistry(EpilogueRegistry):

    def __init__(self):
        super().__init__()
        self.product_root = Path(__file__).parent / "templates"
        self.environment.loader = ChoiceLoader([FileSystemLoader(str(self.product_root)), self.environment.loader])

    def resolve(self, spec):
        for relative in self.candidates(spec):
            if (self.product_root / relative).is_file():
                return relative
        return super().resolve(spec)


def install():
    install_base()
    kernel_call_renderers._FAMILY_ENCODERS["sigmoid_product"] = sigmoid_product_call
    tir_package.TritonTemplateRegistry = SigmoidProductRegistry


def create_target():
    base = base_target()
    implementation = TritonImplementation("tir.sigmoid_product.local", "sigmoid_product", "local",
                                          {"elements_per_program": 128}, {"indexing": "local"},
                                          facts={"portable_triton": True})
    model = base.triton_implementation_model
    target = NvidiaSm90Target(
        triton_implementation_model=replace(model, implementations=(*model.implementations, implementation)))
    registry = base.tir_selection_policy.registry
    registry.add(SigmoidProductProvider())
    target.tir_selection_policy.registry = registry
    return target


__all__ = ["SigmoidProduct", "sigmoid_product", "create_target", "install"]
