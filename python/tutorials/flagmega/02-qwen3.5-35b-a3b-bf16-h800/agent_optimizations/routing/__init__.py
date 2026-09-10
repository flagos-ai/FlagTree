"""Explicit, importable workload-local routing IR and kernel extension."""

from dataclasses import replace
from pathlib import Path

from jinja2 import ChoiceLoader, FileSystemLoader, PrefixLoader
from triton.flagmega.codegen.triton import kernel_call_renderers, tir_package
from triton.flagmega.codegen.triton.candidates import TritonCandidateProposal, default_triton_candidate_registry
from triton.flagmega.codegen.triton.implementation import TritonImplementation
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.targets import NvidiaSm90Target

from .op import StagedRouting, staged_routing
from .render import routing_call


class RoutingCandidateProvider:
    op_names = frozenset({StagedRouting.op_name})

    def propose(self, node, context):
        extent = tensor_of(context.module.node_map[node.inputs[0]].type).shape[-1]
        if not extent.is_fixed or not 0 < extent.fixed_value <= 256:
            raise CodegenError("Staged routing needs a single complete tile (1..256 elements).")
        candidates = tuple(context.configure_implementation(i) for i in context.implementations("staged_routing", indexing="local"))
        return TritonCandidateProposal(candidates, candidates[0].id)


class RoutingRegistry(TritonTemplateRegistry):
    def __init__(self):
        super().__init__()
        self.extension_root = Path(__file__).parent.parent / "templates"
        self.environment.loader = ChoiceLoader([FileSystemLoader(str(self.extension_root)),
            PrefixLoader({"flagmega_core": FileSystemLoader(str(self.root))}), self.environment.loader])

    def resolve(self, spec):
        for relative in self.candidates(spec):
            if (self.extension_root / relative).is_file():
                return relative
        return super().resolve(spec)


def install():
    kernel_call_renderers._FAMILY_ENCODERS["staged_routing"] = routing_call
    tir_package.TritonTemplateRegistry = RoutingRegistry


def create_target(*, tuned=False):
    if tuned:
        # Explicitly reuse the selected workload's target geometry.
        from local_optimizations.target import create_target as base_target
        base = base_target(gate_n=4, gate_k=2048, down_n=16, down_k=512,
                           gdn_value_tile=32, gdn_projection_tile=2048)
    else:
        base = NvidiaSm90Target()
    model = base.triton_implementation_model
    implementation = TritonImplementation("tir.staged_routing.local", "staged_routing", "local",
                                          {"elements_per_program": 256}, {"indexing": "local"},
                                          facts={"portable_triton": True})
    model = replace(model, implementations=(*model.implementations, implementation),
                    preferences={**model.preferences, "staged_routing": (implementation.id, )})
    target = NvidiaSm90Target(triton_implementation_model=model)
    registry = default_triton_candidate_registry()
    registry.add(RoutingCandidateProvider())
    target.tir_selection_policy.registry = registry
    return target


__all__ = ["StagedRouting", "staged_routing", "create_target", "install"]
