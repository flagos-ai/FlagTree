"""Select the existing, guarded CPP scalar-ordered GDN head cache explicitly."""
from __future__ import annotations

from dataclasses import replace

from triton.flagmega.codegen.triton.candidates import TritonCandidateProviderRegistry
from triton.flagmega.codegen.triton.candidates.gdn_head_gates import IMPLEMENTATION
from triton.flagmega.codegen.triton.candidates.simple import GdnCandidateProvider
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.errors import CodegenError
from triton.flagmega.targets import get_target


class HeadGateCandidateProvider:
    """Change only the default after the core provider has applied every guard."""

    def __init__(self, original):
        self.original = original
        self.op_names = original.op_names

    def propose(self, node, context):
        proposal = self.original.propose(node, context)
        if node.op != "nn.gdn_recurrent_core":
            return proposal
        if proposal is None or IMPLEMENTATION not in {c.id for c in proposal.candidates}:
            raise CodegenError(
                "head_gate_cache=True requires the core scalar-head-gates candidate "
                "to admit this GDN call; check cpp/SM120/block128, dense owner map, "
                "IQ4 weights and the nncase numerical contract.")
        return replace(proposal, default_candidate=IMPLEMENTATION)


def enable_head_gate_cache():
    """Retain all policy/provider objects except the model's GDN default selector."""
    target = get_target("nvidia-sm120")
    policy = target.tir_selection_policy
    if not isinstance(policy, TritonTirSelectionPolicy):
        raise CodegenError("head_gate_cache requires the normal TritonTirSelectionPolicy")
    registry = TritonCandidateProviderRegistry()
    replaced = 0
    for provider in policy.registry.providers:
        if isinstance(provider, HeadGateCandidateProvider):
            raise CodegenError("head_gate_cache selector is already installed")
        if isinstance(provider, GdnCandidateProvider):
            provider = HeadGateCandidateProvider(provider)
            replaced += 1
        registry.add(provider)
    if replaced != 1:
        raise CodegenError("head_gate_cache requires exactly one existing GDN provider")
    target.tir_selection_policy = TritonTirSelectionPolicy(policy.workspace_annotator, registry)


def validate_head_gate_selection(descriptor):
    """Reject late resume/overridden selections instead of silently losing the flag."""
    calls = [c for c in descriptor["render_calls"] if c.get("family") == "gdn_recurrent"]
    if not calls or any(c.get("implementation") != IMPLEMENTATION or not c.get("head_gate_cache")
                        for c in calls):
        raise CodegenError(
            "head_gate_cache=True was not selected for every recurrent call; "
            "resume from the original import before TIR selection.")
    return [{"symbol": c["symbol"], "implementation": c["implementation"],
             "local_capacity": c["local_capacity"], "shared_bytes": 16,
             "shared_offset": c["head_gate_offset_bytes"],
             "shared_physical_buffer": c["head_gate_physical_buffer"]} for c in calls]
