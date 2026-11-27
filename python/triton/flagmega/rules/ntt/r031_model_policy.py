# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit corrected T-v6 post-packing H004 model policy; native model packaging is a separate gate.

The coarse effect uses one device-global value domain. Its three native
program meshes, typed scratch fields and completion edges come exclusively
from the qualified column plan. This policy does not turn the coarse call
into a single 384-CTA kernel or reuse the standalone constructor ABI.
"""
from collections.abc import Mapping
from copy import copy
from dataclasses import replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.distributed_inference import broadcast_ir_type
from triton.flagmega.ir.execution_mesh import ExecutionMesh
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidate, DistributedCandidateProviderBase,
    DistributedCandidateProviderRegistry,
)
from .r017_split4lut_bindings import (
    MODULE, IMPLEMENTATION, PLAN_BODY_SHA256, column_phase_plan,
)
from .r004_gdn_phase_bindings import (
    _same_contract, standalone_request, validate_coarse_node, validate_phase_plan,
)

REQUEST_KEY = "r004_model_policy"
SUBJECT_SHA256 = "b7ee29ef5c4f5bc308e2f36ee73fd7c466da90598307769c4421bbcf70b17d58"
PLAN_BODY_SHA256 = column_phase_plan()["plan_body_sha256"]
OP = "nn.gdn_recurrent_core"


def model_request(module):
    request = module.metadata.get(REQUEST_KEY)
    if request is None:
        return None
    expected = dict(schema="flagmega.r031-model-policy/v1", candidate=MODULE,
                    plan_body_sha256=PLAN_BODY_SHA256,
                    numerical_subject_sha256=SUBJECT_SHA256)
    if not isinstance(request, Mapping) or not _same_contract(request, expected):
        raise IRVerificationError("Invalid H004 model policy identity.", stage=module.stage)
    if standalone_request(module) is not None:
        raise IRVerificationError("Model and standalone GDN requests are mutually exclusive.")
    return request


def prepare_model_column_distribution(module):
    """Explicit opt-in at complete packing; verify again on every fresh resume."""
    if module.stage not in ("unused_functions_removed", "distribution_constants_frozen"):
        raise IRVerificationError("H004 model integration must resume before distribution.",
                                  stage=module.stage)
    if standalone_request(module) is not None:
        raise IRVerificationError("A standalone GDN module cannot become a model candidate.")
    request = dict(schema="flagmega.r031-model-policy/v1", candidate=MODULE,
                   plan_body_sha256=PLAN_BODY_SHA256,
                   numerical_subject_sha256=SUBJECT_SHA256)
    if model_request(module) is None:
        module = replace(module, metadata={**module.metadata, REQUEST_KEY: request})
    return configure_model_phases(module)


def configure_model_phases(module):
    """Keep the original inputs, tuple, effects and all neighboring nodes."""
    if model_request(module) is None:
        return module
    if module.stage not in ("unused_functions_removed", "distribution_constants_frozen",
                            "distribution_candidates"):
        raise IRVerificationError("H004 phase ownership is before AutoDistributedPass.",
                                  stage=module.stage)
    if any(s.point_id.startswith(("distribution.", "tir.")) for s in module.selections):
        raise IRVerificationError("H004 model policy cannot reuse downstream selections.")
    if (module.metadata.get("num_hidden_layers") != 64
            or tuple(module.metadata.get("imported_layer_indices", ())) != tuple(range(64))):
        raise IRVerificationError("H004 model policy requires all 64 text layers.")
    from triton.flagmega.passes.functions.graph import static_node_invocation_counts
    counts = static_node_invocation_counts(module)
    coarse = [n for n in module.nodes if n.op == OP and n.id in counts]
    if sum(counts[n.id] for n in coarse) != 48:
        raise IRVerificationError("H004 model policy requires exactly 48 recurrent invocations.")
    # The token bucket's extent is the plan's row count; the batch axis of each
    # phase grid carries it, so Bufferize sees the real extents.
    plan = column_phase_plan(int(module.metadata.get("tokens_per_call", 1) or 1))
    # Dynamic describes the coarse value domain, not a launch grid. Native
    # PREP/CORE/Y retain their own fixed meshes from phase_plan.
    mesh = ExecutionMesh.dynamic().to_data()
    changed = {}
    for node in coarse:
        validate_coarse_node(module, node)
        if node.attrs.get("phase_plan") is not None:
            if not _same_contract(validate_phase_plan(node.attrs["phase_plan"]), plan):
                raise IRVerificationError("Mixed H004 model phase plans.", node_id=node.id)
        changed[node.id] = replace(node, attrs={**node.attrs, "phase_plan": plan},
                                  metadata={**node.metadata, "execution_mesh": mesh,
                                            "kernel_program_kind": "ordinary"})
    return replace(module, nodes=tuple(changed.get(n.id, n) for n in module.nodes))


class ColumnEffectProvider(DistributedCandidateProviderBase):
    """One coarse effect; full tensors reach the phase-owned scalar writers."""
    op_names = frozenset({OP})
    allows_partial_inputs = False
    is_exhaustive = True

    def _enumerate_candidates(self, context):
        node = context.source_call
        if model_request(context.module) is None:
            raise IRVerificationError("Column GDN provider requires its model policy.")
        validate_coarse_node(context.module, node)
        validate_phase_plan(node.attrs.get("phase_plan"))
        if context.placement.physical_level_size("b") != 1:
            raise IRVerificationError("GDN coarse effect must use the device-global value domain.",
                                      node_id=node.id)
        return (DistributedCandidate(
            "r004-column-complete-effect",
            broadcast_ir_type(node.type, context.placement),
            tuple(broadcast_ir_type(context.module.node_map[i].type, context.placement)
                  for i in node.inputs),
            0, "phase-owned-columns-and-complete-head-publication", OP, "agent",
            "flagmega.r031-model-policy/v1", (PLAN_BODY_SHA256, SUBJECT_SHA256)),)


class _SingleOpProvider:
    def __init__(self, op, base):
        self.op_names, self.base = frozenset({op}), base

    def __getattr__(self, name):
        return getattr(self.base, name)


class VerifyNormStatsProvider(DistributedCandidateProviderBase):
    """Keep decode's full hidden-width RMS reduction for verify rows."""
    op_names = frozenset({"nn.norm_stats"})
    allows_partial_inputs = False
    is_exhaustive = True

    def __init__(self, base):
        self.base = base

    def _enumerate_candidates(self, context):
        from math import prod
        from triton.flagmega.ir import VectorType
        from triton.flagmega.ir.distributed_inference import tensor_of
        from triton.flagmega.ir.distributed_type import SBPBroadCast
        candidates = self.base.get_candidates(context)
        if int(context.module.metadata.get("tokens_per_call", 1)) == 1:
            return candidates
        node = context.source_call
        value = tensor_of(context.module.node_map[node.inputs[0]].type)
        axis = int(node.attrs["axis"]) % value.rank
        lanes = prod(value.dtype.lanes) if isinstance(value.dtype, VectorType) else 1
        suffix = value.shape[axis:]
        if not all(d.is_fixed for d in suffix) or prod(d.fixed_value for d in suffix) * lanes != 5120:
            return candidates
        # Additive partial statistics change the F32 reduction order. A last-
        # bit difference can cross a BF16 rounding boundary before Q8 prepare,
        # so verify keeps the same full-width reduction as sequential decode.
        return tuple(c for c in candidates if all(
            isinstance(p, SBPBroadCast) for p in c.input_types[0].axis_policies[axis:]))


class ColumnDistributionPolicy:
    """Column recurrence and decode-compatible RMS statistics for verify."""
    def __init__(self, base):
        self.base = base

    @property
    def identity(self):
        return self.base.identity + "+r031-r017:" + PLAN_BODY_SHA256

    def __getattr__(self, name):
        return getattr(self.base, name)

    def register_candidate_providers(self, registry):
        original = DistributedCandidateProviderRegistry()
        self.base.register_candidate_providers(original)
        registry.set_split_candidate_provider(original.split_candidate_provider)
        for op in original.op_names:
            registry.add(ColumnEffectProvider() if op == OP
                         else VerifyNormStatsProvider(original.try_get(op)) if op == "nn.norm_stats"
                         else _SingleOpProvider(op, original.try_get(op)))
        if OP not in original.op_names:
            registry.add(ColumnEffectProvider())


def model_distribution_target(target, module):
    """Public compiler hook also reconstructs the policy after IR reload."""
    if model_request(module) is None:
        return target
    if target.name != "nvidia-sm120" or getattr(target, "codegen_backend", "triton") != "triton":
        raise IRVerificationError("H004 model integration requires SM120 Triton.")
    if isinstance(target.distribution_policy, ColumnDistributionPolicy):
        return target
    local = copy(target)
    local.distribution_policy = ColumnDistributionPolicy(target.distribution_policy)
    return local


class ColumnTirProvider:
    op_names = frozenset({OP})

    def propose(self, node, context):
        from triton.flagmega.codegen.triton.candidates.core import TritonCandidateProposal
        if model_request(context.module) is None:
            raise IRVerificationError("Model column selection requires its explicit policy.")
        validate_coarse_node(context.module, node)
        plan = validate_phase_plan(node.attrs.get("phase_plan"))
        if plan["module"] != MODULE:
            raise IRVerificationError("Model column selection received another candidate.")
        implementations = tuple(i for i in context.implementations("gdn_recurrent")
                                if i.id == IMPLEMENTATION)
        if len(implementations) != 1:
            raise IRVerificationError("Qualified H004 column implementation is missing.")
        candidate = context.configure_implementation(
            implementations[0], semantic_parameters={"phase_plan": plan})
        return TritonCandidateProposal((candidate,), IMPLEMENTATION)


def propose_model_tir(module, target):
    """Select the existing tuned native algorithm; never fall back to P5-U."""
    from triton.flagmega.targets.nvidia.gdn import r004_reference_implementations
    from triton.flagmega.codegen.triton.implementation import TritonImplementationModel
    local = model_distribution_target(target, module)
    old = local.triton_implementation_model
    catalog = tuple(i for i in r004_reference_implementations() if i.id == IMPLEMENTATION)
    present = tuple(i for i in old.implementations if i.id == IMPLEMENTATION)
    if not present:
        local = copy(local)
        local.triton_implementation_model = TritonImplementationModel(
            (*old.implementations, *catalog), old.preferences, name=old.name + "+r004-column")
    elif present != catalog:
        raise IRVerificationError("Installed H004 algorithm differs from the qualified catalog.")
    policy = copy(local.tir_selection_policy)
    original = policy.registry
    class Registry:
        op_names = original.op_names

        def provider_for(self, op):
            return ColumnTirProvider() if op == OP else original.provider_for(op)
    policy.registry = Registry()
    return policy.propose(module, local)
