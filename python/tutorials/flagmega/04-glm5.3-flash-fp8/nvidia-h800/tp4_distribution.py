"""Explicit GLM-5.3 full-model TP4 distribution override.

Only the named choices are constrained before solving the coupled graph. The
normal providers, reshard planner, materializer and all downstream verifiers
remain responsible for legality. No generated source or old proposal is edited.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from triton.flagmega.ir import ContiguousSplit, DistributedType, SBP, SBPSplit, verify_module
from triton.flagmega.passes.auto_distributed import AutoDistributedPass, solve_search_graph
from triton.flagmega.passes.auto_distributed.materializer import distribution_selection_state
from triton.flagmega.passes.auto_distributed.proposal_analysis import remember_proposal
from triton.flagmega.selection import apply_plan, emit_plan, override_plan
from triton.flagmega.targets import NvidiaSm90Target

POLICY = "glm53.tp4-distribution-override/v2"

# Output-only sharding preserves each output's K accumulation and rounding.
# The small KV-A projection uses 64 owners: 512 scalar rows / 64 = 8 rows,
# exactly one packed output element, rather than padding to 128 block owners.
OUTPUT_SPLITS = {
    "token_embedding": (0, 1, 2),
    "decode_linear_attention_dense_dense_mlp_gate_up_fp8": (0, 1, 2),
    "decode_linear_attention_dense_dense_mlp_down": (0, 1, 2),
    "decode_mla_attention_sparse_self_attn_q_b_proj_weight_apply.packed": (0, 1, 2),
    "decode_mla_attention_sparse_self_attn_o_proj_weight_apply": (0, 1, 2),
    "decode_mla_attention_sparse_self_attn_kv_a_proj_with_mqa_weight_apply.packed": (0, 2),
}


def _layout(*policies, partial=False):
    return tuple(policies), SBP.partial((0,)) if partial else None


def _moe_contracts():
    """Owner-local route, output and K layouts for the full TP4 geometry."""
    b = SBP.broadcast()
    route = SBP.split_contiguous((1,), 1)
    feature = SBP.split_contiguous((2,), 256)
    k = SBP.split_contiguous((0,), 512)
    routed_n = SBP.split_contiguous((0, 2), 32)
    shared_n = SBP.split_contiguous((0, 1, 2), 4)
    shared_down_n = SBP.split_contiguous((1, 2), 32)
    return {
        'routed_experts.dispatch': ((_layout(b, feature), _layout(b, route)),
                                    _layout(b, route, feature)),
        'routed_experts.gate_up': ((_layout(b, route, b), _layout(b, route), _layout(b, b),
                                   _layout(b, routed_n, b), _layout(b, b, b), _layout(b, b),
                                   _layout(b, routed_n, b), _layout(b, b, b)),
                                  _layout(b, route, routed_n)),
        'routed_experts.down': ((_layout(b, route, k), _layout(b, route), _layout(b, b),
                                _layout(b, feature, k), _layout(b, b, b)),
                               _layout(b, route, feature, partial=True)),
        'shared_experts.dispatch': ((_layout(b, feature), _layout(b, b)),
                                    _layout(b, b, feature)),
        'shared_experts.gate_up': ((_layout(b, b, b), _layout(b, b), _layout(b, b),
                                   _layout(b, shared_n, b), _layout(b, b, b), _layout(b, b),
                                   _layout(b, shared_n, b), _layout(b, b, b)),
                                  _layout(b, b, shared_n)),
        'shared_experts.down': ((_layout(b, b, k), _layout(b, b), _layout(b, b),
                                _layout(b, shared_down_n, k), _layout(b, b, b)),
                               _layout(b, b, shared_down_n, partial=True)),
    }


def select_moe_layout(bucket, contract):
    """Match complete typed input/output policies, never historical IDs."""
    def layout(value):
        if not isinstance(value, DistributedType):
            return None
        return tuple(value.axis_policies), value.partial

    inputs, result = contract
    matches = [candidate for candidate in bucket.candidates
               if tuple(layout(value) for value in candidate.input_types) == inputs
               and layout(candidate.return_type) == result]
    if len(matches) != 1:
        raise ValueError(f'{bucket.node_id}: expected one MoE layout, found {[value.id for value in matches]}')
    return matches[0]


def select_output_split(bucket, axes):
    """Select the unique typed, contiguous output-only candidate."""
    matches = []
    for candidate in bucket.candidates:
        result = candidate.return_type
        if not isinstance(result, DistributedType) or result.partial is not None:
            continue
        policy = result.axis_policies[-1]
        if not isinstance(policy, SBPSplit) or tuple(policy.hierarchy_axes) != axes:
            continue
        if not all(isinstance(stage.distribution, ContiguousSplit) for stage in policy.stages):
            continue
        # Avoid a hybrid split-K alternative with the same output layout.
        value = candidate.input_types[0]
        if isinstance(value, DistributedType):
            if (value.partial is not None or value.exclusive is not None
                    or any(isinstance(p, SBPSplit) for p in value.axis_policies)):
                continue
        matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(
            f"{bucket.node_id}: expected one output-only contiguous split on {axes}, "
            f"found {[c.id for c in matches]}"
        )
    return matches[0]


class Tp4DistributionTarget(NvidiaSm90Target):
    """Workload-local constraints over the unchanged SM90 implementation catalog."""

    def __init__(self, *args, evidence_dir=None, gemv_compute_warps=None, **kwargs):
        # Leave implementation preferences with the ordinary SM90 catalog.
        # A descriptor-table trial added producer live state and increased
        # full-model spills from 828 to 1884 bytes; it did not remove the
        # cause of the pressure in the surrounding call graph.
        if gemv_compute_warps is not None:
            if gemv_compute_warps not in (8, 16):
                raise ValueError('The TP4 GEMV trial requires 8 or 16 compute warps.')
            catalog = kwargs.get('triton_implementation_model') or NvidiaSm90Target().triton_implementation_model
            identities = {
                'tir.dense_matmul.split_k_n_packed_k_major_gemv_tn64',
                'tir.dense_matmul_glu.packed_k_major_gemv_tn64',
            }
            found = {value.id for value in catalog.implementations} & identities
            if found != identities:
                raise ValueError(f'GEMV trial implementations missing: {identities - found}')
            kwargs['triton_implementation_model'] = replace(
                catalog,
                implementations=tuple(
                    replace(value, parameters={**value.parameters, 'compute_num_warps': gemv_compute_warps})
                    if value.id in identities else value for value in catalog.implementations),
                name=catalog.name + f'/glm53-sm90-gemv-{gemv_compute_warps}-warps',
            )
        super().__init__(*args, **kwargs)
        self.evidence_dir = None if evidence_dir is None else Path(evidence_dir)

    def propose_distribution(self, module):
        graph = AutoDistributedPass._build_graph(module, self)
        fixed = {
            name: select_output_split(graph.bucket_map[name], axes).id
            for name, axes in OUTPUT_SPLITS.items()
        }
        for function in ('decode_linear_attention_sparse', 'decode_mla_attention_sparse'):
            for suffix, contract in _moe_contracts().items():
                name = f'{function}_{suffix}'
                fixed[name] = select_moe_layout(graph.bucket_map[name], contract).id
        result = solve_search_graph(graph, fixed_selections=fixed, dump_subdirectory="TP4Override")
        points, records = distribution_selection_state(result, policy=self.distribution_policy.identity)
        old_ids = {p.id for p in module.selection_points if p.kind == "distribution"}
        source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        proposed = replace(
            module,
            selection_points=tuple(p for p in module.selection_points if p.id not in old_ids) + points,
            selections=tuple(r for r in module.selections if r.point_id not in old_ids) + records,
            metadata={
                **module.metadata,
                "auto_distribution_proposal": {
                    "schema": "flagmega.auto-distributed-proposal/v1",
                    "placement": graph.placement.to_data(),
                    "solver": "ortools-cp-sat", "status": result.status, "objective": result.objective,
                },
                "local_distribution_override": {
                    "policy": POLICY, "source_sha256": source_digest,
                    "constraints": tuple(sorted(fixed.items())),
                },
            },
        )
        plan = override_plan(
            proposed, tuple((f"distribution.{name}", choice) for name, choice in sorted(fixed.items())),
            policy=POLICY,
            rationale="TP4 output sharding of replicated banks; aligned KV-A owners; preserve native FP8 MoE contracts.",
        )
        if self.evidence_dir is not None:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
            emit_plan(plan, self.evidence_dir / "intended-selection.py")
            (self.evidence_dir / "distribution-solve.json").write_text(json.dumps({
                "policy": POLICY, "source_sha256": source_digest,
                "input_semantic_hash": module.semantic_hash,
                "proposal_semantic_hash": proposed.semantic_hash,
                "status": result.status, "objective": result.objective,
                "hard_constraints": fixed,
                "selected": {name: candidate.id for name, candidate in result.selected.items()},
            }, indent=2) + "\n")
        proposed = verify_module(apply_plan(proposed, plan))
        remember_proposal(proposed, graph, self)
        return proposed
