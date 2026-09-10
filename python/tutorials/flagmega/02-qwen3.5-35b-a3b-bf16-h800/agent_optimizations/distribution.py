"""Choose an output-owned projection domain before normal distribution materialization."""


from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateProviderRegistry, build_search_graph, solve_search_graph,
)
from triton.flagmega.passes.auto_distributed.materializer import distribution_selection_state
from triton.flagmega.selection import emit_plan, override_plan


def choose_projection(module, target, output_dir):
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)
    placement, = target.distributed_placements(module)
    graph = build_search_graph(module, placement, registry, target.distributed_reshard_realization_policy(),
                               target.distributed_reshard_cost_model(), target.distributed_operation_cost_model())
    fixed = {r.point_id.removeprefix("distribution."): r.candidate_id for r in module.selections
             if r.point_id.startswith("distribution.")}
    changed = set()
    for combine in module.nodes:
        if combine.op != "ntt.matmul_norm_stats_combine":
            continue
        node = module.node_map[combine.inputs[0]]
        if node.op != "ntt.packed_matmul":
            raise ValueError("Projection combine must retain its semantic producer")
        extent = node.type.shape[-1].fixed_value
        policy = (fm.SBP.broadcast(), fm.SBP.split_contiguous(tuple(range(placement.rank)), extent // placement.size))
        matches = [c for c in graph.bucket_map[node.id].candidates
                   if isinstance(c.return_type, fm.DistributedType) and c.return_type.partial is None
                   and c.return_type.axis_policies == policy
                   and c.input_types[0].axis_policies == (fm.SBP.broadcast(), fm.SBP.broadcast())]
        if len(matches) != 1:
            raise ValueError(f"Expected one N-sharded/full-K projection candidate for {node.id}, got {len(matches)}")
        selected = matches[0]
        print("projection", node.id, "old", fixed[node.id], "new", selected.id, flush=True)
        fixed[node.id] = selected.id
        # The combine must consume the materialized output, not a stale
        # split-K partial signature. The remaining graph stays constrained.
        fixed.pop(combine.id, None)
        changed.add(combine.id)
    for node in module.nodes:
        if node.op in {"builtin.get_item", "tensors.get_item", "tensors.bitcast", "tensors.reshape"}:
            if any(i in changed for i in node.inputs):
                fixed.pop(node.id, None)
                changed.add(node.id)
    result = solve_search_graph(graph, fixed_selections=fixed)
    _, records = distribution_selection_state(result, policy=target.distribution_policy.identity)
    plan = override_plan(module, [(r.point_id, r.candidate_id) for r in records],
                         rationale="Output-owner projection with full K; re-solve coupled materialization choices")
    emit_plan(plan, output_dir / "distribution.plan.py")
    return plan
