"""Materialized residual/statistics fusion using the existing typed combine."""

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError, StageError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.nn._norm import normalize_axis
from triton.flagmega.ir.ops.ntt.add_norm_stats import AddNormStats


def fuse_residual_stats(module):
    if module.stage not in {"frozen_constants", "tuple_boxing_lowered"}:
        raise StageError("Materialized residual fusion requires pre-TIR distributed IR", stage=module.stage)
    fm.verify_module(module)
    nodes = module.node_map
    groups = {}
    for stats in module.nodes:
        if stats.op == "nn.norm_stats" and stats.effect.is_pure:
            groups.setdefault(stats.inputs[0], []).append(stats)
    inserted, replaced = {}, {}
    occupied = set(nodes)
    for source_id, stats_nodes in groups.items():
        source = nodes[source_id]
        if not source.effect.is_pure or not (
            source.op == "math.add" or source.op == "math.vectorized_binary"
            and source.attrs["binary_op"] == "add"
        ):
            continue
        arguments = tuple(nodes[name] for name in source.inputs)
        # Do not move a collective or reinterpret the already-selected layout.
        if any(argument.type != source.type for argument in arguments):
            continue
        contracts = {(normalize_axis(int(stats.attrs["axis"]), tensor_of(source.type).rank),
                      bool(stats.attrs["use_mean"])) for stats in stats_nodes}
        if len(contracts) != 1 or any(stats.type != stats_nodes[0].type for stats in stats_nodes[1:]):
            continue
        axis, use_mean = next(iter(contracts))
        try:
            prepared = AddNormStats.prepare(arguments, {"axis": axis, "use_mean": use_mean})
        except IRSchemaError:
            continue
        if prepared.result_type != fm.TupleType((source.type, stats_nodes[0].type)):
            continue
        stem = source.id + ".residual_stats"
        identity, suffix = stem, 0
        while identity in occupied:
            suffix += 1
            identity = f"{stem}.{suffix}"
        occupied.add(identity)
        inserted[source.id] = fm.Node(identity, AddNormStats.op_name, source.inputs,
                                      prepared.result_type, prepared.effect, prepared.attrs,
                                      {**dict(source.metadata), "introduced_by": "FuseMaterializedResidualStats"})
        replaced[source.id] = replace(source, op="builtin.get_item", inputs=(identity,), attrs={"index": 0})
        # Each original statistics node may own an independent output buffer.
        # Fuse one producer; other consumers still read the preserved value ID.
        stats = stats_nodes[0]
        replaced[stats.id] = replace(stats, op="builtin.get_item", inputs=(identity,), attrs={"index": 1})
    if not inserted:
        return module
    removed_points = {point.id for point in module.selection_points if point.owner in replaced}
    rewritten = []
    for old in module.nodes:
        if old.id in inserted:
            rewritten.append(inserted[old.id])
        rewritten.append(replaced.get(old.id, old))
    return fm.verify_module(replace(
        module, nodes=tuple(rewritten),
        selection_points=tuple(point for point in module.selection_points if point.id not in removed_points),
        selections=tuple(selection for selection in module.selections if selection.point_id not in removed_points),
    ))
