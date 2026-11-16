# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Keep owner-local RMS statistics and fuse their consumer with packed GLU."""
from dataclasses import replace
from triton.flagmega.ir import verify_module
from triton.flagmega.ir.ops.ntt.partial_rms_packed_dense_matmul_glu import PartialRmsPackedDenseMatMulGlu
from .fuse_rms_packed_dense_glu import find_complete_rms_glu_matches, fuse_rms_packed_dense_glu


def fuse_partial_rms_packed_dense_matmul_glu(module, *, partial_group_size):
    # The existing closed-chain proof also rejects late placement and escapes.
    # Reuse its scalar/broadcast views without broadening its ownership match.
    matches = {match.glu: match for match in find_complete_rms_glu_matches(module)}
    fused = fuse_rms_packed_dense_glu(module, partial_group_size=partial_group_size)
    if fused is module:
        return module
    restored, rewritten = set(), []
    for node in fused.nodes:
        match = matches.get(node.id)
        if match is not None and node.op == 'ntt.rms_packed_dense_matmul_glu':
            stats = module.node_map[match.stats]
            operands = (*tuple(fused.node_map[key] for key in node.inputs), stats)
            prepared = PartialRmsPackedDenseMatMulGlu.prepare(operands, node.attrs)
            rewritten.append(stats)
            restored.add(stats.id)
            node = replace(node, op=PartialRmsPackedDenseMatMulGlu.op_name,
                inputs=tuple(value.id for value in operands), attrs=prepared.attrs,
                effect=prepared.effect, metadata={**node.metadata, 'introduced_by': 'FusePartialRmsPackedDenseMatMulGlu'})
        rewritten.append(node)
    restored_points = tuple(point for point in module.selection_points if point.owner in restored)
    restored_ids = {point.id for point in restored_points}
    return verify_module(replace(fused, nodes=tuple(rewritten),
        selection_points=(*fused.selection_points, *restored_points),
        selections=(*fused.selections, *(choice for choice in module.selections if choice.point_id in restored_ids))))


__all__ = ['fuse_partial_rms_packed_dense_matmul_glu']
