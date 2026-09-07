# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""NTT AutoDistribution policy parameterized by target topology."""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.ir import IRModule, Node, Placement, get_definition
from triton.flagmega.passes.auto_distributed.candidates import DistributedCandidateProviderRegistry
from triton.flagmega.passes.auto_distributed.inference_providers import (
    TypeInferenceCandidateProvider,
)
from triton.flagmega.passes.auto_distributed.greedy_sample_provider import GreedySampleCandidateProvider
from triton.flagmega.passes.auto_distributed.providers import (
    BinaryCandidateProvider,
    BroadcastCandidateProvider,
    EmbeddingCandidateProvider,
    GdnConvolutionCandidateProvider,
    GdnRecurrentCandidateProvider,
    MatMulCandidateProvider,
    MatMulGluCandidateProvider,
    PackedQKVParallelLinearCombineCandidateProvider,
    PackedQKVParallelLinearCandidateProvider,
)
from triton.flagmega.passes.auto_distributed.norm_providers import (
    BindNormStatsCandidateProvider,
    MatMulNormStatsCombineCandidateProvider,
    NormApplyCandidateProvider,
    NormStatsCandidateProvider,
)
from triton.flagmega.passes.auto_distributed.packed_matmul_provider import PackedMatMulCandidateProvider
from triton.flagmega.passes.auto_distributed.paged_attention_providers import (
    PagedAttentionCombineCandidateProvider,
    PagedAttentionPartialCandidateProvider,
)
from triton.flagmega.passes.auto_distributed.qkv_rope_with_cache_provider import (
    QKVRoPEWithCacheCandidateProvider,
)
from triton.flagmega.passes.auto_distributed.realization import (
    DistributedReshardRealizationPolicy,
    NttDistributedReshardRealizationPolicy,
)
from triton.flagmega.passes.auto_distributed.split_candidates import (
    ContiguousDistributedSplitCandidateProvider,
)
from triton.flagmega.passes.vector_contracts import (
    NATIVE_VECTOR_COMPUTE_OPS,
    retained_vectorization_roots,
    vectorization_root,
)


_DISTRIBUTION_ADAPTER_OPS = frozenset({
    "distributed.boxing", "distributed.force_boxing", "distributed.sharded_view",
})


class NttDistributionPolicy:
    """NTT providers and realization over target-supplied physical meshes."""

    def __init__(
        self,
        placements: tuple[Placement, ...],
        split_candidate_provider=None,
        reshard_realization_policy: DistributedReshardRealizationPolicy | None = None,
    ) -> None:
        if not placements:
            raise ValueError("NTT AutoDistribution requires at least one placement.")
        self._placements = tuple(placements)
        self._split_candidate_provider = (
            split_candidate_provider
            or ContiguousDistributedSplitCandidateProvider()
        )
        self._reshard_realization_policy = (
            reshard_realization_policy
            or NttDistributedReshardRealizationPolicy()
        )

    @property
    def identity(self) -> str:
        placements = ";".join(
            f"{value.name}={'x'.join(str(axis) for axis in value.hierarchy)}"
            f"/{value.hierarchy_levels}"
            for value in self._placements
        )
        return (
            "ntt-auto-distributed/v8("
            f"placements={placements},"
            f"split={self._split_candidate_provider.identity})"
        )

    def placements(self, module: IRModule) -> tuple[Placement, ...]:
        del module
        return self._placements

    def register_candidate_providers(
        self,
        registry: DistributedCandidateProviderRegistry,
    ) -> None:
        registry.set_split_candidate_provider(self._split_candidate_provider)
        registry.add(MatMulCandidateProvider())
        registry.add(PackedMatMulCandidateProvider())
        registry.add(MatMulGluCandidateProvider())
        registry.add(PackedQKVParallelLinearCandidateProvider())
        registry.add(PackedQKVParallelLinearCombineCandidateProvider())
        registry.add(BinaryCandidateProvider())
        registry.add(EmbeddingCandidateProvider())
        registry.add(GreedySampleCandidateProvider())
        registry.add(GdnConvolutionCandidateProvider())
        registry.add(GdnRecurrentCandidateProvider())
        registry.add(NormStatsCandidateProvider())
        registry.add(NormApplyCandidateProvider())
        registry.add(BindNormStatsCandidateProvider())
        registry.add(MatMulNormStatsCombineCandidateProvider())
        registry.add(PagedAttentionPartialCandidateProvider())
        registry.add(PagedAttentionCombineCandidateProvider())
        registry.add(QKVRoPEWithCacheCandidateProvider())
        registry.add(TypeInferenceCandidateProvider(frozenset({
            "math.vectorized_unary",
            "nn.rms_norm",
            "nn.rope",
            "nn.update_paged_attention_kv_cache",
            "ntt.vectorized_cast",
            "ntt.vectorized_rope",
            "tensors.bitcast",
            "tensors.pack",
            "tensors.reshape",
            "tensors.unpack",
        })))
        registry.add(BroadcastCandidateProvider(frozenset({
            "math.silu",
            "math.vectorized_matmul",
            "nn.rotary_embedding",
            "nn.vectorized_rms_norm",
            "tensors.cast",
            "tensors.concat",
            "tensors.pad",
            "tensors.permute",
            "tensors.slice_to_shape",
        })))

    def reshard_realization_policy(self) -> DistributedReshardRealizationPolicy:
        return self._reshard_realization_policy


def lower_vectorization_contracts(module: IRModule) -> IRModule:
    """Lower typed EGraph layouts to editable semantic schedule contracts.

    Compiler-internal Pack/Unpack nodes are removed, but the selected axes and
    lanes remain first-class metadata.  TIR selection must turn that metadata
    into an executable ``vector_schedule``; lowering rejects a missing one.
    """

    node_ids = {node.id for node in module.nodes}
    native_vector_ops = NATIVE_VECTOR_COMPUTE_OPS
    native_vector_roots = retained_vectorization_roots(module)
    node_map = module.node_map
    native_dependencies: set[str] = set()
    distribution_adapters = _DISTRIBUTION_ADAPTER_OPS
    dependency_bridges = frozenset({
        "builtin.get_item",
        "builtin.identity",
        "builtin.tuple",
        *_DISTRIBUTION_ADAPTER_OPS,
    })
    pending = [
        node.id
        for node in module.nodes
        if node.op in native_vector_ops
        and vectorization_root(node) in native_vector_roots
    ]
    visited_dependencies = set(pending)
    while pending:
        current = node_map[pending.pop()]
        for input_id in current.inputs:
            dependency = node_map[input_id]
            is_internal = (
                dependency.metadata.get("vectorization_internal") is True
            )
            if is_internal:
                native_dependencies.add(input_id)
            # AutoDistribution is allowed to insert zero-copy views, tuple
            # projections, and a required collective between a selected
            # typed-vector producer and its native consumer.  Those nodes are
            # physical and survive this pass, but must be transparent to the
            # liveness walk so they cannot hide an internal Pack/Reshape.
            if (
                (is_internal or dependency.op in dependency_bridges)
                and input_id not in visited_dependencies
            ):
                visited_dependencies.add(input_id)
                pending.append(input_id)
    compute_roots: dict[str, str] = {}
    for node in module.nodes:
        if node.op in distribution_adapters:
            continue
        if node.metadata.get("vectorization_internal") is not True:
            continue
        if node.id in native_dependencies:
            # This value remains a real typed-vector producer.  Its
            # ``vectorization_semantic_id`` is provenance only and must not
            # redirect physical users to a logical node which is intentionally
            # absent from the selected vector expression.
            continue
        if vectorization_root(node) in native_vector_roots:
            # These operations consume the typed Pack/Unpack expression
            # itself; those nodes are physical IR, not removable schedule
            # scaffolding.
            continue
        semantic_id = node.metadata.get("vectorization_semantic_id")
        if semantic_id is not None and str(semantic_id) not in node_ids:
            compute_roots[node.id] = str(semantic_id)
            continue
        if (
            node.metadata.get("vectorization_role") == "compute"
            and str(node.metadata.get("vectorization_root")) not in node_ids
        ):
            compute_roots[node.id] = str(node.metadata["vectorization_root"])
    specialized_boundary_computes: dict[str, str] = {}
    specialized_vector_inputs: set[str] = set()
    for node in module.nodes:
        if (
            node.metadata.get("vectorized_from") is None
            or node.metadata.get("cloned_for_function_variant") is None
        ):
            continue
        compute = _referenced_vector_expression(node, module)
        if compute is None:
            continue
        specialized_boundary_computes[node.id] = compute.id
        pending_inputs = list(compute.inputs)
        while pending_inputs:
            input_id = pending_inputs.pop()
            value = node_map[input_id]
            if (
                value.metadata.get("vectorization_internal") is not True
                or value.metadata.get("vectorization_role") == "compute"
                or value.op not in {
                    "distributed.sharded_view",
                    "tensors.bitcast",
                    "tensors.pack",
                    "tensors.pad",
                    "tensors.slice_to_shape",
                    "tensors.unpack",
                }
                or input_id in specialized_vector_inputs
            ):
                continue
            specialized_vector_inputs.add(input_id)
            pending_inputs.extend(value.inputs)
    internal = {
        node.id
        for node in module.nodes
        if node.metadata.get("vectorization_internal") is True
        and node.op not in distribution_adapters
        and node.id not in compute_roots
        and node.id not in specialized_boundary_computes.values()
        and node.id not in specialized_vector_inputs
        and vectorization_root(node) not in native_vector_roots
        and node.id not in native_dependencies
    }
    nodes: list[Node] = []
    prepared_by_id: dict[str, Node] = {}
    native_producers: dict[str, str] = {}
    for node in module.nodes:
        root = vectorization_root(node)
        if root not in native_vector_roots or node.op not in native_vector_ops:
            continue
        previous = native_producers.get(root)
        if previous is None or node.metadata.get("vectorization_role") == "compute":
            native_producers[root] = node.id

    def prepared_input(value: str) -> Node:
        if value in prepared_by_id:
            return prepared_by_id[value]
        producer_id = native_producers.get(value)
        if producer_id is None or producer_id not in prepared_by_id:
            return module.node_map[value]
        producer = prepared_by_id[producer_id]
        axes = tuple(
            int(axis)
            for axis in producer.metadata.get(
                "selected_vector_axes",
                producer.metadata.get("vector_axes", ()),
            )
        )
        if not axes:
            raise ValueError(
                f"Native vector producer {producer.id!r} has no unpack axes."
            )
        definition = get_definition("tensors.unpack")
        prepared_call = definition.prepare((producer,), {"axes": axes})
        boundary = Node(
            id=value,
            op="tensors.unpack",
            inputs=(producer.id,),
            type=prepared_call.result_type,
            effect=prepared_call.effect,
            attrs=prepared_call.attrs,
            metadata={
                "introduced_by": "LowerVectorizationContracts",
                "vector_boundary": "native_to_semantic",
            },
        )
        nodes.append(boundary)
        prepared_by_id[value] = boundary
        return boundary

    for node in module.nodes:
        if node.id in internal:
            continue
        if node.op in distribution_adapters:
            # Boundary layout propagation retains the replaced expression's
            # provenance. Boxing and ShardedView are real physical edges, not
            # vector compute wrappers: never re-execute that semantic op on
            # an already-computed value (notably partial NormStats).
            prepared = replace(
                node,
                inputs=tuple(compute_roots.get(value, value) for value in node.inputs),
            )
            nodes.append(prepared)
            prepared_by_id[prepared.id] = prepared
            continue
        if node.id in specialized_boundary_computes.values():
            metadata = {
                key: value
                for key, value in node.metadata.items()
                if not str(key).startswith("vectorization_")
                and key not in {"vectorized_from", "vector_axes", "vector_lanes"}
            }
            metadata.update({
                "selected_vectorization": str(
                    node.metadata["vectorization_candidate"]
                ),
                "selected_vector_axes": tuple(
                    int(value) for value in node.metadata["vector_axes"]
                ),
                "selected_vector_lanes": tuple(
                    int(value) for value in node.metadata["vector_lanes"]
                ),
                "prepared_from_vectorization": True,
            })
            prepared = replace(node, metadata=metadata)
            nodes.append(prepared)
            prepared_by_id[prepared.id] = prepared
            continue
        if node.id in specialized_boundary_computes:
            metadata = {
                key: value
                for key, value in node.metadata.items()
                if not str(key).startswith("vectorization_")
                and key not in {"vectorized_from", "vector_axes", "vector_lanes"}
            }
            metadata.update({
                "introduced_by": "LowerVectorizationContracts",
                "vector_boundary": "specialized_semantic_to_logical",
            })
            prepared = replace(node, metadata=metadata)
            nodes.append(prepared)
            prepared_by_id[prepared.id] = prepared
            continue
        current_vector_root = vectorization_root(node)
        if (
            current_vector_root in native_vector_roots
            and node.op in native_vector_ops
            and node.metadata.get("vectorization_candidate") is not None
        ):
            prepared = replace(
                node,
                metadata={
                    **dict(node.metadata),
                    "selected_vectorization": str(
                        node.metadata["vectorization_candidate"]
                    ),
                    "selected_vector_axes": tuple(
                        int(value) for value in node.metadata["vector_axes"]
                    ),
                    "selected_vector_lanes": tuple(
                        int(value) for value in node.metadata["vector_lanes"]
                    ),
                    "prepared_from_vectorization": True,
                },
            )
            nodes.append(prepared)
            prepared_by_id[prepared.id] = prepared
            continue
        if (
            node.op == "builtin.get_item"
            and len(node.inputs) == 1
        ):
            # Tuple projections are structural even when a function-boundary
            # or combine rewrite preserves the provenance of the semantic op
            # they replaced.  Never reinterpret a tuple input as that op.
            # A combine's value projection still denotes the replaced Add
            # boundary, so publish its vector contract for fused-provider
            # legality checks. Other fields (for example threaded NormStats)
            # remain ordinary structural projections.
            metadata = dict(node.metadata)
            parent = module.node_map[node.inputs[0]]
            candidate = (
                metadata.get("vectorization_candidate")
                if parent.op in {
                    "ntt.matmul_norm_stats_combine",
                    "ntt.matmul_norm_stats",
                }
                and int(node.attrs.get("index", -1)) == 0
                else None
            )
            if candidate is not None:
                metadata.update({
                    "selected_vectorization": str(candidate),
                    "selected_vector_axes": tuple(
                        int(value) for value in metadata["vector_axes"]
                    ),
                    "selected_vector_lanes": tuple(
                        int(value) for value in metadata["vector_lanes"]
                    ),
                    "prepared_from_vectorization": True,
                })
            prepared = replace(node, metadata=metadata)
            nodes.append(prepared)
            prepared_by_id[prepared.id] = prepared
            continue
        if node.id in native_dependencies:
            # A selected native vector op consumes this exact typed value.
            # Its own semantic provenance may name a different logical root;
            # dependency reachability, rather than that annotation, owns the
            # value's lifetime after AutoDistribution inserts bridge nodes.
            nodes.append(node)
            prepared_by_id[node.id] = node
            continue
        if (
            (node.id in native_vector_roots or current_vector_root in native_vector_roots)
            and node.metadata.get("vectorized_from") is not None
        ):
            # Keep the typed-vector result boundary (usually Unpack) as an
            # explicit zero-copy view.  Its native compute node above owns the
            # executable vector schedule.
            nodes.append(node)
            prepared_by_id[node.id] = node
            continue
        vectorized_from = node.metadata.get("vectorized_from")
        if vectorized_from is None:
            prepared = replace(
                node,
                id=compute_roots.get(node.id, node.id),
                inputs=tuple(compute_roots.get(value, value) for value in node.inputs),
            )
            nodes.append(prepared)
            prepared_by_id[prepared.id] = prepared
            continue
        metadata = {
            key: value
            for key, value in node.metadata.items()
            if not str(key).startswith("vectorization_")
            and key not in {"vectorized_from", "vector_axes", "vector_lanes"}
        }
        metadata.update(
            {
                "selected_vectorization": str(
                    node.metadata["vectorization_candidate"]
                ),
                "selected_vector_axes": tuple(
                    int(value) for value in node.metadata["vector_axes"]
                ),
                "selected_vector_lanes": tuple(
                    int(value) for value in node.metadata["vector_lanes"]
                ),
            }
        )
        semantic_op = str(vectorized_from)
        semantic_inputs = _recover_semantic_inputs(
            node,
            module,
            compute_roots=compute_roots,
        )
        semantic_attrs = dict(node.metadata.get("vectorization_attrs", {}))
        definition = get_definition(semantic_op)
        input_nodes = tuple(prepared_input(value) for value in semantic_inputs)
        normalized_attrs = definition.normalize_attrs(semantic_attrs)
        prepared = replace(
            node,
            id=compute_roots.get(node.id, node.id),
            op=semantic_op,
            inputs=semantic_inputs,
            type=definition.infer_type(input_nodes, normalized_attrs),
            effect=definition.infer_effect(input_nodes, normalized_attrs),
            attrs=definition.ir_attrs(normalized_attrs),
            metadata={**metadata, "prepared_from_vectorization": True},
        )
        nodes.append(prepared)
        prepared_by_id[prepared.id] = prepared
    live = {node.id for node in nodes}
    points = tuple(
        replace(point, owner=compute_roots.get(point.owner, point.owner))
        if point.owner is not None
        and compute_roots.get(point.owner, point.owner) in live
        else point
        for point in module.selection_points
        if point.owner is None or compute_roots.get(point.owner, point.owner) in live
    )
    point_ids = {point.id for point in points}
    selections = tuple(record for record in module.selections if record.point_id in point_ids)
    return replace(module, nodes=tuple(nodes), selection_points=points, selections=selections)


def _recover_semantic_inputs(
    root: Node,
    module: IRModule,
    *,
    compute_roots: dict[str, str],
) -> tuple[str, ...]:
    """Recover semantic operands from the selected typed-vector expression.

    ``vectorization_inputs`` is editable provenance, not an IR use-list.  E-graph
    hash-consing may therefore replace an equivalent operand (most visibly a
    splat zero) without rewriting that annotation.  The executable typed-vector
    graph is authoritative: locate the generated compute node and peel only
    compiler-owned Pad/Pack boundaries from each of its real operands.

    This deliberately avoids teaching generic rewriting about target metadata
    keys and keeps every operand that affects liveness represented by a normal
    ``Node.inputs`` edge.
    """

    compute = _referenced_vector_expression(root, module)
    generated = tuple(
        node
        for node in module.nodes
        if node.metadata.get("vectorization_internal") is True
        and str(node.metadata.get("vectorization_semantic_id")) == root.id
    )
    if compute is None and not generated:
        generated = tuple(
            node
            for node in module.nodes
            if node.metadata.get("vectorization_internal") is True
            and node.metadata.get("vectorization_role") == "compute"
            and str(node.metadata.get("vectorization_root")) == root.id
        )
    if compute is None and generated:
        if len(generated) != 1:
            raise ValueError(
                f"Vectorization root {root.id!r} has {len(generated)} generated compute nodes."
            )
        compute = generated[0]
    if compute is None:
        compute = root

    return tuple(
        _peel_vector_operand(value, module, compute_roots=compute_roots, seen=frozenset())
        for value in compute.inputs
    )


def _referenced_vector_expression(root: Node, module: IRModule) -> Node | None:
    """Follow a selected boundary to its nearest semantic vector expression.

    Function-boundary specialization may clone a vectorized compute while
    retaining the original ``vectorization_root`` provenance.  Looking up a
    compute globally by that provenance is then ambiguous and, for a cloned
    result boundary, can peel the compute itself into a single semantic root.
    Propagation creates another case: Pad/Slice/Reshape expressions carry their
    own ``vectorization_semantic_id`` and must not be peeled through to an
    earlier compute.  The real SSA edge remains authoritative, but traversal
    stops at the nearest expression denoting the current semantic boundary.
    """

    if (
        root.metadata.get("vectorization_internal") is True
        and root.metadata.get("vectorization_semantic_id") is not None
        and root.op not in _DISTRIBUTION_ADAPTER_OPS
    ):
        return root
    if len(root.inputs) != 1:
        return None
    value_id = root.inputs[0]
    seen: set[str] = set()
    while value_id not in seen:
        seen.add(value_id)
        value = module.node_map[value_id]
        if value.op in _DISTRIBUTION_ADAPTER_OPS and len(value.inputs) == 1:
            # A cloned result may have a real reshard between its native
            # compute and bitcast. The bridge survives lowering, but neither
            # its provenance nor its missing internal flag owns the compute.
            value_id = value.inputs[0]
            continue
        if str(value.metadata.get("vectorization_semantic_id")) == root.id:
            return value
        if (
            value.metadata.get("vectorization_internal") is True
            and value.metadata.get("vectorization_role") == "compute"
        ):
            return value
        if (
            value.metadata.get("vectorization_internal") is not True
            or value.op not in {
                "distributed.sharded_view",
                "tensors.bitcast",
                "tensors.pack",
                "tensors.pad",
                "tensors.slice_to_shape",
                "tensors.unpack",
            }
            or len(value.inputs) != 1
        ):
            return None
        value_id = value.inputs[0]
    raise ValueError(
        f"Cycle while locating generated compute for vectorization root {root.id!r}."
    )


def _peel_vector_operand(
    value_id: str,
    module: IRModule,
    *,
    compute_roots: dict[str, str],
    seen: frozenset[str],
) -> str:
    if value_id in seen:
        raise ValueError(f"Cycle while recovering vector operand {value_id!r}.")
    node = module.node_map[value_id]
    if node.metadata.get("vectorization_internal") is not True:
        return compute_roots.get(value_id, value_id)
    semantic_id = node.metadata.get("vectorization_semantic_id")
    if semantic_id is not None:
        return str(semantic_id)
    if node.metadata.get("vectorization_role") == "compute":
        return str(node.metadata["vectorization_root"])
    if node.op not in {"tensors.pack", "tensors.pad", "tensors.unpack", "tensors.slice_to_shape"}:
        raise ValueError(
            f"Cannot recover semantic operand through compiler-owned {node.op!r} node {node.id!r}."
        )
    if len(node.inputs) != 1:
        raise ValueError(
            f"Compiler-owned vector boundary {node.id!r} must have one input, got {len(node.inputs)}."
        )
    return _peel_vector_operand(
        node.inputs[0],
        module,
        compute_roots=compute_roots,
        seen=seen | {value_id},
    )


__all__ = ["NttDistributionPolicy", "lower_vectorization_contracts"]
