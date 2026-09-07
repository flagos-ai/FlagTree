# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""NTT AutoPacking candidate generation and selected-layout application."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from triton.flagmega.ir import (
    Candidate,
    DType,
    IRModule,
    Node,
    SelectionPoint,
    TensorType,
    VectorType,
    get_definition,
    tensor_type,
    try_div_exactly,
)
from triton.flagmega.passes.functions import (
    lift_parameter_constant_transforms,
)
from triton.flagmega.rules import DataflowRewriter, RewriteResult, RewriteRule
from triton.flagmega.ir.ops.tensors._k_major import (
    k_major_layout_name,
)


class NttPackingPolicy:
    op_names = frozenset({
        "math.block_scaled_matmul",
        "math.matmul",
        "math.vectorized_matmul",
        "nn.dense_matmul_glu",
        "nn.matmul_glu",
        "nn.qkv_parallel_linear",
    })

    def __init__(
        self,
        *,
        vector_bytes: int,
        k_pack: int,
    ) -> None:
        if any(value <= 0 for value in (vector_bytes, k_pack)):
            raise ValueError("NTT packing geometry must contain positive integers.")
        self.vector_bytes = vector_bytes
        self.k_pack = k_pack

    @property
    def identity(self) -> str:
        """Readable PyNTT rule registration, with no machine identity."""

        return (
            "pyntt-auto-packing/v2("
            f"vector_bytes={self.vector_bytes},"
            f"k_pack={self.k_pack})"
        )

    def propose(self, module: IRModule, target) -> IRModule:
        points = self.candidate_points(module)
        return target.add_default_selections(
            module,
            points,
            "Select a byte-preserving physical layout from semantic shape and dtype facts.",
            policy_version=self.identity,
        )

    def candidate_points(self, module: IRModule) -> tuple[SelectionPoint, ...]:
        """Enumerate only the candidates registered by portable PyNTT rules."""

        existing = {point.id for point in module.selection_points}
        points: list[SelectionPoint] = []
        for node in module.nodes:
            root_id = (
                str(node.metadata.get("vectorization_root"))
                if node.op == "math.vectorized_matmul"
                and node.metadata.get("vectorization_root") is not None
                else node.id
            )
            point_id = f"packing.{root_id}"
            if point_id in existing:
                continue
            if (
                node.op == "math.vectorized_matmul"
                and self._can_pack_vectorized_matmul(node, module)
            ):
                parameters = self._vectorized_k_major_parameters(node, module)
                assert parameters is not None
                layout = str(parameters["layout"])
                points.append(SelectionPoint(
                    id=point_id,
                    kind="packing",
                    candidates=(
                        Candidate(
                            "packing.logical",
                            {"layout": "logical"},
                            {"requires_offline_pack": False},
                        ),
                        Candidate(
                            f"packing.{layout}",
                            parameters,
                            {
                                "requires_offline_pack": True,
                                "byte_count_preserving": True,
                                "constant_recipe": (
                                    (
                                        "unpack", "pack", "pack", "pack", "permute"
                                    )
                                    if bool(node.attrs.get("transpose_b", False))
                                    else (
                                        "unpack", "permute", "pack", "pack", "pack", "permute"
                                    )
                                ),
                                "vector_type": True,
                                "preserves_vectorized_result": True,
                            },
                        ),
                    ),
                    default_candidate=f"packing.{layout}",
                    owner=node.id,
                ))
                continue
            if node.op == "nn.qkv_parallel_linear" and self._can_pack_qkv(node, module):
                parameters = self._qkv_k_major_parameters(node, module)
                assert parameters is not None
                layout = str(parameters["layout"])
                points.append(SelectionPoint(
                    id=point_id,
                    kind="packing",
                    candidates=(
                        Candidate(
                            "packing.logical",
                            {"layout": "logical"},
                            {"requires_offline_pack": False},
                        ),
                        Candidate(
                            f"packing.{layout}",
                            parameters,
                            {
                                "requires_offline_pack": True,
                                "byte_count_preserving": True,
                                "constant_recipe": (
                                    "permute", "pack", "pack", "pack", "permute"
                                ),
                                "vector_type": True,
                                "three_independent_weights": True,
                            },
                        ),
                    ),
                    default_candidate=f"packing.{layout}",
                    owner=node.id,
                ))
                continue
            if node.op == "nn.dense_matmul_glu" and self._can_pack_dense_glu(node, module):
                dense = self._dense_k_major_parameters(DType.BFLOAT16)
                assert dense is not None
                layout = str(dense["layout"])
                points.append(SelectionPoint(
                    id=point_id,
                    kind="packing",
                    candidates=(
                        Candidate(
                            "packing.logical",
                            {"layout": "logical"},
                            {"requires_offline_pack": False},
                        ),
                        Candidate(
                            f"packing.{layout}",
                            dense,
                            {
                                "requires_offline_pack": True,
                                "byte_count_preserving": True,
                                "constant_recipe": ("reshape", "permute", "reshape"),
                            },
                        ),
                    ),
                    default_candidate=f"packing.{layout}",
                    owner=node.id,
                ))
                continue
            if node.op == "math.matmul" and self._can_pack_dense_matmul(node, module):
                dense = self._dense_k_major_parameters(DType.BFLOAT16)
                assert dense is not None
                layout = str(dense["layout"])
                points.append(SelectionPoint(
                    id=point_id,
                    kind="packing",
                    candidates=(
                        Candidate(
                            "packing.logical",
                            {"layout": "logical"},
                            {"requires_offline_pack": False},
                        ),
                        Candidate(
                            f"packing.{layout}",
                            dense,
                            {
                                "requires_offline_pack": True,
                                "byte_count_preserving": True,
                                "constant_recipe": ("reshape", "permute", "reshape"),
                            },
                        ),
                    ),
                    default_candidate=f"packing.{layout}",
                    owner=node.id,
                ))
                continue
            if node.op not in {"math.block_scaled_matmul", "nn.matmul_glu"} or not self._can_pack(node, module):
                continue
            points.append(SelectionPoint(
                id=point_id,
                kind="packing",
                candidates=(
                    Candidate("packing.logical", {"layout": "logical"}, {"requires_offline_pack": False}),
                    Candidate(
                        "packing.n_major_k_packed",
                        {
                            "layout": "n_major_k_packed",
                            "vector_bytes": self.vector_bytes,
                            "k_pack": self.k_pack,
                        },
                        {"requires_offline_pack": False, "byte_preserving_view": True},
                    ),
                ),
                default_candidate="packing.n_major_k_packed",
                owner=node.id,
            ))
        return tuple(points)

    def apply(self, module: IRModule, target) -> IRModule:
        selected = module.selection_map
        points = {point.id: point for point in module.selection_points}
        points_by_owner = {
            point.owner: point
            for point in module.selection_points
            if point.kind == "packing"
        }

        def matches(node: Node, _module: IRModule) -> bool:
            point = points_by_owner.get(node.id)
            record = None if point is None else selected.get(point.id)
            if node.op not in self.op_names or record is None:
                return False
            point = points.get(record.point_id)
            if point is None:
                return False
            candidate = next(
                (value for value in point.candidates if value.id == record.candidate_id),
                None,
            )
            return (
                candidate is not None
                and candidate.parameters.get("layout", "logical") != "logical"
            )

        def rewrite(node: Node, current: IRModule) -> RewriteResult:
            point = points_by_owner[node.id]
            candidate = next(value for value in point.candidates if value.id == selected[point.id].candidate_id)
            if node.op == "math.vectorized_matmul":
                return self._pack_vectorized_matmul(node, current, candidate)
            if node.op == "nn.qkv_parallel_linear":
                return self._pack_qkv_parallel_linear(node, current, candidate)
            if node.op == "nn.dense_matmul_glu":
                return self._pack_dense_glu(node, current, candidate)
            if node.op == "math.matmul":
                return self._pack_dense_matmul(node, current, candidate)
            k_pack = int(candidate.parameters["k_pack"])
            weight_ids = (
                (node.inputs[1],)
                if node.op == "math.block_scaled_matmul"
                else (node.inputs[1], node.inputs[2])
            )
            helpers: list[Node] = []
            replacements: dict[str, str] = {}
            k_vector = 0
            for input_id in weight_ids:
                weight = current.node_map[input_id]
                assert isinstance(weight.type, TensorType) and isinstance(weight.type.dtype, DType)
                k_vector = int(candidate.parameters["vector_bytes"]) // weight.type.dtype.itemsize
                lanes = (k_pack, k_vector)
                packed_k = try_div_exactly(weight.type.shape[1], _product(lanes))
                assert packed_k is not None
                packed_type = tensor_type(
                    VectorType(weight.type.dtype, lanes),
                    (weight.type.shape[0], packed_k),
                    layout=weight.type.layout,
                )
                packed_id = f"{input_id}.packed_for.{node.id}"
                helpers.append(Node(
                    id=packed_id,
                    op="tensors.pack",
                    inputs=(input_id,),
                    type=packed_type,
                    attrs={"lanes": lanes, "axes": (1, 1)},
                    metadata={
                        "packed_from": input_id,
                        "packed_for": node.id,
                        "packed_layout": "n_major_k_packed",
                    },
                ))
                replacements[input_id] = packed_id
            inputs = tuple(replacements.get(input_id, input_id) for input_id in node.inputs)
            packed_op = (
                "math.packed_block_scaled_matmul"
                if node.op == "math.block_scaled_matmul"
                else "nn.packed_matmul_glu"
            )
            packed_attrs = {
                **dict(node.attrs),
                "k_pack": k_pack,
                "k_vector": k_vector,
                "packed_layout": "n_major_k_packed",
            }
            return RewriteResult(replace(
                node,
                op=packed_op,
                inputs=inputs,
                attrs=packed_attrs,
                metadata={
                    **dict(node.metadata),
                    "packed_from": node.op,
                    "packing_candidate": candidate.id,
                },
            ), tuple(helpers))

        packed = DataflowRewriter(
            (RewriteRule("ntt-auto-packing", matches, rewrite),)
        ).rewrite(module)
        return lift_parameter_constant_transforms(packed)

    def _pack_vectorized_matmul(
        self,
        node: Node,
        module: IRModule,
        candidate: Candidate,
    ) -> RewriteResult:
        """Apply nncase ``PackMatMulRhsKMajor`` without scalarizing the graph."""

        n_vector = int(candidate.parameters["n_vector"])
        k_pack = int(candidate.parameters["k_pack"])
        k_vector = int(candidate.parameters["k_vector"])
        rhs = module.node_map[node.inputs[1]]
        helpers: list[Node] = []

        def append(
            suffix: str,
            op: str,
            inputs: tuple[Node, ...],
            attrs: Mapping[str, object],
            *,
            metadata: Mapping[str, object] | None = None,
        ) -> Node:
            definition = get_definition(op)
            prepared = definition.prepare(inputs, attrs)
            helper = Node(
                id=f"{node.id}.rhs_k_major.{suffix}",
                op=op,
                inputs=tuple(value.id for value in prepared.inputs),
                type=prepared.result_type,
                effect=prepared.effect,
                attrs=prepared.attrs,
                metadata={} if metadata is None else metadata,
            )
            helpers.append(helper)
            return helper

        transpose_b = bool(node.attrs.get("transpose_b", False))
        n_axis = 0 if transpose_b else 1
        scalar_rhs = append(
            "unpack_n", "tensors.unpack", (rhs,), {"axes": (n_axis,)}
        )
        n_major = scalar_rhs
        if not transpose_b:
            n_major = append(
                "transpose_kn", "tensors.permute", (scalar_rhs,), {"axes": (1, 0)}
            )
        packed_k_vector = append(
            "k_vector",
            "tensors.pack",
            (n_major,),
            {"lanes": (k_vector,), "axes": (1,)},
        )
        packed_k = append(
            "k_pack",
            "tensors.pack",
            (packed_k_vector,),
            {"lanes": (k_pack,), "axes": (1,)},
        )
        packed_n = append(
            "n_vector",
            "tensors.pack",
            (packed_k,),
            {"lanes": (n_vector,), "axes": (0,)},
        )
        packed_rhs = append(
            "physical",
            "tensors.permute",
            (packed_n,),
            {"axes": (1, 0)},
            metadata={
                "packed_from": _parameter_transform_source(rhs, module),
                "packed_for": node.id,
                "packed_layout": "k_major",
                "vector_lanes": (n_vector, k_pack, k_vector),
            },
        )
        none_definition = get_definition("builtin.none")
        none_prepared = none_definition.prepare((), {})
        none = Node(
            id=f"{node.id}.packed_none",
            op="builtin.none",
            inputs=(),
            type=none_prepared.result_type,
            effect=none_prepared.effect,
            attrs=none_prepared.attrs,
        )
        helpers.append(none)
        definition = get_definition("ntt.packed_matmul")
        output_type = node.type
        assert isinstance(output_type, TensorType)
        output_dtype = (
            output_type.dtype.elem_type
            if isinstance(output_type.dtype, VectorType)
            else output_type.dtype
        )
        prepared = definition.prepare(
            (module.node_map[node.inputs[0]], packed_rhs, none, none),
            {
                "fused_reduce": False,
                "output_data_type": output_dtype,
                "rhs_layout": "k_major",
            },
        )
        if prepared.result_type != node.type:
            raise AssertionError(
                "K-major PackedMatMul must preserve VectorizedMatMul's result type."
            )
        replacement = replace(
            node,
            op="ntt.packed_matmul",
            inputs=tuple(value.id for value in prepared.inputs),
            type=prepared.result_type,
            effect=prepared.effect,
            attrs=prepared.attrs,
            metadata={
                **dict(node.metadata),
                "packed_from": node.op,
                "packing_candidate": candidate.id,
            },
        )
        return RewriteResult(replacement, tuple(helpers))

    def _pack_qkv_parallel_linear(
        self,
        node: Node,
        module: IRModule,
        candidate: Candidate,
    ) -> RewriteResult:
        n_vector = int(candidate.parameters["n_vector"])
        k_vector = int(candidate.parameters["k_vector"])
        k_pack = int(candidate.parameters["k_pack"])
        helpers: list[Node] = []

        def append(
            node_id: str,
            op: str,
            inputs: tuple[Node, ...],
            attrs: Mapping[str, object],
            *,
            metadata: Mapping[str, object] | None = None,
        ) -> Node:
            definition = get_definition(op)
            prepared = definition.prepare(inputs, attrs)
            helper = Node(
                id=node_id,
                op=op,
                inputs=tuple(value.id for value in prepared.inputs),
                type=prepared.result_type,
                effect=prepared.effect,
                attrs=prepared.attrs,
                metadata={} if metadata is None else metadata,
            )
            helpers.append(helper)
            return helper

        packed_weights: list[Node] = []
        for role, input_id in zip(("q", "k", "v"), node.inputs[1:4]):
            weight = module.node_map[input_id]
            packed_from = _parameter_transform_source(weight, module)
            prefix = f"{node.id}.{role}_weight_pack"
            transposed = append(
                f"{prefix}.transpose_nk",
                "tensors.permute",
                (weight,),
                {"axes": (1, 0)},
            )
            packed_k_vector = append(
                f"{prefix}.k_vector",
                "tensors.pack",
                (transposed,),
                {"lanes": (k_vector,), "axes": (1,)},
            )
            packed_k = append(
                f"{prefix}.k_pack",
                "tensors.pack",
                (packed_k_vector,),
                {"lanes": (k_pack,), "axes": (1,)},
            )
            packed_n = append(
                f"{prefix}.n_vector",
                "tensors.pack",
                (packed_k,),
                {"lanes": (n_vector,), "axes": (0,)},
            )
            packed_weights.append(append(
                f"{prefix}.k_major",
                "tensors.permute",
                (packed_n,),
                {"axes": (1, 0)},
                metadata={
                    "packed_from": packed_from,
                    "packed_for": node.id,
                    "packed_layout": "k_major",
                    "vector_lanes": (n_vector, k_pack, k_vector),
                },
            ))

        packed_biases: list[Node] = []
        for role, input_id in zip(("q", "k", "v"), node.inputs[4:7]):
            bias = module.node_map[input_id]
            if isinstance(bias.type, TensorType):
                bias = append(
                    f"{node.id}.{role}_bias_pack.n_vector",
                    "tensors.pack",
                    (bias,),
                    {"lanes": (n_vector,), "axes": (0,)},
                    metadata={
                        "packed_from": bias.id,
                        "packed_for": node.id,
                        "packed_layout": "n_vector",
                    },
                )
            packed_biases.append(bias)

        projection_inputs = (
            module.node_map[node.inputs[0]],
            *packed_weights,
            *packed_biases,
            *(module.node_map[input_id] for input_id in node.inputs[7:13]),
        )
        packed_definition = get_definition("ntt.packed_qkv_parallel_linear")
        packed_prepared = packed_definition.prepare(
            projection_inputs,
            {
                "num_heads": node.attrs["num_heads"],
                "num_kv_heads": node.attrs["num_kv_heads"],
                "output_data_type": node.attrs["output_data_type"],
                "rhs_layout": "k_major",
            },
        )
        packed_projection = Node(
            id=f"{node.id}.packed_projection",
            op="ntt.packed_qkv_parallel_linear",
            inputs=tuple(value.id for value in packed_prepared.inputs),
            type=packed_prepared.result_type,
            effect=packed_prepared.effect,
            attrs=packed_prepared.attrs,
            metadata={
                **dict(node.metadata),
                "packed_from": node.op,
                "packing_candidate": candidate.id,
                # FoldGetItemTuple makes the wrapper tuple dead.  Preserve the
                # editable packing decision by moving its owner to the packed
                # semantic projection that survives DCE.
                "selection_owner_for": node.id,
            },
        )
        helpers.append(packed_projection)

        combine = append(
            f"{node.id}.packed_combine",
            "ntt.packed_qkv_parallel_linear_combine",
            (packed_projection,),
            {"output_type": packed_projection.type},
            metadata={
                "packed_from": node.op,
                "packing_candidate": candidate.id,
            },
        )

        unpacked: list[Node] = []
        for index, role in enumerate(("q", "k", "v")):
            item = append(
                f"{node.id}.{role}.packed",
                "builtin.get_item",
                (combine,),
                {"index": index},
            )
            unpacked.append(append(
                f"{node.id}.{role}",
                "tensors.unpack",
                (item,),
                {"axes": (1,)},
            ))

        tuple_definition = get_definition("builtin.tuple")
        tuple_prepared = tuple_definition.prepare(tuple(unpacked), {})
        replacement = replace(
            node,
            op="builtin.tuple",
            inputs=tuple(value.id for value in tuple_prepared.inputs),
            type=tuple_prepared.result_type,
            effect=tuple_prepared.effect,
            attrs=tuple_prepared.attrs,
            metadata={
                **dict(node.metadata),
                "packed_from": node.op,
                "packing_candidate": candidate.id,
            },
        )
        return RewriteResult(replacement, tuple(helpers))


    def _pack_dense_matmul(
        self,
        node: Node,
        module: IRModule,
        candidate: Candidate,
    ) -> RewriteResult:
        n_lane = int(candidate.parameters["n_lane"])
        k_lane = int(candidate.parameters["k_lane"])
        payload_groups = int(candidate.parameters["payload_groups"])
        payload_width = int(candidate.parameters["payload_width"])
        packed_layout = str(candidate.parameters["layout"])
        weight = module.node_map[node.inputs[1]]
        assert isinstance(weight.type, TensorType)
        n = weight.type.shape[0].fixed_value
        k = weight.type.shape[1].fixed_value
        assert n is not None and k is not None
        helpers: list[Node] = []

        def append(suffix: str, op: str, input_node: Node, attrs, *, metadata=None) -> Node:
            definition = get_definition(op)
            prepared = definition.prepare((input_node,), attrs)
            helper = Node(
                id=f"{node.id}.{weight.id}_pack.{suffix}",
                op=op,
                inputs=tuple(value.id for value in prepared.inputs),
                type=prepared.result_type,
                effect=prepared.effect,
                attrs=prepared.attrs,
                metadata={} if metadata is None else metadata,
            )
            helpers.append(helper)
            return helper

        split = append(
            "split_lanes",
            "tensors.reshape",
            weight,
            {"shape": (n // n_lane, n_lane, k // k_lane, k_lane)},
        )
        lane_major = append(
            "lane_major",
            "tensors.permute",
            split,
            {"axes": (2, 0, 1, 3)},
        )
        metadata = {
            "packed_from": weight.id,
            "packed_for": node.id,
            "packed_layout": packed_layout,
        }
        if "rdata_group" in weight.metadata:
            metadata["rdata_group"] = weight.metadata["rdata_group"]
        physical = append(
            "physical",
            "tensors.reshape",
            lane_major,
            {"shape": (
                k // k_lane, n // n_lane, payload_groups, payload_width
            )},
            metadata=metadata,
        )
        definition = get_definition("math.packed_dense_matmul")
        prepared = definition.prepare(
            (module.node_map[node.inputs[0]], physical),
            {"packed_layout": packed_layout, "logical_n": None},
        )
        replacement = replace(
            node,
            op="math.packed_dense_matmul",
            inputs=tuple(value.id for value in prepared.inputs),
            type=prepared.result_type,
            effect=prepared.effect,
            attrs=prepared.attrs,
            metadata={
                **dict(node.metadata),
                "packed_from": node.op,
                "packing_candidate": candidate.id,
            },
        )
        return RewriteResult(replacement, tuple(helpers))

    def _pack_dense_glu(
        self,
        node: Node,
        module: IRModule,
        candidate: Candidate,
    ) -> RewriteResult:
        n_lane = int(candidate.parameters["n_lane"])
        k_lane = int(candidate.parameters["k_lane"])
        payload_groups = int(candidate.parameters["payload_groups"])
        payload_width = int(candidate.parameters["payload_width"])
        packed_layout = str(candidate.parameters["layout"])
        helpers: list[Node] = []
        packed_weights: list[Node] = []

        def append(
            source_name: str,
            suffix: str,
            op: str,
            inputs: tuple[Node, ...],
            attrs: Mapping[str, object],
            *,
            metadata: Mapping[str, object] | None = None,
        ) -> Node:
            definition = get_definition(op)
            prepared = definition.prepare(inputs, attrs)
            helper = Node(
                id=f"{node.id}.{source_name}_pack.{suffix}",
                op=op,
                inputs=tuple(value.id for value in prepared.inputs),
                type=prepared.result_type,
                effect=prepared.effect,
                attrs=prepared.attrs,
                metadata={} if metadata is None else metadata,
            )
            helpers.append(helper)
            return helper

        for input_id in node.inputs[1:3]:
            weight = module.node_map[input_id]
            assert isinstance(weight.type, TensorType)
            n = weight.type.shape[0].fixed_value
            k = weight.type.shape[1].fixed_value
            assert n is not None and k is not None
            split = append(
                input_id,
                "split_lanes",
                "tensors.reshape",
                (weight,),
                {"shape": (n // n_lane, n_lane, k // k_lane, k_lane)},
            )
            lane_major = append(
                input_id,
                "lane_major",
                "tensors.permute",
                (split,),
                {"axes": (2, 0, 1, 3)},
            )
            metadata = {
                "packed_from": input_id,
                "packed_for": node.id,
                "packed_layout": packed_layout,
            }
            if "rdata_group" in weight.metadata:
                metadata["rdata_group"] = weight.metadata["rdata_group"]
            packed_weights.append(append(
                input_id,
                "physical",
                "tensors.reshape",
                (lane_major,),
                {"shape": (
                    k // k_lane, n // n_lane, payload_groups, payload_width
                )},
                metadata=metadata,
            ))

        inputs = (
            module.node_map[node.inputs[0]], packed_weights[0], packed_weights[1])
        definition = get_definition("nn.packed_dense_matmul_glu")
        prepared = definition.prepare(inputs, {
            "activation": node.attrs["activation"],
            "round_activation": node.attrs.get("round_activation", True),
            "packed_layout": packed_layout,
        })
        replacement = replace(
            node,
            op="nn.packed_dense_matmul_glu",
            inputs=tuple(value.id for value in prepared.inputs),
            type=prepared.result_type,
            effect=prepared.effect,
            attrs=prepared.attrs,
            metadata={
                **dict(node.metadata),
                "packed_from": node.op,
                "packing_candidate": candidate.id,
            },
        )
        return RewriteResult(replacement, tuple(helpers))

    def _can_pack_dense_matmul(self, node: Node, module: IRModule) -> bool:
        if (
            bool(node.attrs.get("transpose_a", False))
            or not bool(node.attrs.get("transpose_b", False))
        ):
            return False
        value = module.node_map[node.inputs[1]]
        value_type = value.type
        dense = self._dense_k_major_parameters(DType.BFLOAT16)
        return dense is not None and (
            value.op in {"builtin.weight", "builtin.var"}
            and (
                value.op == "builtin.weight"
                or "function_parameter" in value.metadata
            )
            and isinstance(value_type, TensorType)
            and value_type.rank == 2
            and value_type.dtype == DType.BFLOAT16
            and try_div_exactly(
                value_type.shape[0], int(dense["n_lane"])
            ) is not None
            and try_div_exactly(
                value_type.shape[1], int(dense["k_lane"])
            ) is not None
        )

    def _vectorized_k_major_parameters(
        self,
        node: Node,
        module: IRModule,
    ) -> dict[str, object] | None:
        lhs = module.node_map[node.inputs[0]].type
        rhs = module.node_map[node.inputs[1]].type
        if (
            not isinstance(lhs, TensorType)
            or not isinstance(rhs, TensorType)
            or isinstance(lhs.dtype, VectorType)
            or not isinstance(rhs.dtype, VectorType)
            or len(rhs.dtype.lanes) != 1
            or not isinstance(rhs.dtype.elem_type, DType)
            or rhs.dtype.elem_type == DType.FLOAT8_E4M3FN
            or self.vector_bytes % rhs.dtype.elem_type.itemsize
        ):
            return None
        n_vector = rhs.dtype.lanes[0]
        k_vector = self.vector_bytes // rhs.dtype.elem_type.itemsize
        if n_vector != k_vector:
            return None
        k_axis = 1 if bool(node.attrs.get("transpose_b", False)) else 0
        if try_div_exactly(rhs.shape[k_axis], self.k_pack * k_vector) is None:
            return None
        return {
            "layout": k_major_layout_name(n_vector, self.k_pack * k_vector),
            "vector_bytes": self.vector_bytes,
            "n_vector": n_vector,
            "k_pack": self.k_pack,
            "k_vector": k_vector,
        }

    def _can_pack_vectorized_matmul(self, node: Node, module: IRModule) -> bool:
        if (
            node.op != "math.vectorized_matmul"
            or bool(node.attrs.get("transpose_a", False))
            or tuple(node.attrs.get("lhs_axes", ()))
            or tuple(node.attrs.get("output_axes", ())) != (1,)
        ):
            return False
        n_axis = 0 if bool(node.attrs.get("transpose_b", False)) else 1
        if tuple(node.attrs.get("rhs_axes", ())) != (n_axis,):
            return False
        parameters = self._vectorized_k_major_parameters(node, module)
        return (
            parameters is not None
            and _is_offline_parameter_transform(node.inputs[1], module)
            and tuple(node.attrs.get("output_lanes", ()))
            == (int(parameters["n_vector"]),)
        )

    def _qkv_k_major_parameters(
        self,
        node: Node,
        module: IRModule,
    ) -> dict[str, object] | None:
        input_type = module.node_map[node.inputs[0]].type
        if not isinstance(input_type, TensorType) or not isinstance(input_type.dtype, DType):
            return None
        output_dtype = DType(node.attrs["output_data_type"])
        weights = tuple(module.node_map[input_id].type for input_id in node.inputs[1:4])
        if (
            self.vector_bytes % input_type.dtype.itemsize
            or self.vector_bytes % output_dtype.itemsize
            or any(
                not isinstance(weight, TensorType)
                or not isinstance(weight.dtype, DType)
                or self.vector_bytes % weight.dtype.itemsize
                for weight in weights
            )
        ):
            return None
        weight_dtype = weights[0].dtype
        assert isinstance(weight_dtype, DType)
        if any(weight.dtype != weight_dtype for weight in weights):
            return None
        n_vector = self.vector_bytes // output_dtype.itemsize
        k_vector = self.vector_bytes // weight_dtype.itemsize
        return {
            "layout": k_major_layout_name(n_vector, self.k_pack * k_vector),
            "vector_bytes": self.vector_bytes,
            "n_vector": n_vector,
            "k_pack": self.k_pack,
            "k_vector": k_vector,
        }

    def _can_pack_qkv(self, node: Node, module: IRModule) -> bool:
        parameters = self._qkv_k_major_parameters(node, module)
        if parameters is None:
            return False
        n_vector = int(parameters["n_vector"])
        k_width = int(parameters["k_pack"]) * int(parameters["k_vector"])
        for input_id in node.inputs[1:4]:
            weight = module.node_map[input_id]
            if (
                not isinstance(weight.type, TensorType)
                or weight.type.rank != 2
                or try_div_exactly(weight.type.shape[0], k_width) is None
                or try_div_exactly(weight.type.shape[1], n_vector) is None
            ):
                return False
        for input_id in node.inputs[4:7]:
            bias = module.node_map[input_id]
            if isinstance(bias.type, TensorType) and (
                bias.type.rank != 1
                or try_div_exactly(bias.type.shape[0], n_vector) is None
            ):
                return False
        return True

    def _can_pack_dense_glu(self, node: Node, module: IRModule) -> bool:
        dense = self._dense_k_major_parameters(DType.BFLOAT16)
        if dense is None:
            return False
        for input_id in node.inputs[1:3]:
            value = module.node_map[input_id]
            value_type = value.type
            if (
                value.op not in {"builtin.weight", "builtin.var"}
                or (
                    value.op == "builtin.var"
                    and "function_parameter" not in value.metadata
                )
                or not isinstance(value_type, TensorType)
                or value_type.rank != 2
                or value_type.dtype != DType.BFLOAT16
                or try_div_exactly(
                    value_type.shape[0], int(dense["n_lane"])
                ) is None
                or try_div_exactly(
                    value_type.shape[1], int(dense["k_lane"])
                ) is None
            ):
                return False
        return True

    def _can_pack(self, node: Node, module: IRModule) -> bool:
        weight_ids = (node.inputs[1],) if node.op == "math.block_scaled_matmul" else (node.inputs[1], node.inputs[2])
        for input_id in weight_ids:
            value_type = module.node_map[input_id].type
            if (
                not isinstance(value_type, TensorType)
                or value_type.rank != 2
                or value_type.dtype != DType.FLOAT8_E4M3FN
            ):
                return False
            if self.vector_bytes % value_type.dtype.itemsize:
                return False
            k_vector = self.vector_bytes // value_type.dtype.itemsize
            if try_div_exactly(
                value_type.shape[1], self.k_pack * k_vector
            ) is None:
                return False
        return True

    def _dense_k_major_parameters(self, dtype: DType):
        if self.vector_bytes % dtype.itemsize:
            return None
        vector_lane = self.vector_bytes // dtype.itemsize
        n_lane = vector_lane
        k_lane = self.k_pack * vector_lane
        return {
            "layout": k_major_layout_name(n_lane, k_lane),
            "n_lane": n_lane,
            "k_lane": k_lane,
            "payload_groups": self.k_pack,
            "payload_width": n_lane * vector_lane,
        }


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _parameter_transform_source(weight: Node, module: IRModule) -> str:
    """Trace an imported [N,K] transpose back to a reusable formal.

    That transpose is part of the offline physical-weight recipe.  Using the
    formal as the transform source lets LiftParameterConstantTransforms move
    transpose and packing together to each call actual.
    """

    current = _parameter_transform_origin(weight, module)
    if current.op == "builtin.var" and "function_parameter" in current.metadata:
        return current.id
    return weight.id


def _is_offline_parameter_transform(node_id: str, module: IRModule) -> bool:
    origin = _parameter_transform_origin(module.node_map[node_id], module)
    return origin.op == "builtin.weight" or (
        origin.op == "builtin.var" and "function_parameter" in origin.metadata
    )


def _parameter_transform_origin(weight: Node, module: IRModule) -> Node:
    current = weight
    seen: set[str] = set()
    while (
        current.id not in seen
        and current.op in {
            "tensors.pack",
            "tensors.pad",
            "tensors.permute",
            "tensors.reshape",
            "tensors.slice_to_shape",
            "tensors.unpack",
        }
        and len(current.inputs) == 1
        and current.effect.is_pure
    ):
        seen.add(current.id)
        current = module.node_map[current.inputs[0]]
    return current


__all__ = ["NttPackingPolicy"]
