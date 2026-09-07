# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""M/N/K vectorization candidates for rank-2 MatMul."""

from __future__ import annotations

from triton.flagmega.ir import DType, IRModule, Node, TensorType, VectorType
from triton.flagmega.ir.ops.math.vectorized_matmul import VectorizedMatMul
from triton.flagmega.rules import RewriteResult
from triton.flagmega.rules.ntt.vectorize.base import VectorizeCandidate
from triton.flagmega.rules.ntt.vectorize.utility import finish_vector_result, padding_for, prepare_packed_input, internal_metadata


class VectorizeMatMul:
    name = "VectorizeMatMul"
    op_names = frozenset({"math.matmul"})

    def __init__(
        self,
        *,
        lane_bytes: int = 16,
        max_axes: int = 2,
    ) -> None:
        self.lane_bytes = lane_bytes
        self.max_axes = max_axes

    def candidates(self, node: Node, module: IRModule) -> tuple[VectorizeCandidate, ...]:
        if node.op != "math.matmul" or not node.effect.is_pure or not isinstance(node.type, TensorType):
            return ()
        lhs = module.node_map[node.inputs[0]].type
        rhs = module.node_map[node.inputs[1]].type
        if (
            not isinstance(lhs, TensorType) or not isinstance(rhs, TensorType)
            or lhs.rank != 2 or rhs.rank != 2 or node.type.rank != 2
            or isinstance(lhs.dtype, VectorType) or lhs.dtype != rhs.dtype or not isinstance(lhs.dtype, DType)
            or lhs.dtype not in {DType.BFLOAT16, DType.FLOAT32}
        ):
            return ()
        lane = self.lane_bytes // lhs.dtype.itemsize
        if lane <= 1:
            return ()
        transpose_a = bool(node.attrs.get("transpose_a", False))
        transpose_b = bool(node.attrs.get("transpose_b", False))
        lhs_m, lhs_k = ((1, 0) if transpose_a else (0, 1))
        rhs_k, rhs_n = ((1, 0) if transpose_b else (0, 1))
        # Match nncase's VectorizeMatMul alternatives. ``max_axes`` is the
        # maximum vector rank of each tensor, not a global M/N/K whitelist:
        # rank=1 exposes RHS-N; rank>=2 additionally exposes MxN and the full
        # LHS(M,K)/RHS(K,N) packed form.  The previously generated standalone
        # M/K/MK/NK forms do not exist in nncase and cannot describe its
        # coupled operand layouts.
        layouts = [
            ("n", (), (rhs_n,), (1,)),
        ]
        if self.max_axes > 1:
            layouts.extend((
                ("mn", (lhs_m,), (rhs_n,), (0, 1)),
                ("mkn", (lhs_m, lhs_k), (rhs_k, rhs_n), (0, 1)),
            ))
        results: list[VectorizeCandidate] = []
        for kind, lhs_axes, rhs_axes, out_axes in layouts:
            lhs_lanes = (lane,) * len(lhs_axes)
            rhs_lanes = (lane,) * len(rhs_axes)
            out_lanes = (lane,) * len(out_axes)
            lhs_pads = padding_for(lhs, lhs_axes, lhs_lanes) if lhs_axes else (0, 0)
            rhs_pads = padding_for(rhs, rhs_axes, rhs_lanes) if rhs_axes else (0, 0)
            output_pads = padding_for(node.type, out_axes, out_lanes)
            if lhs_pads is None or rhs_pads is None or output_pads is None:
                continue
            results.append(VectorizeCandidate(
                f"vectorization.matmul.{kind}",
                self.name,
                out_axes,
                out_lanes,
                {
                    "kind": kind,
                    "lhs_axes": list(lhs_axes), "rhs_axes": list(rhs_axes),
                    "output_axes": list(out_axes), "lhs_lanes": list(lhs_lanes),
                    "rhs_lanes": list(rhs_lanes), "output_lanes": list(out_lanes),
                    "vector_bytes": self.lane_bytes,
                },
                {
                    "lhs_padding": list(lhs_pads), "rhs_padding": list(rhs_pads),
                    "output_padding": list(output_pads), "egraph_equivalent": True,
                },
            ))
        return tuple(results)

    def rewrite(self, node: Node, module: IRModule, candidate: VectorizeCandidate) -> RewriteResult:
        parameters = candidate.parameters
        lhs_axes = tuple(parameters["lhs_axes"])
        rhs_axes = tuple(parameters["rhs_axes"])
        lhs_lanes = tuple(parameters["lhs_lanes"])
        rhs_lanes = tuple(parameters["rhs_lanes"])
        output_axes = tuple(parameters["output_axes"])
        output_lanes = tuple(parameters["output_lanes"])
        prefix: list[Node] = []
        lhs = module.node_map[node.inputs[0]]
        rhs = module.node_map[node.inputs[1]]
        if lhs_axes:
            helpers, lhs = prepare_packed_input(
                lhs, axes=lhs_axes, lanes=lhs_lanes, root_id=node.id, input_index=0,
                forced_pad=tuple(candidate.facts["lhs_padding"]),
            )
            prefix.extend(helpers)
        if rhs_axes:
            helpers, rhs = prepare_packed_input(
                rhs, axes=rhs_axes, lanes=rhs_lanes, root_id=node.id, input_index=1,
                forced_pad=tuple(candidate.facts["rhs_padding"]),
            )
            prefix.extend(helpers)
        attrs = {
            "lhs_axes": lhs_axes,
            "rhs_axes": rhs_axes,
            "output_axes": output_axes,
            "output_lanes": output_lanes,
            "transpose_a": bool(node.attrs.get("transpose_a", False)),
            "transpose_b": bool(node.attrs.get("transpose_b", False)),
        }
        compute_type = VectorizedMatMul.infer_type((lhs, rhs), attrs)
        compute = Node(
            f"{node.id}.vectorized.compute",
            "math.vectorized_matmul",
            (lhs.id, rhs.id),
            compute_type,
            attrs=attrs,
            metadata=internal_metadata(node.id, "compute"),
        )
        return finish_vector_result(
            node,
            compute,
            prefix,
            axes=output_axes,
            pads=tuple(candidate.facts["output_padding"]),
            candidate=candidate,
        )


__all__ = ["VectorizeMatMul"]
