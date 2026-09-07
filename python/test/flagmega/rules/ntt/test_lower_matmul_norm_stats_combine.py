# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import form_matmul_norm_stats_combine
from triton.flagmega.passes.norm_stats import lower_matmul_norm_stats_combine


class _Graph(fm.Module):
    def __init__(self, *, shared_projection=False):
        super().__init__(dialect="ntt", stage="norm_bindings_finalized", entry="main")
        self.shared_projection = shared_projection

    def forward(self):
        lhs = self.input("lhs", fm.tensor_type("float32", (2, 8)), id="lhs")
        rhs = self.input("rhs", fm.tensor_type("float32", (4, 8)), id="rhs")
        residual = self.input(
            "residual", fm.tensor_type("float32", (2, 4)), id="residual")
        projection = fm.F.math.matmul(
            lhs, rhs, transpose_b=True, name="projection")
        value = fm.F.math.add(projection, residual, name="value")
        stats = fm.F.nn.norm_stats(value, axis=-1, use_mean=False, name="stats")
        outputs = [value, stats]
        if self.shared_projection:
            outputs.append(projection)
        self.function("main", (lhs, rhs, residual), outputs)


def test_keeps_dense_matmul_combine_out_of_packed_kernel_rule():
    original = _Graph().build()
    formed = form_matmul_norm_stats_combine(original)
    rewritten = lower_matmul_norm_stats_combine(formed)

    assert rewritten == formed
    assert rewritten.node_map["projection"].op == "math.matmul"
    assert any(node.op == "ntt.matmul_norm_stats_combine" for node in rewritten.nodes)
    assert not any(node.op == "ntt.matmul_norm_stats" for node in rewritten.nodes)

    feeds = {
        "lhs": torch.randn(2, 8),
        "rhs": torch.randn(4, 8),
        "residual": torch.randn(2, 4),
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))
    expected = evaluator.run(original, feeds)
    actual = evaluator.run(rewritten, feeds)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_keeps_combine_when_matmul_result_has_another_user():
    formed = form_matmul_norm_stats_combine(_Graph(shared_projection=True).build())
    rewritten = lower_matmul_norm_stats_combine(formed)

    assert "projection" in rewritten.node_map
    assert any(node.op == "ntt.matmul_norm_stats_combine" for node in rewritten.nodes)
    assert not any(node.op == "ntt.matmul_norm_stats" for node in rewritten.nodes)


class _PartialGraph(fm.Module):
    def __init__(self):
        super().__init__(
            dialect="ntt", stage="norm_bindings_finalized", entry="main"
        )

    def forward(self):
        placement = fm.Placement((2, 2), "ab", "bb")
        lhs = self.input(
            "lhs",
            fm.DistributedType(
                fm.tensor_type("bfloat16", (1, 8)),
                (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
                placement,
            ),
            id="lhs",
        )
        rhs = self.input(
            "rhs",
            fm.DistributedType(
                fm.tensor_type("bfloat16", (4, 8)),
                (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
                placement,
            ),
            id="rhs",
        )
        residual = self.input(
            "residual",
            fm.DistributedType(
                fm.tensor_type("bfloat16", (1, 4)),
                (fm.SBP.broadcast(), fm.SBP.broadcast()),
                placement,
            ),
            id="residual",
        )
        projection = fm.F.math.matmul(
            lhs, rhs, transpose_b=True, name="projection"
        )
        combined = fm.F.ntt.matmul_norm_stats_combine(
            projection,
            residual,
            axis=-1,
            use_mean=False,
            name="combined",
        )
        self.function("main", (lhs, rhs, residual), (combined,))


def test_keeps_partial_matmul_and_combine_as_two_tir_roles_like_nncase():
    original = _PartialGraph().build()

    rewritten = lower_matmul_norm_stats_combine(original)

    assert rewritten == original
    assert rewritten.node_map["projection"].type.partial.axes == (0, 1)
    assert rewritten.node_map["combined"].op == "ntt.matmul_norm_stats_combine"
    assert not any(node.op == "ntt.matmul_norm_stats" for node in rewritten.nodes)


class _PackedViewGraph(fm.Module):
    def __init__(self, *, shared_projection: bool = False, broadcast_view: bool = False):
        super().__init__(
            dialect="ntt", stage="norm_bindings_finalized", entry="main"
        )
        self.shared_projection = shared_projection
        self.broadcast_view = broadcast_view

    def forward(self):
        placement = fm.Placement((2, 2), "ab", "bb")
        lhs_type = fm.DistributedType(
            fm.tensor_type("bfloat16", (1, 64)),
            (fm.SBP.broadcast(), fm.SBP.broadcast()),
            placement,
        )
        rhs_type = fm.DistributedType(
            fm.tensor_type(
                fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)), (4, 4)
            ),
            (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
            placement,
        )
        output_type = fm.DistributedType(
            fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16, (8,)), (1, 4)),
            (
                fm.SBP.broadcast(),
                fm.SBP.broadcast() if self.broadcast_view else fm.SBP.split_contiguous((0, 1)),
            ),
            placement,
        )
        lhs = self.input("lhs", lhs_type, id="lhs")
        rhs = self.input("rhs", rhs_type, id="rhs")
        residual = self.input("residual", output_type, id="residual")
        none = fm.F.builtin.none(name="none")
        projection = fm.F.ntt.packed_matmul(
            lhs,
            rhs,
            none,
            none,
            output_data_type=fm.DType.BFLOAT16,
            name="projection",
            metadata={
                "selected_vectorization": "axes_0_1.variant_5",
                "selected_vector_axes": (0, 1),
                "selected_vector_lanes": (1, 8),
            },
        )
        materialized = (
            fm.F.distributed.sharded_view(
                projection, output_type, name="projection_view"
            ) if self.broadcast_view else projection
        )
        combined = fm.F.ntt.matmul_norm_stats_combine(
            materialized,
            residual,
            axis=1,
            use_mean=False,
            name="combined",
        )
        outputs = [combined]
        if self.shared_projection:
            outputs.append(projection)
        self.function("main", (lhs, rhs, residual), outputs)


def test_lowers_private_packed_matmul_with_matching_local_shard():
    original = _PackedViewGraph().build()

    rewritten = lower_matmul_norm_stats_combine(original)

    fused = rewritten.node_map["combined"]
    assert fused.op == "ntt.matmul_norm_stats"
    assert fused.inputs == ("lhs", "rhs", "residual")
    assert fused.attrs["rhs_layout"] == "k_major"
    assert fused.metadata["fused_matmul"] == "projection"
    assert fused.metadata["projection_adapters"] == ()
    assert fused.metadata["matmul_vectorization"] == {
        "selected_vectorization": "axes_0_1.variant_5",
        "selected_vector_axes": (0, 1),
        "selected_vector_lanes": (1, 8),
    }
    assert "projection" not in rewritten.node_map
    assert "projection_view" not in rewritten.node_map


def test_keeps_packed_matmul_adapter_chain_when_producer_is_shared():
    original = _PackedViewGraph(shared_projection=True).build()

    rewritten = lower_matmul_norm_stats_combine(original)

    assert rewritten == original
    assert rewritten.node_map["combined"].op == "ntt.matmul_norm_stats_combine"


def test_keeps_split_to_broadcast_publication_before_normalization():
    original = _PackedViewGraph(broadcast_view=True).build()

    rewritten = lower_matmul_norm_stats_combine(original)

    # Each owner computes only N/4. Publishing that canonical storage as
    # Broadcast requires all owners before a full-N normalization; a local
    # fused epilogue cannot swallow the inter-owner publication boundary.
    assert rewritten == original
    assert rewritten.node_map["projection_view"].op == "distributed.sharded_view"
    assert rewritten.node_map["combined"].op == "ntt.matmul_norm_stats_combine"
