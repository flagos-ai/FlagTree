# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.rules import RewriteRedirect
from triton.flagmega.rules.ntt.vectorize.propagation import fold_boundary_rules


def _rule(name):
    return next(rule for rule in fold_boundary_rules() if rule.name == name)


def test_fold_unpack_pack_redirects_to_scalar_source():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            source = self.input("source", fm.tensor_type("bfloat16", (2, 16)), id="source")
            packed = fm.F.tensors.pack(source, (8,), axes=(1,), name="packed")
            root = fm.F.tensors.unpack(packed, axes=(1,), name="root")
            self.function("main", (source,), (root,))

    module = Graph().build()
    result = _rule("FoldUnpackPack").apply(module.node_map["root"], module)
    assert result == RewriteRedirect("source")


def test_fold_pack_unpack_redirects_to_vector_source():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            source = self.input(
                "source", fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 2)), id="source",
            )
            unpacked = fm.F.tensors.unpack(source, axes=(1,), name="unpacked")
            root = fm.F.tensors.pack(unpacked, (8,), axes=(1,), name="root")
            self.function("main", (source,), (root,))

    module = Graph().build()
    result = _rule("FoldPackUnpack").apply(module.node_map["root"], module)
    assert result == RewriteRedirect("source")


def test_fold_pack_unpack_rejects_lane_mismatch():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            source = self.input(
                "source", fm.tensor_type(fm.vector_type("bfloat16", (2, 4)), (4,)), id="source",
            )
            unpacked = fm.F.tensors.unpack(source, axes=(0,), name="unpacked")
            root = fm.F.tensors.pack(unpacked, (4,), axes=(0,), name="root")
            self.function("main", (source,), (root,))

    module = Graph().build()
    assert _rule("FoldPackUnpack").apply(module.node_map["root"], module) is None


def test_fold_unpack_pack_preserves_vectorization_semantic_root():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            source = self.input("source", fm.tensor_type("bfloat16", (16,)), id="source")
            packed = fm.F.tensors.pack(source, (8,), axes=(0,), name="packed")
            root = fm.F.tensors.unpack(
                packed, axes=(0,), name="root", metadata={"vectorized_from": "math.silu"},
            )
            self.function("main", (source,), (root,))

    module = Graph().build()
    assert _rule("FoldUnpackPack").apply(module.node_map["root"], module) is None


def test_fold_rules_accept_singular_axis_spelling():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            source = self.input("source", fm.tensor_type("bfloat16", (16,)), id="source")
            packed = fm.F.tensors.pack(source, (8,), axis=0, name="packed")
            root = fm.F.tensors.unpack(packed, axis=0, name="root")
            self.function("main", (source,), (root,))

    module = Graph().build()
    assert _rule("FoldUnpackPack").apply(module.node_map["root"], module) == RewriteRedirect("source")


def test_fold_rules_reject_non_boundary_roots():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            source = self.input("source", fm.tensor_type("bfloat16", (16,)), id="source")
            root = fm.F.math.silu(source, name="root")
            self.function("main", (source,), (root,))

    module = Graph().build()
    assert _rule("FoldPackUnpack").apply(module.node_map["root"], module) is None
    assert _rule("FoldUnpackPack").apply(module.node_map["root"], module) is None
