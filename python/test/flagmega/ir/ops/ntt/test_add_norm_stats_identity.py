# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm


def test_add_norm_stats_has_a_producer_independent_public_identity(tmp_path):
    class Graph(fm.Module):
        def forward(self):
            value = self.input("value", fm.tensor_type("float32", (2, 32)))
            addend = self.input("addend", value.type)
            combined = fm.F.ntt.add_norm_stats(value, addend, axis=-1, use_mean=False, name="residual_stats")
            self.function("main", (value, addend), (combined,))

    module = fm.verify_module(Graph(dialect="high_level", stage="imported", entry="main").build())
    node = module.node_map["residual_stats"]
    assert node.op == "ntt.add_norm_stats"
    assert fm.get_definition(node.op).__name__ == "AddNormStats"
    assert "NTT.AddNormStats(" in fm.text_source(module)
    checkpoint = fm.emit_module(module, tmp_path / "add_norm_stats.py")
    assert fm.load_module(checkpoint) == module
    assert "matmul_norm_stats_combine" not in checkpoint.read_text()


def test_obsolete_matmul_combine_identity_is_not_registered():
    with pytest.raises(KeyError, match="No FlagMega op definition is registered"):
        fm.get_definition("ntt.matmul_norm_stats_combine")
    assert not hasattr(fm.F.ntt, "matmul_norm_stats_combine")
