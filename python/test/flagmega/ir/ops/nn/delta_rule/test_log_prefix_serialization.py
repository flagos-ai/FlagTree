# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm, pattern_match as pm


def test_rounding_attributes_survive_python_edit_resume(tmp_path):

    class Graph(fm.Module):

        def forward(self):
            alpha = self.input("alpha", fm.tensor_type("float32", (65, 8)))
            result = fm.F.nn.delta_rule_log_prefix(alpha, block_size=32, scan_group_size=8, epsilon=0.25,
                                                   log2_mode="accurate", name="prefix")
            self.function("main", (alpha, ), (result, ))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    fm.emit_module(module, tmp_path / "prefix.py")
    loaded = fm.load_module(tmp_path / "prefix.py")
    assert module.semantic_hash == loaded.semantic_hash
    assert "F.nn.delta_rule_log_prefix" in (tmp_path / "prefix.py").read_text()
    pattern = pm.F.nn.is_delta_rule_log_prefix(block_size=32, scan_group_size=8, epsilon=0.25, log2_mode="accurate")
    assert pm.try_match_root(loaded.node_map["prefix"], pattern, loaded) is not None
