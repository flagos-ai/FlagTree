# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm


def test_prepare_builders_and_patterns_survive_python_round_trip(tmp_path):
    class Graph(fm.Module):
        def forward(self):
            query = self.input("query", fm.tensor_type("bfloat16", (3, 4, 8)))
            a = self.input("a", fm.tensor_type("bfloat16", (3, 8)))
            b = self.input("b", a.type)
            a_log = self.input("a_log", fm.tensor_type("float32", (8,)))
            bias = self.input("bias", fm.tensor_type("bfloat16", (8,)))
            norm = fm.F.nn.l2_normalization(query, epsilon=1e-6, epsilon_mode="add",
                                           division_mode="reciprocal_multiply", name="norm")
            gates = fm.F.nn.delta_rule_gates(a, b, a_log, bias, alpha_exp_mode="accurate", name="gates")
            self.function("main", (query, a, b, a_log, bias), (norm, gates))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    path = tmp_path / "prepare.py"
    fm.emit_module(module, path)
    restored = fm.load_module(path)
    assert restored.semantic_hash == module.semantic_hash
    assert "F.nn.l2_normalization" in path.read_text() and "F.nn.delta_rule_gates" in path.read_text()
    assert pm.try_match_root(restored.node_map["norm"], pm.F.nn.is_l2_normalization(epsilon_mode="add"), restored) is not None
    assert pm.try_match_root(restored.node_map["gates"], pm.F.nn.is_delta_rule_gates(alpha_exp_mode="accurate"), restored) is not None
