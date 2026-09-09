# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm


def test_handwritten_builder_pattern_and_python_resume(tmp_path):

    class Graph(fm.Module):

        def forward(self):
            key = self.input("key", fm.tensor_type("bfloat16", (17, 2, 16)))
            beta = self.input("beta", fm.tensor_type("float32", (17, 4)))
            coefficients = fm.F.nn.delta_rule_coefficients(key, beta, block_size=16, name="coefficients")
            self.function("main", (key, beta), (coefficients, ))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    fm.emit_module(module, tmp_path / "module.py")
    loaded = fm.load_module(tmp_path / "module.py")
    assert loaded.semantic_hash == module.semantic_hash
    assert "F.nn.delta_rule_coefficients" in (tmp_path / "module.py").read_text()
    pattern = pm.F.nn.is_delta_rule_coefficients(block_size=16)
    assert pm.try_match_root(loaded.node_map["coefficients"], pattern, loaded) is not None
