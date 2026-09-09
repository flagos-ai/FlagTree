# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm, pattern_match as pm


def test_named_state_layout_and_rounding_survive_python_resume(tmp_path):
    state = fm.RefType("cache", (("recurrent", fm.tensor_type(fm.VectorType(fm.DType.FLOAT32, (4, )), (1, 4, 8, 4))), ))

    class Graph(fm.Module):

        def forward(self):
            q = self.input("query", fm.tensor_type("bfloat16", (65, 2, 16)))
            k = self.input("key", q.type)
            v = self.input("value", fm.tensor_type("bfloat16", (65, 4, 8)))
            c = self.input("coefficients", fm.tensor_type("bfloat16", (2, 4, 64, 64)))
            p = self.input("prefix", fm.tensor_type("float32", (2, 4, 64)))
            s = self.input("state", state)
            output = fm.F.nn.delta_rule_block_update(q, k, v, c, p, s, scale=0.125, state_field="recurrent",
                                                     state_layout=("layer", "head", "value", "key"),
                                                     state_vector_axes=("key", ), name="update")
            self.function("main", (q, k, v, c, p, s), (output, ))

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    path = tmp_path / "block.py"
    fm.emit_module(module, path)
    loaded = fm.load_module(path)
    assert loaded.semantic_hash == module.semantic_hash
    assert "F.nn.delta_rule_block_update" in path.read_text()
    node = loaded.node_map["update"]
    assert node.type.fields[1] == state
    assert node.effect == fm.effect("read_write", "delta_rule_state")
    pattern = pm.F.nn.is_delta_rule_block_update(scale=0.125, state_field="recurrent",
                                                 state_layout=("layer", "head", "value", "key"),
                                                 state_vector_axes=("key", ))
    assert pm.try_match_root(node, pattern, loaded) is not None
