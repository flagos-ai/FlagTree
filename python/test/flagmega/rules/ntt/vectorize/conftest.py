# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Small rule-test harness mirroring nncase's TransformTestBase usage."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm


@pytest.fixture
def make_op_module():
    def make(op, input_types, *, attrs=None, output_id="root"):
        builder = fm.IRBuilder(dialect="high_level", stage="imported")
        inputs = tuple(
            builder.var(f"arg{index}", value_type, id=f"arg{index}")
            for index, value_type in enumerate(input_types)
        )
        definition = fm.get_definition(op)
        prepared = definition.prepare(inputs, attrs or {})
        output = builder.call(
            op,
            prepared.inputs,
            prepared.result_type,
            id=output_id,
            effect=prepared.effect,
            attrs=definition.ir_attrs(prepared.attrs),
        )
        builder.function("main", inputs, (output,))
        return fm.verify_module(builder.build(entry="main"))

    return make


@pytest.fixture
def materialize_rewrite():
    def materialize(module, result, *, root_id="root"):
        root_index = next(index for index, node in enumerate(module.nodes) if node.id == root_id)
        assert result.replacement.id == root_id
        rewritten = replace(
            module,
            nodes=(
                *module.nodes[:root_index],
                *result.prefix_nodes,
                result.replacement,
                *module.nodes[root_index + 1:],
            ),
        )
        return fm.verify_module(rewritten)

    return materialize


@pytest.fixture
def rewrite_candidate(materialize_rewrite):
    def rewrite(module, rule, candidate_id, *, root_id="root"):
        root = module.node_map[root_id]
        candidate = next(value for value in rule.candidates(root, module) if value.id == candidate_id)
        result = rule.rewrite(root, module, candidate)
        return candidate, result, materialize_rewrite(module, result, root_id=root_id)

    return rewrite
