# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Structured payload identity includes the complete field path."""

import json

import torch

from triton.flagmega import ir as fm
from triton.flagmega.diagnostics import DumpFlags, DumpManager, DumpScope
from triton.flagmega.evaluator.dump import EvaluatorDumpWriter


def test_mapping_and_nested_field_paths_do_not_overwrite_payloads(tmp_path):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    node = builder.var("value", fm.tensor_type("float32", (1,)), id="value")
    builder.function("main", (node,), (node,))
    module = builder.build(entry="main")
    # This tests the dump writer's serialization protocol, independently of
    # evaluator op type checking. The writer explicitly supports structured
    # state payloads as well as tensors.
    values = {
        "a_b": torch.tensor([1.]),
        "a": {"b": torch.tensor([2.])},
        "a/b": torch.tensor([3.]),
        "a?b": torch.tensor([4.]),
        "a%2Fb": torch.tensor([5.]),
        "tuple_0": torch.tensor([6.]),
        "tuple": (torch.tensor([7.]),),
    }
    manager = DumpManager(tmp_path, DumpFlags.EVALUATOR)
    with DumpScope(manager.root):
        writer = EvaluatorDumpWriter(module)
        token = writer.before("main", node, ())
        writer.after(token, "main", node, values)
    directory = tmp_path / "Evaluate" / "Run0000" / "0000_value"
    result = json.loads((directory / "result.json").read_text())["result"]
    payloads = []

    def check(value, expected):
        if isinstance(expected, torch.Tensor):
            payloads.append(value["payload"])
            assert (directory / value["payload"]).read_bytes() == expected.view(torch.uint8).numpy().tobytes()
        elif isinstance(expected, dict):
            for key, child in expected.items():
                check(value["fields"][key], child)
        else:
            for child, reference in zip(value["fields"], expected, strict=True):
                check(child, reference)

    check(result, values)
    assert len(payloads) == len(set(payloads)) == 7
    indexed = [value for value in manager.artifacts if value.kind == "tensor-bytes"]
    assert len(indexed) == 7
