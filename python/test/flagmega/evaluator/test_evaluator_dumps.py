# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import hashlib
import json

import torch

from triton.flagmega import ir as fm
from triton.flagmega.diagnostics import DumpFlags, DumpManager, DumpScope
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


def _module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("bfloat16", [2])
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    output = builder.call("math.add", (lhs, rhs), value_type, id="output")
    builder.function("main", (lhs, rhs), (output,))
    return fm.verify_module(builder.build(entry="main"))


def test_evaluator_flag_dumps_arguments_results_and_exact_tensor_bytes(tmp_path):
    module = _module()
    lhs = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    rhs = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
    manager = DumpManager(tmp_path, DumpFlags.EVALUATOR)

    with DumpScope(manager.root):
        output = TorchEvaluator(DictWeightResolver({})).run(
            module, {"lhs": lhs, "rhs": rhs}
        )[0]

    call = tmp_path / "Evaluate" / "Run0000" / "0000_output"
    arguments = json.loads((call / "arguments.json").read_text(encoding="utf-8"))
    result = json.loads((call / "result.json").read_text(encoding="utf-8"))
    assert [value["name"] for value in arguments["arguments"]] == ["lhs", "rhs"]
    assert result["result"]["shape"] == [2]
    assert result["result"]["dtype"] == "bfloat16"
    expected = output.contiguous().view(torch.uint8).numpy().tobytes()
    assert (call / "result.bin").read_bytes() == expected

    manifest = json.loads((tmp_path / "artifacts.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "flagmega.diagnostic-artifacts/v1"
    indexed = {value["relative_path"]: value for value in manifest["artifacts"]}
    record = indexed["Evaluate/Run0000/0000_output/result.bin"]
    assert record["source_semantic_hash"] == module.semantic_hash
    assert record["sha256"] == hashlib.sha256(expected).hexdigest()


def test_evaluator_flag_disabled_creates_no_value_artifacts(tmp_path):
    module = _module()
    with DumpScope(DumpManager(tmp_path, DumpFlags.PASS_IR).root):
        TorchEvaluator(DictWeightResolver({})).run(
            module,
            {
                "lhs": torch.ones(2, dtype=torch.bfloat16),
                "rhs": torch.ones(2, dtype=torch.bfloat16),
            },
        )

    assert not (tmp_path / "Evaluate").exists()
    assert not (tmp_path / "artifacts.json").exists()
