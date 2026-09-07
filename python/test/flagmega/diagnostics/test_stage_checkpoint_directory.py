# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import json
import time
from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.diagnostics import DiagnosticSession, DumpFlags


def _two_function_module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("float32", [2])
    layer_arg = builder.var("layer_arg", value_type, id="layer_arg")
    layer_out = builder.call("math.silu", (layer_arg,), value_type, id="layer_out")
    builder.function("decode_layer", (layer_arg,), (layer_out,))
    main_arg = builder.var("main_arg", value_type, id="main_arg")
    main_out = builder.call(
        "builtin.call", (main_arg,), value_type, id="main_out",
        attrs={"callee": "decode_layer"},
    )
    builder.function("main", (main_arg,), (main_out,))
    return fm.verify_module(builder.build(entry="main"))


def test_stage_report_points_to_mergeable_function_directory(tmp_path):
    source = _two_function_module()
    result = replace(source, metadata={"iteration": 1})
    session = DiagnosticSession(tmp_path, dump_flags=DumpFlags.COMPILE)
    report = session.record("unit-stage", source, result, time.perf_counter())

    checkpoint = tmp_path / "Compile" / "00_imported" / "After"
    assert report.checkpoint == str(checkpoint)
    assert fm.load_module(report.checkpoint) == result
    assert {path.stem for path in checkpoint.glob("*.py")} == {"main", "decode_layer"}

    manifest = json.loads((tmp_path / "stages.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "flagmega.stage-reports/v1"
    assert manifest["stages"][0]["checkpoint"] == str(checkpoint)
