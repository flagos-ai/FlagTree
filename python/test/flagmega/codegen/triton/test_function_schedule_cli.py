# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import json
from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.cli import main
from triton.flagmega.passes.tir import (
    bind_prim_function_buffers,
    materialize_kernel_prim_functions,
)


def _bufferized_schedule_module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value_type = fm.tensor_type("bfloat16", (1, 16))
    source = builder.var("source", value_type, id="source")
    output = builder.call(
        "tir.kernel",
        (source,),
        value_type,
        id="output",
        attrs={
            "semantic_op": "math.silu",
            "candidate": "tir.silu.local",
            "parameters": {"family": "test", "variant": "local"},
            "facts": {},
            "semantic_attrs": {},
        },
    )
    builder.function("main", (source,), (output,))
    selected = materialize_kernel_prim_functions(builder.build(entry="main"))
    plan = fm.make_buffer_plan(selected)
    bound = bind_prim_function_buffers(selected)
    return fm.verify_module(replace(
        bound,
        stage="bufferized_tir",
        dialect="bufferized_tir",
        metadata={**bound.metadata, "buffer_plan": plan.to_data()},
    ))


def test_cli_exposes_editable_ir_physical_schedule(tmp_path, capsys):
    checkpoint = fm.emit_module(
        _bufferized_schedule_module(),
        tmp_path / "bufferized.py",
    )

    assert main(["schedule", str(checkpoint), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["ok"] is True
    assert result["schedule"]["schema"] == (
        "flagmega.triton-function-schedule/v2"
    )
    assert result["schedule"]["physical_strategy"] == "entry_schedule"
    assert result["runtime_binding"]["signature"] == [
        "source",
        "output",
    ]
    assert [
        region["kind"] for region in result["schedule"]["regions"]
    ] == ["local_segment"]
