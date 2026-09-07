# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import json

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.diagnostics import DumpFlags, DumpManager


def _module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported", metadata={"model": "dump-add"})
    value_type = fm.tensor_type("float32", [17])
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    output = builder.call("math.add", [lhs, rhs], value_type, id="output")
    builder.function("main", [lhs, rhs], [output])
    return builder.build(entry="main")


def test_schedule_and_codegen_flags_index_real_renderer_artifacts(tmp_path):
    module = Compiler().compile(_module()).module
    manager = DumpManager(tmp_path / "dumps", DumpFlags.SCHEDULE | DumpFlags.CODEGEN)
    write_artifact(
        module,
        tmp_path / "artifact",
        target="nvidia-sm90",
        emit_executable=True,
        dumper=manager.root,
    )

    manifest = json.loads((tmp_path / "dumps" / "artifacts.json").read_text(encoding="utf-8"))
    indexed = {(value["category"], value["kind"]): value for value in manifest["artifacts"]}
    assert ("schedule", "tir.schedule/v1") in indexed
    assert ("schedule", "tir.function-schedule/v2") in indexed
    assert ("schedule", "tir.runtime-binding/v1") in indexed
    assert ("codegen", "triton.package-descriptor/v1") in indexed
    assert ("codegen", "triton.python-source/v1") in indexed
    schedules = tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "dumps" / "Schedule").glob("*.schedule.json")
    )
    schedule_data = next(
        value
        for value in schedules
        if value["dispatch"]["candidate"] == "tir.elementwise.add"
    )
    assert schedule_data["dispatch"]["candidate"] == "tir.elementwise.add"
    assert schedule_data["function"].startswith("kernel_elementwise_add")
    package = json.loads((tmp_path / "dumps" / "CodeGen" / "package.json").read_text(encoding="utf-8"))
    assert package["renderer_spec"] == "bufferized-tir"
    assert package["kind"] == "tir_call_graph/v1"


def test_schedule_dump_does_not_require_executable_codegen(tmp_path):
    module = Compiler().compile(_module()).module
    manager = DumpManager(tmp_path / "dumps", DumpFlags.SCHEDULE)

    write_artifact(
        module,
        tmp_path / "artifact",
        target="nvidia-sm90",
        emit_executable=False,
        dumper=manager.root,
    )

    call_schedule = json.loads(
        (tmp_path / "dumps" / "Schedule" / "main.call_schedule.json")
        .read_text(encoding="utf-8")
    )
    assert call_schedule["schema"] == "flagmega.triton-function-schedule/v2"
    assert call_schedule["regions"][0]["execution_domain"] == (
        "dense_local_shard"
    )
    runtime_binding = json.loads(
        (tmp_path / "dumps" / "Schedule" / "main.runtime_binding.json")
        .read_text(encoding="utf-8")
    )
    assert runtime_binding["signature"][:3] == ["lhs", "rhs", "output"]
    assert runtime_binding["signature"][3:] == ["workspace", "block_local_data"]
    assert [pool["storage"] for pool in runtime_binding["pools"]] == [
        "workspace",
        "block_local_data",
    ]
    assert not (tmp_path / "dumps" / "CodeGen").exists()


def test_schedule_flag_does_not_leak_codegen_source(tmp_path):
    module = Compiler().compile(_module()).module
    manager = DumpManager(tmp_path / "dumps", DumpFlags.SCHEDULE)
    write_artifact(
        module,
        tmp_path / "artifact",
        target="nvidia-sm90",
        emit_executable=True,
        dumper=manager.root,
    )

    assert (tmp_path / "dumps" / "Schedule").is_dir()
    assert not (tmp_path / "dumps" / "CodeGen").exists()
