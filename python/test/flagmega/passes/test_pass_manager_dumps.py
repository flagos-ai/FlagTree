# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.diagnostics import DumpFlags, DumpManager
from triton.flagmega.passes import FunctionalPass, PassManager


def _two_function_module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("float32", (8,))
    main_input = builder.var("main_input", value_type, id="main_input")
    main_output = builder.call("math.silu", (main_input,), value_type, id="main_output")
    helper_input = builder.var("helper_input", value_type, id="helper_input")
    helper_output = builder.call("math.silu", (helper_input,), value_type, id="helper_output")
    builder.function("main", (main_input,), (main_output,))
    builder.function("helper", (helper_input,), (helper_output,))
    return builder.build(entry="main")


def _mark_transformed(module):
    return replace(module, metadata={**dict(module.metadata), "transformed": True})


def test_pass_manager_owns_before_after_dumps_for_every_function(tmp_path):
    dumper = DumpManager(tmp_path, DumpFlags.PASS_IR).root.create_sub_dumper("LargeStage")
    result = PassManager("LargeStage", dumper=dumper).add(
        FunctionalPass("MarkTransformed", _mark_transformed)
    ).run(_two_function_module())

    assert result.before_dump == str(tmp_path / "LargeStage" / "Before")
    assert result.after_dump == str(tmp_path / "LargeStage" / "After")
    assert {item.function for item in result.before_functions} == {"main", "helper"}
    assert {item.function for item in result.after_functions} == {"main", "helper"}
    assert result.module.metadata["transformed"] is True
    assert result.before_functions[0].semantic_hash != result.after_functions[0].semantic_hash
    for boundary in ("Before", "After"):
        for function in ("main", "helper"):
            assert (tmp_path / "LargeStage" / boundary / f"{function}.py").is_file()
            assert (tmp_path / "LargeStage" / boundary / f"{function}.il").is_file()


def test_pass_manager_dump_flag_still_controls_large_stage_boundaries(tmp_path):
    dumper = DumpManager(tmp_path, DumpFlags.EGRAPH_COST).root.create_sub_dumper("LargeStage")
    result = PassManager("LargeStage", dumper=dumper).add(
        FunctionalPass("MarkTransformed", _mark_transformed)
    ).run(_two_function_module())

    assert result.before_dump is None
    assert result.after_dump is None
    assert not (tmp_path / "LargeStage" / "Before").exists()
    assert not (tmp_path / "LargeStage" / "After").exists()
