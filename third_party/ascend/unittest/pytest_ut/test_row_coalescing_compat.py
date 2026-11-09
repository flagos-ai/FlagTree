# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
"""IR contracts for the pure-SIMT Row graph-optimization rule.

Row is selected by graph-rule bit 8 and ``compile_mode="simt_only"``.  The
test enters through the public graph-optimization binding, which is the same
pass scheduled by ``make_ttir()``.
"""

import pytest

from triton._C.libtriton import ascend, ir
from triton._C.libtriton.ascend import ir as ascend_ir

if not hasattr(ascend.passes.ttir, "add_graph_optimize"):
    pytest.skip(
        "requires the TritonAscend build containing graph optimization",
        allow_module_level=True,
    )


def _row_module(
    name,
    width,
    *,
    axis="x",
    reads_num_programs=False,
    pid_derived_outside_work=False,
    bound_before_pid=False,
    multiply_pid_in_work=False,
    has_direct_call=False,
    body="copy",
):
    num_programs = (f"    %num_programs = tt.get_num_programs {axis} : i32\n" if reads_num_programs else "")
    pre_load = ""
    pre_pid = ""
    helper = ""
    if has_direct_call:
        # Keep the call outside the work region so this test exercises the
        # V1 entry-closure gate rather than merely Row's liftable-op filter.
        pre_pid = "    tt.call @row_helper() : () -> ()\n"
        helper = """  tt.func private @row_helper() {
    tt.return
  }
"""
    pid_prefix = ""
    pid_in_work = "%pid"
    if pid_derived_outside_work:
        pid_prefix = """    %one = arith.constant 1 : i32
    %pid_base = arith.addi %pid, %one : i32
"""
        pid_in_work = "%pid_base"
    work_pid_prefix = ""
    if multiply_pid_in_work:
        work_pid_prefix = f"""    %block = arith.constant {width} : i32
    %pid_scaled = arith.muli %pid, %block : i32
"""
        pid_in_work = "%pid_scaled"
    bound_prefix = ""
    count_definition = "    %count = tt.load %valid : !tt.ptr<i32>\n"
    guard_count = "%count"
    if bound_before_pid:
        bound_prefix = "    %bound = arith.constant 32768 : i32\n"
        count_definition = ""
        guard_count = "%bound"
    load = f"    %value = tt.load %src_ptr : tensor<{width}x!tt.ptr<f32>>"
    post_load = ""
    store_value = "%value"
    store_mask = ""

    if body == "masked_copy":
        pre_load = f"""    %count_splat = tt.splat %count : i32 -> tensor<{width}xi32>
    %input_mask = arith.cmpi slt, %offsets, %count_splat : tensor<{width}xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<{width}xf32>
"""
        load = f"    %value = tt.load %src_ptr, %input_mask, %zero : tensor<{width}x!tt.ptr<f32>>"
        store_mask = ", %input_mask"
    elif body == "reduce":
        post_load = f"""    %reduced = "tt.reduce"(%value) <{{axis = 0 : i32}}> ({{
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }}) : (tensor<{width}xf32>) -> f32
    %out = tt.splat %reduced : f32 -> tensor<{width}xf32>
"""
        store_value = "%out"
    elif body == "scan":
        post_load = f"""    %out = "tt.scan"(%value) <{{axis = 0 : i32, reverse = false}}> ({{
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.scan.return %sum : f32
    }}) : (tensor<{width}xf32>) -> tensor<{width}xf32>
"""
        store_value = "%out"
    elif body == "for":
        post_load = f"""    %for_lb = arith.constant 0 : index
    %for_step = arith.constant 1 : index
    %for_ub = arith.constant 2 : index
    %out = scf.for %i = %for_lb to %for_ub step %for_step iter_args(%acc = %value) -> (tensor<{width}xf32>) {{
      %one = arith.constant dense<1.000000e+00> : tensor<{width}xf32>
      %next = arith.addf %acc, %one : tensor<{width}xf32>
      scf.yield %next : tensor<{width}xf32>
    }}
"""
        store_value = "%out"
    elif body == "for_pid_lower_bound":
        post_load = f"""    %pid_index = arith.index_cast %pid : i32 to index
    %for_step = arith.constant 1 : index
    %for_ub = arith.constant 2 : index
    %out = scf.for %i = %pid_index to %for_ub step %for_step iter_args(%acc = %value) -> (tensor<{width}xf32>) {{
      scf.yield %acc : tensor<{width}xf32>
    }}
"""
        store_value = "%out"
    elif body == "non_whitelisted_y_num_programs":
        post_load = "    %not_whitelisted = tt.get_num_programs y : i32\n"
    elif body != "copy":
        raise ValueError(f"unsupported row test body: {body}")

    return f"""
module {{
  tt.func public @{name}(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %valid: !tt.ptr<i32>) {{
{pre_pid}{bound_prefix}    %pid = tt.get_program_id {axis} : i32
{num_programs}{pid_prefix}{count_definition}    %past_end = arith.cmpi sge, %pid, {guard_count} : i32
    cf.cond_br %past_end, ^bb_return, ^bb_work
  ^bb_return:
    tt.return
  ^bb_work:
{work_pid_prefix}    %pid_splat = tt.splat {pid_in_work} : i32 -> tensor<{width}xi32>
    %range = tt.make_range {{end = {width} : i32, start = 0 : i32}} : tensor<{width}xi32>
    %offsets = arith.addi %pid_splat, %range : tensor<{width}xi32>
    %src_splat = tt.splat %src : !tt.ptr<f32> -> tensor<{width}x!tt.ptr<f32>>
    %src_ptr = tt.addptr %src_splat, %offsets : tensor<{width}x!tt.ptr<f32>>, tensor<{width}xi32>
{pre_load}{load}
{post_load}    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<{width}x!tt.ptr<f32>>
    %dst_ptr = tt.addptr %dst_splat, %offsets : tensor<{width}x!tt.ptr<f32>>, tensor<{width}xi32>
    tt.store %dst_ptr, {store_value}{store_mask} : tensor<{width}x!tt.ptr<f32>>
    tt.return
  }}
{helper}}}
"""


def _run_row(text, tmp_path, *, compile_mode="simt_only", rule_mask=8):
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    path = tmp_path / "row_coalescing_compat.mlir"
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), context)
    pm = ir.pass_manager(context)
    ascend.passes.ttir.add_graph_optimize(
        pm,
        rule_mask=rule_mask,
        compile_mode=compile_mode,
    )
    pm.run(module, "row-coalescing-graph-rule-test")
    return str(module)


def _assert_row_hit(text, *, factor=8, axis=0):
    assert f"hacc.coalesce_factor = {factor} : i32" in text
    assert f"hacc.coalesce_axis = {axis} : i32" in text
    assert "hacc.coalesce_grid_ceil_div = 1 : i32" in text


def _assert_row_bailout(text):
    assert "hacc.coalesce_factor" not in text
    assert "hacc.coalesce_axis" not in text
    assert "hacc.coalesce_grid_ceil_div" not in text
    assert "cf.cond_br" in text


@pytest.mark.parametrize(
    ("width", "factor"),
    ((16, 8), (32, 4), (1024, 2)),
)
def test_row_coalescing_graph_rule_preserves_h_selection(width, factor, tmp_path):
    text = _run_row(_row_module(f"row_h{factor}", width), tmp_path)

    assert f"hacc.coalesce_factor = {factor} : i32" in text
    assert "hacc.coalesce_axis = 0 : i32" in text
    assert "hacc.coalesce_grid_ceil_div = 1 : i32" in text
    assert f"tensor<{factor}x{width}xf32>" in text
    assert "cf.cond_br" not in text


def test_row_coalescing_scaffold_follows_pid_when_bound_is_hoisted(tmp_path):
    text = _run_row(
        _row_module(
            "row_bound_before_pid",
            16,
            bound_before_pid=True,
            multiply_pid_in_work=True,
        ),
        tmp_path,
    )

    _assert_row_hit(text)
    assert text.index("arith.constant 32768") < text.index("tt.get_program_id")
    assert text.index("tt.get_program_id") < text.index("tt.make_range")
    assert "cf.cond_br" not in text


def test_row_coalescing_rejects_visible_num_programs_on_row_axis(tmp_path):
    text = _run_row(
        _row_module("row_reads_num_programs", 16, reads_num_programs=True),
        tmp_path,
    )

    _assert_row_bailout(text)


def test_row_coalescing_adds_tail_mask_to_masked_load_and_store(tmp_path):
    text = _run_row(_row_module("row_tail_mask", 16, body="masked_copy"), tmp_path)

    _assert_row_hit(text)
    assert "cf.cond_br" not in text
    assert "arith.cmpi slt" in text
    assert text.count("arith.andi") >= 2
    assert text.count("tensor<8x16xi1>") >= 4

    lifted_loads = [line for line in text.splitlines() if "tt.load" in line and "tensor<8x16x!tt.ptr<f32>>" in line]
    lifted_stores = [line for line in text.splitlines() if "tt.store" in line and "tensor<8x16x!tt.ptr<f32>>" in line]
    assert len(lifted_loads) == 1 and lifted_loads[0].count(",") >= 2
    assert len(lifted_stores) == 1 and lifted_stores[0].count(",") >= 2


def test_row_coalescing_lifts_reduce_along_original_row_axis(tmp_path):
    text = _run_row(_row_module("row_reduce", 16, body="reduce"), tmp_path)

    _assert_row_hit(text)
    assert '"tt.reduce"' in text
    assert "<{axis = 1 : i32}>" in text
    assert "(tensor<8x16xf32>) -> tensor<8xf32>" in text
    assert "tensor<8x1xf32> -> tensor<8x16xf32>" in text


def test_row_coalescing_lifts_scan_along_original_row_axis(tmp_path):
    text = _run_row(_row_module("row_scan", 16, body="scan"), tmp_path)

    _assert_row_hit(text)
    assert '"tt.scan"' in text
    assert "<{axis = 1 : i32, reverse = false}>" in text
    assert "(tensor<8x16xf32>) -> tensor<8x16xf32>" in text


def test_row_coalescing_clones_scf_for_with_lifted_iter_args(tmp_path):
    text = _run_row(_row_module("row_for", 16, body="for"), tmp_path)

    _assert_row_hit(text)
    assert "scf.for" in text
    assert "iter_args" in text
    assert "-> (tensor<8x16xf32>)" in text
    assert "scf.yield" in text


def test_row_coalescing_declines_late_sandbox_failure_without_committing(tmp_path):
    text = _run_row(
        _row_module("row_late_sandbox_decline", 16, body="for_pid_lower_bound"),
        tmp_path,
    )

    _assert_row_bailout(text)
    assert "arith.index_cast" in text
    assert "scf.for" in text


def test_row_coalescing_rejects_non_whitelisted_y_num_programs(tmp_path):
    text = _run_row(
        _row_module(
            "row_non_whitelisted_y_num_programs",
            16,
            body="non_whitelisted_y_num_programs",
        ),
        tmp_path,
    )

    _assert_row_bailout(text)
    assert "tt.get_num_programs y" in text


def test_row_coalescing_preserves_nonzero_row_axis(tmp_path):
    text = _run_row(_row_module("row_axis_y", 16, axis="y"), tmp_path)

    _assert_row_hit(text, axis=1)
    assert "tt.get_program_id y" in text


def test_row_coalescing_requires_force_simt_only_and_rule_bit(tmp_path):
    source = _row_module("row_force_gate", 16)

    mode_disabled = _run_row(source, tmp_path, compile_mode="simd")
    mask_disabled = _run_row(source, tmp_path, rule_mask=7)

    _assert_row_bailout(mode_disabled)
    _assert_row_bailout(mask_disabled)


def test_row_coalescing_rejects_pid_value_defined_before_work_block(tmp_path):
    text = _run_row(
        _row_module("row_pid_derived_before_work", 16, pid_derived_outside_work=True),
        tmp_path,
    )

    _assert_row_bailout(text)
    # The Python MLIR printer may renumber SSA values, so check the preserved
    # scalar add structurally rather than relying on its textual value name.
    assert text.count("arith.addi") >= 2


def test_row_coalescing_rejects_direct_call_even_outside_work_region(tmp_path):
    text = _run_row(_row_module("row_with_call", 16, has_direct_call=True), tmp_path)

    _assert_row_bailout(text)
    assert "tt.call @row_helper()" in text


def test_row_coalescing_requires_one_public_entry_but_ignores_unused_private_funcs(tmp_path, ):
    source = _row_module("row_one_public", 16)
    multiple_public = source.replace(
        "\n}\n",
        """
  tt.func public @another_entry() {
    tt.return
  }
}
""",
        1,
    )
    with_unused_private = source.replace(
        "\n}\n",
        """
  tt.func private @unused_helper() {
    tt.return
  }
}
""",
        1,
    )

    _assert_row_bailout(_run_row(multiple_public, tmp_path))
    _assert_row_hit(_run_row(with_unused_private, tmp_path))
