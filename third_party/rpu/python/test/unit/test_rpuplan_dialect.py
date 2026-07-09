import re
from pathlib import Path

import pytest


def _load_rpuplan_context(monkeypatch):
    monkeypatch.setenv("TRITON_RPU_ACTIVE", "1")
    monkeypatch.setenv("RPU_ARCH", "rpu")
    monkeypatch.setenv("RPU_LOGICAL_LANES", "16")
    monkeypatch.setenv("TRITON_CODEGEN_BACKENDS", "rpu")
    monkeypatch.delenv("RPU_BOARD_LOAD_CONTIG_NVEC", raising=False)

    from triton._C.libtriton import ir, rpu

    context = ir.context()
    ir.load_dialects(context)
    rpu.load_dialects(context)
    return ir, context


def _parse_mlir(ir, context, tmp_path, text):
    path = tmp_path / "rpuplan_dialect.mlir"
    path.write_text(text.strip())
    return ir.parse_mlir_module(str(path), context)


def _emit_rpurc_from_plan_text(text, tmp_path, monkeypatch):
    ir, context = _load_rpuplan_context(monkeypatch)
    module = _parse_mlir(ir, context, tmp_path, text)

    from triton._C.libtriton import rpu

    return rpu._emit_rpurc_from_plan(module)


def _get_rpuplan_summary_from_text(text, tmp_path, monkeypatch):
    ir, context = _load_rpuplan_context(monkeypatch)
    module = _parse_mlir(ir, context, tmp_path, text)

    from triton._C.libtriton import rpu

    return rpu._get_rpuplan_kernel_summary(module)


def _repo_root():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "third_party" / "rpu" / "backend" / "compiler.py").exists():
            return parent
    return here.parents[5]


def _source_between(source, start_marker, end_marker):
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def _kernel_plan_module(pattern, shape, args, signature_params, source_name):
    return f"""
module {{
  tt.func public @{source_name}(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {{noinline = false}} {{
    tt.return
  }}

  rpu_plan.kernel @{source_name}_plan {{
    args = {args},
    emission = {{kind = "{pattern}"}},
    kernel_name = "{source_name}",
    layout = {{access = "linear", memory = "contiguous_vector"}},
    mask = {{masked = false}},
    pattern = "{pattern}",
    required_dsl_features = ["ctx.load_contig"],
    shape = {shape},
    signature = {{
      params = {signature_params},
      return_type = "void"
    }},
    source_func = @{source_name},
    version = 1 : i32
  }}
}}
"""


def _three_ptr_signature():
    return """[
        {element_type = "f16", index = 0 : i32, kind = "ptr", name = "arg0"},
        {element_type = "f16", index = 1 : i32, kind = "ptr", name = "arg1"},
        {element_type = "f16", index = 2 : i32, kind = "ptr", name = "arg2"}
      ]"""


def _four_ptr_signature():
    return """[
        {element_type = "f16", index = 0 : i32, kind = "ptr", name = "arg0"},
        {element_type = "f16", index = 1 : i32, kind = "ptr", name = "arg1"},
        {element_type = "f16", index = 2 : i32, kind = "ptr", name = "arg2"},
        {element_type = "f16", index = 3 : i32, kind = "ptr", name = "arg3"}
      ]"""


def _five_ptr_signature():
    return """[
        {element_type = "f16", index = 0 : i32, kind = "ptr", name = "arg0"},
        {element_type = "f16", index = 1 : i32, kind = "ptr", name = "arg1"},
        {element_type = "f16", index = 2 : i32, kind = "ptr", name = "arg2"},
        {element_type = "f16", index = 3 : i32, kind = "ptr", name = "arg3"},
        {element_type = "f16", index = 4 : i32, kind = "ptr", name = "arg4"}
      ]"""


def _direct_add_module(source_name, emission_attrs, shape_attrs, mask_attr):
    return f"""
module {{
  tt.func public @{source_name}(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {{noinline = false}} {{
    tt.return
  }}

  rpu_plan.kernel @{source_name}_plan {{
    args = {{lhs = 1 : i64, out = 0 : i64, rhs = 2 : i64}},
    emission = {emission_attrs},
    kernel_name = "{source_name}",
    layout = {{access = "linear", memory = "contiguous_vector"}},
    mask = {mask_attr},
    pattern = "add",
    required_dsl_features = ["ctx.load_contig", "tile.add", "ctx.store_contig"],
    shape = {shape_attrs},
    signature = {{
      params = {_three_ptr_signature()},
      return_type = "void"
    }},
    source_func = @{source_name},
    version = 1 : i32
  }}
}}
"""


def _direct_gemm_module(source_name, m, n, k):
    return f"""
module {{
  tt.func public @{source_name}(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {{noinline = false}} {{
    tt.return
  }}

  rpu_plan.kernel @{source_name}_plan {{
    args = {{lhs = 1 : i64, out = 0 : i64, rhs = 2 : i64}},
    emission = {{kind = "gemm", k = {k} : i64, lhs = 1 : i64, m = {m} : i64, n = {n} : i64, out = 0 : i64, rhs = 2 : i64}},
    kernel_name = "{source_name}",
    layout = {{access = "matrix_tile", memory = "array2d", order = "row_major"}},
    mask = {{masked = false}},
    pattern = "gemm",
    required_dsl_features = ["rpu.Array", "ctx.load", "ctx.zeros", "ctx.mma", "ctx.store"],
    shape = {{k = {k} : i64, m = {m} : i64, n = {n} : i64}},
    signature = {{
      params = {_three_ptr_signature()},
      return_type = "void"
    }},
    source_func = @{source_name},
    version = 1 : i32
  }}
}}
"""


def _direct_softmax_module(source_name, n):
    return f"""
module {{
  tt.func public @{source_name}(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) attributes {{noinline = false}} {{
    tt.return
  }}

  rpu_plan.kernel @{source_name}_plan {{
    args = {{input = 1 : i64, out = 0 : i64}},
    emission = {{input = 1 : i64, kind = "softmax", n = {n} : i64, out = 0 : i64}},
    kernel_name = "{source_name}",
    layout = {{access = "linear", memory = "contiguous_vector"}},
    mask = {{masked = false}},
    pattern = "softmax",
    required_dsl_features = ["ctx.load_contig", "ctx.reduce_max_all", "rpu.exp", "ctx.reduce_sum_all", "rpu.reciprocal", "ctx.store_contig"],
    shape = {{n = {n} : i64}},
    signature = {{
      params = [
        {{element_type = "f16", index = 0 : i32, kind = "ptr", name = "arg0"}},
        {{element_type = "f16", index = 1 : i32, kind = "ptr", name = "arg1"}}
      ],
      return_type = "void"
    }},
    source_func = @{source_name},
    version = 1 : i32
  }}
}}
"""


def _direct_convkxk_module(source_name, kernel_size):
    return f"""
module {{
  tt.func public @{source_name}(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {{noinline = false}} {{
    tt.return
  }}

  rpu_plan.kernel @{source_name}_plan {{
    args = {{input = 1 : i64, out = 0 : i64, weight = 2 : i64}},
    emission = {{in_channels = 16 : i64, input = 1 : i64, input_width = 16 : i64, kernel_size = {kernel_size} : i64, kind = "convkxk", m = 16 : i64, out = 0 : i64, out_channels = 16 : i64, weight = 2 : i64}},
    kernel_name = "{source_name}",
    layout = {{access = "row_window", memory = "array2d", order = "row_major", tile = {{m = 16 : i64, n = 16 : i64}}, window = {{input_width = 16 : i64, kernel_size = {kernel_size} : i64, padding = [0 : i64, 0 : i64], stride = [1 : i64, 1 : i64]}}}},
    mask = {{masked = false}},
    pattern = "convkxk",
    required_dsl_features = ["rpu.Array", "ctx.load", "ctx.zeros", "ctx.mma", "ctx.store"],
    shape = {{in_channels = 16 : i64, input_width = 16 : i64, kernel_size = {kernel_size} : i64, m = 16 : i64, out_channels = 16 : i64}},
    signature = {{
      params = {_three_ptr_signature()},
      return_type = "void"
    }},
    source_func = @{source_name},
    version = 1 : i32
  }}
}}
"""


def _direct_resnet_block_module(source_name):
    return f"""
module {{
  tt.func public @{source_name}(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: !tt.ptr<f16>) attributes {{noinline = false}} {{
    tt.return
  }}

  rpu_plan.kernel @{source_name}_plan {{
    args = {{out = 0 : i64, w1 = 2 : i64, w2 = 3 : i64, x = 1 : i64}},
    emission = {{channels = 16 : i64, hidden = 16 : i64, kind = "resnet_block", m = 16 : i64, out = 0 : i64, w1 = 2 : i64, w2 = 3 : i64, x = 1 : i64}},
    kernel_name = "{source_name}",
    layout = {{access = "matrix_tile", memory = "array2d", order = "row_major"}},
    mask = {{masked = false}},
    pattern = "resnet_block",
    required_dsl_features = ["rpu.Array", "ctx.load", "ctx.zeros", "ctx.mma", "tile.add", "rpu.max_binop", "ctx.store"],
    shape = {{channels = 16 : i64, hidden = 16 : i64, m = 16 : i64}},
    signature = {{
      params = {_four_ptr_signature()},
      return_type = "void"
    }},
    source_func = @{source_name},
    version = 1 : i32
  }}
}}
"""


def _direct_resnet50_bottleneck_module(source_name):
    return f"""
module {{
  tt.func public @{source_name}(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: !tt.ptr<f16>, %arg4: !tt.ptr<f16>) attributes {{noinline = false}} {{
    tt.return
  }}

  rpu_plan.kernel @{source_name}_plan {{
    args = {{input = 1 : i64, out = 0 : i64, w1 = 2 : i64, w2 = 3 : i64, w3 = 4 : i64}},
    emission = {{bottleneck = 16 : i64, channels = 16 : i64, input = 1 : i64, input_width = 16 : i64, kernel_size = 3 : i64, kind = "resnet50_bottleneck", m = 16 : i64, out = 0 : i64, w1 = 2 : i64, w2 = 3 : i64, w3 = 4 : i64}},
    kernel_name = "{source_name}",
    layout = {{access = "bottleneck_row_window", memory = "array2d", order = "row_major", tile = {{m = 16 : i64, n = 16 : i64}}, window = {{input_width = 16 : i64, kernel_size = 3 : i64, padding = [0 : i64, 0 : i64], stride = [1 : i64, 1 : i64]}}}},
    mask = {{masked = false}},
    pattern = "resnet50_bottleneck",
    required_dsl_features = ["rpu.Array", "ctx.load", "ctx.zeros", "ctx.mma", "tile.add", "rpu.max_binop", "ctx.store"],
    shape = {{bottleneck = 16 : i64, channels = 16 : i64, input_width = 16 : i64, kernel_size = 3 : i64, m = 16 : i64}},
    signature = {{
      params = {_five_ptr_signature()},
      return_type = "void"
    }},
    source_func = @{source_name},
    version = 1 : i32
  }}
}}
"""


@pytest.mark.parametrize(
    "pattern,shape,args,source_name",
    [
        (
            "add",
            "{logical_n = 16 : i64, n = 16 : i64}",
            "{lhs = 1 : i64, out = 0 : i64, rhs = 2 : i64}",
            "rpu_add_kernel",
        ),
        (
            "gemm",
            "{k = 16 : i64, m = 16 : i64, n = 16 : i64}",
            "{lhs = 0 : i64, out = 2 : i64, rhs = 1 : i64}",
            "rpu_gemm_kernel",
        ),
        (
            "softmax",
            "{n = 16 : i64}",
            "{input = 0 : i64, out = 1 : i64}",
            "rpu_softmax_kernel",
        ),
    ],
)
def test_rpuplan_kernel_round_trips(pattern, shape, args, source_name, tmp_path, monkeypatch):
    ir, context = _load_rpuplan_context(monkeypatch)
    module = _parse_mlir(
        ir,
        context,
        tmp_path,
        _kernel_plan_module(pattern, shape, args, _three_ptr_signature(), source_name),
    )

    printed = str(module)
    assert f"rpu_plan.kernel @{source_name}_plan" in printed
    assert f'kernel_name = "{source_name}"' in printed
    assert f'pattern = "{pattern}"' in printed
    assert "source_func = @" + source_name in printed

    reparsed = _parse_mlir(ir, context, tmp_path, printed)
    assert f"rpu_plan.kernel @{source_name}_plan" in str(reparsed)


def _invalid_module(case):
    valid_signature = _three_ptr_signature()
    valid_attrs = f"""
    args = {{lhs = 1 : i64, out = 0 : i64, rhs = 2 : i64}},
    emission = {{kind = "add"}},
    kernel_name = "rpu_add_kernel",
    layout = {{access = "linear", memory = "contiguous_vector"}},
    mask = {{masked = false}},
    pattern = "add",
    required_dsl_features = ["ctx.load_contig"],
    shape = {{logical_n = 16 : i64, n = 16 : i64}},
    signature = {{
      params = {valid_signature},
      return_type = "void"
    }},
    source_func = @rpu_add_kernel,
    version = 1 : i32
"""

    replacements = {
        "missing_mask": valid_attrs.replace("    mask = {masked = false},\n", ""),
        "unknown_pattern": valid_attrs.replace('pattern = "add"', 'pattern = "unknown"'),
        "bad_arg_index": valid_attrs.replace("rhs = 2 : i64", "rhs = 99 : i64"),
        "non_positive_shape": valid_attrs.replace("n = 16 : i64", "n = 0 : i64"),
        "private_source": valid_attrs,
        "same_symbol": valid_attrs,
        "nested_op": valid_attrs,
    }

    source_visibility = "private" if case == "private_source" else "public"
    plan_symbol = "rpu_add_kernel" if case == "same_symbol" else "rpu_add_kernel_plan"
    attrs = replacements[case]

    if case == "nested_op":
        return f"""
module {{
  tt.func public @rpu_add_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {{noinline = false}} {{
    rpu_plan.kernel @{plan_symbol} {{
{attrs}
    }}
    tt.return
  }}
}}
"""

    return f"""
module {{
  tt.func {source_visibility} @rpu_add_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {{noinline = false}} {{
    tt.return
  }}

  rpu_plan.kernel @{plan_symbol} {{
{attrs}
  }}
}}
"""


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        ("missing_mask", "requires attribute 'mask'|requires mask dictionary"),
        ("unknown_pattern", "unsupported pattern"),
        ("bad_arg_index", "argument index 99 is outside signature parameter range"),
        ("non_positive_shape", "shape field n must be a positive integer"),
        ("private_source", "source_func must reference a public tt.func"),
        ("same_symbol", "sym_name must not equal source_func|redefinition of symbol"),
        ("nested_op", "must be a top-level operation in builtin.module|SymbolTable"),
    ],
)
def test_rpuplan_kernel_verifier_rejects_invalid_plans(case, diagnostic, tmp_path, monkeypatch, capfd):
    ir, context = _load_rpuplan_context(monkeypatch)
    with pytest.raises(RuntimeError):
        _parse_mlir(ir, context, tmp_path, _invalid_module(case))
    assert re.search(diagnostic, capfd.readouterr().err)


def test_rpuplan_json_export_is_derived_from_kernel_op():
    transforms_dir = _repo_root() / "third_party" / "rpu" / "lib" / "RPUTransforms"
    recognizer = (transforms_dir / "RecognizeRPUPlan.cpp").read_text()
    json_export = (transforms_dir / "RPUPlanJSON.cpp").read_text()

    assert not re.search(r"serializeRPUPlanToJson\s*\(", recognizer)
    assert "serializeRPUPlanKernelOpToJson" in recognizer
    body = _source_between(
        recognizer,
        "LogicalResult emitPlanOpJsonAndTrace(",
        "RPUPlan makeAddPlan",
    )
    create_idx = body.index("FailureOr<plan::KernelOp> op = createRPUPlanKernelOp")
    export_idx = body.index("serializeRPUPlanKernelOpToJson(*op)")
    set_json_idx = body.index('module->setAttr("rpu.plan.json"')
    assert create_idx < export_idx < set_json_idx
    assert "rpuPlanFromKernelOp" in json_export
    assert "serializeRPUPlanKernelOpToJson" in json_export


def test_rpuplan_dto_json_serializer_is_private_to_kernel_export():
    transforms_dir = _repo_root() / "third_party" / "rpu" / "lib" / "RPUTransforms"
    model_header = (transforms_dir / "RPUPlanModel.h").read_text()
    json_export = (transforms_dir / "RPUPlanJSON.cpp").read_text()

    assert not re.search(
        r"^\s*std::string\s+serializeRPUPlanToJson\s*\(",
        model_header,
        flags=re.M,
    )
    assert re.search(
        r"^static\s+std::string\s+serializeRPUPlanToJson\s*\(",
        json_export,
        flags=re.M,
    )
    helper_body = _source_between(
        json_export,
        "std::optional<std::string> serializeRPUPlanKernelOpToJson(",
        "\n}\n\n} // namespace rpu",
    )
    serializer_refs = list(re.finditer(r"serializeRPUPlanToJson\s*\(", json_export))
    assert len(serializer_refs) == 2
    assert "return serializeRPUPlanToJson(*plan);" in helper_body


def test_rpuplan_trace_softmax_failure_anchor_allows_reduce_region():
    source_path = (_repo_root() / "third_party/rpu/lib/RPUTransforms/RPUTTIRPatternMatcher.cpp")
    source = source_path.read_text()
    start = source.index("bool isAllowedSoftmaxTopLevelOp(")
    end = source.index("rankedFailureAnchor(", start)
    helper = source[start:end]

    assert "triton::ReduceOp" in helper
    assert "op.getNumRegions()" not in helper


def test_rpuplan_direct_emitter_lists_supported_patterns(monkeypatch):
    _load_rpuplan_context(monkeypatch)

    from triton._C.libtriton import rpu

    assert set(rpu._direct_rpurc_supported_patterns()) == {
        "add",
        "gemm",
        "softmax",
        "convkxk",
        "resnet_block",
        "resnet50_bottleneck",
        "sqrt",
        "reduce_sum_all",
    }


def test_rpuplan_kernel_summary_reports_identity(tmp_path, monkeypatch):
    result = _get_rpuplan_summary_from_text(
        _kernel_plan_module(
            "add",
            "{logical_n = 16 : i64, n = 16 : i64}",
            "{lhs = 1 : i64, out = 0 : i64, rhs = 2 : i64}",
            _three_ptr_signature(),
            "rpu_add_kernel",
        ),
        tmp_path,
        monkeypatch,
    )

    assert result == {
        "kernel_name": "rpu_add_kernel",
        "pattern": "add",
    }


def test_rpuplan_kernel_summary_rejects_missing_or_multiple_plans(tmp_path, monkeypatch, capfd):
    ir, context = _load_rpuplan_context(monkeypatch)

    no_plan = _parse_mlir(
        ir,
        context,
        tmp_path,
        """
module {
  tt.func public @rpu_no_plan(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    tt.return
  }
}
""",
    )
    from triton._C.libtriton import rpu

    with pytest.raises(RuntimeError, match="RPU rpu_plan.kernel summary failed"):
        rpu._get_rpuplan_kernel_summary(no_plan)
    assert "found none" in capfd.readouterr().err

    signature = _three_ptr_signature()
    multi_plan = _parse_mlir(
        ir,
        context,
        tmp_path,
        f"""
module {{
  tt.func public @rpu_add_kernel_a(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {{noinline = false}} {{
    tt.return
  }}

  rpu_plan.kernel @rpu_add_kernel_a_plan {{
    args = {{lhs = 1 : i64, out = 0 : i64, rhs = 2 : i64}},
    emission = {{kind = "add", logical_n = 16 : i64, masked = false, n = 16 : i64, out = 0 : i64, lhs = 1 : i64, rhs = 2 : i64}},
    kernel_name = "rpu_add_kernel_a",
    layout = {{access = "linear", memory = "contiguous_vector"}},
    mask = {{masked = false}},
    pattern = "add",
    required_dsl_features = ["ctx.load_contig", "tile.add", "ctx.store_contig"],
    shape = {{logical_n = 16 : i64, n = 16 : i64}},
    signature = {{
      params = {signature},
      return_type = "void"
    }},
    source_func = @rpu_add_kernel_a,
    version = 1 : i32
  }}

  tt.func public @rpu_add_kernel_b(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {{noinline = false}} {{
    tt.return
  }}

  rpu_plan.kernel @rpu_add_kernel_b_plan {{
    args = {{lhs = 1 : i64, out = 0 : i64, rhs = 2 : i64}},
    emission = {{kind = "add", logical_n = 16 : i64, masked = false, n = 16 : i64, out = 0 : i64, lhs = 1 : i64, rhs = 2 : i64}},
    kernel_name = "rpu_add_kernel_b",
    layout = {{access = "linear", memory = "contiguous_vector"}},
    mask = {{masked = false}},
    pattern = "add",
    required_dsl_features = ["ctx.load_contig", "tile.add", "ctx.store_contig"],
    shape = {{logical_n = 16 : i64, n = 16 : i64}},
    signature = {{
      params = {signature},
      return_type = "void"
    }},
    source_func = @rpu_add_kernel_b,
    version = 1 : i32
  }}
}}
""",
    )
    with pytest.raises(RuntimeError, match="RPU rpu_plan.kernel summary failed"):
        rpu._get_rpuplan_kernel_summary(multi_plan)
    assert "found multiple" in capfd.readouterr().err


def test_rpuplan_kernel_summary_reads_identity_attrs_without_dto_conversion(tmp_path, monkeypatch):
    ir, context = _load_rpuplan_context(monkeypatch)
    module = _parse_mlir(
        ir,
        context,
        tmp_path,
        """
module {
  tt.func public @rpu_bad_plan(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {noinline = false} {
    tt.return
  }

  rpu_plan.kernel @rpu_bad_plan_plan {
    args = {lhs = 1 : i64, out = 0 : i64, rhs = 2 : i64},
    emission = {
      kind = "add",
      logical_n = 16 : i64,
      masked = false,
      n = 16 : i64,
      out = 0 : i64,
      lhs = 1 : i64,
      rhs = 2 : i64,
      unsupported = 1.000000e+00 : f32
    },
    kernel_name = "rpu_bad_plan",
    layout = {access = "linear", memory = "contiguous_vector"},
    mask = {masked = false},
    pattern = "add",
    required_dsl_features = ["ctx.load_contig", "tile.add", "ctx.store_contig"],
    shape = {logical_n = 16 : i64, n = 16 : i64},
    signature = {
      params = [
        {element_type = "f16", index = 0 : i32, kind = "ptr", name = "arg0"},
        {element_type = "f16", index = 1 : i32, kind = "ptr", name = "arg1"},
        {element_type = "f16", index = 2 : i32, kind = "ptr", name = "arg2"}
      ],
      return_type = "void"
    },
    source_func = @rpu_bad_plan,
    version = 1 : i32
  }
}
""",
    )
    from triton._C.libtriton import rpu

    summary = rpu._get_rpuplan_kernel_summary(module)

    assert summary["kernel_name"] == "rpu_bad_plan"
    assert summary["pattern"] == "add"


def test_rpuplan_direct_emitter_add_contiguous(tmp_path, monkeypatch):
    result = _emit_rpurc_from_plan_text(
        _direct_add_module(
            "rpu_add_kernel",
            '{kind = "add", logical_n = 16 : i64, masked = false, n = 16 : i64, out = 0 : i64, lhs = 1 : i64, rhs = 2 : i64}',
            "{logical_n = 16 : i64, n = 16 : i64}",
            "{masked = false}",
        ),
        tmp_path,
        monkeypatch,
    )

    assert result["kernel_name"] == "rpu_add_kernel"
    assert result["pattern"] == "add"
    assert '#include "rpu_tile.h"\nusing namespace rpu;\n' in result["source"]
    assert "\\n" not in result["source"]
    assert "ctx.load_contig<16>(arg1)" in result["source"]
    assert "ctx.load_contig<16>(arg2)" in result["source"]
    assert "ctx.store_contig<16>(arg0, result)" in result["source"]


def test_rpuplan_direct_emitter_add_large_and_masked(tmp_path, monkeypatch):
    large = _emit_rpurc_from_plan_text(
        _direct_add_module(
            "rpu_large_add_kernel",
            '{kind = "add", logical_n = 256 : i64, masked = false, n = 256 : i64, out = 0 : i64, lhs = 1 : i64, rhs = 2 : i64}',
            "{logical_n = 256 : i64, n = 256 : i64}",
            "{masked = false}",
        ),
        tmp_path,
        monkeypatch,
    )
    masked = _emit_rpurc_from_plan_text(
        _direct_add_module(
            "rpu_masked_add_kernel",
            '{kind = "add", logical_n = 17 : i64, masked = true, n = 32 : i64, out = 0 : i64, lhs = 1 : i64, rhs = 2 : i64}',
            "{logical_n = 17 : i64, n = 32 : i64}",
            "{masked = true}",
        ),
        tmp_path,
        monkeypatch,
    )

    assert "auto frame = ctx.tile_frame();" in large["source"]
    assert "\\n" not in large["source"]
    assert "\\n" not in masked["source"]
    assert "make_shape(4096, 16)" in large["source"]
    assert "rpu::make_coord(2048, 0)" in large["source"]
    assert "make_shape(1, 17)" in masked["source"]
    assert "ctx.load<half, 16, 32>" in masked["source"]
    assert "ctx.store_contig<32>" not in masked["source"]


def test_rpuplan_direct_emitter_gemm_and_softmax(tmp_path, monkeypatch):
    gemm = _emit_rpurc_from_plan_text(
        _direct_gemm_module("rpu_gemm_kernel", 32, 16, 16),
        tmp_path,
        monkeypatch,
    )
    softmax = _emit_rpurc_from_plan_text(
        _direct_softmax_module("rpu_softmax_kernel", 16),
        tmp_path,
        monkeypatch,
    )

    assert "\\n" not in gemm["source"]
    assert "\\n" not in softmax["source"]
    assert "rpu::Array<half, 2> lhs{arg1, 32, 16}" in gemm["source"]
    assert "ctx.mma<32, 16, 16>" in gemm["source"]
    assert "ctx.store<half, 32, 16>" in gemm["source"]
    assert "ctx.reduce_max_all" in softmax["source"]
    assert "rpu::exp" in softmax["source"]
    assert "ctx.store_contig<1>(arg0, y)" in softmax["source"]


def test_rpuplan_direct_emitter_convkxk(tmp_path, monkeypatch):
    result = _emit_rpurc_from_plan_text(
        _direct_convkxk_module("rpu_convkxk_kernel", 3),
        tmp_path,
        monkeypatch,
    )

    assert result["kernel_name"] == "rpu_convkxk_kernel"
    assert result["pattern"] == "convkxk"
    assert "\\n" not in result["source"]
    assert result["source"].count("ctx.mma<16, 16, 16>") == 9
    assert "rpu::IndexList{34, 0}" in result["source"]
    assert "rpu::IndexList{128, 0}" in result["source"]
    assert "ctx.store<half, 16, 16>" in result["source"]


def test_rpuplan_direct_emitter_resnet_block(tmp_path, monkeypatch):
    result = _emit_rpurc_from_plan_text(
        _direct_resnet_block_module("rpu_resnet_block_kernel"),
        tmp_path,
        monkeypatch,
    )

    assert result["kernel_name"] == "rpu_resnet_block_kernel"
    assert result["pattern"] == "resnet_block"
    assert "\\n" not in result["source"]
    assert result["source"].count("ctx.mma<16, 16, 16>") == 2
    assert result["source"].count("rpu::max_binop(") == 2
    assert "auto residual = conv2 + x;" in result["source"]
    assert "ctx.store<half, 16, 16>" in result["source"]


def test_rpuplan_direct_emitter_resnet50_bottleneck(tmp_path, monkeypatch):
    result = _emit_rpurc_from_plan_text(
        _direct_resnet50_bottleneck_module("rpu_resnet50_bottleneck_kernel"),
        tmp_path,
        monkeypatch,
    )

    assert result["kernel_name"] == "rpu_resnet50_bottleneck_kernel"
    assert result["pattern"] == "resnet50_bottleneck"
    assert "\\n" not in result["source"]
    assert result["source"].count("ctx.mma<16, 16, 16>") == 19
    assert result["source"].count("rpu::max_binop(") == 11
    assert "rpu::IndexList{34, 0}" in result["source"]
    assert "rpu::IndexList{128, 0}" in result["source"]
    assert "auto residual = conv3 + x_skip;" in result["source"]
    assert "ctx.store<half, 16, 16>" in result["source"]


def test_rpuplan_direct_emitter_rejects_missing_or_multiple_plans(tmp_path, monkeypatch, capfd):
    ir, context = _load_rpuplan_context(monkeypatch)

    from triton._C.libtriton import rpu

    no_plan = _parse_mlir(
        ir,
        context,
        tmp_path,
        """
module {
  tt.func public @rpu_no_plan(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    tt.return
  }
}
""",
    )
    with pytest.raises(RuntimeError, match="RPU direct .rc emission failed"):
        rpu._emit_rpurc_from_plan(no_plan)
    assert "found none" in capfd.readouterr().err

    signature = _three_ptr_signature()
    multi_plan = _parse_mlir(
        ir,
        context,
        tmp_path,
        f"""
module {{
  tt.func public @rpu_add_kernel_a(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {{noinline = false}} {{
    tt.return
  }}

  rpu_plan.kernel @rpu_add_kernel_a_plan {{
    args = {{lhs = 1 : i64, out = 0 : i64, rhs = 2 : i64}},
    emission = {{kind = "add", logical_n = 16 : i64, masked = false, n = 16 : i64, out = 0 : i64, lhs = 1 : i64, rhs = 2 : i64}},
    kernel_name = "rpu_add_kernel_a",
    layout = {{access = "linear", memory = "contiguous_vector"}},
    mask = {{masked = false}},
    pattern = "add",
    required_dsl_features = ["ctx.load_contig", "tile.add", "ctx.store_contig"],
    shape = {{logical_n = 16 : i64, n = 16 : i64}},
    signature = {{
      params = {signature},
      return_type = "void"
    }},
    source_func = @rpu_add_kernel_a,
    version = 1 : i32
  }}

  tt.func public @rpu_add_kernel_b(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {{noinline = false}} {{
    tt.return
  }}

  rpu_plan.kernel @rpu_add_kernel_b_plan {{
    args = {{lhs = 1 : i64, out = 0 : i64, rhs = 2 : i64}},
    emission = {{kind = "add", logical_n = 16 : i64, masked = false, n = 16 : i64, out = 0 : i64, lhs = 1 : i64, rhs = 2 : i64}},
    kernel_name = "rpu_add_kernel_b",
    layout = {{access = "linear", memory = "contiguous_vector"}},
    mask = {{masked = false}},
    pattern = "add",
    required_dsl_features = ["ctx.load_contig", "tile.add", "ctx.store_contig"],
    shape = {{logical_n = 16 : i64, n = 16 : i64}},
    signature = {{
      params = {signature},
      return_type = "void"
    }},
    source_func = @rpu_add_kernel_b,
    version = 1 : i32
  }}
}}
""",
    )
    with pytest.raises(RuntimeError, match="RPU direct .rc emission failed"):
        rpu._emit_rpurc_from_plan(multi_plan)
    assert "found multiple" in capfd.readouterr().err


def test_rpuplan_direct_emitter_stays_op_derived_without_json_roundtrip():
    transforms_dir = _repo_root() / "third_party" / "rpu" / "lib" / "RPUTransforms"
    emitter_header = transforms_dir / "RPUDSLEmitter.h"
    emitter_source = transforms_dir / "RPUDSLEmitter.cpp"

    assert emitter_header.exists()
    assert emitter_source.exists()
    text = emitter_source.read_text()
    assert "rpuPlanFromKernelOp" in text
    assert "serializeRPUPlanKernelOpToJson" not in text
    assert "serializeRPUPlanToJson" not in text
    assert "found none" in text
    assert "found multiple" in text
