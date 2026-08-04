"""Compile-only coverage for the mthreads TLE warp-specialize container."""

import re

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton._C.libtriton import ir
from triton.backends.compiler import Language
from triton.compiler import ASTSource
from triton.compiler.errors import CompilationError

from test_tle_utils import mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _ws_default(out, value):
    tl.store(out, value + 1)


@triton.jit
def _ws_worker(out, value, smem, duplicate_out, BIAS: tl.constexpr):
    local = tle.gpu.local_ptr(smem, (0, ))
    tl.store(local, value)
    loaded = tl.load(local)
    tl.store(out + 1, loaded + BIAS)
    tl.store(duplicate_out + 2, loaded + BIAS + 1)


@triton.jit
def _ws_container_kernel(out, value, BIAS: tl.constexpr):
    smem = tle.gpu.alloc(
        (1, ),
        dtype=tl.int32,
        nv_mma_shared_layout=False,
    )
    tle.gpu.warp_specialize(
        [
            (_ws_default, (out, value)),
            (_ws_worker, (out, value, smem, out, BIAS)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_simple_default(out):
    tl.store(out, 1)


@triton.jit
def _ws_simple_worker(out):
    tl.store(out + 1, 2)


@triton.jit
def _ws_dynamic_warps_kernel(out, worker_warps):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[worker_warps],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_dynamic_regs_kernel(out, worker_regs):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[4],
        worker_num_regs=[worker_regs],
    )


@triton.jit
def _ws_non_sequence_warps_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=4,
        worker_num_regs=[24],
    )


@triton.jit
def _ws_non_sequence_regs_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[4],
        worker_num_regs=24,
    )


@triton.jit
def _ws_float_warps_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[4.0],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_float_regs_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[4],
        worker_num_regs=[24.0],
    )


@triton.jit
def _ws_zero_warps_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[0],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_non_power_of_two_warps_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[3],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_warp_count_mismatch_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_register_count_mismatch_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[4],
        worker_num_regs=[],
    )


@triton.jit
def _ws_empty_functions_kernel(out):
    tle.gpu.warp_specialize([], worker_num_warps=[], worker_num_regs=[])


@triton.jit
def _ws_invalid_entry_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), _ws_simple_worker],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_invalid_args_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, out)],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_bool_warps_kernel(out):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=[True],
        worker_num_regs=[24],
    )


@triton.jit
def _ws_constexpr_tuple_config_kernel(out, WORKER_WARPS: tl.constexpr, WORKER_REGS: tl.constexpr):
    tle.gpu.warp_specialize(
        [(_ws_simple_default, (out, )), (_ws_simple_worker, (out, ))],
        worker_num_warps=(WORKER_WARPS, ),
        worker_num_regs=(WORKER_REGS, ),
    )


def _compile_ws_ir(fn=_ws_container_kernel, signature=None, constexprs=None):
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 16, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(
        fn=fn,
        signature=signature or {"out": "*i32", "value": "i32"},
        constexprs={"BIAS": 7} if constexprs is None else constexprs,
    )
    module = src.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    stages = {}
    backend.add_stages(stages, options, Language.TRITON)
    metadata = {}
    module = stages["ttir"](module, metadata)
    ttir = module.str_nodebug()
    module = stages["ttgir"](module, metadata)
    ttgir = module.str_nodebug()
    return ttir, ttgir


def _split_top_level(text):
    values = []
    start = 0
    depth = 0
    pairs = {"<": ">", "[": "]", "{": "}"}
    closing = set(pairs.values())
    for index, char in enumerate(text):
        if char in pairs:
            depth += 1
        elif char in closing:
            depth -= 1
        elif char == "," and depth == 0:
            values.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        values.append(tail)
    return values


def _assert_ws_container(ir_text):
    assert ir_text.count("ttg.warp_specialize") == 1, ir_text
    assert ir_text.count("ttg.warp_yield") == 1, ir_text
    assert ir_text.count("ttg.warp_return") == 1, ir_text
    assert "partition0" in ir_text, ir_text
    assert "partition1" not in ir_text, ir_text
    assert "num_warps(4)" in ir_text, ir_text
    assert re.search(r"requestedRegisters\s*=\s*array<i32:\s*24>", ir_text), ir_text
    assert "actualRegisters" not in ir_text, ir_text
    assert "ttg.maxnreg" not in ir_text, ir_text
    assert "tle.wgmma_pipeline_mode" not in ir_text, ir_text
    assert "tt.call" not in ir_text, ir_text

    ws_match = re.search(
        r"ttg\.warp_specialize\((?P<captures>[^)]*)\).*?"
        r"partition0\((?P<args>[^)]*)\)\s*num_warps\(4\).*?"
        r":\s*\((?P<types>[^)]*)\)\s*->\s*\(\)",
        ir_text,
        re.DOTALL,
    )
    assert ws_match, ir_text
    captures = _split_top_level(ws_match.group("captures"))
    args = _split_top_level(ws_match.group("args"))
    types = _split_top_level(ws_match.group("types"))

    # Five Python worker arguments flatten to three unique SSA captures:
    # pointer, dynamic i32, and shared memdesc.  The repeated pointer is
    # remapped to the first block argument and constexpr BIAS consumes no SSA.
    assert len(captures) == 3, ir_text
    assert len(args) == 3, ir_text
    assert len(types) == 3, ir_text
    assert sum("ptr" in ty for ty in types) == 1, ir_text
    assert sum(ty == "i32" for ty in types) == 1, ir_text
    assert sum("memdesc" in ty for ty in types) == 1, ir_text
    assert any(re.search(r"\bi32\b", arg) for arg in args), ir_text
    assert any("memdesc" in arg for arg in args), ir_text

    assert re.search(r"(?:arith\.)?constant\s+7\s*:\s*i32", ir_text), ir_text
    assert re.search(r"ttg\.warp_yield\s*(?:\n|\})", ir_text), ir_text
    assert re.search(r"ttg\.warp_return\s*(?:\n|\})", ir_text), ir_text
    assert re.search(r"\)\s*->\s*\(\)\s*\n\s*tt\.return", ir_text), ir_text


def test_tle_warp_specialize_mthreads_container_contract():
    ttir, ttgir = _compile_ws_ir()
    _assert_ws_container(ttir)
    _assert_ws_container(ttgir)


def test_tle_warp_specialize_builder_bindings_are_available():
    builder = libtriton.ir.builder
    for method in (
            "create_warp_specialize",
            "create_warp_specialize_partitions",
            "create_warp_yield",
            "create_warp_return",
    ):
        assert hasattr(builder, method)
    assert hasattr(libtriton.mthreads.ir, "WarpSpecializeOp")


def test_tle_warp_specialize_accepts_constexpr_tuple_configuration():
    ttir, ttgir = _compile_ws_ir(
        _ws_constexpr_tuple_config_kernel,
        {"out": "*i32", "WORKER_WARPS": "constexpr", "WORKER_REGS": "constexpr"},
        {"WORKER_WARPS": 4, "WORKER_REGS": 24},
    )
    for ir_text in (ttir, ttgir):
        assert "num_warps(4)" in ir_text, ir_text
        assert re.search(r"requestedRegisters\s*=\s*array<i32:\s*24>", ir_text), ir_text


@pytest.mark.parametrize(
    "kernel,signature,diagnostic",
    [
        (
            _ws_dynamic_warps_kernel,
            {"out": "*i32", "worker_warps": "i32"},
            r"mthreads TLE warp_specialize worker_num_warps\[0\] must be a compile-time integer",
        ),
        (
            _ws_dynamic_regs_kernel,
            {"out": "*i32", "worker_regs": "i32"},
            r"mthreads TLE warp_specialize worker_num_regs\[0\] must be a compile-time integer",
        ),
        (
            _ws_non_sequence_warps_kernel,
            {"out": "*i32"},
            "mthreads TLE warp_specialize worker_num_warps must be a static sequence",
        ),
        (
            _ws_non_sequence_regs_kernel,
            {"out": "*i32"},
            "mthreads TLE warp_specialize worker_num_regs must be a static sequence",
        ),
        (
            _ws_float_warps_kernel,
            {"out": "*i32"},
            r"mthreads TLE warp_specialize worker_num_warps\[0\] must be a compile-time integer",
        ),
        (
            _ws_float_regs_kernel,
            {"out": "*i32"},
            r"mthreads TLE warp_specialize worker_num_regs\[0\] must be a compile-time integer",
        ),
        (
            _ws_zero_warps_kernel,
            {"out": "*i32"},
            r"mthreads TLE warp_specialize worker_num_warps\[0\] must be positive",
        ),
        (
            _ws_bool_warps_kernel,
            {"out": "*i32"},
            r"mthreads TLE warp_specialize worker_num_warps\[0\] must be a compile-time integer",
        ),
        (
            _ws_warp_count_mismatch_kernel,
            {"out": "*i32"},
            "warp_specialize got 1 worker functions but 0 warp counts",
        ),
        (
            _ws_register_count_mismatch_kernel,
            {"out": "*i32"},
            "warp_specialize got 1 worker functions but 0 register counts",
        ),
        (
            _ws_empty_functions_kernel,
            {"out": "*i32"},
            "warp_specialize requires at least a default partition function",
        ),
        (
            _ws_invalid_entry_kernel,
            {"out": "*i32"},
            "warp_specialize entry 1 must be a tuple",
        ),
        (
            _ws_invalid_args_kernel,
            {"out": "*i32"},
            "warp_specialize entry 1 args must be a tuple",
        ),
    ],
)
def test_tle_warp_specialize_frontend_rejects_invalid_static_configuration(kernel, signature, diagnostic):
    with pytest.raises(CompilationError, match=diagnostic):
        _compile_ws_ir(kernel, signature, constexprs={})


def test_tle_warp_specialize_dialect_rejects_non_power_of_two_warps(capfd):
    with pytest.raises(RuntimeError, match="error encountered during parsing"):
        _compile_ws_ir(_ws_non_power_of_two_warps_kernel, {"out": "*i32"}, constexprs={})
    diagnostic = "'ttg.warp_specialize' op partition #0 number of warps (3) must be a power of 2"
    assert diagnostic in capfd.readouterr().err


_WS_INVALID_DIALECT_FIXTURES = [
    (
        """
module attributes {"ttg.num-warps" = 16 : i32} {
  tt.func @bad_partition_count() {
    "ttg.warp_specialize"() ({
      "ttg.warp_yield"() : () -> ()
    }, {
      "ttg.warp_specialize.partitions"() : () -> ()
    }) {partitionNumWarps = array<i32: 4>} : () -> ()
    tt.return
  }
}
""",
        "'ttg.warp_specialize' op has 0 partitions but `partitionNumWarps` has 1 elements",
    ),
    (
        """
module attributes {"ttg.num-warps" = 16 : i32} {
  tt.func @bad_capture_count(%arg0: i32) {
    ttg.warp_specialize(%arg0)
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : (i32) -> ()
    tt.return
  }
}
""",
        "'ttg.warp_specialize.partitions' op partition region #0 has 0 arguments but expected 1",
    ),
    (
        """
module attributes {"ttg.num-warps" = 16 : i32} {
  tt.func @bad_capture_type(%arg0: i32) {
    ttg.warp_specialize(%arg0)
    default {
      ttg.warp_yield
    }
    partition0(%arg1: i64) num_warps(4) {
      ttg.warp_return
    } : (i32) -> ()
    tt.return
  }
}
""",
        "'ttg.warp_specialize.partitions' op partition region #0 argument #0 has type 'i64' "
        "but corresponding capture has type 'i32'",
    ),
]


@pytest.mark.parametrize("fixture,diagnostic", _WS_INVALID_DIALECT_FIXTURES)
def test_tle_warp_specialize_dialect_rejects_invalid_structure(tmp_path, capfd, fixture, diagnostic):
    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    fixture_path = tmp_path / "invalid_ws_capture.mlir"
    fixture_path.write_text(fixture)

    with pytest.raises(RuntimeError):
        ir.parse_mlir_module(str(fixture_path), context)
    assert diagnostic in capfd.readouterr().err
