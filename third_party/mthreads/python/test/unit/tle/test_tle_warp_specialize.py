"""Compile-only coverage for the mthreads TLE warp-specialize container."""

import re

import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton._C.libtriton import ir
from triton.backends.compiler import Language
from triton.compiler import ASTSource

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


def _compile_ws_ir():
    target, backend = mthreads_backend()
    options = backend.parse_options({"num_warps": 16, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(
        fn=_ws_container_kernel,
        signature={"out": "*i32", "value": "i32"},
        constexprs={"BIAS": 7},
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
