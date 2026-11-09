"""MThreads RLC pass regressions, collected by the backend's pytest CI."""

from pathlib import Path
import re

import pytest
import triton
from triton._C import libtriton
from triton._C.libtriton import ir, passes
from triton._filecheck import run_filecheck
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import make_backend

pytestmark = pytest.mark.skipif(
    not hasattr(libtriton, "mthreads"), reason="MThreads backend is required"
)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("per_thread", [1, 8])
@pytest.mark.parametrize("other", [None, 7.0])
def test_split_mask_scalar_loads(dtype, per_thread, other, paired, tmp_path):
    # Keep shared masks unchanged. Split masks use a full-group fast path and
    # a guarded tail; the bound permits partial and wholly empty tiles.
    other_arg = "" if other is None else ", %other"
    peer = f"""
    %base_b = tt.splat %input_b : !tt.ptr<{dtype}> -> tensor<1024x!tt.ptr<{dtype}>, #b>
    %p_b = tt.addptr %base_b, %offset : tensor<1024x!tt.ptr<{dtype}>, #b>, tensor<1024xi32, #b>
    %v_b = tt.load %p_b, %mask{other_arg} : tensor<1024x!tt.ptr<{dtype}>, #b>
    %sum = arith.addf %v, %v_b : tensor<1024x{dtype}, #b>
""" if paired else ""
    result = "%sum" if paired else "%v"
    source = f"""
#b = #ttg.blocked<{{sizePerThread = [{per_thread}], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}}>
module attributes {{"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "musa:31"}} {{
  tt.func public @masked_scalar(%input: !tt.ptr<{dtype}>, %input_b: !tt.ptr<{dtype}>, %output: !tt.ptr<{dtype}>, %n: i32 {{tt.divisibility = 16 : i32}}) {{
    %r = tt.make_range {{start = 0 : i32, end = 1024 : i32}} : tensor<1024xi32, #b>
    %two = arith.constant dense<2> : tensor<1024xi32, #b>
    %offset = arith.muli %r, %two : tensor<1024xi32, #b>
    %base = tt.splat %input : !tt.ptr<{dtype}> -> tensor<1024x!tt.ptr<{dtype}>, #b>
    %p = tt.addptr %base, %offset : tensor<1024x!tt.ptr<{dtype}>, #b>, tensor<1024xi32, #b>
    %bound = tt.splat %n : i32 -> tensor<1024xi32, #b>
    %mask = arith.cmpi slt, %r, %bound : tensor<1024xi32, #b>
    %other = arith.constant dense<7.0> : tensor<1024x{dtype}, #b>
    %v = tt.load %p, %mask{other_arg} : tensor<1024x!tt.ptr<{dtype}>, #b>
    {peer}
    %out = tt.splat %output : !tt.ptr<{dtype}> -> tensor<1024x!tt.ptr<{dtype}>, #b>
    %q = tt.addptr %out, %r : tensor<1024x!tt.ptr<{dtype}>, #b>, tensor<1024xi32, #b>
    tt.store %q, {result} : tensor<1024x!tt.ptr<{dtype}>, #b>
    tt.return
  }}
}}
"""
    path = tmp_path / "masked_scalar.ttgir"
    path.write_text(source)
    kernel = triton.compile(str(path), target=GPUTarget("musa", 31, 32))
    branches = len(re.findall(r"\bbr i1\b", kernel.asm["llir"]))
    assert branches <= (2 if per_thread == 1 else 1)


def run_rlc(source, tmp_path, phase_mask):
    path = tmp_path / "input.ttgir"
    path.write_text(source)
    context = ir.context()
    ir.load_dialects(context)
    make_backend(GPUTarget("musa", 31, 32)).load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    pm = ir.pass_manager(context)
    passes.ttgpuir.add_remove_layout_conversions(pm, phase_mask != 0, phase_mask)
    pm.run(module, "musa_rlc")
    return module.str_nodebug()


@pytest.mark.parametrize("phase_mask", [0, 5, 8, 15], ids=["off", "phase2", "phase3", "all"])
def test_rlc_regressions(phase_mask, tmp_path, monkeypatch, capfd):
    path = Path(__file__).resolve().parents[4] / "test/TritonGPU/remove-layout-conversions.mlir"
    source = path.read_text()
    monkeypatch.setenv("FLAGTREE_RLC_TRACE_REJECTS", str(int(phase_mask == 5)))
    result = "\n".join(run_rlc(part, tmp_path, phase_mask) for part in source.split("// -----"))
    trace = capfd.readouterr().err
    prefixes = ["CONTRACT"]
    # Phase 2 checks its intermediate layouts before later rematerialization.
    if phase_mask == 5:
        prefixes.append("P2")
    if phase_mask & 8:
        prefixes.append("P3")
    checks = re.sub(r"// (?:" + "|".join(prefixes) + r")(?=[-:])", "// CHECK", source)
    run_filecheck("musa_rlc", result, checks)
    if phase_mask == 5:
        run_filecheck("musa_rlc_trace", trace, source.replace("// TRACE", "// CHECK"))


@pytest.mark.parametrize("phase_mask", [1, 5, 15])
@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("warps,columns,stale", [(4, 256, False), (8, 512, False), (8, 256, False), (4, 256, True)])
def test_widened_loop_layout(phase_mask, dtype, warps, columns, stale, tmp_path):
    # Real GEMV topology: coalesced load, widening, broadcast multiply, loop
    # accumulation and row reduction. Keep small/FP32 positive controls.
    width = 4 if dtype == "f32" else 8
    compute_width = 2 if columns // warps == 64 else 1
    load = f"#ttg.blocked<{{sizePerThread = [1, {width}], threadsPerWarp = [1, 32], warpsPerCTA = [{warps}, 1], order = [1, 0]}}>"
    compute = f"#ttg.blocked<{{sizePerThread = [1, {compute_width}], threadsPerWarp = [1, 32], warpsPerCTA = [1, {warps}], order = [1, 0]}}>"
    original = compute.replace(f"[1, {compute_width}]", "[1, 1]", 1) if stale else compute
    tile = f"64x{columns}"
    ext = (
        f"%a = arith.extf %cvt : tensor<{tile}x{dtype}, #compute> to tensor<{tile}xf32, #compute>"
        if dtype != "f32" else ""
    )
    value = "%a" if ext else "%cvt"
    source = f"""
#load = {load}
#compute = {original}
#vector = {compute}
module attributes {{"ttg.num-warps" = {warps} : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "musa:31"}} {{
  tt.func @loop(%ptr: tensor<{tile}x!tt.ptr<{dtype}>, #load>, %vptr: tensor<1x{columns}x!tt.ptr<f32>, #vector>, %n: i32) -> tensor<64xf32, #ttg.slice<{{dim = 1, parent = #compute}}>> {{
    %zero = arith.constant dense<0.0> : tensor<{tile}xf32, #compute>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %acc = scf.for %i = %c0 to %n step %c1 iter_args(%carry = %zero) -> tensor<{tile}xf32, #compute> : i32 {{
      %loaded = tt.load %ptr : tensor<{tile}x!tt.ptr<{dtype}>, #load>
      %cvt = ttg.convert_layout %loaded : tensor<{tile}x{dtype}, #load> -> tensor<{tile}x{dtype}, #compute>
      {ext}
      %v = tt.load %vptr : tensor<1x{columns}x!tt.ptr<f32>, #vector>
      %vector = ttg.convert_layout %v : tensor<1x{columns}xf32, #vector> -> tensor<1x{columns}xf32, #compute>
      %b = tt.broadcast %vector : tensor<1x{columns}xf32, #compute> -> tensor<{tile}xf32, #compute>
      %product = arith.mulf {value}, %b : tensor<{tile}xf32, #compute>
      %next = arith.addf %carry, %product : tensor<{tile}xf32, #compute>
      scf.yield %next : tensor<{tile}xf32, #compute>
    }}
    %result = "tt.reduce"(%acc) <{{axis = 1 : i32}}> ({{
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }}) : (tensor<{tile}xf32, #compute>) -> tensor<64xf32, #ttg.slice<{{dim = 1, parent = #compute}}>>
    tt.return %result : tensor<64xf32, #ttg.slice<{{dim = 1, parent = #compute}}>>
  }}
}}
"""
    result = run_rlc(source, tmp_path, phase_mask)
    accumulator = re.search(r"scf.for[^\n]*tensor<" + tile + r"xf32, (#[\w]+)>", result).group(1)
    selected = re.search(re.escape(accumulator) + r" = (.*)", result).group(1)
    protected = dtype != "f32" and columns // warps == 64
    assert selected == (compute if protected else load)


@pytest.mark.parametrize("mma", ["wmma", "sqmma"])
@pytest.mark.parametrize("phase_mask", [8, 15])
@pytest.mark.parametrize("operand", ["address", "mask"])
@pytest.mark.parametrize("length", [8, 300])
def test_writeback_slice_limit(mma, phase_mask, operand, length, tmp_path):
    ty = "tensor<16x64xi32, #b>"
    chain = "\n".join(
        f"%v{i} = arith.addi %v{i - 1}, %seed : {ty}" for i in range(1, length + 1)
    )
    offset = f"%v{length}" if operand == "address" else "%v0"
    mask_value = f"%v{length}" if operand == "mask" else "%v0"
    source = f"""
#b = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}}>
#mma = #ttg.musa_{mma}<{{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}}>
module attributes {{"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "musa:31"}} {{
  tt.func @writeback(%acc: tensor<16x64xf32, #mma>, %base: !tt.ptr<f32>, %a: i32, %b: i32) {{
    %v0 = tt.splat %a : i32 -> {ty}
    %seed = tt.splat %b : i32 -> {ty}
    {chain}
    %bases = tt.splat %base : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>, #b>
    %ptrs = tt.addptr %bases, {offset} : tensor<16x64x!tt.ptr<f32>, #b>, {ty}
    %mask = arith.cmpi slt, {mask_value}, %seed : {ty}
    %value = ttg.convert_layout %acc : tensor<16x64xf32, #mma> -> tensor<16x64xf32, #b>
    %unused = tt.atomic_rmw fadd, acq_rel, gpu, %ptrs, %value, %mask : (tensor<16x64x!tt.ptr<f32>, #b>, tensor<16x64xf32, #b>, tensor<16x64xi1, #b>) -> tensor<16x64xf32, #b>
    tt.return
  }}
}}
"""
    result = run_rlc(source, tmp_path, phase_mask)
    assert result.count("tt.atomic_rmw") == 1
    assert result.count("ttg.convert_layout") == (1 if length > 256 else 0)


def reduction_writeback_source(dtype, num_warps):
    matrix = "32x128"
    row = "tensor<32xf32, #row>"
    index = "tensor<32xi32, #row>"
    loaded = "%loaded" if dtype == "f32" else "%extended"
    extension = "" if dtype == "f32" else (
        f"%extended = arith.extf %loaded : tensor<{matrix}x{dtype}, #b> "
        f"to tensor<{matrix}xf32, #b>"
    )
    source = f"""
#b = #ttg.blocked<{{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [{num_warps}, 1], order = [1, 0]}}>
#row = #ttg.slice<{{dim = 1, parent = #b}}>
#col = #ttg.slice<{{dim = 0, parent = #b}}>
module attributes {{"ttg.num-warps" = {num_warps} : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "musa:31"}} {{
  tt.func @reduce_rows(%input: !tt.ptr<{dtype}> {{tt.divisibility = 16 : i32}}, %values: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}, %indices: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}, %n: i32) {{
    %zero = arith.constant dense<0.0> : {row}
    %izero = arith.constant dense<0> : {index}
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    %r = tt.make_range {{start = 0 : i32, end = 32 : i32}} : {index}
    %c = tt.make_range {{start = 0 : i32, end = 128 : i32}} : tensor<128xi32, #col>
    %ce = tt.expand_dims %c {{axis = 0 : i32}} : tensor<128xi32, #col> -> tensor<1x128xi32, #b>
    %columns = tt.broadcast %ce : tensor<1x128xi32, #b> -> tensor<{matrix}xi32, #b>
    %base = tt.splat %input : !tt.ptr<{dtype}> -> tensor<{matrix}x!tt.ptr<{dtype}>, #b>
    %p = tt.addptr %base, %columns : tensor<{matrix}x!tt.ptr<{dtype}>, #b>, tensor<{matrix}xi32, #b>
    %results:2 = scf.for %i = %c0 to %n step %c128 iter_args(%carry = %zero, %ic = %izero) -> ({row}, {index}) : i32 {{
      %offset = tt.splat %i : i32 -> tensor<{matrix}xi32, #b>
      %next = tt.addptr %p, %offset : tensor<{matrix}x!tt.ptr<{dtype}>, #b>, tensor<{matrix}xi32, #b>
      %loaded = tt.load %next : tensor<{matrix}x!tt.ptr<{dtype}>, #b>
      {extension}
      %reduced:2 = "tt.reduce"({loaded}, %columns) <{{axis = 1 : i32}}> ({{
      ^bb0(%a: f32, %ia: i32, %b: f32, %ib: i32):
        %less = arith.cmpf olt, %a, %b : f32
        %v = arith.select %less, %a, %b : f32
        %idx = arith.select %less, %ia, %ib : i32
        tt.reduce.return %v, %idx : f32, i32
      }}) : (tensor<{matrix}xf32, #b>, tensor<{matrix}xi32, #b>) -> ({row}, {index})
      %update = arith.cmpf olt, %reduced#0, %carry : {row}
      %v = arith.select %update, %reduced#0, %carry : tensor<32xi1, #row>, {row}
      %idx = arith.select %update, %reduced#1, %ic : tensor<32xi1, #row>, {index}
      scf.yield %v, %idx : {row}, {index}
    }}
    %vp = tt.splat %values : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #row>
    %ip = tt.splat %indices : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #row>
    %vptr = tt.addptr %vp, %r : tensor<32x!tt.ptr<f32>, #row>, {index}
    %iptr = tt.addptr %ip, %r : tensor<32x!tt.ptr<i32>, #row>, {index}
    tt.store %vptr, %results#0 : tensor<32x!tt.ptr<f32>, #row>
    tt.store %iptr, %results#1 : tensor<32x!tt.ptr<i32>, #row>
    tt.return
  }}
}}
"""
    return source


@pytest.mark.parametrize(
    "dtype,phase_mask",
    [("f32", phase) for phase in (0, 1, 3, 5, 8, 15)]
    + [(dtype, phase) for dtype in ("f16", "bf16") for phase in (0, 5, 15)],
)
@pytest.mark.parametrize("num_warps", [4, 8, 16])
def test_consecutive_reduction_writeback(dtype, phase_mask, num_warps, tmp_path):
    result = run_rlc(reduction_writeback_source(dtype, num_warps), tmp_path, phase_mask)
    enabled = dtype == "f32" and phase_mask in (5, 15)
    assert ("sizePerThread = [2, 4]" in result) == (enabled and num_warps == 16)
    assert ("sizePerThread = [4, 4]" in result) == (enabled and num_warps == 8)
    assert "ttg.convert_layout" not in result


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize(
    "phase_mask,variant",
    [(phase, "matched") for phase in (0, 1, 3, 4, 5, 8, 15)]
    + [(phase, variant) for phase in (5, 15)
       for variant in ("float_output", "unaligned_mask", "wrong_layout")],
)
def test_narrow_reduction_writeback(dtype, phase_mask, variant, tmp_path):
    # Keep widening inside the loop and truncation outside it. A dynamic tail
    # mask without alignment proof must not be regrouped.
    warps, rows, columns, packed_rows = (4, 16, 256, 4) if dtype == "f16" else (8, 32, 128, 2)
    source = reduction_writeback_source(dtype, warps)
    source = source.replace("sizePerThread = [1, 4]", "sizePerThread = [1, 8]")
    if dtype == "f16":
        source = source.replace("32x", "16x").replace("end = 32 :", "end = 16 :")
        source = source.replace("128", "256")
    elif variant != "wrong_layout":
        source = source.replace("threadsPerWarp = [1, 32]", "threadsPerWarp = [2, 16]")
    if variant == "wrong_layout" and dtype == "f16":
        source = source.replace("sizePerThread = [1, 8]", "sizePerThread = [1, 4]")
    if variant != "float_output":
        source = source.replace("!tt.ptr<f32>", f"!tt.ptr<{dtype}>")
        source = source.replace(
            "    tt.store %vptr, %results#0",
            f"    %narrow = arith.truncf %results#0 : tensor<{rows}xf32, #row> to tensor<{rows}x{dtype}, #row>\n"
            "    tt.store %vptr, %narrow",
        )
    if variant == "unaligned_mask":
        source = source.replace(
            "    tt.store %vptr,",
            f"    %bound = tt.splat %n : i32 -> tensor<{rows}xi32, #row>\n"
            f"    %mask = arith.cmpi slt, %r, %bound : tensor<{rows}xi32, #row>\n"
            "    tt.store %vptr,",
        ).replace("tt.store %vptr, %narrow :", "tt.store %vptr, %narrow, %mask :")
    result = run_rlc(source, tmp_path, phase_mask)
    enabled = variant == "matched" and phase_mask in (5, 15)
    assert (f"sizePerThread = [{packed_rows}, 8]" in result) == enabled
    assert "ttg.convert_layout" not in result
    assert "arith.extf" in result
    assert ("arith.truncf" in result) == (variant != "float_output")


@pytest.mark.parametrize("dtype,packed_rows", [("f32", 8), ("f16", 2)])
@pytest.mark.parametrize("phase_mask", [0, 1, 3, 5, 8, 15])
@pytest.mark.parametrize("unaligned", [False, True])
def test_cross_warp_reduction_writeback(dtype, packed_rows, phase_mask, unaligned, tmp_path):
    # Four row warps and two reduction warps; retain the reduction warp tree.
    source = reduction_writeback_source(dtype, 8).replace("warpsPerCTA = [8, 1]", "warpsPerCTA = [4, 2]")
    source = source.replace("128", "256")
    if dtype == "f16":
        source = source.replace("!tt.ptr<f32>", "!tt.ptr<f16>")
        source = source.replace(
            "    tt.store %vptr, %results#0",
            "    %narrow = arith.truncf %results#0 : tensor<32xf32, #row> to tensor<32xf16, #row>\n"
            "    tt.store %vptr, %narrow",
        )
    if unaligned:
        source = source.replace("%indices: !tt.ptr<i32> {tt.divisibility = 16 : i32}",
                                "%indices: !tt.ptr<i32> {tt.divisibility = 1 : i32}")
    result = run_rlc(source, tmp_path, phase_mask)
    enabled = not unaligned and phase_mask in (5, 15)
    assert (f"sizePerThread = [{packed_rows}, 4]" in result) == enabled
    assert "warpsPerCTA = [4, 2]" in result
    assert "ttg.convert_layout" not in result


def _bounded_writeback_ir(proof):
 attrs=" {musa.rlc_positive_even_divisor = 4 : i32, musa.rlc_positive_even_block_upper_bound = 32 : i32}" if proof else ""
 return """
#b = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "musa:31"} {
 tt.func public @two_store(%inp: !tt.ptr<f32>, %out: !tt.ptr<f32>, %idx: !tt.ptr<i64>, %width: i32__ATTR__) {
  %pid = tt.get_program_id x : i32
  %c31 = arith.constant 31 : i32
  %c32 = arith.constant 32 : i32
  %n = arith.addi %width, %c31 : i32
  %blocks = arith.divsi %n, %c32 : i32
  %h = arith.divsi %pid, %blocks : i32
  %w = arith.remsi %pid, %blocks : i32
  %offset0 = arith.addi %h, %w : i32
  %offset = tt.splat %offset0 : i32 -> tensor<16x32xi32, #b>
  %inbase = tt.splat %inp : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>, #b>
  %ip = tt.addptr %inbase, %offset : tensor<16x32x!tt.ptr<f32>, #b>, tensor<16x32xi32, #b>
  %val = tt.load %ip : tensor<16x32x!tt.ptr<f32>, #b>
  %outbase = tt.splat %out : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>, #b>
  %op = tt.addptr %outbase, %offset : tensor<16x32x!tt.ptr<f32>, #b>, tensor<16x32xi32, #b>
  %idxbase = tt.splat %idx : !tt.ptr<i64> -> tensor<16x32x!tt.ptr<i64>, #b>
  %ixp = tt.addptr %idxbase, %offset : tensor<16x32x!tt.ptr<i64>, #b>, tensor<16x32xi32, #b>
  %bound = tt.splat %width : i32 -> tensor<16x32xi32, #b>
  %mask = arith.cmpi slt, %offset, %bound : tensor<16x32xi32, #b>
  %ival = arith.extsi %offset : tensor<16x32xi32, #b> to tensor<16x32xi64, #b>
  tt.store %op, %val, %mask : tensor<16x32x!tt.ptr<f32>, #b>
  tt.store %ixp, %ival, %mask : tensor<16x32x!tt.ptr<i64>, #b>
  tt.return
 }
}
""".replace('__ATTR__',attrs)

@pytest.mark.parametrize("mask", [4, 5, 6, 7, 12, 14, 15])
@pytest.mark.parametrize("proof", [False, True])
def test_bounded_writeback_proof(mask, proof, tmp_path):
    result = run_rlc(_bounded_writeback_ir(proof), tmp_path, mask)
    signature = next(line for line in result.splitlines() if "tt.func public" in line)
    assert ("tt.divisibility = 4" in signature) == (proof and mask & 5 == 5)

def test_bounded_codegen_cache_class_and_optouts():
    from triton.backends.mthreads.compiler import MUSABackend

    class Param:
        def __init__(self, name, constexpr=False):
            self.name = name
            self.is_constexpr = constexpr
            self.do_not_specialize = False
            self.do_not_specialize_on_alignment = False

    class Jit:
        cache_key = "bounded-test"
        src = """@triton.jit
def f(out_w, BLOCK_W, x, y):
    n = tl.cdiv(out_w, BLOCK_W)
    i = tl.arange(0, BLOCK_W)
    mask = i < out_w
    tl.store(x + i + out_w, n, mask)
    tl.store(y + i + out_w, n, mask)
"""

    params = [Param("out_w"), Param("BLOCK_W", True), Param("x"), Param("y")]
    jit = Jit()
    backend = MUSABackend(GPUTarget("musa", 31, 32))
    def specialize(width):
        bound = {"out_w": width, "BLOCK_W": 32, "x": None, "y": None}
        specs = [("i32", None), ("constexpr", 32), ("*fp32", None), ("*i64", None)]
        return backend._refine_bounded_writeback(bound, params, specs, jit)[0]
    assert specialize(28) == specialize(12) == ("i32", "RLC2B32D4")
    assert specialize(29) == specialize(36) == ("i32", None)
    params[0].do_not_specialize_on_alignment = True
    assert specialize(28) == ("i32", None)
    params[0].do_not_specialize_on_alignment = False
    assert specialize(28) == ("i32", "RLC2B32D4")


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mask", range(16))
def test_bounded_codegen_phase_gate(enabled, mask, monkeypatch):
    from triton.backends.mthreads.compiler import MUSABackend
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_ENHANCE", str(int(enabled)))
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_PHASE_MASK", str(mask))
    monkeypatch.setattr(MUSABackend, "_refine_bounded_writeback", lambda *args: [("i32", "REFINED")])
    backend = MUSABackend(GPUTarget("musa", 31, 32))
    monkeypatch.setattr(backend, "_uses_device_descriptor", lambda _: False)
    class Param:
        do_not_specialize = False
        do_not_specialize_on_alignment = False
    result = backend.refine_specialization({"x": 28}, [Param()], [("i32", None)])
    assert (result[0][1] == "REFINED") == (enabled and mask & 5 == 5)


@pytest.mark.parametrize("arch", [22, 30, 32])
def test_bounded_codegen_rejects_other_arches(arch, monkeypatch):
    from triton.backends.mthreads.compiler import MUSABackend
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_ENHANCE", "1")
    monkeypatch.setenv("FLAGTREE_MUSA_RLC_PHASE_MASK", "15")
    monkeypatch.setattr(MUSABackend, "_refine_bounded_writeback", lambda *args: [("i32", "REFINED")])
    backend = MUSABackend(GPUTarget("musa", arch, 32))
    monkeypatch.setattr(backend, "_uses_device_descriptor", lambda _: False)
    class Param:
        do_not_specialize = False
        do_not_specialize_on_alignment = False
    assert backend.refine_specialization({"x": 28}, [Param()], [("i32", None)])[0][1] is None


def test_bounded_codegen_proof_attributes():
    from triton.backends.mthreads.compiler import MUSABackend
    assert MUSABackend.parse_attr("RLC2B32D4") == [
        ["musa.rlc_positive_even_divisor", 4],
        ["musa.rlc_positive_even_block_upper_bound", 32],
    ]
    assert MUSABackend.parse_attr("RLC2B1025D4") == []
