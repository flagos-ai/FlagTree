# auto shared layout
"""Automatic shared charts follow producers/consumers through closed views."""
import re

import pytest
from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)

from triton.backends.hcu.compiler import HIPBackend

import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler import ASTSource


@triton.jit
def _auto_shared_layout_frontend_alloc(C, MODE: tl.constexpr):
    if MODE == 2:
        chart: tl.constexpr = tle.gpu.swizzled_shared_layout(8, 1, 8, [1, 2, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                                                             [2, 1, 0])
        owner = tle.gpu.alloc([2, 64, 64], tl.bfloat16, layout=chart)
    elif MODE == 1:
        owner = tle.gpu.alloc([2, 64, 64], tl.bfloat16, nv_mma_shared_layout=False)
    else:
        owner = tle.gpu.alloc([2, 64, 64], tl.bfloat16)
    owner.slot(0).store(tl.full((64, 64), 1, tl.bfloat16))
    value = owner.slot(0).load()
    offsets = tl.arange(0, 64)[:, None] * 64 + tl.arange(0, 64)[None, :]
    tl.store(C + offsets, value)


@pytest.mark.parametrize("backend_name", ["", "nvidia", "hcu"])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_allocator_policy_does_not_require_runtime_device_discovery(monkeypatch, backend_name, mode):
    import triton._flagtree_backend as configured
    from triton.compiler.code_generator import CodeGenerator

    def unexpected_device_discovery():
        raise AssertionError("Shared allocator must not discover an active GPU")

    # The compilation target is already known. Pin the pre-existing TLE
    # capability lookup separately; only the allocator policy is under test.
    monkeypatch.setattr(CodeGenerator, "_get_tle_backend_name", lambda self: "hcu")
    monkeypatch.setattr(configured, "FLAGTREE_BACKEND", backend_name)
    monkeypatch.setattr(configured, "get_active_backend_name", unexpected_device_discovery, raising=False)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    target = GPUTarget("hip", "gfx936", 64)
    backend = HIPBackend(target)
    backend.load_dialects(context)
    options = backend.parse_options({"num_warps": 4})
    source = ASTSource(_auto_shared_layout_frontend_alloc, {"C": "*bf16", "MODE": "constexpr"}, {"MODE": mode})
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)
    text = str(module)
    assert ("hcu.auto_shared_layout" in text) == (backend_name == "hcu" and mode == 0)
    assert ("nvmma_shared" in text) == (backend_name != "hcu" and mode == 0)


def _auto_shared_layout_source(dtype="bf16", rank=3, k=64, negative=None):
    half = k // 2
    plane = f"!ttg.memdesc<{k}x64x{dtype}, #plane, #smem, mutable>"
    root = plane if rank == 2 else f"!ttg.memdesc<2x{k}x64x{dtype}, #root, #smem, mutable>"
    view = f"!ttg.memdesc<{half}x64x{dtype}, #plane, #smem, mutable, {k}x64>"
    pointers = f"tensor<{k}x64x!tt.ptr<{dtype}>, #blocked>"
    loaded = f"tensor<{k}x64x{dtype}, #blocked>"
    a = f"tensor<128x{half}x{dtype}, #a>"
    b = f"tensor<{half}x64x{dtype}, #b>"
    marker = "" if negative == "explicit" else " {hcu.auto_shared_layout = true}"
    body = f"%owner = ttg.local_alloc{marker} : () -> {root}\n"
    if negative == "alias":
        body += f"%alias = ttg.memdesc_reinterpret %owner : {root} -> {root}\n"
    if rank == 3:
        body += f"%window = ttg.memdesc_index %owner[%zero] : {root} -> {plane}\n"
    else:
        body += f"%window = ttg.memdesc_subslice %owner[0, 0] : {root} -> {plane}\n"
    body += f"%pointers = tt.splat %p : !tt.ptr<{dtype}> -> {pointers}\n"
    body += f"%value = tt.load %pointers : {pointers}\n"
    store = "%value"
    if negative == "non_global":
        body += f"%zeros = arith.constant dense<0.0> : {loaded}\n"
        store = "%zeros"
    body += f"ttg.local_store {store}, %window : {loaded} -> {plane}\n"
    if negative == "producer_conflict":
        reverse_ptrs = f"tensor<{k}x64x!tt.ptr<{dtype}>, #reverse>"
        reverse_value = f"tensor<{k}x64x{dtype}, #reverse>"
        body += f"%p2 = tt.splat %p : !tt.ptr<{dtype}> -> {reverse_ptrs}\n"
        body += f"%v2 = tt.load %p2 : {reverse_ptrs}\n"
        body += f"ttg.local_store %v2, %window : {reverse_value} -> {plane}\n"
    body += f"%low = ttg.memdesc_subslice %window[0, 0] : {plane} -> {view}\n"
    body += f"%read = ttg.local_load %low : {view} -> {b}\n"
    if negative == "read_escape":
        body += f"%escaped = ttg.local_alloc %read : ({b}) -> !ttg.memdesc<{half}x64x{dtype}, #plane, #smem>\n"
    body += f"%out = tt.dot %lhs, %read, %acc : {a} * {b} -> tensor<128x64xf32, #mma>\n"
    return f"""
#blocked = #ttg.blocked<{{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}}>
#reverse = #ttg.blocked<{{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}}>
#mma = #ttg.amd_mfma<{{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true, mmacLayout = 1}}>
#a = #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 8}}>
#b = #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = 8}}>
#plane = #ttg.swizzled_shared<{{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}}>
#root = #ttg.swizzled_shared<{{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}}>
#smem = #ttg.shared_memory
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx936", "ttg.threads-per-warp" = 64 : i32}} {{
  tt.func public @automatic(%p: !tt.ptr<{dtype}>, %lhs: {a}, %acc: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma> {{
    %zero = arith.constant 0 : i32
    {body}
    tt.return %out : tensor<128x64xf32, #mma>
  }}
}}
"""


def _auto_shared_layout_lower(tmp_path, **kwargs):
    path = tmp_path / "auto.mlir"
    path.write_text(_auto_shared_layout_source(**kwargs))
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    before = str(module)
    manager = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_optimize_dot_operands(manager, "gfx936")
    manager.run(module, "auto_shared_layout")
    return before, str(module)


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("k", [32, 64])
def test_infers_mfma_chart_over_full_allocation(tmp_path, dtype, rank, k):
    before, after = _auto_shared_layout_lower(tmp_path, dtype=dtype, rank=rank, k=k)
    assert "hcu.auto_shared_layout" not in after
    phase = 64 // k
    assert f"vec = 8, perPhase = {phase}, maxPhase = {k // 8}, order = [0, 1]" in after
    if rank == 3:
        assert f"vec = 8, perPhase = {phase}, maxPhase = {k // 8}, order = [1, 2, 0]" in after
    for operation in ("tt.load ", "ttg.local_load ", "ttg.local_store ", "tt.dot "):
        assert before.count(operation) == after.count(operation) == 1
    assert after.index("tt.load ") < after.index("ttg.local_store ") < after.index("ttg.local_load ")
    assert f"mutable, {k}x64>" in after


@pytest.mark.parametrize("negative", ["explicit", "alias", "non_global", "producer_conflict", "read_escape"])
@pytest.mark.parametrize("rank", [2, 3])
def test_unproven_or_explicit_chart_keeps_original_types(tmp_path, negative, rank):
    before, after = _auto_shared_layout_lower(tmp_path, negative=negative, rank=rank)
    assert "vec = 8" not in after
    assert ("hcu.auto_shared_layout" in after) == (negative != "explicit")
    # The ordinary canonicalizer can renumber SSA and location aliases.
    assert re.findall(r"ttg.local_load .*? : (.*?) loc\(", before) == re.findall(r"ttg.local_load .*? : (.*?) loc\(",
                                                                                 after)


# local load view layout


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("fanout", [False, True])
@pytest.mark.parametrize("through_loop", [False, True])
def test_shared_read_view_conversion_preserves_token_and_fanout(tmp_path, dtype, fanout, through_loop):
    shared = "!ttg.memdesc<16x256x" + dtype + ", #shared, #smem, mutable>"
    original = "tensor<16x256x" + dtype + ", #blocked>"
    split = "tensor<16x8x16x2x" + dtype + ", #split>"
    transposed = "tensor<16x8x2x16x" + dtype + ", #transposed>"
    output = "tensor<16x8x2x16x" + dtype + ", #linear>"
    returns = "(" + output + ", " + original + ")" if fanout else output
    operands = "%out, %value" if fanout else "%out"
    return_types = output + ", " + original if fanout else output
    read = f"%value = ttg.local_load %mem token %ready : {shared} -> {original}"
    if through_loop:
        read = f"""
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %initial = ttg.local_load %mem token %ready : {shared} -> {original}
    %value = scf.for %i = %zero to %iterations step %one
        iter_args(%held = %initial) -> ({original}) : i32 {{
      %next = ttg.local_load %mem token %ready : {shared} -> {original}
      scf.yield %next : {original}
    }}
"""
    source = f"""
#blocked = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}}>
#split = #ttg.blocked<{{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 2, 16, 2], warpsPerCTA = [1, 4, 1, 1], order = [3, 2, 1, 0]}}>
#transposed = #ttg.blocked<{{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 2, 2, 16], warpsPerCTA = [1, 4, 1, 1], order = [2, 3, 1, 0]}}>
#linear = #ttg.linear<{{register = [[1, 0, 0, 0], [2, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 4, 0, 0]], lane = [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 8], [4, 0, 0, 0], [8, 0, 0, 0]], warp = [[0, 2, 0, 0], [0, 0, 0, 0]], block = []}}>
#shared = #ttg.shared_linear<{{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [1, 0], [2, 0], [4, 0], [8, 0]]}}, alignment = 16>
#smem = #ttg.shared_memory
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx936", "ttg.threads-per-warp" = 64 : i32}} {{
  tt.func public @view(%mem: {shared}, %ready: !ttg.async.token, %iterations: i32) -> {returns} {{
    {read}
    %split = tt.reshape %value : {original} -> {split}
    %transposed = tt.trans %split {{order = array<i32: 0, 1, 3, 2>}} : {split} -> {transposed}
    %out = ttg.convert_layout %transposed : {transposed} -> {output}
    tt.return {operands} : {return_types}
  }}
}}
"""
    path = tmp_path / "view.mlir"
    path.write_text(source)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    before = str(module)
    manager = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_mls_lowering_pass(manager)
    manager.run(module, "shared_read_view_layout")
    after = str(module)
    assert after.count("ttg.local_load ") == (2 if through_loop else 1)
    if through_loop:
        assert "scf.for" in after
    token = r"ttg.local_load (\S+) token (\S+)"
    assert re.findall(token, before) == re.findall(token, after)
    assert ("ttg.convert_layout " in after) == fanout


# register descriptor views


def _register_descriptor_views_lower(tmp_path, dtype, negative=None, pre_split=False):
    root = f"tensor<128x64x{dtype}, #blocked>"
    reshaped = f"tensor<128x2x32x{dtype}, #reshaped>"
    transposed = f"tensor<128x32x2x{dtype}, #transposed>"
    half = f"tensor<128x32x{dtype}, #blocked>"
    dot = f"tensor<128x32x{dtype}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 8}}>>"
    other = dot.replace("kWidth = 8", "kWidth = 4") if negative == "incompatible" else dot
    volatile = ", isVolatile = true" if negative == "volatile" else ""
    extra = f"tt.store %pointers, %value : tensor<128x64x!tt.ptr<{dtype}>, #blocked>" if negative == "fanout" else ""
    text = f'''#blocked = #ttg.blocked<{{sizePerThread = [1, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}}>
#reshaped = #ttg.blocked<{{sizePerThread = [1, 2, 2], threadsPerWarp = [4, 1, 16], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}}>
#transposed = #ttg.blocked<{{sizePerThread = [1, 2, 2], threadsPerWarp = [4, 16, 1], warpsPerCTA = [4, 1, 1], order = [1, 2, 0]}}>
#mma = #ttg.amd_mfma<{{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true, mmacLayout = 1}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"}} {{
  tt.func public @views(%p: !tt.ptr<{dtype}>, %q: !tt.ptr<{dtype}>) {{
    ttg.warp_specialize() attributes {{requestedRegisters = array<i32: 32>}}
    default {{
      %pointers = tt.splat %p : !tt.ptr<{dtype}> -> tensor<128x64x!tt.ptr<{dtype}>, #blocked>
      %extent = arith.constant 128 : i32
      %columns = arith.constant 64 : i32
      %stride = arith.constant 64 : i64
      %unit_stride = arith.constant 1 : i64
      %origin = arith.constant 0 : i32
      %enabled = arith.constant 1 : i32
      %value = tt.matrix_load %p, shape = [%extent, %columns], strides = [%stride, %unit_stride], tensorShape = [128, 64], indices = [%origin, %origin, %enabled] {{hcu.structured_register, tt.load.async = false{volatile}}} : <{dtype}> -> {root}
      %zero = arith.constant 0 : i32
      "hcug.cta_barrier"() : () -> ()
      %reshape = tt.reshape %value : {root} -> {reshaped}
      %transpose = tt.trans %reshape {{order = array<i32: 0, 2, 1>}} : {reshaped} -> {transposed}
      %lo, %hi = tt.split %transpose : {transposed} -> {half}
      %a = ttg.convert_layout %lo : {half} -> {dot}
      %b = ttg.convert_layout %hi : {half} -> {other}
      %qa = tt.splat %q : !tt.ptr<{dtype}> -> tensor<128x32x!tt.ptr<{dtype}>, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 8}}>>
      %qb = tt.splat %q : !tt.ptr<{dtype}> -> tensor<128x32x!tt.ptr<{dtype}>, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {4 if negative == 'incompatible' else 8}}}>>
      tt.store %qa, %a : tensor<128x32x!tt.ptr<{dtype}>, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 8}}>>
      tt.store %qb, %b : tensor<128x32x!tt.ptr<{dtype}>, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {4 if negative == 'incompatible' else 8}}}>>
      {extra}
      ttg.warp_yield
    }}
    partition0() num_warps(4) {{ ttg.warp_return }} : () -> ()
    tt.return
  }}
}}'''
    # The real typed CTA barrier is retained as an ordering witness.
    text = text.replace('"hcug.cta_barrier"() : () -> ()', 'ttg.local_barrier {hcu.cta_scope}')
    if pre_split:
        chart = "#target = #ttg.linear<{register = [[0, 0, 1], [0, 1, 0], [0, 2, 0], [0, 4, 0], [64, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 8, 0], [0, 16, 0]], warp = [[16, 0, 0], [32, 0, 0]], block = []}>\n"
        text = chart + text
        target = f"tensor<128x32x2x{dtype}, #target>"
        text = text.replace(
            f"%lo, %hi = tt.split %transpose : {transposed} -> {half}",
            f"%view = ttg.convert_layout %transpose : {transposed} -> {target}\n"
            f"      %lo, %hi = tt.split %view : {target} -> {dot}")
        text = text.replace(f"%lo : {half} ->", f"%lo : {dot} ->")
        text = text.replace(f"%hi : {half} ->", f"%hi : {dot} ->")
    path = tmp_path / "views.mlir"
    path.write_text(text)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    before = str(module)
    manager = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_optimize_dot_operands(manager, "gfx936")
    manager.run(module, "register_load_views")
    return before, str(module)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_split_consumers_infer_one_full_load(tmp_path, dtype):
    before, after = _register_descriptor_views_lower(tmp_path, dtype)
    assert before.count("tt.matrix_load ") == after.count("tt.matrix_load ") == 1
    assert re.search(r"tt.matrix_load .*tensor<128x64x" + dtype + r", #linear", after)
    assert after.index("tt.matrix_load ") < after.index("ttg.local_barrier")
    assert "ttg.local_alloc" not in after


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("negative", ["incompatible", "fanout", "volatile"])
def test_view_load_falls_back_without_changing_memory(tmp_path, dtype, negative):
    before, after = _register_descriptor_views_lower(tmp_path, dtype, negative)
    assert before.count("tt.matrix_load ") == after.count("tt.matrix_load ") == 1
    load = next(line for line in after.splitlines() if "tt.matrix_load " in line)
    assert "#blocked" in load
    assert after.index("tt.matrix_load ") < after.index("ttg.local_barrier")


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_conversion_before_split_retargets_original_descriptor(tmp_path, dtype):
    before, after = _register_descriptor_views_lower(tmp_path, dtype, pre_split=True)
    assert before.count("tt.matrix_load ") == after.count("tt.matrix_load ") == 1
    assert re.search(r"tt.matrix_load .*tensor<128x64x" + dtype + r", #linear", after)
    assert after.index("tt.matrix_load ") < after.index("ttg.local_barrier")
    assert "ttg.local_alloc" not in after


# alt matrix read offsets

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource


@g.constexpr_function
def _alt_matrix_read_offsets_fragment_layout(waves, transpose):
    registers = [[1, 0], [2, 0], [4, 0], [0, 1], [0, 32], [0, 64]]
    lanes = [[0, 2], [0, 4], [0, 8], [0, 16], [8, 0], [16, 0]]
    if transpose:
        registers = [basis[::-1] for basis in registers]
        lanes = [basis[::-1] for basis in lanes]
    return gl.DistributedLinearLayout(registers, lanes, [[0, 0]] * (waves.bit_length() - 1), [],
                                      [128, 32] if transpose else [32, 128])


@g.jit
def _alt_matrix_read_offsets_read_slice(A, Out, START: gl.constexpr, TRANSPOSE: gl.constexpr = False,
                                        WAVES: gl.constexpr = 4):
    blocked: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [WAVES, 1], [1, 0])
    shared: gl.constexpr = gl.SwizzledSharedLayout(32, 1, 4, [1, 0])
    fragment: gl.constexpr = _alt_matrix_read_offsets_fragment_layout(WAVES, TRANSPOSE)
    rows = gl.arange(0, 64, layout=gl.SliceLayout(1, blocked))
    cols = gl.arange(0, 128, layout=gl.SliceLayout(0, blocked))
    owner = gl.allocate_shared_memory(A.dtype.element_ty, [64, 128], shared,
                                      gl.load(A + rows[:, None] * 128 + cols[None, :]))
    window = owner.slice(START, 32)
    if TRANSPOSE:
        window = window.permute((1, 0))
    values = window.load(fragment)
    rows = gl.arange(0, 128 if TRANSPOSE else 32, layout=gl.SliceLayout(1, fragment))
    cols = gl.arange(0, 32 if TRANSPOSE else 128, layout=gl.SliceLayout(0, fragment))
    if TRANSPOSE:
        gl.store(Out + cols[None, :] * 128 + rows[:, None], values)
    else:
        gl.store(Out + rows[:, None] * 128 + cols[None, :], values)


@pytest.mark.parametrize("start", [0, 32])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("waves", [1, 2, 4, 8])
def test_alt_matrix_read_reuses_base_for_aligned_subslice(start, transpose, waves):
    source = GluonASTSource(
        _alt_matrix_read_offsets_read_slice,
        {"A": "*bf16", "Out": "*bf16", "START": "constexpr", "TRANSPOSE": "constexpr", "WAVES": "constexpr"},
        {"START": start, "TRANSPOSE": transpose, "WAVES": waves},
        attrs={(i, ): [("tt.divisibility", 16)]
               for i in range(2)},
    )
    kernel = triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options={"num_warps": waves})
    reads = re.findall(
        r"^\s*ds_read_m32x16_b16_alt\s+v\[\d+:\d+\],\s+(v\d+)(?:\s+offset:(\d+))?",
        kernel.asm["amdgcn"],
        re.M,
    )
    by_base = {}
    for base, offset in reads:
        by_base.setdefault(base, set()).add(int(offset or 0))
    assert len(reads) == 8
    assert len(by_base) == 4 and all(offsets == {0, 1024} for offsets in by_base.values())


@pytest.mark.parametrize("start", [0, 32])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("waves", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
def test_transposed_physical_alt_map_changed_replay(start, transpose, waves, dtype_name):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a Hygon device")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx936":
        pytest.skip("requires gfx936")
    source = torch.empty((64, 128), dtype=getattr(torch, dtype_name), device="cuda")
    storage = torch.full((32 * 128 + 256, ), 7, dtype=source.dtype, device=source.device)
    output = storage[128:-128].view(32, 128)
    _alt_matrix_read_offsets_read_slice[(1, )](source, output, start, transpose, waves, num_warps=waves)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _alt_matrix_read_offsets_read_slice[(1, )](source, output, start, transpose, waves, num_warps=waves)
    for _ in range(3):
        source.copy_(torch.randn_like(source))
        graph.replay()
        torch.testing.assert_close(output, source[start:start + 32], atol=0, rtol=0)
        assert torch.all(storage[:128] == 7) and torch.all(storage[-128:] == 7)


@pytest.mark.parametrize("start", [0, 32])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("waves", [1, 2, 4, 8])
def test_alt_transpose_isomorphism_on_device(start, transpose, waves):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires an HCU device")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx936":
        pytest.skip("requires gfx936")
    source = torch.randn((64, 128), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((32, 128), dtype=source.dtype, device=source.device)
    _alt_matrix_read_offsets_read_slice[(1, )](source, output, start, transpose, waves, num_warps=waves)
    torch.testing.assert_close(output, source[start:start + 32], rtol=0, atol=0)


def test_paired_mmac_layout_roundtrip():
    from triton.experimental.gluon.language.amd import AMDMFMALayout
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    libtriton.hcu.load_dialects(context)
    builder = libtriton.gluon_ir.GluonOpBuilder(context)
    module = builder.create_module()
    fn = builder.get_or_insert_function(module, "layout_roundtrip", builder.get_function_ty([], []), "public", False)
    module.push_back(fn)
    builder.set_insertion_point_to_end(fn.add_entry_block())
    layout = AMDMFMALayout(3, [16, 16, 16], True, [4, 2], tiles_per_warp=[1, 2], mmac_layout=5)
    ty = builder.clone_tensor_type_with_encoding(builder.get_block_ty(builder.get_float_ty(), [32, 64]),
                                                 layout._to_ir(builder))
    tensor = builder.create_poison(ty)
    restored = builder.get_gluon_layout_from_tensor(tensor)
    assert restored == layout
    assert restored.mangle() == layout.mangle()
    assert hash(restored) == hash(layout)
