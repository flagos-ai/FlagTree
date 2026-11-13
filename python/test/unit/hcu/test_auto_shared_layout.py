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
def frontend_alloc(C, MODE: tl.constexpr):
    if MODE == 2:
        chart: tl.constexpr = tle.gpu.swizzled_shared_layout(
            8, 1, 8, [1, 2, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 1, 0])
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
    monkeypatch.setattr(configured, "get_active_backend_name", unexpected_device_discovery)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    target = GPUTarget("hip", "gfx936", 64)
    backend = HIPBackend(target)
    backend.load_dialects(context)
    options = backend.parse_options({"num_warps": 4})
    source = ASTSource(frontend_alloc, {"C": "*bf16", "MODE": "constexpr"}, {"MODE": mode})
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context)
    text = str(module)
    assert ("hcu.auto_shared_layout" in text) == (backend_name == "hcu" and mode == 0)
    assert ("nvmma_shared" in text) == (backend_name != "hcu" and mode == 0)


def source(dtype="bf16", rank=3, k=64, negative=None):
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


def lower(tmp_path, **kwargs):
    path = tmp_path / "auto.mlir"
    path.write_text(source(**kwargs))
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
    before, after = lower(tmp_path, dtype=dtype, rank=rank, k=k)
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
    before, after = lower(tmp_path, negative=negative, rank=rank)
    assert "vec = 8" not in after
    assert ("hcu.auto_shared_layout" in after) == (negative != "explicit")
    # The ordinary canonicalizer can renumber SSA and location aliases.
    assert re.findall(r"ttg.local_load .*? : (.*?) loc\(", before) == re.findall(
        r"ttg.local_load .*? : (.*?) loc\(", after)
