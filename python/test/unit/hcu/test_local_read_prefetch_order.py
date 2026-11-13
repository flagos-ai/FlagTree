"""Keep loop-carried register prefetch ahead of independent current math."""
import re

import pytest
from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)

from triton.backends.hcu.compiler import HIPBackend

LAYOUTS = r"""
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true, mmacLayout = 1}>
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 8], [4, 16], [8, 0], [16, 0], [32, 0]]}, alignment = 16>
#shared1 = #ttg.shared_linear<{offset = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1], [8, 2], [16, 4], [0, 8], [0, 16], [0, 32], [0, 64]]}, alignment = 16>
#smem = #ttg.shared_memory
"""


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize("case", ["eligible", "one_side", "chain", "noncarried",
                                  "nohint", "wrong_target", "early_use", "barrier",
                                  "wait", "unknown_call"])
def test_prefetched_reads_precede_math_without_moving_memory(tmp_path, dtype, case):
    a = f"tensor<64x32x{dtype}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 8}}>>"
    b = f"tensor<32x128x{dtype}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = 8}}>>"
    acc = "tensor<64x128xf32, #mma>"
    ma = f"!ttg.memdesc<64x32x{dtype}, #shared, #smem, mutable>"
    mb = f"!ttg.memdesc<32x128x{dtype}, #shared1, #smem, mutable>"
    hint = "" if case == "nohint" else ", ttg.hcu.local_prefetch = 1 : i32"
    target = "gfx90a" if case == "wrong_target" else "gfx936"
    left = "%initialA" if case == "noncarried" else "%a"
    right = "%initialB" if case in ("noncarried", "one_side") else "%b"
    math = f"%current = tt.dot {left}, {right}, %held : {a} * {b} -> {acc}"
    result = "%current"
    if case == "chain":
        math += f"\n%second = tt.dot {left}, {right}, %current : {a} * {b} -> {acc}"
        result = "%second"
    if case == "early_use":
        math += f"\n%observed = arith.addf %current, %held : {acc}"
        result = "%observed"
    boundary = {
        "barrier": "ttg.local_barrier",
        "wait": "%wait = ttg.async_wait %ready {num = 0 : i32}",
        "unknown_call": "tt.call @opaque() : () -> ()",
    }.get(case, "")
    token = "%wait" if case == "wait" else "%ready"
    text = LAYOUTS + f"""
module attributes {{ttg.target = "hip:{target}", "ttg.num-ctas" = 1 : i32,
    "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32{hint}}} {{
 tt.func private @opaque()
 tt.func public @prefetch(%memA: {ma}, %memB: {mb},
     %initialA: {a}, %initialB: {b}, %initial: {acc},
     %ready: !ttg.async.token, %count: i32) -> ({acc}, {a}, {b}) {{
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %out:3 = scf.for %i = %zero to %count step %one
      iter_args(%held = %initial, %a = %initialA, %b = %initialB)
      -> ({acc}, {a}, {b}) : i32 {{
   {math}
   {boundary}
   ttg.local_store %initialA, %memA : {a} -> {ma}
   %nextA = ttg.local_load %memA token {token} : {ma} -> {a}
   ttg.local_store %initialB, %memB : {b} -> {mb}
   %nextB = ttg.local_load %memB token {token} : {mb} -> {b}
   scf.yield {result}, %nextA, %nextB : {acc}, {a}, {b}
  }}
  tt.return %out#0, %out#1, %out#2 : {acc}, {a}, {b}
 }}
}}
"""
    path = tmp_path / "prefetch.mlir"
    path.write_text(text)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    before = str(module)
    manager = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_mls_lowering_pass(manager)
    manager.run(module, "loop_carried_local_prefetch_order")
    after = str(module)
    dots = [x.start() for x in re.finditer(r"= tt.dot ", after)]
    reads = [x.start() for x in re.finditer(r"= ttg.local_load ", after)]
    assert len(dots) == (2 if case == "chain" else 1)
    assert len(reads) == 2
    if case in ("eligible", "chain"):
        assert all(dot > reads[-1] for dot in dots)
    elif case == "one_side":
        assert dots[0] > reads[0]
    else:
        assert dots[0] < reads[0]
    memory = r"ttg\.(?:local_store|local_load|local_barrier|async_wait)|tt\.call"
    assert re.findall(memory, before) == re.findall(memory, after)
    assert "rocdl.sched.barrier" not in after
