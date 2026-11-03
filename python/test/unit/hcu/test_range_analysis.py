"""CPU-only regressions for HCU scalar integer-range analysis."""
import pytest

from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler bindings are required", allow_module_level=True)

compiler = pytest.importorskip("triton.backends.hcu.compiler")


@pytest.mark.parametrize("lower,other,upper,step,limit", [
    (0, 8, 16, 2, 4),
    (1, 10, 19, 3, 3),
    (-8, 0, 8, 2, 4),
])
def test_dynamic_lower_bound_keeps_variable_trip_count(tmp_path, lower, other, upper, step, limit):
    # One input runs more iterations than limit; the other does not. Using
    # the maximum lower bound to bound tripcount incorrectly proves true.
    assert (upper - lower + step - 1) // step > limit
    assert (upper - other + step - 1) // step <= limit
    source = tmp_path / "dynamic_lower.mlir"
    source.write_text(f'''module attributes {{"ttg.num-warps" = 4 : i32}} {{
  tt.func @dynamic_lower(%choose_lower: i1) -> i1 {{
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %low = arith.constant {lower} : i32
    %other = arith.constant {other} : i32
    %upper = arith.constant {upper} : i32
    %step = arith.constant {step} : i32
    %limit = arith.constant {limit} : i32
    %begin = arith.select %choose_lower, %low, %other : i32
    %count = scf.for %iv = %begin to %upper step %step iter_args(%acc = %zero) -> (i32) : i32 {{
      %next = arith.addi %acc, %one : i32
      scf.yield %next : i32
    }}
    %result = arith.cmpi sle, %count, %limit : i32
    tt.return %result : i1
  }}
}}
''')
    ir, passes, hcu = libtriton.ir, libtriton.passes, libtriton.hcu
    context = ir.context()
    ir.load_dialects(context)
    compiler.HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    pm = ir.pass_manager(context)
    hcu.passes.ttgpuir.add_fold_true_cmpi(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(module, "test_dynamic_lower_range")
    result = str(module)
    assert "scf.for" in result
    assert "arith.cmpi sle" in result
