"""Retired marker attributes cannot elide generic warp-specialization scratch."""
import pytest

from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler bindings are required", allow_module_level=True)

compiler = pytest.importorskip("triton.backends.hcu.compiler")


def allocation_size(tmp_path, extra_attributes, terminal=True):
    source = tmp_path / "warp-scratch.mlir"
    extra = ", " + extra_attributes if extra_attributes else ""
    source.write_text('''module attributes {"ttg.num-warps" = 4 : i32,
      "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32,
      ttg.target = "hip:gfx936"EXTRA} {
      tt.func public @capture(%input: i32, %out: !tt.ptr<i32>) {
        ttg.warp_specialize(%input)
        default {
          ttg.warp_yield
        }
        partition0(%captured: i32) num_warps(4) {
          ttg.warp_return
        } : (i32) -> ()
        EXIT
        tt.return
      }
    }
    '''.replace("EXTRA", extra).replace("EXIT", "" if terminal else "tt.store %out, %input : !tt.ptr<i32>"))
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    compiler.HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(source), context)
    pm = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_allocate_shared_memory(pm)
    pm.run(module, "generic_ws_scratch_without_issued_raw_proof")
    return module.get_int_attr("ttg.shared")


@pytest.mark.parametrize("attributes", [
    "ttg.hcu.cta_ws_protocol = 1 : i32",
    "ttg.hcu.cta_ws_verified",
    "ttg.hcu.cta_ws_protocol = 1 : i32, ttg.hcu.cta_ws_verified",
    "ttg.hcu.cta_ws_protocol = 2 : i32, ttg.hcu.cta_ws_verified",
    "ttg.hcu.raw_window_ws = 1 : i32",
    "ttg.hcu.raw_window_ws = 1 : i32, ttg.hcu.raw_window_ws.internal_witness",
])
@pytest.mark.parametrize("terminal", [True, False])
def test_unissued_markers_do_not_override_actual_domain(tmp_path, attributes, terminal):
    normal = allocation_size(tmp_path, "", terminal)
    # Actual terminal uniform-capture IR uses direct partition branches and
    # needs no scratch. A store after WS defeats that proof and must retain it.
    assert normal == 0 if terminal else normal > 0
    assert allocation_size(tmp_path, attributes, terminal) == normal
