"""Ordinary target operations remain usable after retiring custom protocols."""
import pytest
import triton
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU bindings required", allow_module_level=True)
from triton.backends.hcu.compiler import HIPBackend
from triton.experimental.gluon.language.hcu import warp_group_id


def parse(text, tmp_path):
    path = tmp_path / "target.mlir"
    path.write_text(text)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    return libtriton.ir.parse_mlir_module(str(path), context), context


@pytest.mark.parametrize("name", ["raw_window_copy", "raw_window_read", "raw_window_wait"])
def test_retired_raw_ops_are_unregistered(name, tmp_path, capfd):
    # A quoted generic syntax avoids relying on a retired custom parser.
    with pytest.raises(RuntimeError):
        parse('module { "hcug.' + name + '"() : () -> () }', tmp_path)
    assert "unregistered operation" in capfd.readouterr().err


def mls_argument(version):
    return '''#layout = #ttg.mls_shared<{opIdx = 0, mlsTile = [64, 32],
      elemBitWidth = 16, alt2Kind = 0, version = VERSION, order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32,
    "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @test(%tile: !ttg.memdesc<64x32xbf16, #layout,
      #ttg.shared_memory, mutable>) { tt.return }
}'''.replace("VERSION", str(version))


@pytest.mark.parametrize("version", [0, 256, 257, 258, 259])
def test_retired_physical_versions_fail_before_vendor_lowering(version, tmp_path, capfd):
    module, context = parse(mls_argument(version), tmp_path)
    pm = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_to_llvmir(pm, "gfx936", False)
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        pm.run(module, "retired_physical_encoding")
    assert "retired custom HCU MLS physical encoding" in capfd.readouterr().err


def test_vendor_mls_layout_parser_remains_available(tmp_path):
    module, _ = parse(mls_argument(1), tmp_path)
    assert "version = 1" in str(module)


@g.jit
def group_query(O, SIZE: gl.constexpr):
    gl.store(O, warp_group_id(SIZE))


def compile_query(size):
    source = GluonASTSource(group_query, {"O": "*i32", "SIZE": "constexpr"}, {"SIZE": size})
    return triton.compile(source, target=GPUTarget("hip", "gfx936", 64),
                          options=dict(num_warps=4, num_stages=1))


@pytest.mark.parametrize("size", [1, 2, 4])
def test_partition_relative_query_without_raw_api(size):
    kernel = compile_query(size)
    assert "hcug.warp_group_id" in kernel.asm["ttgir"]
    assert "raw_window" not in kernel.asm["ttgir"]


@pytest.mark.parametrize("size", [8, 16])
def test_query_cannot_exceed_actual_partition(size, capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        compile_query(size)
    assert "within this partition" in capfd.readouterr().err
