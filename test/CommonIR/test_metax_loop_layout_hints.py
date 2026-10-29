"""Explicit layout constraints on control-flow values survive optimization."""

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend

from test_metax_set_layout import operations, lower, optimize, pytestmark


@triton.jit
def loop_boundary(src, dst, steps):
    i = tl.arange(0, 1024)
    x = tl.load(src + i)
    for _ in range(steps):
        y = tle.gpu.set_layout(x, tle.gpu.BlockEncoding([2], [64], [4], [0]))
        x = y + y
    tl.store(dst + i, x)


def loop_frontend():
    backend = make_backend(GPUTarget("maca", 80, 64))
    options = backend.parse_options({"num_warps": 4})
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(loop_boundary, {"src": "*fp32", "dst": "*fp32", "steps": "i32"}, {})
    module = source.make_ir(backend.target, options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    module.context = context
    return module, backend, options


def test_loop_argument_keeps_explicit_layout(tmp_path):
    module, backend, options = loop_frontend()
    lower(module, backend, options)
    for stage in ("converted", "optimized"):
        if stage == "optimized":
            optimize(module)
        (tmp_path / f"loop-{stage}.ttgir").write_text(module.str_nodebug())
        assert "tile.explicit_encoding.0" in module.str_nodebug()
        add, = operations(module, "arith.addf")
        assert "sizePerThread = [2]" in str(add.get_result(0).get_type())
        assert not operations(module, "tile.set_layout")


def test_normal_pipeline_keeps_constraint_until_llvm(tmp_path):
    module, backend, options = loop_frontend()
    metadata = {}
    backend.make_ttir(module, metadata, options)
    backend.make_ttgir(module, metadata, options, 80)
    add, = operations(module, "arith.addf")
    assert "sizePerThread = [2]" in str(add.get_result(0).get_type())
    assert "tile.explicit_encoding.0" in module.str_nodebug()
    (tmp_path / "loop-final.ttgir").write_text(module.str_nodebug())
    llvm_dialect = backend.make_mlir(module, metadata, options, 80)
    assert "llvm.func" in llvm_dialect
    assert "tile.explicit_" not in llvm_dialect


def parse_ir(tmp_path, body):
    module, _, _ = loop_frontend()
    path = tmp_path / "boundary.mlir"
    path.write_text("""#L = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#S = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
""" + body + "\n}")
    context = module.context
    module = libtriton.ir.parse_mlir_module(str(path), context)
    module.context = context
    return module


def canonicalize(module):
    pm = libtriton.ir.pass_manager(module.context)
    libtriton.passes.common.add_canonicalizer(pm)
    pm.run(module, "canonicalize_hint")
    assert module.verify()


def test_removing_leading_loop_argument_keeps_constraint(tmp_path):
    module = parse_ir(tmp_path, """
  tt.func @boundary(%x: tensor<1024xf32, #L>, %n: i32) -> tensor<1024xf32, #L> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%dead=%c0, %a=%x) -> (i32, tensor<1024xf32, #L>) : i32 {
      %fixed = ttg.convert_layout %a {tile.explicit_encoding.0 = #L} : tensor<1024xf32, #L> -> tensor<1024xf32, #L>
      %b = arith.addf %fixed, %fixed : tensor<1024xf32, #L>
      scf.yield %dead, %b : i32, tensor<1024xf32, #L>
    }
    tt.return %r#1 : tensor<1024xf32, #L>
  }
""")
    canonicalize(module)
    loop, = operations(module, "scf.for")
    assert loop.get_num_results() == 1
    assert "tile.explicit_encoding.0" in module.str_nodebug()
    optimize(module)
    add, = operations(module, "arith.addf")
    assert "sizePerThread = [2]" in str(add.get_result(0).get_type())
    (tmp_path / "loop-reindexed.ttgir").write_text(module.str_nodebug())


@pytest.mark.parametrize("placement", ["argument", "result"])
def test_while_boundaries_survive_canonicalization(tmp_path, placement):
    before = ("%fixed = ttg.convert_layout %a {tile.explicit_encoding.0 = #L} : tensor<1024xf32, #L> -> tensor<1024xf32, #L>"
              if placement == "argument" else "")
    arg = "%fixed" if placement == "argument" else "%a"
    after = ("%fixed = ttg.convert_layout %r {tile.explicit_encoding.0 = #L} : tensor<1024xf32, #L> -> tensor<1024xf32, #L>"
             if placement == "result" else "")
    result = "%fixed" if placement == "result" else "%r"
    module = parse_ir(tmp_path, f"""
  tt.func @boundary(%x: tensor<1024xf32, #L>, %p: !tt.ptr<i32>) -> tensor<1024xf32, #L> {{
    %r = scf.while (%a = %x) : (tensor<1024xf32, #L>) -> tensor<1024xf32, #L> {{
      {before}
      %c0 = arith.constant 0 : i32
      %c = tt.load %p : !tt.ptr<i32>
      %cond = arith.cmpi ne, %c, %c0 : i32
      scf.condition(%cond) {arg} : tensor<1024xf32, #L>
    }} do {{
    ^bb0(%a: tensor<1024xf32, #L>):
      %b = arith.addf %a, %a : tensor<1024xf32, #L>
      scf.yield %b : tensor<1024xf32, #L>
    }}
    {after}
    tt.return {result} : tensor<1024xf32, #L>
  }}
""")
    canonicalize(module)
    assert "tile.explicit_encoding.0" in module.str_nodebug()


def test_canonicalization_transfers_hint_to_regular_producer(tmp_path):
    module = parse_ir(tmp_path, """
  tt.func @boundary(%p: tensor<1024x!tt.ptr<f32>, #L>) -> tensor<1024xf32, #L> {
    %x = tt.load %p : tensor<1024x!tt.ptr<f32>, #L>
    %y = ttg.convert_layout %x {tile.explicit_encoding.0 = #L} : tensor<1024xf32, #L> -> tensor<1024xf32, #L>
    tt.return %y : tensor<1024xf32, #L>
  }
""")
    canonicalize(module)
    assert not operations(module, "ttg.convert_layout")
    assert any("tt.load" in line and "tile.explicit_encoding.0" in line
               for line in module.str_nodebug().splitlines())


def test_conversion_chain_does_not_bypass_fixed_value(tmp_path):
    module = parse_ir(tmp_path, """
  tt.func @boundary(%x: tensor<1024xf32, #S>) -> tensor<1024xf32, #S> {
    %y = ttg.convert_layout %x {tile.explicit_encoding.0 = #L} : tensor<1024xf32, #S> -> tensor<1024xf32, #L>
    %z = ttg.convert_layout %y : tensor<1024xf32, #L> -> tensor<1024xf32, #S>
    tt.return %z : tensor<1024xf32, #S>
  }
""")
    canonicalize(module)
    assert "tile.explicit_encoding.0" in module.str_nodebug()
    assert len(operations(module, "ttg.convert_layout")) == 2


def test_consumer_does_not_bypass_explicit_conversion(tmp_path):
    module = parse_ir(tmp_path, """
  tt.func @boundary(%x: tensor<1024xi32, #S>) -> tensor<256xi32, #L> {
    %y = ttg.convert_layout %x {tile.explicit_encoding.0 = #L} : tensor<1024xi32, #S> -> tensor<1024xi32, #L>
    %z = tt.histogram %y : tensor<1024xi32, #L> -> tensor<256xi32, #L>
    tt.return %z : tensor<256xi32, #L>
  }
""")
    canonicalize(module)
    assert "tile.explicit_encoding.0" in module.str_nodebug()
    assert len(operations(module, "ttg.convert_layout")) == 1


@pytest.mark.parametrize("steps", [0, 1, 4])
def test_loop_boundary_gpu(steps):
    import torch
    source = torch.arange(1024, dtype=torch.float32, device="cuda")
    output = torch.empty_like(source)
    kernel = loop_boundary[(1,)](source, output, steps, num_warps=4)
    torch.testing.assert_close(output, source * (2 ** steps), rtol=0, atol=0)
    assert "tile.set_layout" not in kernel.asm["ttgir"]
    assert "tile.explicit_" not in kernel.asm["mlir"]
