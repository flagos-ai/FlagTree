import os

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import pytest
import triton
import triton.language as tl
from triton._C import libtriton
from pathlib import Path

if not hasattr(libtriton, "musa"):
    pytest.skip("musa backend not built in libtriton", allow_module_level=True)

from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton._C.libtriton import ir


def _get_musa_backend():
    if "musa" not in backends:
        pytest.skip("musa backend not discovered")
    target = GPUTarget("musa", "ph1", 32)
    return backends["musa"].compiler(target)


def _compile_to_llir(fn, signature, constexprs=None):
    target = GPUTarget("musa", "ph1", 32)
    backend = _get_musa_backend()

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    options = backend.parse_options({})
    module_map = backend.get_module_map()
    codegen_fns = backend.get_codegen_implementation(options)
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs or {})

    ttir = src.make_ir(target, options, codegen_fns, module_map, context)
    stages = {}
    backend.add_stages(stages, options, src.language)
    meta = {}
    ttir = stages["ttir"](ttir, meta)
    ttgir = stages["ttgir"](ttir, meta)
    llir = stages["llir"](ttgir, meta)
    return llir, meta


def test_musa_056_default_libdevice_path(fresh_knobs):
    backend = _get_musa_backend()
    from triton.backends.musa import compiler as musa_compiler

    with fresh_knobs.musa.scope():
        del fresh_knobs.musa.libdevice_path
        options = backend.parse_options({})

    expected = Path(musa_compiler.__file__).resolve().parent / "lib" / "libdevice.31.bc"
    assert Path(dict(options.extern_libs)["libdevice"]).resolve() == expected


def test_musa_056_libdevice_path_override(fresh_knobs, tmp_path):
    backend = _get_musa_backend()
    override = tmp_path / "libdevice.override.bc"
    override.write_bytes(b"")

    with fresh_knobs.musa.scope():
        fresh_knobs.musa.libdevice_path = str(override)
        options = backend.parse_options({})

    assert dict(options.extern_libs)["libdevice"] == str(override)


def test_musa_056_cast_compile_only():

    @triton.jit
    def kernel_cast(inp, out):
        offs = tl.arange(0, 64)
        x = tl.load(inp + offs)
        y = x.to(tl.float16)
        z = y.to(tl.float32)
        tl.store(out + offs, z)

    llir, _ = _compile_to_llir(kernel_cast, {"inp": "*fp32", "out": "*fp32"})
    assert "fptrunc" in llir
    assert "fpext" in llir


def test_musa_056_chained_dot_compile_only():

    @triton.jit
    def kernel_chained_dot(out):
        a = tl.full((16, 16), 1.0, tl.float16)
        b = tl.full((16, 16), 2.0, tl.float16)
        c = tl.dot(a, b)
        d = tl.dot(c.to(tl.float16), a)
        row = tl.sum(d, axis=1)
        offs = tl.arange(0, 16)
        tl.store(out + offs, row.to(tl.float32))

    llir, meta = _compile_to_llir(kernel_chained_dot, {"out": "*fp32"})
    assert "target datalayout" in llir
    assert "shared" in meta


@pytest.mark.parametrize("input_precision", ["bf16x3", "bf16x6"])
def test_musa_056_bf16xN_dot_compile_only(input_precision):

    @triton.jit
    def kernel_bf16_dot(out, INPUT_PRECISION: tl.constexpr):
        a = tl.full((16, 16), 1.0, tl.float32)
        b = tl.full((16, 16), 2.0, tl.float32)
        c = tl.dot(a, b, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        row = tl.sum(c, axis=1)
        offs = tl.arange(0, 16)
        tl.store(out + offs, row)

    llir, _ = _compile_to_llir(
        kernel_bf16_dot,
        {"out": "*fp32", "INPUT_PRECISION": "constexpr"},
        constexprs={"INPUT_PRECISION": input_precision},
    )
    assert "target datalayout" in llir


def test_musa_056_functional_vecmat_compile_only():

    @triton.jit
    def kernel_vecmat(inp, out):
        offs = tl.arange(0, 16)
        vec = tl.load(inp + offs)
        mat = tl.full((16, 16), 0.5, tl.float32)
        prod = mat * tl.expand_dims(vec, 0)
        red = tl.sum(prod, axis=1)
        tl.store(out + offs, red)

    llir, _ = _compile_to_llir(kernel_vecmat, {"inp": "*fp32", "out": "*fp32"})
    assert "fadd" in llir
    assert "fmul" in llir


def test_musa_056_constexpr_annotation_compile_only():

    @triton.jit
    def kernel_constexpr(inp, out, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(inp + offs)
        tl.store(out + offs, x)

    llir, _ = _compile_to_llir(kernel_constexpr, {"inp": "*fp32", "out": "*fp32", "BLOCK": "constexpr"},
                               constexprs={"BLOCK": 32})
    assert "target datalayout" in llir
