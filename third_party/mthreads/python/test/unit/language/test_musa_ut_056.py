import os

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import pytest
import triton
import triton.language as tl
from triton._C import libtriton
from pathlib import Path

# The in-tree MTT plugin is registered as ``mthreads`` even though its Python
# backend package is exposed as ``musa``.  Accept both names so compiler tests
# actually run against the supported in-tree build instead of being skipped.
if not (hasattr(libtriton, "musa") or hasattr(libtriton, "mthreads")):
    pytest.skip("mthreads/musa backend not built in libtriton", allow_module_level=True)

from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton._C.libtriton import ir


@pytest.mark.parametrize(
    ("arch", "expected"),
    [("ph1", 31), ("ph1s", 32), ("mp31", 31), ("mp_31", 31),
     ("mp32", 32), ("mp_32", 32)],
)
def test_musa_056_arch_alias_capability(arch, expected):
    from triton.backends.mthreads.compiler import _capability_from_arch

    assert _capability_from_arch(arch) == expected


def _get_musa_backend():
    backend_name = "musa" if "musa" in backends else "mthreads"
    if backend_name not in backends:
        pytest.skip("musa backend not discovered")
    # ``mthreads`` is the plugin registry key, while the backend contract and
    # driver still use the canonical target name ``musa``.
    target = GPUTarget("musa", "ph1", 32)
    return backends[backend_name].compiler(target)


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
    from triton.backends.mthreads import compiler as musa_compiler

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


def test_musa_056_can_disable_max_ilp_scheduler():
    backend = _get_musa_backend()
    from triton.backends.mthreads.compiler import _llc_extra_options

    enabled = backend.parse_options({"enable_backend_opt": True})
    enabled_args = _llc_extra_options({"uses_sqmma": True}, enabled)
    assert "-misched=mtgpu-max-ilp" in enabled_args

    disabled = backend.parse_options(
        {
            "enable_backend_opt": True,
            "disable_max_ilp_scheduler": True,
        }
    )
    disabled_args = _llc_extra_options({"uses_sqmma": True}, disabled)
    assert "-misched=mtgpu-max-ilp" not in disabled_args
    assert "-mtgpu-opt-level=1" in disabled_args


def test_musa_056_llc_user_scheduler_option_overrides_backend_default():
    backend = _get_musa_backend()
    from triton.backends.mthreads.compiler import _llc_extra_options

    options = backend.parse_options(
        {
            "enable_backend_opt": True,
            "llc_options": "-misched=mtgpu-max-occupancy -mtgpu-opt-level=2",
        }
    )
    args = _llc_extra_options({"uses_sqmma": True}, options)
    assert args.count("-misched=mtgpu-max-occupancy") == 1
    assert not any(arg == "-misched=mtgpu-max-ilp" for arg in args)
    assert args.count("-mtgpu-opt-level=2") == 1
    assert not any(arg == "-mtgpu-opt-level=1" for arg in args)


def test_musa_056_llc_user_opt_level_replaces_default():
    backend = _get_musa_backend()
    from triton.backends.mthreads.compiler import _llc_extra_options, _llc_opt_level

    default = backend.parse_options({})
    assert _llc_opt_level(default) == "-O2"

    options = backend.parse_options({"llc_options": "-O3 -O1"})
    assert _llc_opt_level(options) == "-O1"
    args = _llc_extra_options({"uses_sqmma": True}, options)
    assert not any(arg in ("-O0", "-O1", "-O2", "-O3") for arg in args)


def test_musa_056_llc_register_allocator_failure_is_recoverable(monkeypatch, tmp_path):
    """Known MTT llc allocator aborts must be prunable by autotuning."""
    from triton.backends.mthreads import compiler as musa_compiler
    from triton.runtime.errors import PTXASError

    class FailedProcess:
        returncode = 134
        stdout = ""
        stderr = "LLVM ERROR: no registers from class available to allocate"

    monkeypatch.setattr(musa_compiler.subprocess, "run", lambda *args, **kwargs: FailedProcess())
    with pytest.raises(PTXASError, match="MTGPU register allocation"):
        musa_compiler._run_tool_command(
            "llc",
            ["llc", "kernel.ll", "-O2"],
            repro_dir=tmp_path,
        )


def test_musa_056_unknown_llc_error_is_not_swallowed(monkeypatch, tmp_path):
    from triton.backends.mthreads import compiler as musa_compiler

    class FailedProcess:
        returncode = 134
        stdout = ""
        stderr = "LLVM ERROR: unexpected backend failure"

    monkeypatch.setattr(musa_compiler.subprocess, "run", lambda *args, **kwargs: FailedProcess())
    with pytest.raises(RuntimeError, match="`llc` failed with error code 134"):
        musa_compiler._run_tool_command(
            "llc",
            ["llc", "kernel.ll", "-O2"],
            repro_dir=tmp_path,
        )


def test_musa_056_llc_post_ra_allocator_crash_is_recoverable(monkeypatch, tmp_path):
    from triton.backends.mthreads import compiler as musa_compiler
    from triton.runtime.errors import PTXASError

    class FailedProcess:
        returncode = 139
        stdout = ""
        stderr = "MTGPU Post-RA Internal Registers Related Optimization"

    monkeypatch.setattr(musa_compiler.subprocess, "run", lambda *args, **kwargs: FailedProcess())
    with pytest.raises(PTXASError, match="MTGPU register allocation"):
        musa_compiler._run_tool_command(
            "llc",
            ["llc", "kernel.ll", "-O0"],
            repro_dir=tmp_path,
        )


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
