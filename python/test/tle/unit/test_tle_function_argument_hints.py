"""TLE semantic hints on function parameters survive frontend and inlining."""
import re

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir, passes
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend

pytestmark = pytest.mark.skipif(not hasattr(ir.builder, "ensure_ttg_layout_attrs"), reason="requires __TLE__ build")
_TARGET = GPUTarget("cuda", 90, 32)


def _module(kernel, signature):
    backend = make_backend(_TARGET)
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ASTSource(kernel, signature, attrs={}).make_ir(
        _TARGET,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    return module, context


@triton.jit(noinline=True)
def _pointer_copy(pointer, output):
    tl.store(output, tl.load(pointer))


@triton.jit
def _direct_pointer(pointer, output):
    _pointer_copy(tl.multiple_of(pointer, 256), output)


@triton.jit
def _integer_argument(value, output):
    tl.store(output, tl.multiple_of(value, 256))


@triton.jit
def _tensor_argument(value):
    return tl.max_constancy(tl.max_contiguous(tl.multiple_of(value, 8), 8), 1)


@triton.jit
def _tensor_root(output):
    value = _tensor_argument(tl.arange(0, 128))
    tl.store(output + tl.arange(0, 128), value)


@pytest.mark.parametrize(
    "kernel,signature,function,hints",
    [
        (_direct_pointer, {"pointer": "*bf16", "output": "*bf16"}, "_direct_pointer", ("tt.divisibility", )),
        (_integer_argument, {"value": "i64", "output": "*i64"}, "_integer_argument", ("tt.divisibility", )),
        (_tensor_root, {"output": "*i32"}, "_tensor_argument", ("tt.divisibility", "tt.contiguity", "tt.constancy")),
    ],
)
def test_function_argument_hints_are_saved(kernel, signature, function, hints):
    module, _ = _module(kernel, signature)
    declaration = next(line for line in module.str_nodebug().splitlines() if "tt.func" in line and function in line)
    assert all(hint in declaration for hint in hints), declaration


@triton.jit
def _promised_copy(pointer, output):
    offsets = tl.arange(0, 1024)
    tl.store(output + offsets, tl.load(tl.multiple_of(pointer, 256) + offsets))


@triton.jit
def _nested_copy(pointer, output):
    _promised_copy(pointer, output)


@triton.jit
def _mixed_root(pointer, other, output):
    tl.store(output, tl.load(pointer + 1))
    _nested_copy(pointer, output + 1)
    _promised_copy(other, output + 1025)


def test_inlined_hints_remain_local_to_each_call():
    module, context = _module(_mixed_root, {"pointer": "*bf16", "other": "*bf16", "output": "*bf16"})
    pm = ir.pass_manager(context)
    passes.common.add_inliner(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(module, "tle-function-argument-hints")
    text = module.str_nodebug()
    declaration = next(line for line in text.splitlines() if "tt.func public" in line)
    assert "tt.divisibility" not in declaration
    carriers = [line for line in text.splitlines() if "tt.bitcast" in line and "tt.divisibility" in line]
    assert len(carriers) == 2, text
    assert not re.search(r"@.*(?:_promised_copy|_nested_copy)", text)


def test_inlined_tensor_hints_survive_canonicalization():
    module, context = _module(_tensor_root, {"output": "*i32"})
    pm = ir.pass_manager(context)
    passes.common.add_inliner(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(module, "tle-tensor-argument-hints")
    text = module.str_nodebug()
    assert any("tt.bitcast" in line and all(hint in line
                                            for hint in ("tt.divisibility", "tt.contiguity", "tt.constancy"))
               for line in text.splitlines()), text


@triton.jit
def _hinted_root(pointer, output):
    _promised_copy(pointer, output)


@triton.jit
def _caller_promise(pointer, output):
    offsets = tl.arange(0, 1024)
    tl.store(output + offsets, tl.load(tl.multiple_of(pointer, 256) + offsets))


def test_inlined_pointer_hint_preserves_vector_load_without_pointer_cast():
    compiled = [
        triton.compile(
            ASTSource(fn, {"pointer": "*bf16", "output": "*bf16"}, attrs={}),
            target=_TARGET,
            options={"num_warps": 4},
        ) for fn in (_hinted_root, _caller_promise)
    ]
    loads = [[line.strip()
              for line in kernel.asm["ptx"].splitlines()
              if re.search(r"\bld.global", line)]
             for kernel in compiled]
    assert loads[0] == loads[1]
    assert any("ld.global.v4.b32" in line for line in loads[0])
    assert all(not re.search(r"= bitcast ptr", kernel.asm["llir"]) for kernel in compiled)


@triton.jit(noinline=True)
def _noinline_promise(pointer, output):
    offsets = tl.arange(0, 1024)
    tl.store(output + offsets, tl.load(tl.multiple_of(pointer, 256) + offsets))


@triton.jit
def _noinline_root(pointer, output):
    _noinline_promise(pointer, output)


def test_noinline_callee_hint_survives_caller_axis_inference():
    compiled = triton.compile(
        ASTSource(_noinline_root, {"pointer": "*bf16", "output": "*bf16"}, attrs={}),
        target=_TARGET,
        options={"num_warps": 4},
    )
    assert "ld.global.v4.b32" in compiled.asm["ptx"]
