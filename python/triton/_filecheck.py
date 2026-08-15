# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import functools
import os
import inspect
import subprocess
import tempfile

import triton
from triton.compiler import ASTSource, make_backend
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource
from triton.runtime.jit import create_function_from_signature
from triton._C.libtriton import ir
from triton._flagtree_spec import spec_call

# ===-----------------------------------------------------------------------===#
# filecheck_test
# ===-----------------------------------------------------------------------===#

# Stub target for testing the frontend.
# flagtree backend call specialization
stub_target = spec_call("spec_get_stub_target")
if not stub_target:
    stub_target = GPUTarget("cuda", 100, 32)

triton_dir = os.path.dirname(__file__)
filecheck_path = os.path.join(triton_dir, "FileCheck")


class MatchError(ValueError):

    def __init__(self, message, module_str):
        super().__init__(message)
        self.module_str = module_str

    def __str__(self):
        return f"{super().__str__()}\n{self.module_str}"


def run_filecheck(name, module_str, check_template):
    with tempfile.TemporaryDirectory() as tempdir:
        temp_module = os.path.join(tempdir, "module")
        with open(temp_module, "w") as temp:
            temp.write(module_str)

        temp_expected = os.path.join(tempdir, "expected")
        with open(temp_expected, "w") as temp:
            temp.write(check_template)

        try:
            subprocess.check_output(
                [filecheck_path, temp_expected, "--input-file", temp_module, "--dump-input-context=50"],
                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            decoded = error.output.decode('unicode_escape')
            raise ValueError(decoded)


def run_parser(kernel_fn, args=(), kwargs={}, target=stub_target):
    if "sanitize_overflow" not in kwargs:
        kwargs = dict(kwargs)
        kwargs["sanitize_overflow"] = False
    backend = make_backend(target)
    binder = create_function_from_signature(
        kernel_fn.signature,
        kernel_fn.params,
        backend,
    )

    bound_args, specialization, options = binder(*args, **kwargs)
    options, signature, constexprs, attrs = kernel_fn._pack_args(backend, kwargs, bound_args, specialization, options)
    source_cls = GluonASTSource if kernel_fn.is_gluon() else ASTSource
    src = source_cls(kernel_fn, signature, constexprs, attrs)

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    module = spec_call("filecheck_make_ir", src, target, options, codegen_fns, module_map, context)
    if module is None:
        module = backend.make_ir(src, options, codegen_fns, module_map, context)
    return module


def run_filecheck_test(kernel_fn):
    assert isinstance(kernel_fn, triton.runtime.JITFunction)
    check_template = inspect.getsource(kernel_fn.fn)
    if check_template is None:
        raise ValueError("kernel function must have a docstring with FileCheck template")
    mlir_module = run_parser(kernel_fn)

    run_filecheck("placeholder", mlir_module.str_nodebug(), check_template)


def filecheck_test(fn):

    @functools.wraps(fn)
    def test_fn():
        run_filecheck_test(fn)

    return test_fn
