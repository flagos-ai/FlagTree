# Copyright 2026 FlagOS Contributors
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

import os

import pytest
import triton
from triton._C import libtriton
from triton._C.libtriton import ir
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


def require_mthreads_backend():
    if not hasattr(libtriton, "mthreads"):
        pytest.skip("mthreads backend not built in libtriton")
    if "mthreads" not in backends:
        pytest.skip("mthreads backend not discovered")


def require_mthreads_libtriton():
    if not hasattr(libtriton, "mthreads"):
        pytest.skip("mthreads backend not built in libtriton", allow_module_level=True)


def musa_target():
    arch = os.environ.get("TRITON_OVERRIDE_ARCH") or os.environ.get("TRITON_MUSA_ARCH") or "ph1"
    return GPUTarget("musa", arch, 32)


def mthreads_backend():
    require_mthreads_backend()
    target = musa_target()
    return target, backends["mthreads"].compiler(target)


def compile_musa(fn, signature, constexprs=None):
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs or {})
    return triton.compile(src, target=musa_target())


def compile_to_ttir(fn, signature, constexprs=None):
    target, backend = mthreads_backend()

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    options = backend.parse_options({})
    module_map = backend.get_module_map()
    codegen_fns = backend.get_codegen_implementation(options)
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs or {})
    return src.make_ir(target, options, codegen_fns, module_map, context).str_nodebug()
