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

from triton.language import core as tl
from triton.language.core import builtin
from triton._C.libtriton import proton as triton_proton
from triton.language.semantic import TritonSemantic
from triton.experimental.gluon.language._semantic import GluonSemantic

from .flags import flags

_ALL_SEMANTICS = {
    "triton": TritonSemantic,
    "gluon": GluonSemantic,
}
"""
By default **only Gluon** semantic is enabled.
Instrumenting kernels written in Triton DSL is disable because Triton's higher-level IR undergoes
aggressive compiler rewrites (loop pipelining, instruction re-ordering, IR duplication, etc.).
These transformations can invalidate naïve instrumentation and lead to misleading results.
"""
_SEMANTICS = {_ALL_SEMANTICS["gluon"]}


def _check_supported_semantic(semantic):
    if not isinstance(semantic, tuple(_SEMANTICS)):
        raise TypeError(f"Unsupported semantic type: {type(semantic)}. "
                        f"Supported semantics are: {_SEMANTICS}")


def enable_semantic(semantic_name: str):
    _SEMANTICS.add(_ALL_SEMANTICS[semantic_name])


def disable_semantic(semantic_name: str):
    _SEMANTICS.remove(_ALL_SEMANTICS[semantic_name])


def record(is_start: tl.constexpr, scope_name: tl.constexpr, semantic):
    if not flags.instrumentation_on:
        return
    _check_supported_semantic(semantic)
    is_start = tl._unwrap_if_constexpr(is_start)
    scope_name = tl._unwrap_if_constexpr(scope_name)
    return tl.tensor(triton_proton.create_proton_record(semantic.builder, is_start, scope_name), tl.void)


@builtin
def enter_scope(name: tl.constexpr, _semantic=None):
    record(is_start=True, scope_name=name, semantic=_semantic)


@builtin
def exit_scope(name: tl.constexpr, _semantic=None):
    record(is_start=False, scope_name=name, semantic=_semantic)


class scope:

    def __init__(self, name: str, _semantic=None):
        self.name = name
        self.semantic = _semantic

    def __enter__(self):
        enter_scope(self.name, _semantic=self.semantic)

    def __exit__(self, exc_type, exc_value, traceback):
        exit_scope(self.name, _semantic=self.semantic)
