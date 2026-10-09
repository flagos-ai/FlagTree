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
"""Explicit MetaX synchronization primitives."""

import triton.language.core as tl


@tl.builtin
def gvm_arrive(num, _semantic=None):
    """Emit a GVM wait with a compile-time outstanding-request threshold."""
    num = tl._unwrap_if_constexpr(num)
    if isinstance(num, bool) or not isinstance(num, int) or num < 0:
        raise ValueError("gvm_arrive num must be a compile-time non-negative integer")
    _semantic.builder.create_gvm_arrive(num)


@tl.builtin
def barrier(_semantic=None):
    """Emit the MetaX instruction barrier."""
    _semantic.builder.create_maca_barrier()


@tl.builtin
def barrier_shared(_semantic=None):
    """Emit the MetaX shared-memory barrier."""
    _semantic.builder.create_maca_barrier_shared()
