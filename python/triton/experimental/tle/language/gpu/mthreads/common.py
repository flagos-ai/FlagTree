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

import os

import triton.language.core as tl

try:
    from triton._flagtree_backend import FLAGTREE_BACKEND
except ModuleNotFoundError:
    FLAGTREE_BACKEND = os.environ.get("FLAGTREE_BACKEND", "")


def _has_mthreads_libtriton() -> bool:
    try:
        from triton._C import libtriton
    except ImportError:
        return False
    return hasattr(libtriton, "mthreads")


def enabled() -> bool:
    return FLAGTREE_BACKEND == "mthreads" or _has_mthreads_libtriton()


@tl.builtin
def local_barrier(_semantic=None) -> None:
    """Synchronize a CTA and make shared-memory accesses visible."""
    builder = _semantic.builder
    if not hasattr(builder, "create_mthreads_local_barrier"):
        raise RuntimeError(
            "mthreads TLE local barrier requires a newer native libtriton"
        )
    builder.create_mthreads_local_barrier()
