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
"""MetaX shared-memory layout policy for TLE allocations."""

from triton._C import libtriton
from triton._common_ir import ENABLED as COMMON_IR_ENABLED
from triton._flagtree_backend import FLAGTREE_BACKEND

from .. import types as tle


def use_auto_shared_layout(layout):
    """Allow CommonIR to select a shared layout when the user omitted it."""
    is_metax = FLAGTREE_BACKEND == "metax" or hasattr(libtriton, "metax")
    return is_metax and COMMON_IR_ENABLED and layout is None


def select_default_smem_layout(shape):
    """Return a Python default and no IR layout, allowing CommonIR to infer it."""
    layout = tle.swizzled_shared_layout.make_default(rank=len(shape))
    return layout, None
