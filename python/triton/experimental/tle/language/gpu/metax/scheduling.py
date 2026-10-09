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
"""Explicit MetaX instruction scheduling controls."""

import triton.language.core as tl


@tl.builtin
def sched_bound(_semantic=None):
    """Emit a MetaX instruction scheduling boundary."""
    _semantic.builder.create_maca_sched_bound()


@tl.builtin
def iglp(config_0=0, config_1=-1, config_2=-1, config_3=-1, config_4=-1, config_5=-1, config_6=-1, config_7=-1,
         _semantic=None):
    """Set MetaX instruction-group scheduling with eight compile-time integers."""
    configs = [
        tl._unwrap_if_constexpr(value)
        for value in (config_0, config_1, config_2, config_3, config_4, config_5, config_6, config_7)
    ]
    for index, value in enumerate(configs):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"iglp config_{index} must be a compile-time integer")
    _semantic.builder.create_maca_iglp(*configs)
