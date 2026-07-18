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

from triton.language import core


@core.extern
def memrealtime(_semantic=None):
    """
    Returns a 64-bit real time-counter value
    """
    target_arch = _semantic.builder.options.arch
    if 'gfx11' in target_arch or 'gfx12' in target_arch:
        return core.inline_asm_elementwise(
            """
            s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)
            s_waitcnt lgkmcnt(0)
            """,
            "=r",
            [],
            dtype=core.int64,
            is_pure=False,
            pack=1,
            _semantic=_semantic,
        )
    else:
        return core.inline_asm_elementwise(
            """
            s_memrealtime $0
            s_waitcnt vmcnt(0)
            """,
            "=r",
            [],
            dtype=core.int64,
            is_pure=False,
            pack=1,
            _semantic=_semantic,
        )
