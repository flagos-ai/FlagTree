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

import triton
import triton.language as tl

from triton.backends.compiler import GPUTarget
from triton.knobs import CompileTimes
from triton.compiler.compiler import ASTSource, IRSource

from typing import Any, Union

import torch


@triton.jit
def cumsum_kernel(ptr):
    block = ptr + tl.arange(0, 4)
    x = tl.load(block)
    tl.store(block, tl.cumsum(x, 0))


def test_compile_stats(device: str, fresh_knobs: Any, fresh_triton_cache: str) -> None:
    captured: Union[tuple[Union[ASTSource, IRSource], dict[str, Any], dict[str, Any], CompileTimes, bool], None] = None

    def compile_listener(src: Union[ASTSource, IRSource], metadata: dict[str, str], metadata_group: dict[str, Any],
                         times: CompileTimes, cache_hit: bool) -> None:
        nonlocal captured
        assert captured is None
        captured = (src, metadata, metadata_group, times, cache_hit)

    fresh_knobs.compilation.listener = compile_listener

    x = torch.randn(4, device=device)
    cumsum_kernel[(1, )](x)

    assert captured is not None

    # No cache hit at first
    assert not captured[4]

    # Expected metadata
    assert len(captured[1]["hash"]) > 0
    assert isinstance(captured[1]["target"], GPUTarget)

    # It in fact did take some time to do compilation
    assert captured[3].ir_initialization > 0
    assert captured[3].total_lowering > 0
    assert captured[3].store_results > 0
    assert captured[3].total > 0

    # Now lets create a new instance of the same kernel to pick up cache_hit=True
    cumsum_kernel.device_caches.clear()
    captured = None
    cumsum_kernel[(1, )](x)

    assert captured is not None
    # Cache hit!
    assert captured[4]

    # Expected metadata
    assert len(captured[1]["hash"]) > 0
    assert isinstance(captured[1]["target"], GPUTarget)

    # It in fact did take some time to do compilation
    assert captured[3].ir_initialization > 0
    assert captured[3].total_lowering == 0
    assert captured[3].store_results == 0
    assert captured[3].total > 0
