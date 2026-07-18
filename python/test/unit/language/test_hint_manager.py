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

from types import SimpleNamespace

from triton.backends.nvidia.nvidia_hint_handler import NvidiaHintHandler


class ParseMustNotRun:

    def parse(self):
        raise AssertionError("hint lookup must not reparse the JIT function")


def test_nvidia_hint_lookup_uses_codegen_attached_map():
    code_generator = SimpleNamespace(
        flagtree_line_hints={17: "cache_global"},
        jit_fn=ParseMustNotRun(),
    )
    node = SimpleNamespace(lineno=17)

    assert NvidiaHintHandler.get_node_hints(code_generator, node) == "cache_global"


def test_nvidia_hint_source_cache_returns_independent_dicts():
    jit_fn = SimpleNamespace(src="def kernel(x):\n    y = x  # @hint:cache_global\n    return y\n")

    first = NvidiaHintHandler.maps_line_numbers_to_comment_hints(jit_fn)
    first[2] = "mutated"
    second = NvidiaHintHandler.maps_line_numbers_to_comment_hints(jit_fn)

    assert second == {2: "cache_global"}
