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
