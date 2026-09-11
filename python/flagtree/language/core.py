"""FlagTree DSL builtins lowered through Triton's frontend."""

# FlagPrism: define the frontend hooks consumed by the external debugger.
from triton.language.core import builtin


@builtin
def debug_collect_start(level=1, addr_level=None, _semantic=None):
    """Begin a FlagTree debug collect region."""
    from flagtree import _flagprism

    return _flagprism.debug_collect_start(_semantic, level, addr_level)


@builtin
def debug_collect_end(_semantic=None):
    """End a FlagTree debug collect region."""
    from flagtree import _flagprism

    return _flagprism.debug_collect_end(_semantic)
