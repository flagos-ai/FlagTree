"""FlagTree DSL builtins lowered through Triton's frontend."""

from triton import knobs
from triton.language.core import builtin


@builtin
def debug_collect_start(level=1, addr_level=None, _semantic=None):
    """Begin a FlagTree debug collect region."""
    from flagtree import _flagprism

    if knobs.runtime.use_prism:
        return _flagprism.debug_collect_start(_semantic, level, addr_level)
    return None


@builtin
def debug_collect_end(_semantic=None):
    """End a FlagTree debug collect region."""
    from flagtree import _flagprism

    if knobs.runtime.use_prism:
        return _flagprism.debug_collect_end(_semantic)
    return None
