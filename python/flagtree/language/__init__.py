"""FlagTree-specific Triton language extensions."""

# FlagPrism: export debugger collect builtins from the FlagTree namespace.
from .core import debug_collect_end, debug_collect_start

__all__ = ["debug_collect_end", "debug_collect_start"]
