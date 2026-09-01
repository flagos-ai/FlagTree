"""Semantic validation for tle.chiplet.* (called from chiplet/core.py)."""
import triton.language.core as tl


class ChipletSemanticError(Exception):

    def __init__(self, message, operation=None):
        self.operation = operation
        self.message = message
        super().__init__(
            f"Chiplet semantic error in {operation}: {message}" if operation else f"Chiplet semantic error: {message}")


def validate_putmem(nelements):
    if isinstance(nelements, int):
        if nelements <= 0:
            raise ChipletSemanticError(f"nelements must be positive, got {nelements}", "putmem")
    elif not isinstance(nelements, (tl.constexpr, tl.tensor)):
        raise ChipletSemanticError(f"nelements must be int/tl.constexpr/tl.tensor, got {type(nelements).__name__}",
                                   "putmem")


def validate_getmem(nelements):
    if isinstance(nelements, int):
        if nelements <= 0:
            raise ChipletSemanticError(f"nelements must be positive, got {nelements}", "getmem")
    elif not isinstance(nelements, (tl.constexpr, tl.tensor)):
        raise ChipletSemanticError(f"nelements must be int/tl.constexpr/tl.tensor, got {type(nelements).__name__}",
                                   "getmem")


def validate_wait(barrier_ptr):
    if not (hasattr(barrier_ptr, 'type') and hasattr(barrier_ptr.type, 'element_ty')
            and str(barrier_ptr.type.element_ty) in ('i64', 'int64')):
        raise ChipletSemanticError(
            "barrier_ptr must point to int64",
            "wait",
        )


def validate_notify(sig_op):
    if sig_op not in ("set", "add"):
        raise ChipletSemanticError(f"sig_op must be 'set' or 'add', got '{sig_op}'", "notify")
