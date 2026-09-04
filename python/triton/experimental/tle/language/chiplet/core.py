"""tle.chiplet.* — inter-die RMA and synchronization primitives."""
import os
import triton.language.core as tl
from . import semantic

try:
    from triton._flagtree_backend import FLAGTREE_BACKEND
except ModuleNotFoundError:
    FLAGTREE_BACKEND = os.environ.get("FLAGTREE_BACKEND", "nvidia")


def _get_chiplet_impl():
    if FLAGTREE_BACKEND == "thrive":
        from .thrive import distributed as thrive_dist
        return thrive_dist
    raise NotImplementedError(f"tle.chiplet requires a valid chiplet backend; "
                              f"got FLAGTREE_BACKEND={FLAGTREE_BACKEND!r}")


@tl.builtin
def putmem(dest, source, nelements, rank, blocking=True, _semantic=None):
    """Write `nelements` elements from local `source` to `dest` on `rank`.

    Block-scope remote memory write between dies. `nelements` is counted in
    elements of `source` dtype, and the byte size is derived automatically.

    Args:
        dest:       Local symmetric pointer to the destination data object.
        source:     Pointer to the local source data object to be written.
        nelements:  Number of elements to write. Must be an int, constexpr or
                    scalar tensor.
        rank:       Rank of the die on which `dest` resides.
        blocking:   When True, the write is blocking; when False, the write is
                    nonblocking and must be ordered/awaited with `fence`/`sync`.
    """
    blocking = tl._unwrap_if_constexpr(blocking)
    semantic.validate_putmem(nelements)
    _get_chiplet_impl().putmem_impl(dest, source, nelements, rank, blocking, _semantic=_semantic)


@tl.builtin
def getmem(dest, source, nelements, rank, blocking=True, _semantic=None):
    """Read `nelements` elements from `source` on `rank` into local `dest`.

    Block-scope remote memory read between dies. `nelements` is counted in
    elements of `source` dtype.

    Args:
        dest:       Local symmetric pointer to the destination data object.
        source:     Pointer to the remote source data object to be read.
        nelements:  Number of elements to read. Must be an int, constexpr or
                    scalar tensor.
        rank:       Rank of the die on which `source` resides.
        blocking:   When True, the read is blocking; when False, the read is
                    nonblocking and must be ordered/awaited with `fence`/`sync`.
    """
    blocking = tl._unwrap_if_constexpr(blocking)
    semantic.validate_getmem(nelements)
    _get_chiplet_impl().getmem_impl(dest, source, nelements, rank, blocking, _semantic=_semantic)


@tl.builtin
def wait(barrier_ptr, wait_value=1, _semantic=None):
    """Block until the signal at `barrier_ptr` equals `wait_value`.

    Block-scope wait on a local signal address; all threads in the block must
    reach this call site.

    Args:
        barrier_ptr:  Local symmetric pointer to the signal data object.
        wait_value:   Value to wait for (comparison is equality, i.e. cmp_eq).
    """
    semantic.validate_wait(barrier_ptr)
    return _get_chiplet_impl().wait_impl(barrier_ptr, wait_value, _semantic=_semantic)


@tl.builtin
def notify(ptr, rank, signal=1, sig_op="set", _semantic=None):
    """Atomically update the signal at `ptr` on `rank`.

    Nonblocking, block-scope remote signal update on a die.

    Args:
        ptr:     Local symmetric pointer to the signal data object on `rank`.
        rank:    Rank of the die whose signal is updated.
        signal:  Value to write (for `sig_op="set"`) or add (for `sig_op="add"`).
        sig_op:  Signal operation, either "set" or "add".
    """
    sig_op = tl._unwrap_if_constexpr(sig_op)
    semantic.validate_notify(sig_op)
    _get_chiplet_impl().notify_impl(ptr, rank, signal, sig_op, _semantic=_semantic)


@tl.builtin
def fence(_semantic=None):
    """Order remote memory accesses to a die.

    Ensures operations on symmetric data objects issued to a die before this
    call are delivered before subsequent operations to the same die.
    """
    _get_chiplet_impl().fence_impl(_semantic=_semantic)


@tl.builtin
def sync(_semantic=None):
    """Complete all previously issued remote memory and atomic operations."""
    _get_chiplet_impl().sync_impl(_semantic=_semantic)
