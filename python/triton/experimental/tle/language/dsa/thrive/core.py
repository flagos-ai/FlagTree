# flagtree tle
"""Thrive vendor-specific DSA primitives: inter-die RMA and synchronization.

Exposed as ``triton.experimental.tle.language.dsa.thrive``. These builtins
generate ``thvtile.libdevice_call`` ops with ``__shmem_*`` symbols directly,
so no C++ side changes are involved. Thrive-only dependencies are imported
inside function bodies so this module stays importable on other backends.
"""

import triton.language.core as tl

# Signal op codes for __shmem_signal_op_nbi_block (see libdevice_shmem).
SIGNAL_SET = 0
SIGNAL_ADD = 1
CMP_EQ = 0


def _as_uint64(value, _semantic):
    if isinstance(value, int):
        value = tl.constexpr(value)
    if isinstance(value, tl.constexpr):
        return _semantic.tensor(_semantic.builder.get_uint64(value), tl.uint64)
    if isinstance(value, tl.tensor):
        return value
    raise TypeError("Only support int/constexpr/tensor type of pe/sig_val/cmp_val")


def _as_uint32(value, _semantic):
    if isinstance(value, int):
        return _semantic.tensor(_semantic.builder.get_uint32(value), tl.uint32)
    raise TypeError("invalid input type of sig_op/cmp_op")


def _element_byte_size(element_ty):
    sizes = {
        'fp16': 2,
        'bf16': 2,
        'fp32': 4,
        'fp64': 8,
        'i8': 1,
        'i16': 2,
        'i32': 4,
        'i64': 8,
        'u8': 1,
        'u16': 2,
        'u32': 4,
        'u64': 8,
    }
    return sizes.get(str(element_ty), 4)


def _size_of(nelements, element_ty, _semantic):
    elem_size = _element_byte_size(element_ty)
    if isinstance(nelements, tl.constexpr):
        return _semantic.tensor(_semantic.builder.get_uint32(nelements * elem_size), tl.uint32)
    if isinstance(nelements, tl.tensor):
        size_ir = _semantic.builder.get_uint32(elem_size)
        return _semantic.tensor(_semantic.builder.create_mul(nelements.handle, size_ir), nelements.type)
    raise TypeError("Only support constexpr and tensor type of size")


def _validate_nelements(nelements, op):
    if isinstance(nelements, int):
        if nelements <= 0:
            raise ValueError(f"nelements must be positive, got {nelements}")
    elif not isinstance(nelements, (tl.constexpr, tl.tensor)):
        raise TypeError(f"nelements must be int/tl.constexpr/tl.tensor, got {type(nelements).__name__} ({op})")


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
    from triton.language.extra.thrive.thrive_semantic import get_thrive_semantic
    blocking = tl._unwrap_if_constexpr(blocking)
    _validate_nelements(nelements, "putmem")
    semantic = get_thrive_semantic(_semantic)
    size_input = _size_of(nelements, source.type.element_ty, _semantic)
    pe_input = _as_uint64(rank, _semantic)
    suffix = "" if blocking else "_nbi"
    semantic.extern_call(
        f"__shmem_putmem{suffix}_block",
        [dest, source, size_input, pe_input],
        [tl.void],
        False,
    )


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
    from triton.language.extra.thrive.thrive_semantic import get_thrive_semantic
    blocking = tl._unwrap_if_constexpr(blocking)
    _validate_nelements(nelements, "getmem")
    semantic = get_thrive_semantic(_semantic)
    size_input = _size_of(nelements, source.type.element_ty, _semantic)
    pe_input = _as_uint64(rank, _semantic)
    suffix = "" if blocking else "_nbi"
    semantic.extern_call(
        f"__shmem_getmem{suffix}_block",
        [dest, source, size_input, pe_input],
        [tl.void],
        False,
    )


@tl.builtin
def wait(barrier_ptr, wait_value=1, _semantic=None):
    """Block until the signal at `barrier_ptr` equals `wait_value`.

    Block-scope wait on a local signal address; all threads in the block must
    reach this call site.

    Args:
        barrier_ptr:  Local symmetric pointer to the signal data object.
        wait_value:   Value to wait for (comparison is equality, i.e. cmp_eq).
    """
    from triton.language.extra.thrive.thrive_semantic import get_thrive_semantic
    if not (hasattr(barrier_ptr, 'type') and hasattr(barrier_ptr.type, 'element_ty')
            and str(barrier_ptr.type.element_ty) in ('i64', 'int64')):
        raise TypeError("barrier_ptr must point to int64 (wait)")
    semantic = get_thrive_semantic(_semantic)
    cmp_input = _as_uint32(CMP_EQ, _semantic)
    cmp_val_input = _as_uint64(wait_value, _semantic)
    return semantic.extern_call(
        "__shmem_signal_wait_until_block",
        [barrier_ptr, cmp_input, cmp_val_input],
        [tl.void],
        False,
    )


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
    from triton.language.extra.thrive.thrive_semantic import get_thrive_semantic
    sig_op = tl._unwrap_if_constexpr(sig_op)
    if sig_op not in ("set", "add"):
        raise ValueError(f"sig_op must be 'set' or 'add', got '{sig_op}' (notify)")
    semantic = get_thrive_semantic(_semantic)
    sig_op_code = {"set": SIGNAL_SET, "add": SIGNAL_ADD}[sig_op]
    signal_input = _as_uint64(signal, _semantic)
    sig_op_input = _as_uint32(sig_op_code, _semantic)
    pe_input = _as_uint64(rank, _semantic)
    semantic.extern_call(
        "__shmem_signal_op_nbi_block",
        [ptr, signal_input, sig_op_input, pe_input],
        [tl.void],
        False,
    )


@tl.builtin
def fence(_semantic=None):
    """Order remote memory accesses to a die.

    Ensures operations on symmetric data objects issued to a die before this
    call are delivered before subsequent operations to the same die.
    """
    from triton.language.extra.thrive.thrive_semantic import get_thrive_semantic
    semantic = get_thrive_semantic(_semantic)
    semantic.extern_call("__shmem_fence", [], [tl.void], False)


@tl.builtin
def sync(_semantic=None):
    """Complete all previously issued remote memory and atomic operations."""
    from triton.language.extra.thrive.thrive_semantic import get_thrive_semantic
    semantic = get_thrive_semantic(_semantic)
    semantic.extern_call("__shmem_quiet", [], [tl.void], False)
