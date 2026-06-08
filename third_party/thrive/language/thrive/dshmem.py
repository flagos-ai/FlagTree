import triton.language as tl
from triton.language import core
from .thrive_semantic import get_thrive_semantic

# team scope:
TEAM_WORLD = 0
TEAM_SHARED = 1
TEAM_NODE = 2

# sig_op:
SIGNAL_SET = 0
SIGNAL_ADD = 1

# cmp_op:
CMP_EQ = 0
CMP_NE = 1
CMP_GT = 2
CMP_LE = 3
CMP_LT = 4
CMP_GE = 5
# CMP_SENTINEL = sys.maxsize

# reduce_op:
AMO_AND = 0
AMO_OR = 1
AMO_XOR = 2
AMO_MIN = 3
AMO_MAX = 4
AMO_SUM = 5
AMO_PROD = 6

u64_t = tl.pointer_type(tl.dtype("int64"))


def helper(N, _semantic):
    if isinstance(N, tl.constexpr):
        return _semantic.tensor(_semantic.builder.get_uint64(N), tl.uint64)
    elif isinstance(N, tl.tensor):
        return N
    else:
        raise TypeError("Only support constexpr and tensor type of pe/sig_val/cmp_val")


def helper_ptrcheck(dest, src):
    if (dest.type.element_ty != src.type.element_ty):
        TypeError("dst type != src type")


def helper_sigptr(sig_addr):
    if (sig_addr.type.element_ty != tl.uint64):
        TypeError("signal ptr/buffer should always be int64")


def helper_size(nelements, element_size, _semantic):
    if isinstance(nelements, tl.constexpr):
        return _semantic.tensor(_semantic.builder.get_uint32(nelements * element_size), tl.uint32)
    elif isinstance(nelements, tl.tensor):
        size_ir = _semantic.builder.get_uint32(element_size)
        return _semantic.tensor(_semantic.builder.create_mul(nelements.handle, size_ir), nelements.type)
    else:
        raise TypeError("Only support constexpr and tensor type of size")


def helper_sigcmp_op(op, _semantic):
    if isinstance(op, int):
        return _semantic.tensor(_semantic.builder.get_uint32(op), tl.uint32)
    else:
        raise TypeError("invalid input type of cig_op/cmp_op")


def helper_cctype(N):
    if N == tl.int8:
        return tl.constexpr("char")
    elif N == tl.int16:
        return tl.constexpr("short")
    elif N == tl.int32:
        return tl.constexpr("int")
    elif N == tl.int64:
        return tl.constexpr("long")
    elif N == tl.uint8:
        return tl.constexpr("uchar")
    elif N == tl.uint16:
        return tl.constexpr("ushort")
    elif N == tl.uint32:
        return tl.constexpr("uint")
    elif N == tl.uint64:
        return tl.constexpr("ulong")
    elif N == tl.float16:
        return tl.constexpr("fp16")
    # elif N == tl.bfloat16:
    #     return tl.constexpr("bf16")
    elif N == tl.float32:
        return tl.constexpr("float")
    elif N == tl.float64:
        return tl.constexpr("double")
    else:
        TypeError("invalid input type of dest/source")


def helper_ptrtypesize(N):
    if N == tl.int8:
        return 1
    elif N == tl.int16:
        return 2
    elif N == tl.int32:
        return 4
    elif N == tl.int64:
        return 8
    elif N == tl.uint8:
        return 1
    elif N == tl.uint16:
        return 2
    elif N == tl.uint32:
        return 4
    elif N == tl.uint64:
        return 8
    elif N == tl.float16:
        return 2
    elif N == tl.bfloat16:
        return 2
    elif N == tl.float32:
        return 4
    elif N == tl.float64:
        return 8
    else:
        TypeError("invalid input type of dest/source")


def helper_reduceop(N):
    if N == 0:
        return tl.constexpr("and")
    elif N == 1:
        return tl.constexpr("or")
    elif N == 2:
        return tl.constexpr("xor")
    elif N == 3:
        return tl.constexpr("min")
    elif N == 4:
        return tl.constexpr("max")
    elif N == 5:
        return tl.constexpr("sum")
    elif N == 6:
        return tl.constexpr("prod")
    else:
        TypeError("invalid input type of reduceOp")


@core.extern
def my_pe(_semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.extern_call("__shmem_my_pe", [], [tl.int32], True)


@core.extern
def n_pes(_semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.extern_call("__shmem_n_pes", [], [tl.int32], True)


@core.extern
def team_my_pe(_semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.extern_call("__shmem_team_my_pe", [], [tl.int32], True)


@core.extern
def team_n_pes(_semantic=None):
    semantic = get_thrive_semantic(_semantic)
    return semantic.extern_call("__shmem_team_n_pes", [], [tl.int32], True)


@core.extern
def shmem_ptr(local_ptr, pe, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    pe_input = helper(pe, _semantic)
    return semantic.extern_call("__shmem_ptr", [
        local_ptr,
        pe_input,
    ], [tl.void], False)


def _putmem_impl(
        dest,
        src,
        n_elements,
        pe,
        _semantic,
        SCOPE_SUFFIX: tl.constexpr,
        NBI: tl.constexpr = tl.constexpr(""),
):
    helper_ptrcheck(dest, src)
    semantic = get_thrive_semantic(_semantic)
    size_input = helper_size(n_elements, helper_ptrtypesize(src.type.element_ty), _semantic)
    pe_input = helper(pe, _semantic)
    semantic.extern_call(
        f"__shmem_putmem{NBI.value}{SCOPE_SUFFIX.value}",
        [
            dest,
            src,
            size_input,
            pe_input,
        ],
        [tl.void],
        False,
    )


@core.extern
def putmem(dest, src, n_elements, pe, _semantic=None):
    _putmem_impl(
        dest,
        src,
        n_elements,
        pe,
        _semantic,
        tl.constexpr(""),
        tl.constexpr(""),
    )


@core.extern
def putmem_block(dest, src, n_elements, pe, _semantic=None):
    _putmem_impl(
        dest,
        src,
        n_elements,
        pe,
        _semantic,
        tl.constexpr("_block"),
        tl.constexpr(""),
    )


@core.extern
def putmem_cluster(dest, src, n_elements, pe, _semantic=None):
    _putmem_impl(
        dest,
        src,
        n_elements,
        pe,
        _semantic,
        tl.constexpr("_cluster"),
        tl.constexpr(""),
    )


@core.extern
def putmem_nbi(dest, src, n_elements, pe, _semantic=None):
    _putmem_impl(
        dest,
        src,
        n_elements,
        pe,
        _semantic,
        tl.constexpr(""),
        tl.constexpr("_nbi"),
    )


@core.extern
def putmem_nbi_block(dest, src, n_elements, pe, _semantic=None):
    _putmem_impl(
        dest,
        src,
        n_elements,
        pe,
        _semantic,
        tl.constexpr("_block"),
        tl.constexpr("_nbi"),
    )


@core.extern
def putmem_nbi_cluster(dest, src, n_elements, pe, _semantic=None):
    _putmem_impl(
        dest,
        src,
        n_elements,
        pe,
        _semantic,
        tl.constexpr("_cluster"),
        tl.constexpr("_nbi"),
    )


def _getmem_impl(
        dest,
        src,
        n_elements,
        pe,
        _semantic,
        SCOPE_SUFFIX: tl.constexpr,
        NBI: tl.constexpr = tl.constexpr(""),
):
    helper_ptrcheck(dest, src)
    semantic = get_thrive_semantic(_semantic)
    size_input = helper_size(n_elements, helper_ptrtypesize(src.type.element_ty), _semantic)
    pe_input = helper(pe, _semantic)
    semantic.extern_call(
        f"__shmem_getmem{NBI.value}{SCOPE_SUFFIX.value}",
        [
            dest,
            src,
            size_input,
            pe_input,
        ],
        [tl.void],
        False,
    )


@core.extern
def getmem(dest, src, n_elements, pe, _semantic=None):
    _getmem_impl(
        dest,
        src,
        n_elements,
        pe,
        _semantic,
        tl.constexpr(""),
        tl.constexpr(""),
    )


@core.extern
def getmem_block(dest, src, n_elements, pe, _semantic=None):
    _getmem_impl(
        dest,
        src,
        n_elements,
        pe,
        _semantic,
        tl.constexpr("_block"),
        tl.constexpr(""),
    )


@core.extern
def getmem_cluster(dest, src, n_elements, pe, _semantic=None):
    _getmem_impl(
        dest,
        src,
        n_elements,
        pe,
        _semantic,
        tl.constexpr("_cluster"),
        tl.constexpr(""),
    )


@core.extern
def getmem_nbi(dest, src, n_elements, pe, _semantic=None):
    _getmem_impl(
        dest,
        src,
        n_elements,
        pe,
        _semantic,
        tl.constexpr(""),
        tl.constexpr("_nbi"),
    )


@core.extern
def getmem_nbi_block(dest, src, n_elements, pe, _semantic=None):
    _getmem_impl(
        dest,
        src,
        n_elements,
        pe,
        _semantic,
        tl.constexpr("_block"),
        tl.constexpr("_nbi"),
    )


@core.extern
def getmem_nbi_cluster(dest, src, n_elements, pe, _semantic=None):
    _getmem_impl(
        dest,
        src,
        n_elements,
        pe,
        _semantic,
        tl.constexpr("_cluster"),
        tl.constexpr("_nbi"),
    )


@core.extern
def quiet(_semantic=None):
    semantic = get_thrive_semantic(_semantic)
    semantic.extern_call(f"__shmem_quiet", [], [tl.void], False)


@core.extern
def fence(_semantic=None):
    semantic = get_thrive_semantic(_semantic)
    semantic.extern_call(f"__shmem_fence", [], [tl.void], False)


def _barrier_impl(SCOPE_SUFFIX: tl.constexpr, _semantic=None):
    semantic = get_thrive_semantic(_semantic)
    semantic.extern_call(f"__shmem_barrier{SCOPE_SUFFIX.value}", [], [tl.void], False)


@core.extern
def barrier(_semantic=None):
    _barrier_impl(tl.constexpr(""), _semantic)


@core.extern
def barrier_block(_semantic=None):
    _barrier_impl(tl.constexpr("_block"), _semantic)


@core.extern
def barrier_cluster(_semantic=None):
    _barrier_impl(tl.constexpr("_cluster"), _semantic)


@core.extern
def signal_op(sig_addr, signal, sig_op, pe, _semantic=None):
    helper_sigptr(sig_addr)
    semantic = get_thrive_semantic(_semantic)
    signal_input = helper(signal, _semantic)
    sig_op_input = helper_sigcmp_op(sig_op, _semantic)
    pe_input = helper(pe, _semantic)
    semantic.extern_call(f"__shmem_signal_op", [sig_addr, signal_input, sig_op_input, pe_input], [tl.void], False)


@core.extern
def signal_wait_until(sig_addr, cmp_, cmp_val, _semantic=None):
    helper_sigptr(sig_addr)
    semantic = get_thrive_semantic(_semantic)
    cmp_input = helper_sigcmp_op(cmp_, _semantic)
    cmp_val_input = helper(cmp_val, _semantic)
    semantic.extern_call(f"__shmem_signal_wait_until", [sig_addr, cmp_input, cmp_val_input], [tl.void], False)


def _putmem_signal_impl(dest, source, n_elements, sig_addr, signal, sig_op, pe, _semantic, SCOPE_SUFFIX: tl.constexpr,
                        NBI: tl.constexpr):
    helper_ptrcheck(dest, source)
    helper_sigptr(sig_addr)
    semantic = get_thrive_semantic(_semantic)
    size_input = helper_size(n_elements, helper_ptrtypesize(source.type.element_ty), _semantic)
    signal_input = helper(signal, _semantic)
    sig_op_input = helper_sigcmp_op(sig_op, _semantic)
    pe_input = helper(pe, _semantic)
    semantic.extern_call(f"__shmem_putmem_signal{NBI.value}{SCOPE_SUFFIX.value}",
                         [dest, source, size_input, sig_addr, signal_input, sig_op_input, pe_input], [tl.void], False)


@core.extern
def putmem_signal(dest, source, n_elements, sig_addr, signal, sig_op, pe, _semantic=None):
    _putmem_signal_impl(dest, source, n_elements, sig_addr, signal, sig_op, pe, _semantic, tl.constexpr(""),
                        tl.constexpr(""))


@core.extern
def putmem_signal_block(dest, source, n_elements, sig_addr, signal, sig_op, pe, _semantic=None):
    _putmem_signal_impl(dest, source, n_elements, sig_addr, signal, sig_op, pe, _semantic, tl.constexpr("_block"),
                        tl.constexpr(""))


@core.extern
def putmem_signal_cluster(dest, source, n_elements, sig_addr, signal, sig_op, pe, _semantic=None):
    _putmem_signal_impl(dest, source, n_elements, sig_addr, signal, sig_op, pe, _semantic, tl.constexpr("_cluster"),
                        tl.constexpr(""))


@core.extern
def putmem_signal_nbi(dest, source, n_elements, sig_addr, signal, sig_op, pe, _semantic=None):
    _putmem_signal_impl(dest, source, n_elements, sig_addr, signal, sig_op, pe, _semantic, tl.constexpr(""),
                        tl.constexpr("_nbi"))


@core.extern
def putmem_signal_nbi_block(dest, source, n_elements, sig_addr, signal, sig_op, pe, _semantic=None):
    _putmem_signal_impl(dest, source, n_elements, sig_addr, signal, sig_op, pe, _semantic, tl.constexpr("_block"),
                        tl.constexpr("_nbi"))


@core.extern
def putmem_signal_nbi_cluster(dest, source, n_elements, sig_addr, signal, sig_op, pe, _semantic=None):
    _putmem_signal_impl(dest, source, n_elements, sig_addr, signal, sig_op, pe, _semantic, tl.constexpr("_cluster"),
                        tl.constexpr("_nbi"))


# collective operations
def broadcast_impl(dest, source, nelements, pe_root, team, _semantic, SCOPE_SUFFIX: tl.constexpr):
    helper_ptrcheck(dest, source)
    semantic = get_thrive_semantic(_semantic)
    team_input = _semantic.scalar_constant(0, tl.uint32)
    nelements_input = helper(nelements, _semantic)
    pe_input = helper(pe_root, _semantic)
    type_input = helper_cctype(dest.type.element_ty)
    semantic.extern_call(f"__shmem_{type_input.value}_broadcast{SCOPE_SUFFIX.value}",
                         [team_input, dest, source, nelements_input, pe_input], [tl.void], False)


@core.extern
def broadcast(dest, source, nelements, pe_root, team=TEAM_WORLD, _semantic=None):
    return broadcast_impl(dest, source, nelements, pe_root, team, _semantic, tl.constexpr(""))


@core.extern
def broadcast_block(dest, source, nelements, pe_root, team=TEAM_WORLD, _semantic=None):
    return broadcast_impl(dest, source, nelements, pe_root, team, _semantic, tl.constexpr("_block"))


@core.extern
def broadcast_cluster(dest, source, nelements, pe_root, team=TEAM_WORLD, _semantic=None):
    return broadcast_impl(dest, source, nelements, pe_root, team, _semantic, tl.constexpr("_cluster"))


def fcollect_impl(dest, source, nelements, team, _semantic, SCOPE_SUFFIX: tl.constexpr):
    helper_ptrcheck(dest, source)
    semantic = get_thrive_semantic(_semantic)
    team_input = _semantic.scalar_constant(0, tl.uint32)
    nelements_input = helper(nelements, _semantic)
    type_input = helper_cctype(dest.type.element_ty)
    semantic.extern_call(f"__shmem_{type_input.value}_fcollect{SCOPE_SUFFIX.value}",
                         [team_input, dest, source, nelements_input], [tl.void], False)


@core.extern
def fcollect(dest, source, nelements, team=TEAM_WORLD, _semantic=None):
    return fcollect_impl(dest, source, nelements, team, _semantic, tl.constexpr(""))


@core.extern
def fcollect_block(dest, source, nelements, team=TEAM_WORLD, _semantic=None):
    return fcollect_impl(dest, source, nelements, team, _semantic, tl.constexpr("_block"))


@core.extern
def fcollect_cluster(dest, source, nelements, team=TEAM_WORLD, _semantic=None):
    return fcollect_impl(dest, source, nelements, team, _semantic, tl.constexpr("_cluster"))


def alltoall_impl(dest, source, nelements, team, _semantic, SCOPE_SUFFIX: tl.constexpr):
    helper_ptrcheck(dest, source)
    semantic = get_thrive_semantic(_semantic)
    team_input = _semantic.scalar_constant(0, tl.uint32)
    nelements_input = helper(nelements, _semantic)
    type_input = helper_cctype(dest.type.element_ty)
    semantic.extern_call(f"__shmem_{type_input.value}_alltoall{SCOPE_SUFFIX.value}",
                         [team_input, dest, source, nelements_input], [tl.void], False)


@core.extern
def alltoall(dest, source, nelements, team=TEAM_WORLD, _semantic=None):
    return alltoall_impl(dest, source, nelements, team, _semantic, tl.constexpr(""))


@core.extern
def alltoall_block(dest, source, nelements, team=TEAM_WORLD, _semantic=None):
    return alltoall_impl(dest, source, nelements, team, _semantic, tl.constexpr("_block"))


@core.extern
def alltoall_cluster(dest, source, nelements, team=TEAM_WORLD, _semantic=None):
    return alltoall_impl(dest, source, nelements, team, _semantic, tl.constexpr("_cluster"))


def reduce_impl(dest, source, nelements, reduce_op, team, _semantic, SCOPE_SUFFIX: tl.constexpr):
    helper_ptrcheck(dest, source)
    semantic = get_thrive_semantic(_semantic)
    team_input = _semantic.scalar_constant(0, tl.uint32)
    nelements_input = helper(nelements, _semantic)
    type_input = helper_cctype(dest.type.element_ty)
    reduce_input = helper_reduceop(reduce_op)
    semantic.extern_call(f"__shmem_{type_input.value}_{reduce_input.value}_reduce{SCOPE_SUFFIX.value}",
                         [team_input, dest, source, nelements_input], [tl.void], False)


@core.extern
def reduce(dest, source, nelements, reduce_op, team=TEAM_WORLD, _semantic=None):
    return reduce_impl(dest, source, nelements, reduce_op, team, _semantic, tl.constexpr(""))


@core.extern
def reduce_block(dest, source, nelements, reduce_op, team=TEAM_WORLD, _semantic=None):
    return reduce_impl(dest, source, nelements, reduce_op, team, _semantic, tl.constexpr("_block"))


@core.extern
def reduce_cluster(dest, source, nelements, reduce_op, team=TEAM_WORLD, _semantic=None):
    return reduce_impl(dest, source, nelements, reduce_op, team, _semantic, tl.constexpr("_cluster"))


def reduce_workspace_impl(dest, source, nelements, reduce_op, workspace, team, _semantic, SCOPE_SUFFIX: tl.constexpr):
    helper_ptrcheck(dest, source)
    semantic = get_thrive_semantic(_semantic)
    team_input = _semantic.scalar_constant(0, tl.uint32)
    nelements_input = helper(nelements, _semantic)
    type_input = helper_cctype(dest.type.element_ty)
    reduce_input = helper_reduceop(reduce_op)
    semantic.extern_call(f"__shmem_{type_input.value}_{reduce_input.value}_reduce{SCOPE_SUFFIX.value}_workspace",
                         [team_input, dest, source, nelements_input, workspace], [tl.void], False)


@core.extern
def reduce_workspace(dest, source, nelements, reduce_op, workspace, team=TEAM_WORLD, _semantic=None):
    return reduce_workspace_impl(dest, source, nelements, reduce_op, workspace, team, _semantic, tl.constexpr(""))


@core.extern
def reduce_workspace_block(dest, source, nelements, reduce_op, workspace, team=TEAM_WORLD, _semantic=None):
    return reduce_workspace_impl(dest, source, nelements, reduce_op, workspace, team, _semantic, tl.constexpr("_block"))


@core.extern
def reduce_workspace_cluster(dest, source, nelements, reduce_op, workspace, team=TEAM_WORLD, _semantic=None):
    return reduce_workspace_impl(dest, source, nelements, reduce_op, workspace, team, _semantic,
                                 tl.constexpr("_cluster"))
