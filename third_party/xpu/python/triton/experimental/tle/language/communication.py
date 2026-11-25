# Triton 3.6 XPU TLE - distributed host runtime (xccl xshmem symmetric heap)
#
# Host-side companion to the in-kernel `tle.remote` primitive. Provides the
# interfaces the multi-rank tle.remote tests need, mirroring the flagtree
# surface (get_mem_pool / create_dist_tensor / cleanup_communicator) but backed
# by xccl's KL5 xshmem symmetric heap instead of NVIDIA FlagCX.
#
# It reaches the xshmem C ABI through a tiny shim (.so) compiled once at runtime
# against the pinned xccl.h (staged in _deps): the shim exposes extern "C"
# wrappers (tle_xshmem_init/get_info/alloc/free) so ctypes never has to name the
# C++-mangled xccl symbols, and torch tensors allocated inside
# `with torch.cuda.use_mem_pool(get_mem_pool())` live on the symmetric heap
# (xccl_xshmem_malloc), exactly like flagtree does for flagcxMemAlloc.
#
# The per-peer byte offset a kernel needs is the xshmem "p2p xlink offset":
#     off[peer] = peer_heap_base_p2p[peer] - heap_base
# read from xccl_xshmem_get_xshmem_info and carried on the handle returned by
# create_dist_tensor. The kernel just passes that handle to
#     tle.remote(dctx, shard_id=peer, offset=i, dtype=...)
# and a plain tl.load/tl.store does the intra-node P2P get/put.
import ctypes
import os
import subprocess
import tempfile

def _find_xccl_shim():
    """Locate the xccl NCCL-shim that exports the xshmem C ABI.

    Prefer an explicit ``XCCL_SHIM_PATH``; else search ``LD_LIBRARY_PATH`` (that
    is where xcuda installs it, and it differs across hosts); else fall back to
    the common xcuda location. ctypes raises a clear OSError later if the result
    is genuinely absent.
    """
    override = os.environ.get("XCCL_SHIM_PATH")
    if override:
        return override
    names = ("libxpunccl.so.2", "libxpunccl.so")
    search_dirs = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if d]
    search_dirs.append("/usr/local/xcuda/lib64")
    for d in search_dirs:
        for n in names:
            cand = os.path.join(d, n)
            if os.path.exists(cand):
                return cand
    return "/usr/local/xcuda/lib64/libxpunccl.so.2"


# The xccl shim that exports the xshmem C ABI (auto-discovered from
# LD_LIBRARY_PATH; override with XCCL_SHIM_PATH for non-default installs).
_XCCL_SHIM = _find_xccl_shim()

# xshmem (xccl_xshmem_init / _get_xshmem_info / _malloc / _free) is C++ (mangled,
# and xccl_xshmem_init is overloaded); rather than hardcode mangled names we call
# it through the shim's extern "C" wrappers below (compiled against the real
# xccl.h), which also lets the compiler pick the 1-arg
# xccl_xshmem_init(XCCLContext_t) overload for us.
#
# The comm is bootstrapped through the NCCL-shim entry points (plain extern "C"),
# mirroring the proven deepep path (xccl/deepep/csrc/deep_ep.cpp:82):
# ncclCommInitRank then ncclCommGetXcclContext to recover the XCCL context for
# xshmem. The shim's init is what sets up the device channels under the simulator.
_SYM_NCCL_GET_ID = "ncclGetUniqueId"
_SYM_NCCL_INIT_RANK = "ncclCommInitRank"
_SYM_NCCL_GET_CTX = "ncclCommGetXcclContext"

# ncclUniqueId is struct { char internal[128]; } (xccl/include/nccl.h).
_UNIQUE_ID_BYTES = 128


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * _UNIQUE_ID_BYTES)]


_lib = None
_shim = None           # compiled xshmem shim (ctypes.CDLL) exposing extern "C" wrappers
_ctx = None            # XCCLContext_t (ctypes.c_void_p)
_comm = None           # ncclComm_t (ctypes.c_void_p)
_rank = None
_world = None
_local_rank = None
_local_world = None
_mem_pool = None
_alloc_so_path = None
_init_done = False

# One int32 per signal slot, in a pad allocated next to every distributed
# tensor (tle.signal / tle.signal_wait address it by slot id).  flagtree's
# tutorials use one slot per rank; keep a floor so a kernel that keys slots on
# something else does not run off the end of a 2-rank pad.
#
# The floor has to cover a slot PER SOURCE CTA, not per source rank.  `signal`
# is a remote get-add-put (no cross-chip atomic on this machine), so a slot
# takes a single writer per epoch -- which means the CTA that wrote the payload
# is the one that must flag it, because a local grid barrier cannot flush
# another CTA's remote write and the flag would overtake the data.  A kernel
# with G co-resident CTAs therefore needs (teams * G) slots: 10-inter-node-\
# all-reduce at 4 chips x grid 8 asks for (4 + 1 + 4) * 8 = 72.  256 int32 is
# 1 KiB per handle, so the floor is set well above the largest grid in tree.
_SIGNAL_SLOTS_MIN = 256

def _load_lib():
    """Load the xccl shim and declare the ctypes signatures once."""
    global _lib
    if _lib is not None:
        return _lib
    lib = ctypes.CDLL(_XCCL_SHIM, mode=ctypes.RTLD_GLOBAL)
    # NCCL-shim bootstrap (extern "C"); comm_id is passed by value. xshmem itself
    # is reached through the compiled shim's extern "C" wrappers (see _load_shim).
    getattr(lib, _SYM_NCCL_GET_ID).argtypes = [ctypes.POINTER(_NcclUniqueId)]
    getattr(lib, _SYM_NCCL_GET_ID).restype = ctypes.c_int
    getattr(lib, _SYM_NCCL_INIT_RANK).argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _NcclUniqueId, ctypes.c_int
    ]
    getattr(lib, _SYM_NCCL_INIT_RANK).restype = ctypes.c_int
    getattr(lib, _SYM_NCCL_GET_CTX).argtypes = [ctypes.c_void_p]
    getattr(lib, _SYM_NCCL_GET_CTX).restype = ctypes.c_void_p
    _lib = lib
    return lib


# xshmem shim compiled at runtime against the pinned xccl.h. It exposes extern
# "C" wrappers so ctypes calls stable names (no C++ mangling) and the compiler
# resolves the overloaded xccl_xshmem_init for us. torch's MemPool calls
# tle_xshmem_alloc/free; tle_xshmem_init/get_info back init_communicator() and
# create_dist_tensor(). The XCCLContext_t is stashed via tle_set_xshmem_ctx.
# Compiling needs -I for xccl.h and the xre-xcn headers it pulls (cuda_runtime.h,
# xpu/runtime.h); both are staged in _deps (see _xshmem_include_dirs).
_ALLOC_SHIM_SRC = r"""
#include "xccl.h"
#include <cstddef>
static XCCLContext_t g_ctx = nullptr;
extern "C" void  tle_set_xshmem_ctx(void* c) { g_ctx = (XCCLContext_t)c; }
extern "C" int   tle_xshmem_init(void* ctx) { return (int)xccl_xshmem_init((XCCLContext_t)ctx); }
extern "C" int   tle_xshmem_get_info(void* ctx, unsigned long* heap_base,
                                     unsigned long* flag_base,
                                     unsigned long* peer_heap, unsigned long* peer_flag) {
    return (int)xccl_xshmem_get_xshmem_info((XCCLContext_t)ctx,
        (uint64_t*)heap_base, (uint64_t*)flag_base,
        (uint64_t*)peer_heap, (uint64_t*)peer_flag);
}
extern "C" void* tle_xshmem_alloc(size_t size, int /*device*/, void* /*stream*/) {
    return (void*)xccl_xshmem_malloc(g_ctx, (uint64_t)size, 4096UL, 0);
}
extern "C" void  tle_xshmem_free(void* ptr, size_t /*size*/, int /*device*/, void* /*stream*/) {
    if (ptr && g_ctx) xccl_xshmem_free(g_ctx, ptr, 0);
}
"""


def _xshmem_include_dirs():
    """Locate the xccl + xre-xcn header dirs to compile the shim against xccl.h.
    Both are staged by setup.py (download_and_copy_xputools): the xccl headers
    next to the jupiter backend, and the xre-xcn headers (cuda_runtime.h,
    xpu/runtime.h) that xccl.h pulls in. Falls back to the _deps checkout tree.
    Override with XCCL_INCLUDE_DIR / XRE_INCLUDE_DIR.
    """
    def _pick(env_var, cands, marker):
        v = os.environ.get(env_var)
        if v:
            return v
        return next((d for d in cands if os.path.isfile(os.path.join(d, marker))), None)

    try:
        import triton
        troot = os.path.dirname(os.path.abspath(triton.__file__))
    except Exception:
        troot = None
    repo = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * 5)))
    xccl_cands, xre_cands = [], []
    if troot:
        xccl_cands.append(os.path.join(troot, "backends", "jupiter", "xccl", "include"))
        xre_cands.append(os.path.join(troot, "backends", "jupiter", "include"))
    xccl_cands += [os.path.join(repo, "third_party", "jupiter", "backend", "xccl", "include"),
                   os.path.join(repo, "_deps", "xccl", "xccl_Linux_x86_64", "include")]
    xre_cands += [os.path.join(repo, "third_party", "jupiter", "backend", "include"),
                  os.path.join(repo, "_deps", "xre-xcn", "xre-emu-Linux-x86_64", "include")]
    return (_pick("XCCL_INCLUDE_DIR", xccl_cands, "xccl.h"),
            _pick("XRE_INCLUDE_DIR", xre_cands, "cuda_runtime.h"))


def _build_alloc_shim():
    """Compile the xshmem shim .so once per process (cached in tmpdir).

    Every rank of a run compiles it, and they all want the same cached path, so
    build into a private file and `os.replace` it into place: a half-written
    .so is otherwise visible to the other ranks, which fail to load it with
    "file too short" and then die in xccl_xshmem_init (hit with 4 ranks, where
    the window is widest).  Replace is atomic and keeps any already-open
    mapping valid, so a rank either sees the previous complete file or the new
    complete one.
    """
    global _alloc_so_path
    if _alloc_so_path is not None and os.path.isfile(_alloc_so_path):
        return _alloc_so_path
    xccl_inc, xre_inc = _xshmem_include_dirs()
    if not xccl_inc or not xre_inc:
        raise RuntimeError(
            "tle distributed runtime needs the xccl + xre-xcn headers to compile "
            "the xshmem shim; reinstall (pip install -e .) to stage them, or set "
            f"XCCL_INCLUDE_DIR / XRE_INCLUDE_DIR (xccl_inc={xccl_inc}, xre_inc={xre_inc}).")
    out_dir = tempfile.gettempdir()
    src = os.path.join(out_dir, f"tle_xshmem_alloc.{os.getpid()}.cpp")
    so = os.path.join(out_dir, "tle_xshmem_alloc.so")
    so_tmp = f"{so}.{os.getpid()}"
    lib_dir = os.path.dirname(_XCCL_SHIM)
    lib_name = os.path.basename(_XCCL_SHIM)
    # libxpunccl.so.2 -> link name "xpunccl"
    link = lib_name[3:].split(".so")[0]
    with open(src, "w") as f:
        f.write(_ALLOC_SHIM_SRC)
    cxx = os.environ.get("CXX", "g++")
    cmd = [
        cxx, "-O2", "-shared", "-fPIC", "-o", so_tmp, src,
        f"-I{xccl_inc}", f"-I{xre_inc}",
        f"-L{lib_dir}", f"-l{link}", f"-Wl,-rpath,{lib_dir}",
    ]
    try:
        subprocess.run(cmd, check=True)
        os.replace(so_tmp, so)
    finally:
        for path in (src, so_tmp):
            try:
                os.unlink(path)
            except OSError:
                pass
    _alloc_so_path = so
    return so



def _load_shim():
    """Build + load the xshmem shim and declare its extern "C" signatures once."""
    global _shim
    if _shim is not None:
        return _shim
    so = _build_alloc_shim()
    shim = ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    shim.tle_set_xshmem_ctx.argtypes = [ctypes.c_void_p]
    shim.tle_set_xshmem_ctx.restype = None
    shim.tle_xshmem_init.argtypes = [ctypes.c_void_p]
    shim.tle_xshmem_init.restype = ctypes.c_int
    shim.tle_xshmem_get_info.argtypes = [ctypes.c_void_p, u64p, u64p, u64p, u64p]
    shim.tle_xshmem_get_info.restype = ctypes.c_int
    _shim = shim
    return shim


def init_communicator():
    """Bootstrap torch.distributed (nccl), then the NCCL shim:
    ncclGetUniqueId -> broadcast -> ncclCommInitRank ->
    ncclCommGetXcclContext -> xccl_xshmem_init.

    Going through the NCCL shim (rather than raw xccl_init_rank) matches the
    proven deepep path and lets the shim set up the device channels correctly
    under the simulator.

    The group is nccl-only, as in flagtree's own tle/language/communication.py:
    everything that crosses ranks in this stack is a device tensor, the unique
    id included, so there is nothing for a CPU backend to carry.
    """
    global _ctx, _comm, _rank, _world, _local_rank, _local_world, _init_done
    if _init_done:
        return _ctx
    import torch
    import torch.distributed as dist

    lib = _load_lib()
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for tle.get_mem_pool()")
    if not dist.is_initialized():
        # nccl only -- the xccl nccl-shim.  No CPU backend in the group: the
        # unique-id broadcast below rides a device tensor, and so does every
        # cross-rank check in the tests, so a gloo half would only add a host
        # TCP full mesh to the bootstrap for nothing.  No fallback either: this
        # function goes on to set_device() and ncclCommInitRank() through the
        # shim regardless, so a rank with no usable device cannot work, and
        # downgrading the backend would only move the failure later and away
        # from its cause.
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    # Which chip this rank runs on.  Normally LOCAL_RANK, but when several
    # launcher agents share ONE host to emulate a multi-node topology every
    # process can see every chip and LOCAL_RANK repeats across agents, so the
    # global rank is the only non-colliding index.
    device_index = local_rank
    if local_world < world and torch.cuda.device_count() >= world:
        device_index = rank
        # Several agents on ONE host emulating a multi-node topology: the xccl
        # shim itself reads LOCAL_RANK / OMPI_COMM_WORLD_LOCAL_RANK to pick this
        # rank's chip, and LOCAL_RANK repeats across agents (every agent counts
        # from 0), so hand it the global rank through the MPI variable and leave
        # LOCAL_RANK alone -- the application needs that one for its node-local
        # role math.
        os.environ.setdefault("OMPI_COMM_WORLD_LOCAL_RANK", str(rank))
    torch.cuda.set_device(device_index)

    uid = _NcclUniqueId()
    if rank == 0:
        if getattr(lib, _SYM_NCCL_GET_ID)(ctypes.byref(uid)) != 0:
            raise RuntimeError("ncclGetUniqueId failed")
    # Broadcast the raw 128 id bytes from rank 0 on a DEVICE tensor -- the group
    # is nccl, same as flagtree's `.cuda()` here.  bytes(uid) captures the whole
    # struct (embedded NULs and all), unlike .internal.
    buf = torch.frombuffer(bytearray(bytes(uid)), dtype=torch.uint8).clone().cuda()
    dist.broadcast(buf, src=0)
    ctypes.memmove(ctypes.byref(uid), buf.cpu().numpy().tobytes(), _UNIQUE_ID_BYTES)

    comm = ctypes.c_void_p()
    if getattr(lib, _SYM_NCCL_INIT_RANK)(ctypes.byref(comm), world, uid, rank) != 0:
        raise RuntimeError("ncclCommInitRank failed")
    ctx_val = getattr(lib, _SYM_NCCL_GET_CTX)(comm)
    if not ctx_val:
        raise RuntimeError("ncclCommGetXcclContext returned null")
    ctx = ctypes.c_void_p(ctx_val)
    shim = _load_shim()
    shim.tle_set_xshmem_ctx(ctx)
    if shim.tle_xshmem_init(ctx) != 0:
        raise RuntimeError("xccl_xshmem_init failed")

    _ctx, _comm, _rank, _world, _init_done = ctx, comm, rank, world, True
    # Topology for tle.signal/tle.remote's team -> world-rank mapping.
    _local_rank, _local_world = local_rank, local_world
    return _ctx


def get_mem_pool():
    """Return a torch MemPool whose allocations come from the xccl symmetric
    heap (xccl_xshmem_malloc). Tensors created under
    ``with torch.cuda.use_mem_pool(get_mem_pool()): ...`` are symmetric across
    ranks, so ``buf - heap_base`` is identical on every rank.
    """
    global _mem_pool
    if _mem_pool is not None:
        return _mem_pool
    import torch
    from torch.cuda.memory import CUDAPluggableAllocator

    init_communicator()          # builds+loads the shim, sets ctx, xshmem_init
    so = _build_alloc_shim()     # cached path
    # torch's allocator dlopens the same .so path, so it shares the g_ctx that
    # init_communicator() set via tle_set_xshmem_ctx.
    allocator = CUDAPluggableAllocator(so, "tle_xshmem_alloc", "tle_xshmem_free").allocator()
    _mem_pool = torch.cuda.MemPool(allocator)
    return _mem_pool


class DistTensorCtx:
    """Handle for a symmetric-heap tensor plus its per-peer xlink offsets.

    This is the object ``tle.remote`` consumes as its first argument (the xpu5
    analogue of flagtree's ``DistributedRtContext``). It carries the buffer base
    address (``data_ptr``) and, for each peer, the byte offset to add to a local
    pointer into this buffer to reach that peer's copy (0 for self):
        offsets[peer] = peer_heap_base_p2p[peer] - heap_base
    Pass it straight to ``tle.remote(dctx, shard_id=peer, offset=i, dtype=...)``;
    ``xlink_offset(peer)`` is kept as a host-side convenience/debug accessor.

    It also carries the **signal pad** (``signal_ptr``, ``signal_slots``) that
    ``tle.signal``/``tle.signal_wait`` address -- the xpu5 stand-in for FlagCX's
    signal slots -- and the topology (``rank``, ``world``, ``local_rank``,
    ``local_world_size``, ``node_rank``) those primitives need to map a
    team-local peer index onto a world rank.

    Hashable by ``(data_ptr, offsets)`` so it can be passed as a ``tl.constexpr``
    kernel argument (its per-peer offsets are folded into the kernel at compile
    time).
    """

    def __init__(self, buf, heap_base, offsets, signal_pad=None,
                 rank=0, world=1, local_rank=0, local_world_size=1,
                 flag_base=0, peer_flags=()):
        self.buf = buf
        self.data_ptr = int(buf.data_ptr())
        self.heap_base = int(heap_base)
        self.offsets = list(offsets)
        # Keep a reference: the pad lives as long as the handle does.
        self.signal_pad = signal_pad
        self.signal_ptr = 0 if signal_pad is None else int(signal_pad.data_ptr())
        self.signal_slots = 0 if signal_pad is None else int(signal_pad.numel())
        # xshmem's flag range, mine and every peer's.  NOT where the signal pad
        # lives -- the pad stays in the heap.  These addresses exist so a rank
        # can poke a peer's flag range right before publishing a flag, which is
        # what orders the flag after the data it covers; see tle.signal.
        self.flag_base = int(flag_base)
        self.peer_flags = [int(x) for x in peer_flags]
        self.rank = int(rank)
        self.world = int(world)
        self.local_rank = int(local_rank)
        self.local_world_size = int(local_world_size)
        self.node_rank = int(rank) // max(int(local_world_size), 1)

    def xlink_offset(self, peer):
        return int(self.offsets[int(peer)])

    def flush_addr(self, peer):
        """Address in world rank ``peer``'s xshmem flag range."""
        return int(self.peer_flags[int(peer)])

    def _key(self):
        return (self.data_ptr, tuple(self.offsets), self.signal_ptr, self.rank,
                tuple(self.peer_flags))

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        return isinstance(other, DistTensorCtx) and self._key() == other._key()

    def __repr__(self):
        return (f"DistTensorCtx(data_ptr={self.data_ptr:#x}, "
                f"heap_base={self.heap_base:#x}, offsets={self.offsets}, "
                f"signal_ptr={self.signal_ptr:#x}, signal_slots={self.signal_slots}, "
                f"rank={self.rank}/{self.world}, "
                f"local={self.local_rank}/{self.local_world_size})")



def create_dist_tensor(buf_tensor):
    """Register a symmetric-heap tensor and compute its per-peer xlink offsets.

    ``buf_tensor`` must have been allocated on the symmetric heap, i.e. inside
    ``with torch.cuda.use_mem_pool(get_mem_pool()): ...``. Returns a
    :class:`DistTensorCtx` carrying ``heap_base`` and ``offsets[peer] =
    peer_heap_base_p2p[peer] - heap_base`` from xccl_xshmem_get_xshmem_info,
    plus a freshly zeroed signal pad for ``tle.signal``/``tle.signal_wait``.

    Call it on every rank, in the same order: the pad is a symmetric allocation
    and the zeroing is published with a barrier, so a peer cannot signal into a
    pad this rank has not finished clearing.
    """
    import torch
    import torch.distributed as dist

    shim = _load_shim()
    if _ctx is None:
        raise RuntimeError("call get_mem_pool()/init_communicator() before create_dist_tensor()")
    heap_base = ctypes.c_uint64(0)
    flag_base = ctypes.c_uint64(0)
    peer_heap = (ctypes.c_uint64 * _world)()
    peer_flag = (ctypes.c_uint64 * _world)()
    if shim.tle_xshmem_get_info(_ctx, ctypes.byref(heap_base), ctypes.byref(flag_base),
                                peer_heap, peer_flag) != 0:
        raise RuntimeError("xccl_xshmem_get_xshmem_info failed")
    base = int(heap_base.value)
    offsets = [int(peer_heap[p]) - base for p in range(_world)]

    # Every peer must be P2P-mapped, because that mapping is the ONLY transport
    # this backend lowers to -- including tle.remote(space="node").  xccl's
    # xlink offset is (port << 48) + (1 << 52), never 0, so a zero offset for a
    # peer that is not me means "no mapping": refuse the handle here rather than
    # let a kernel put to an address on a host it cannot reach.  Lowering a real
    # off-node transfer needs xccl's KL5 device IBGDA exported as a device ABI.
    unreachable = [p for p, off in enumerate(offsets) if off == 0 and p != _rank]
    if unreachable:
        raise RuntimeError(
            f"tle.create_dist_tensor: rank {_rank} has no P2P mapping for peer(s) "
            f"{unreachable} (xshmem reported no xlink offset). This backend can only "
            "reach ranks on the same C2C/XLINK fabric; in-kernel inter-node transfer "
            "would need xccl to export its KL5 device-side RDMA (IBGDA) as a linkable "
            "device ABI, which it does not.")

    nslots = max(_world, _SIGNAL_SLOTS_MIN)
    with torch.cuda.use_mem_pool(get_mem_pool()):
        signal_pad = torch.zeros(nslots, dtype=torch.int32, device=buf_tensor.device)
    torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return DistTensorCtx(buf_tensor, base, offsets, signal_pad=signal_pad,
                         rank=_rank, world=_world, local_rank=_local_rank,
                         local_world_size=_local_world,
                         flag_base=int(flag_base.value),
                         peer_flags=[int(peer_flag[p]) for p in range(_world)])



def cleanup_communicator():
    """Barrier and tear down the torch process group. The symmetric heap is
    reclaimed at process exit."""
    global _mem_pool
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    finally:
        _mem_pool = None



