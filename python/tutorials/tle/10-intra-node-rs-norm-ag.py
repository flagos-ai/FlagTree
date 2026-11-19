"""
Intra-node fused ring collectives with norms.

Port of FlagGems ``fused/fused_rs_norm_ag.py``, two kernels on one shared buffer:

* ``rs_norm_ag_ring_kernel``: reduce-scatter + residual add + RMSNorm + all-gather for pure
  TP, replaces ``all_reduce(x); x, residual = fused_add_rms_norm(x, residual)``.
  gather=False stops after the norm (RS + add + RMSNorm, every rank keeps its rows).
* ``qkv_norm_ag_ring_kernel``: the other end of gather=False for MLA. RMSNorm q / kv of the
  scattered qkv_a output and all-gather it, replaces ``all_gather; q_a_layernorm;
  kv_a_layernorm``.

Rank r owns rows [r * chunk, min(T, (r + 1) * chunk)), chunk = ceil(T / W). Data moves on C
channels at once; channel c is a ring order[c] over the ranks. Program pid belongs to channel
pid % C and handles local rows pid, pid + G, ... of every chunk. Ring position p:
  RS step s = 0 .. W-2   send x + received partial of chunk order[p-s-1] -> succ slot s
  owner                  slot W-2 + my x + residual, RMSNorm, keep the residual
  AG step t = 0 .. W-2   forward my normed chunk (t = 0) or the one received -> succ slot W-1+t
Every send is followed by one flag write on the successor, so the receiver waits for
flag >= base + step. Flags only grow and are written with absolute values, so calls may
launch different grids. The parity of the data slots flips every call: a predecessor that
runs ahead into call n + 1 never touches what call n still reads. Every rank must make the
same calls in the same order on one stream (eager only).

At setup RingComm selects the host platform, ring orders and peer transport. The
launchers pass the peer address, receive load, remote store and flag operations as
constexpr JIT functions. IPC uses a peer address table. TLE uses a FlagCX window.
The default remote store is an ordinary masked store on all platforms.

Run from the repository root on one node with at least two GPUs:

    NPROC_PER_NODE=8 bash python/tutorials/tle/10-intra-node-rs-norm-ag.sh

Env: H (7168), Q / KV / D (1536 / 512 / 2112, DeepSeek-V3 MLA), DTYPE (bf16 / fp16),
TOKENS (comma list for the benchmark), ITERS, NCHANNELS, PEER (auto / tle / ipc),
RS_NORM_AG_PLATFORM (auto / ppu / hcu / metax / xpu / mthreads / cuda), RINGS (auto / identity / ppu810e), OUT_JSON (write the benchmark there),
COMPARE_FLAGGEMS (path of FlagGems fused_rs_norm_ag.py, benchmarked alongside).
TLE needs matching FlagTree/FlagCX runtime and device bitcode (FLAGCX_MODULE_PATH or
~/.flagtree/flagcx).
"""

import ctypes
import importlib.util
import json
import os
import statistics
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle
except ImportError:  # vendor Triton without TLE: only the IPC path
    tle = None


@triton.jit
def _signal(flag, value):
    # all stores of this program are issued before thread 0 releases the flag
    tl.debug_barrier()
    tl.atomic_xchg(flag, value, sem="release", scope="sys")


@triton.jit
def _wait(flag, target, err_ptr, max_spin):
    # cas(-1, -1) is an acquire read that is not hoisted out of the loop; the
    # difference keeps the compare right across int32 wrap
    i = 0
    v = tl.atomic_cas(flag, -1, -1, sem="acquire", scope="sys")
    while ((v - target) < 0) & (i < max_spin):
        v = tl.atomic_cas(flag, -1, -1, sem="acquire", scope="sys")
        i += 1
    if i >= max_spin:
        tl.atomic_add(err_ptr, 1)
    tl.debug_barrier()


@triton.jit
def _ipc_ptr(ctx: tl.constexpr, peer_ptr, rank, dtype: tl.constexpr, offset):
    # IPC path: element `offset` of `rank`'s buffer
    return tl.load(peer_ptr + rank).to(tl.pointer_type(dtype)) + offset


@triton.jit
def _tle_ptr(ctx: tl.constexpr, peer_ptr, rank, dtype: tl.constexpr, offset):
    return tle.remote(ctx, shard_id=rank, space="device", dtype=dtype, offset=offset)


@triton.jit
def _recv_load(ptr, mask):
    return tl.load(ptr, mask=mask, other=0.0, cache_modifier=".cg")


@triton.jit
def _remote_store(ptr, value, mask):
    tl.store(ptr, value, mask=mask)


@triton.jit(do_not_specialize=[
    "num_tokens", "chunk", "slot_elems", "par_elems", "flag_off", "rank", "nchan", "eps", "base", "max_spin"
])
def rs_norm_ag_ring_kernel(
    x_ptr,  # [T, H] partial sums
    res_ptr,  # [rows of this rank, H], updated in place
    w_ptr,  # [H]
    out_ptr,  # [T, H], may alias x
    buf_ptr,  # local data slots (start of the shared buffer), x dtype
    flag_ptr,  # local flags, one int32 per program
    peer_ptr,  # int64 [W], every rank's buffer (IPC path)
    order_ptr,  # int32 [C, W], ring of every channel
    pos_ptr,  # int32 [C], my position in every ring
    err_ptr,  # int32 [1], wait timeouts
    dist_ctx: tl.constexpr,  # FlagCX window (TLE path)
    num_tokens,
    chunk,
    slot_elems,
    par_elems,  # parity * slots per parity * slot_elems
    flag_off,  # int32 offset of the flags in the shared buffer
    rank,
    nchan,
    eps,
    base,  # flag value at the start of this call (int32, wraps)
    max_spin,
    WORLD: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    DO_AG: tl.constexpr,  # False: stop after the norm, out holds only my rows
    PEER_PTR: tl.constexpr,
    RECV_LOAD: tl.constexpr,
    REMOTE_STORE: tl.constexpr,
    SIGNAL: tl.constexpr,
    WAIT: tl.constexpr,
):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_H)
    m = cols < H
    dt = x_ptr.dtype.element_ty

    c = pid % nchan
    p = tl.load(pos_ptr + c)
    ring = order_ptr + c * WORLD
    nxt_rank = tl.load(ring + (p + 1) % WORLD)

    pe = par_elems.to(tl.int64)
    se = slot_elems.to(tl.int64)
    my_flag = flag_ptr + pid
    mine = buf_ptr + pe
    succ_flag = PEER_PTR(dist_ctx, peer_ptr, nxt_rank, tl.int32, flag_off + pid)
    nxt = PEER_PTR(dist_ctx, peer_ptr, nxt_rank, dt, pe)

    # RS step 0: my partial sums of chunk order[p - 1] -> succ slot 0
    k = tl.load(ring + (p + WORLD - 1) % WORLD)
    b = tl.minimum(k * chunk, num_tokens)
    n = tl.minimum((k + 1) * chunk, num_tokens) - b
    for lr in range(pid, n, G):
        v = tl.load(x_ptr + (b + lr).to(tl.int64) * H + cols, mask=m)
        REMOTE_STORE(nxt + lr.to(tl.int64) * H + cols, v, mask=m)
    SIGNAL(succ_flag, base + 1)

    # RS steps 1 .. W-2: add my partial sums to the received chunk, pass it on
    for s in range(1, WORLD - 1):
        WAIT(my_flag, base + s, err_ptr, max_spin)
        k = tl.load(ring + (p + 2 * WORLD - s - 1) % WORLD)
        b = tl.minimum(k * chunk, num_tokens)
        n = tl.minimum((k + 1) * chunk, num_tokens) - b
        for lr in range(pid, n, G):
            row = lr.to(tl.int64) * H
            # written by a peer: bypass L1
            acc = RECV_LOAD(mine + (s - 1) * se + row + cols, mask=m).to(tl.float32)
            acc += tl.load(x_ptr + (b + lr).to(tl.int64) * H + cols, mask=m).to(tl.float32)
            REMOTE_STORE(nxt + s * se + row + cols, acc.to(dt), mask=m)
        SIGNAL(succ_flag, base + s + 1)

    # owner: the last hop brings the sum of the other W - 1 ranks of my rows
    WAIT(my_flag, base + WORLD - 1, err_ptr, max_spin)
    mb = tl.minimum(rank * chunk, num_tokens)
    mn = tl.minimum((rank + 1) * chunk, num_tokens) - mb
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    for lr in range(pid, mn, G):
        row = lr.to(tl.int64) * H
        g = (mb + lr).to(tl.int64) * H
        x = RECV_LOAD(mine + (WORLD - 2) * se + row + cols, mask=m).to(tl.float32)
        x += tl.load(x_ptr + g + cols, mask=m, other=0.0).to(tl.float32)
        x += tl.load(res_ptr + row + cols, mask=m, other=0.0).to(tl.float32)
        rrms = 1 / tl.sqrt(tl.sum(x * x) / H + eps)
        tl.store(res_ptr + row + cols, x.to(dt), mask=m)
        y = (x * rrms * w).to(dt)
        if DO_AG:
            tl.store(out_ptr + g + cols, y, mask=m)
            REMOTE_STORE(nxt + (WORLD - 1) * se + row + cols, y, mask=m)
        else:
            tl.store(out_ptr + row + cols, y, mask=m)

    if DO_AG:
        SIGNAL(succ_flag, base + WORLD)
        # AG steps: chunk order[p - t] arrives in slot W - 2 + t; keep it, pass it on
        for t in range(1, WORLD):
            WAIT(my_flag, base + WORLD - 1 + t, err_ptr, max_spin)
            k = tl.load(ring + (p + WORLD - t) % WORLD)
            b = tl.minimum(k * chunk, num_tokens)
            n = tl.minimum((k + 1) * chunk, num_tokens) - b
            for lr in range(pid, n, G):
                row = lr.to(tl.int64) * H
                v = RECV_LOAD(mine + (WORLD - 2 + t) * se + row + cols, mask=m)
                tl.store(out_ptr + (b + lr).to(tl.int64) * H + cols, v, mask=m)
                if t < WORLD - 1:
                    REMOTE_STORE(nxt + (WORLD - 1 + t) * se + row + cols, v, mask=m)
            if t < WORLD - 1:
                SIGNAL(succ_flag, base + WORLD + t)


@triton.jit(do_not_specialize=[
    "num_tokens", "chunk", "slot_elems", "par_elems", "flag_off", "rank", "nchan", "q_eps", "kv_eps", "base",
    "max_spin"
])
def qkv_norm_ag_ring_kernel(
    x_ptr,  # [rows of this rank, D] qkv_a output: [q | kv | rest]
    qw_ptr,  # [Q]
    kvw_ptr,  # [KV]
    out_ptr,  # [T, D]
    buf_ptr,  # local data slots (start of the shared buffer), out dtype
    flag_ptr,  # local flags, one int32 per program
    peer_ptr,  # int64 [W], every rank's buffer (IPC path)
    order_ptr,  # int32 [C, W], ring of every channel
    pos_ptr,  # int32 [C], my position in every ring
    err_ptr,  # int32 [1], wait timeouts
    dist_ctx: tl.constexpr,  # FlagCX window (TLE path)
    num_tokens,
    chunk,
    slot_elems,
    par_elems,  # parity * slots per parity * slot_elems
    flag_off,
    rank,
    nchan,
    q_eps,
    kv_eps,
    base,  # flag value at the start of this call (int32, wraps)
    max_spin,
    WORLD: tl.constexpr,
    D: tl.constexpr,
    Q: tl.constexpr,
    KV: tl.constexpr,
    BLOCK: tl.constexpr,
    PEER_PTR: tl.constexpr,
    RECV_LOAD: tl.constexpr,
    REMOTE_STORE: tl.constexpr,
    SIGNAL: tl.constexpr,
    WAIT: tl.constexpr,
):
    # owner: norm q and kv of my rows, keep them, send them to succ slot 0;
    # AG step t = 1 .. W-1: chunk order[p - t] arrives in slot t - 1, keep it, pass it on
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    in_row = cols < D
    is_q = cols < Q
    is_kv = (cols >= Q) & (cols < Q + KV)
    dt = out_ptr.dtype.element_ty

    c = pid % nchan
    p = tl.load(pos_ptr + c)
    ring = order_ptr + c * WORLD
    nxt_rank = tl.load(ring + (p + 1) % WORLD)

    pe = par_elems.to(tl.int64)
    se = slot_elems.to(tl.int64)
    my_flag = flag_ptr + pid
    mine = buf_ptr + pe
    succ_flag = PEER_PTR(dist_ctx, peer_ptr, nxt_rank, tl.int32, flag_off + pid)
    nxt = PEER_PTR(dist_ctx, peer_ptr, nxt_rank, dt, pe)

    wq = tl.load(qw_ptr + cols, mask=is_q, other=0.0).to(tl.float32)
    wkv = tl.load(kvw_ptr + (cols - Q), mask=is_kv, other=0.0).to(tl.float32)
    w = tl.where(is_q, wq, wkv)

    mb = tl.minimum(rank * chunk, num_tokens)
    mn = tl.minimum((rank + 1) * chunk, num_tokens) - mb
    for lr in range(pid, mn, G):
        x = tl.load(x_ptr + lr.to(tl.int64) * D + cols, mask=in_row, other=0.0).to(tl.float32)
        xx = x * x
        rq = 1 / tl.sqrt(tl.sum(tl.where(is_q, xx, 0.0)) / Q + q_eps)
        rkv = 1 / tl.sqrt(tl.sum(tl.where(is_kv, xx, 0.0)) / KV + kv_eps)
        y = tl.where(is_q, x * rq * w, tl.where(is_kv, x * rkv * w, x)).to(dt)
        tl.store(out_ptr + (mb + lr).to(tl.int64) * D + cols, y, mask=in_row)
        REMOTE_STORE(nxt + lr.to(tl.int64) * D + cols, y, mask=in_row)
    SIGNAL(succ_flag, base + 1)

    for t in range(1, WORLD):
        WAIT(my_flag, base + t, err_ptr, max_spin)
        k = tl.load(ring + (p + WORLD - t) % WORLD)
        b = tl.minimum(k * chunk, num_tokens)
        n = tl.minimum((k + 1) * chunk, num_tokens) - b
        for lr in range(pid, n, G):
            row = lr.to(tl.int64) * D
            v = RECV_LOAD(mine + (t - 1) * se + row + cols, mask=in_row)
            tl.store(out_ptr + (b + lr).to(tl.int64) * D + cols, v, mask=in_row)
            if t < WORLD - 1:
                REMOTE_STORE(nxt + t * se + row + cols, v, mask=in_row)
        if t < WORLD - 1:
            SIGNAL(succ_flag, base + t + 1)


def ppu810e_rings(phys_ids: Sequence[int]) -> List[List[int]]:
    """pccl's 810E rings restricted to the PPUs of the group (rank i = phys_ids[i]).
    Keeps the rings whose every hop is still an ICN link; returns rank-space rings."""
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ImportError("Loading the PPU 810E ring configuration requires PyYAML") from exc

    path = Path(__file__).resolve().parent / "config" / "topology" / "ppu810e.yml"
    with path.open(encoding="utf-8") as f:
        topology = yaml.safe_load(f)
    neighbors = topology["icn_neighbors"]
    where = {d: i for i, d in enumerate(phys_ids)}
    rings, fallback = [], []
    for full in topology["rings"]:
        r = [d for d in full if d in where]
        ok = all(r[(i + 1) % len(r)] in neighbors[r[i]] for i in range(len(r)))
        (rings if ok else fallback).append([where[d] for d in r])
    return rings or fallback


def row_range(num_tokens: int, world_size: int, rank: int) -> Tuple[int, int]:
    """[begin, end) rows of `rank`, chunk = ceil(T / W); trailing ranks may own 0 rows."""
    chunk = (num_tokens + world_size - 1) // world_size
    return min(num_tokens, rank * chunk), min(num_tokens, (rank + 1) * chunk)


def _wrap_i32(v: int) -> int:
    return (v + 2**31) % 2**32 - 2**31


class _CudaIpcHandle(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 64)]


class _Platform:
    """Host runtime policy and device callbacks. The CUDA defaults keep the original NVIDIA path."""

    name = "cuda"
    soname = "libcudart.so"
    prefix = "cuda"
    device_type = "cuda"
    ipc_supported = True
    tle_supported = True
    recv_load = staticmethod(_recv_load)
    remote_store = staticmethod(_remote_store)
    signal = staticmethod(_signal)
    wait = staticmethod(_wait)

    @property
    def api(self):
        return getattr(torch, self.device_type)

    def load_runtime(self):
        # Bind the runtime torch already loaded, not a different SDK version on LD_LIBRARY_PATH.
        path = self.soname
        with open("/proc/self/maps") as maps:
            for line in maps:
                if "/" in line:
                    candidate = line[line.index("/"):].strip()
                    name = os.path.basename(candidate)
                    if (name == path or name.startswith(path + ".")) and "/stubs/" not in candidate:
                        path = candidate
                        break
        self.lib = ctypes.CDLL(path)
        self.fn("GetErrorString").restype = ctypes.c_char_p

    def fn(self, suffix):
        return getattr(self.lib, self.prefix + suffix)

    def check(self, status):
        if status:
            detail = self.fn("GetErrorString")(status)
            raise RuntimeError(f"{self.name} runtime error {status}: {detail!r}")

    def allocate(self, ptr, size):
        return self.fn("Malloc")(ptr, size)

    def malloc(self, nbytes):
        ptr, size = ctypes.c_void_p(), ctypes.c_size_t(nbytes)
        self.check(self.allocate(ctypes.byref(ptr), size))
        try:
            self.check(self.fn("Memset")(ptr, 0, size))
        except Exception:
            self.free(ptr.value)
            raise
        return ptr.value

    def free(self, ptr):
        self.check(self.fn("Free")(ctypes.c_void_p(ptr)))

    def ipc_get(self, ptr):
        handle = _CudaIpcHandle()
        self.check(self.fn("IpcGetMemHandle")(ctypes.byref(handle), ctypes.c_void_p(ptr)))
        return bytes(handle.internal)

    def ipc_open(self, raw):
        handle = _CudaIpcHandle.from_buffer_copy(raw)
        ptr = ctypes.c_void_p()
        self.check(self.fn("IpcOpenMemHandle")(ctypes.byref(ptr), handle, ctypes.c_uint(1)))
        return ptr.value

    def ipc_close(self, ptr):
        self.check(self.fn("IpcCloseMemHandle")(ctypes.c_void_p(ptr)))

    def default_rings(self, world):
        return [list(range(world))], "identity"

    def num_warps(self, block):
        return min(16, max(4, block // 256))

    def kernel_args(self):
        return dict(RECV_LOAD=self.recv_load, REMOTE_STORE=self.remote_store, SIGNAL=self.signal, WAIT=self.wait)


class PpuPlatform(_Platform):
    name = "ppu"
    tle_supported = False

    def default_rings(self, world):
        if "810E" in self.api.get_device_name():
            return ppu810e_rings(_phys_ids(world)), "ppu810e"
        return super().default_rings(world)


class HcuPlatform(_Platform):
    name, soname, prefix = "hcu", "libamdhip64.so", "hip"
    tle_supported = False

    def num_warps(self, block):
        # BW1101 uses 64 lanes per wave: four waves give 256 threads per program.
        return 4

    def allocate(self, ptr, size):
        # hipDeviceMallocUncached: coarse-grained hipMalloc breaks peer flag visibility.
        return self.fn("ExtMallocWithFlags")(ptr, size, ctypes.c_uint(3))


class MetaxPlatform(_Platform):
    name = "metax"
    tle_supported = ipc_supported = False  # Native host IPC ABI has not been validated.


class KunlunPlatform(_Platform):
    name = "xpu"
    # XPU3's default CreateGM2LM pass rejects our release/sys atomic_xchg.
    tle_supported = ipc_supported = False


class MthreadsPlatform(_Platform):
    name, device_type = "mthreads", "musa"
    tle_supported = ipc_supported = False  # Native host IPC ABI has not been validated.


_PLATFORMS = {cls.name: cls for cls in (_Platform, PpuPlatform, HcuPlatform, MetaxPlatform, KunlunPlatform,
                                        MthreadsPlatform)}


def _platform(name="auto"):
    if name == "auto":
        try:
            from triton._flagtree_backend import FLAGTREE_BACKEND
        except ImportError:
            FLAGTREE_BACKEND = None
        name = FLAGTREE_BACKEND or ""
        if not name:
            target = triton.runtime.driver.active.get_current_target().backend
            if target == "cuda" and "PPU" in torch.cuda.get_device_name().upper():
                name = "ppu"
            else:
                name = target
        name = {"hip": "hcu", "amd": "hcu", "maca": "metax", "musa": "mthreads", "nvidia": "cuda"}.get(name, name)
    if name not in _PLATFORMS:
        raise ValueError(f"unknown platform {name!r}; choose RS_NORM_AG_PLATFORM from {tuple(_PLATFORMS)}")
    return _PLATFORMS[name]()


class _RawCuda:
    """Zero-copy torch view of raw device memory."""

    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {"data": (ptr, False), "shape": (nbytes, ), "typestr": "|u1", "version": 3}


class _IpcPeer:
    ptr = staticmethod(_ipc_ptr)
    ctx = None

    def __init__(self, platform):
        if not platform.ipc_supported:
            raise NotImplementedError(f"{platform.name}: host IPC is unavailable; use the compiler for offline checks")
        self.platform = platform
        platform.load_runtime()
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    def buffers(self, nbytes, device, world, rank):
        rt = self.platform
        self.local = rt.malloc(nbytes)
        self.opened = []
        try:
            buf = torch.as_tensor(_RawCuda(self.local, nbytes), device=device)
            if self.local != buf.data_ptr():
                raise RuntimeError("IPC buffer must alias its allocation")
            handles = [None] * world
            dist.all_gather_object(handles, rt.ipc_get(self.local))
            ptrs = [self.local] * world
            for r, h in enumerate(handles):
                if r != rank:
                    ptrs[r] = rt.ipc_open(h)
                    self.opened.append(ptrs[r])
            return buf, torch.tensor(ptrs, dtype=torch.int64, device=device)
        except Exception:
            self._release()
            raise

    def _release(self):
        for ptr in self.opened:
            self.platform.ipc_close(ptr)
        self.platform.free(self.local)

    def close(self):
        dist.barrier()
        self._release()
        dist.destroy_process_group()


class _TlePeer:
    ptr = staticmethod(_tle_ptr)

    def __init__(self):
        self.pool = tle.get_mem_pool()  # also initializes the process group
        if self.pool is None:
            raise RuntimeError("FlagCX memory pool unavailable")

    def buffers(self, nbytes, device, world, rank):
        with torch.cuda.use_mem_pool(self.pool):
            buf = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        self.ctx = tle.create_dist_tensor(buf)
        return buf, torch.zeros(world, dtype=torch.int64, device=device)  # unused address table

    def close(self):
        tle.cleanup_communicator()


class RingComm:
    """Shared buffer and ring channels of every op below.

    One buffer per process: 2 parities of `parity_bytes` data slots, then one int32 flag per
    program. Size it with the ops' ``parity_bytes``. Ops share the parity and the flag
    counter: any mix of calls is safe as long as every rank makes the same calls in the same
    order.

    peer: "tle" registers the buffer as a FlagCX window and reaches peers with tle.remote
    (TLE keeps one window per process, which is why every op lives in this buffer); "ipc"
    uses the platform allocator and IPC handles; "auto" takes tle when supported.
    rings: rank orders, one per distinct ring (default from platform); nchannels: channels per
    call, the rings repeat to fill them. Creation runs collectives: call it on every rank.
    """

    def __init__(
        self,
        parity_bytes: int,
        peer: str = "auto",
        rings: Optional[Sequence[Sequence[int]]] = None,
        nchannels: int = 16,
        max_spin: int = 1 << 22,
        platform="auto",
    ):
        self.platform = platform if isinstance(platform, _Platform) else _platform(platform)
        can_tle = self.platform.tle_supported and tle is not None and tle.communication.enabled
        if peer == "auto":
            peer = "tle" if can_tle else "ipc"
        if peer not in ("tle", "ipc"):
            raise ValueError(f"peer must be auto, tle or ipc, got {peer!r}")
        self.use_tle = peer == "tle"
        if self.use_tle and not can_tle:
            raise RuntimeError(f"{self.platform.name}: TLE device windows unavailable")
        self.peer = _TlePeer() if self.use_tle else _IpcPeer(self.platform)
        W, rank = dist.get_world_size(), dist.get_rank()
        if W < 2:
            raise ValueError("needs at least 2 ranks")
        self.world_size, self.rank = W, rank
        self.max_spin = max_spin
        self.device = torch.device(self.platform.device_type, self.platform.api.current_device())
        self._seq = 0
        self._signals = 0
        self.closed = False

        if rings is None:
            rings, self.ring_kind = self.platform.default_rings(W)
        else:
            self.ring_kind = "custom"
        rings = [list(r) for r in rings]
        if not rings:
            raise ValueError("at least one ring is required")
        for r in rings:
            if sorted(r) != list(range(W)):
                raise ValueError(f"ring {r} is not a permutation of the {W} ranks")
        self.max_programs = self.platform.api.get_device_properties(self.device).multi_processor_count
        # a program waits on the same pid of its predecessor: every program of the grid
        # must be resident at once, so at most one per SM
        self.nchannels = max(1, min(nchannels, self.max_programs))
        order = [rings[c % len(rings)] for c in range(self.nchannels)]
        self.rings = rings
        self.order = torch.tensor(order, dtype=torch.int32, device=self.device)
        self.pos = torch.tensor([r.index(rank) for r in order], dtype=torch.int32, device=self.device)

        self.parity_bytes = triton.cdiv(parity_bytes, 512) * 512
        data_bytes = 2 * self.parity_bytes
        nbytes = data_bytes + self.max_programs * 4
        self.buf, self.peers = self.peer.buffers(nbytes, self.device, W, rank)
        self.flags = self.buf[data_bytes:].view(torch.int32)
        self.flag_off = data_bytes // 4
        self.err = torch.zeros(1, dtype=torch.int32, device=self.device)
        # every flag buffer is zeroed before any peer can signal into it
        self.platform.api.synchronize()
        dist.barrier()

    def data(self, dtype: torch.dtype) -> torch.Tensor:
        return self.buf[:2 * self.parity_bytes].view(dtype)

    def next_call(self, signals: int) -> Tuple[int, int]:
        """(parity, flag base) of the next call, which signals `signals` times."""
        parity = self._seq & 1
        base = _wrap_i32(self._signals)
        self._seq += 1
        self._signals += signals
        return parity, base

    def row_range(self, num_tokens: int) -> Tuple[int, int]:
        return row_range(num_tokens, self.world_size, self.rank)

    def launch_shape(self, num_tokens: int) -> Tuple[int, int]:
        """(channels, programs) of a call with `num_tokens`; the same on every rank."""
        chunk = (num_tokens + self.world_size - 1) // self.world_size
        nchan = max(1, min(self.nchannels, chunk))
        per_chan = max(1, min(self.max_programs // nchan, triton.cdiv(chunk, nchan)))
        return nchan, nchan * per_chan

    def errors(self) -> int:
        """Wait timeouts so far (syncs the device)."""
        return int(self.err.item())

    def close(self):
        """Release the buffer and destroy the process group."""
        if self.closed:
            return
        self.closed = True
        self.platform.api.synchronize()
        self.peer.close()
        self.buf = self.flags = None

    def launch_args(self):
        return dict(dist_ctx=self.peer.ctx, PEER_PTR=self.peer.ptr, **self.platform.kernel_args())


class fused_rs_add_rms_norm_ag:
    """Launcher of ``rs_norm_ag_ring_kernel`` on a ``RingComm``.

    gather=True (default): RS + add + RMSNorm + AG, replaces all_reduce + add norm.
    gather=False: RS + add + RMSNorm only, every rank keeps its normed rows.
    """

    @staticmethod
    def parity_bytes(world_size: int, max_tokens: int, hidden_size: int, dtype: torch.dtype) -> int:
        # 2W - 2 slots of ceil(max_tokens / W) rows
        elt = torch.empty((), dtype=dtype).element_size()
        return (2 * world_size - 2) * triton.cdiv(max_tokens, world_size) * hidden_size * elt

    def __init__(self, comm: RingComm, max_tokens: int, hidden_size: int, dtype: torch.dtype = torch.bfloat16):
        W = comm.world_size
        if self.parity_bytes(W, max_tokens, hidden_size, dtype) > comm.parity_bytes:
            raise ValueError("comm buffer too small for this op")
        self.comm, self.hidden_size, self.dtype, self.max_tokens = comm, hidden_size, dtype, max_tokens
        self.slot_elems = triton.cdiv(max_tokens, W) * hidden_size
        self.parity_elems = comm.parity_bytes // torch.empty((), dtype=dtype).element_size()
        self.data = comm.data(dtype)

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        out: Optional[torch.Tensor] = None,
        gather: bool = True,
    ) -> torch.Tensor:
        """x: [T, H] partial sums, residual: this rank's rows (updated in place).
        gather=True: returns the normed full [T, H], written into x unless `out` is given.
        gather=False: returns the normed rows of this rank [e - b, H], written into
        x[b:e] unless `out` is given."""
        comm = self.comm
        T, H = x.shape
        W = comm.world_size
        b, e = comm.row_range(T)
        if H != self.hidden_size or x.dtype != self.dtype:
            raise ValueError(f"expected [*, {self.hidden_size}] {self.dtype}, got {tuple(x.shape)} {x.dtype}")
        if T > self.max_tokens:
            raise ValueError(f"{T} tokens > max_tokens {self.max_tokens}")
        if tuple(residual.shape) != (e - b, H):
            raise ValueError(f"residual must hold rows [{b}, {e}), got {tuple(residual.shape)}")
        if not (x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()):
            raise ValueError("tensors must be contiguous")
        if out is None:
            out = x if gather else x[b:e]
        elif tuple(out.shape) != ((T, H) if gather else (e - b, H)) or not out.is_contiguous():
            raise ValueError(f"bad out shape {tuple(out.shape)}")
        if T == 0:
            return out

        # RS + AG: 2W - 2 signals; RS only: the W - 1 RS hops
        parity, base = comm.next_call(2 * W - 2 if gather else W - 1)
        nchan, grid = comm.launch_shape(T)
        BLOCK_H = triton.next_power_of_2(H)
        rs_norm_ag_ring_kernel[(grid, )](
            x,
            residual if residual.numel() else x,
            weight,
            out if out.numel() else x,
            self.data,
            comm.flags,
            comm.peers,
            comm.order,
            comm.pos,
            comm.err,
            num_tokens=T,
            chunk=triton.cdiv(T, W),
            slot_elems=self.slot_elems,
            par_elems=parity * self.parity_elems,
            flag_off=comm.flag_off,
            rank=comm.rank,
            nchan=nchan,
            eps=eps,
            base=base,
            max_spin=comm.max_spin,
            WORLD=W,
            H=H,
            BLOCK_H=BLOCK_H,
            DO_AG=gather,
            num_warps=comm.platform.num_warps(BLOCK_H),
            **comm.launch_args(),
        )
        return out


class fused_qkv_norm_ag:
    """Launcher of ``qkv_norm_ag_ring_kernel`` on a ``RingComm``.

    The qkv_a output of MLA attention when its input is scattered: every rank holds the rows
    row_range(T, W, rank) of [T, D] = [q | kv | rest], q has q_dim columns, kv kv_dim.
    RMSNorm q and kv of those rows (rest, e.g. k_pe, is copied as is) and all-gather them.
    """

    @staticmethod
    def parity_bytes(world_size: int, max_tokens: int, row_dim: int, dtype: torch.dtype) -> int:
        # W - 1 slots of ceil(max_tokens / W) rows
        elt = torch.empty((), dtype=dtype).element_size()
        return (world_size - 1) * triton.cdiv(max_tokens, world_size) * row_dim * elt

    def __init__(self, comm: RingComm, max_tokens: int, q_dim: int, kv_dim: int, row_dim: int,
                 dtype: torch.dtype = torch.bfloat16):
        W = comm.world_size
        if q_dim <= 0 or kv_dim <= 0 or q_dim + kv_dim > row_dim:
            raise ValueError(f"bad split q {q_dim} + kv {kv_dim} of {row_dim}")
        if self.parity_bytes(W, max_tokens, row_dim, dtype) > comm.parity_bytes:
            raise ValueError("comm buffer too small for this op")
        self.comm, self.dtype, self.max_tokens = comm, dtype, max_tokens
        self.q_dim, self.kv_dim, self.row_dim = q_dim, kv_dim, row_dim
        self.slot_elems = triton.cdiv(max_tokens, W) * row_dim
        self.parity_elems = comm.parity_bytes // torch.empty((), dtype=dtype).element_size()
        self.data = comm.data(dtype)

    def __call__(
        self,
        x: torch.Tensor,
        num_tokens: int,
        q_weight: torch.Tensor,
        q_eps: float,
        kv_weight: torch.Tensor,
        kv_eps: float,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: [e - b, D] rows of this rank, num_tokens: T of the whole batch (same on every
        rank). Returns [T, D] with q and kv normed, in `out` if given."""
        comm = self.comm
        T, D, W = num_tokens, self.row_dim, comm.world_size
        b, e = comm.row_range(T)
        if tuple(x.shape) != (e - b, D) or x.dtype != self.dtype:
            raise ValueError(f"x must be rows [{b}, {e}) of [*, {D}] {self.dtype}, got {tuple(x.shape)} {x.dtype}")
        if T > self.max_tokens:
            raise ValueError(f"{T} tokens > max_tokens {self.max_tokens}")
        if q_weight.numel() != self.q_dim or kv_weight.numel() != self.kv_dim:
            raise ValueError("weights do not match q_dim / kv_dim")
        if out is None:
            out = torch.empty((T, D), dtype=self.dtype, device=x.device)
        elif tuple(out.shape) != (T, D) or not out.is_contiguous():
            raise ValueError(f"out must be a contiguous [{T}, {D}]")
        if not (x.is_contiguous() and q_weight.is_contiguous() and kv_weight.is_contiguous()):
            raise ValueError("tensors must be contiguous")
        if T == 0:
            return out

        parity, base = comm.next_call(W - 1)
        nchan, grid = comm.launch_shape(T)
        BLOCK = triton.next_power_of_2(D)
        qkv_norm_ag_ring_kernel[(grid, )](
            x if x.numel() else out,
            q_weight,
            kv_weight,
            out,
            self.data,
            comm.flags,
            comm.peers,
            comm.order,
            comm.pos,
            comm.err,
            num_tokens=T,
            chunk=triton.cdiv(T, W),
            slot_elems=self.slot_elems,
            par_elems=parity * self.parity_elems,
            flag_off=comm.flag_off,
            rank=comm.rank,
            nchan=nchan,
            q_eps=q_eps,
            kv_eps=kv_eps,
            base=base,
            max_spin=comm.max_spin,
            WORLD=W,
            D=D,
            Q=self.q_dim,
            KV=self.kv_dim,
            BLOCK=BLOCK,
            num_warps=comm.platform.num_warps(BLOCK),
            **comm.launch_args(),
        )
        return out


# ---- references and benchmark ----


def _rms(z, w, eps):
    return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def _rank_input(r, T, cols, dtype, seed):
    g = torch.Generator("cuda").manual_seed(seed * 1000003 + 7919 * r + T)
    return torch.empty(T, cols, dtype=dtype, device="cuda").uniform_(-1, 1, generator=g)


def _shared_input(T, cols, dtype, seed):
    g = torch.Generator("cuda").manual_seed(seed * 1000003 + T)
    return torch.randn(T, cols, dtype=dtype, device="cuda", generator=g)


def _assert_close(actual, expected, tol=5e-2, max_frac=1e-5, hard=0.25):
    # the ring rounds the partial sum to bf16 at every one of the W - 1 hops, the reference
    # sums in fp32 once: allow a few elements past `tol`, none past `hard`
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    if actual.numel() == 0:  # trailing ranks own no rows
        return
    a, e = actual.float(), expected.float()
    d = (a - e).abs()
    bad = d > tol + tol * e.abs()
    frac = bad.float().mean().item()
    if frac > max_frac or d.max().item() > hard or not torch.isfinite(a).all():
        raise AssertionError(f"mismatch: {bad.sum().item()} / {d.numel()} past {tol}, max abs diff {d.max().item()}")


def _check_rs(op, T, weight, eps, gather, seed):
    comm = op.comm
    W, rank, H, dt = comm.world_size, comm.rank, op.hidden_size, op.dtype
    b, e = comm.row_range(T)
    xs = [_rank_input(r, T, H, dt, seed) for r in range(W)]
    residual_full = _shared_input(T, H, dt, seed)
    # fp32 sum of every rank's x, then add + RMSNorm
    z = torch.stack(xs).float().sum(0) + residual_full.float()
    ref_out, ref_res = _rms(z, weight, eps).to(dt), z[b:e].to(dt)
    x, residual = xs[rank].clone(), residual_full[b:e].clone()
    out = op(x, residual, weight, eps, gather=gather)
    torch.cuda.synchronize()
    _assert_close(out, ref_out if gather else ref_out[b:e])
    _assert_close(residual, ref_res)


def _check_qkv(op, T, qw, kvw, eps, seed):
    comm = op.comm
    b, e = comm.row_range(T)
    Q, KV, D, dt = op.q_dim, op.kv_dim, op.row_dim, op.dtype
    full = _shared_input(T, D, dt, seed)
    ref = full.float().clone()
    ref[:, :Q] = _rms(ref[:, :Q], qw, eps)
    ref[:, Q:Q + KV] = _rms(ref[:, Q:Q + KV], kvw, eps)
    out = op(full[b:e].contiguous(), T, qw, eps, kvw, eps)
    torch.cuda.synchronize()
    # no reduction: every rank must get exactly what the owner normed
    torch.testing.assert_close(out, ref.to(dt), atol=5e-2, rtol=5e-2)


def _bench(fn, iters, reps=7, warmup=5):
    """us per call: median over reps of `iters` back-to-back calls, max over ranks."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        dist.barrier()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / iters)
    med = torch.tensor(statistics.median(times), device="cuda")
    dist.all_reduce(med, op=dist.ReduceOp.MAX)
    return med.item()


def _load_flaggems(path):
    spec = importlib.util.spec_from_file_location("flaggems_fused_rs_norm_ag", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _default_rings(platform, world):
    kind = os.environ.get("RINGS", "auto")
    if kind == "auto":
        return platform.default_rings(world)
    if kind == "identity":
        return [list(range(world))], kind
    if kind == "ppu810e" and platform.name == "ppu":
        return ppu810e_rings(_phys_ids(world)), kind
    raise ValueError(f"invalid RINGS={kind!r} for {platform.name}")


def _phys_ids(W):
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    return [int(d) for d in vis.split(",")][:W] if vis else list(range(W))


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    platform = _platform(os.environ.get("RS_NORM_AG_PLATFORM", "auto"))
    platform.api.set_device(local_rank)
    W0 = int(os.environ["WORLD_SIZE"])
    if W0 != int(os.environ.get("LOCAL_WORLD_SIZE", W0)):
        raise ValueError("This tutorial is designed for a single node")
    H = int(os.environ.get("H", "7168"))
    Q, KV, D = (int(os.environ.get(k, v)) for k, v in (("Q", "1536"), ("KV", "512"), ("D", "2112")))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[os.environ.get("DTYPE", "bf16").lower()]
    tokens = [int(t) for t in os.environ.get("TOKENS", "1,8,32,128,512,2048,8192").split(",")]
    iters = int(os.environ.get("ITERS", "20"))
    nchannels = int(os.environ.get("NCHANNELS", "16"))
    T_max = max(tokens)
    eps = 1e-6
    rings, ring_kind = _default_rings(platform, W0)

    comm = RingComm(
        max(fused_rs_add_rms_norm_ag.parity_bytes(W0, T_max, H, dtype),
            fused_qkv_norm_ag.parity_bytes(W0, T_max, D, dtype)),
        platform=platform,
        peer=os.environ.get("PEER", "auto"),
        rings=rings,
        nchannels=nchannels,
    )
    rank, W = comm.rank, comm.world_size
    log = print if rank == 0 else (lambda *a, **k: None)
    rs = fused_rs_add_rms_norm_ag(comm, T_max, H, dtype)
    qkv = fused_qkv_norm_ag(comm, T_max, Q, KV, D, dtype)
    log(f"{platform.name} {platform.api.get_device_name()} W={W} {dtype} H={H} Q/KV/D={Q}/{KV}/{D} max_tokens={T_max} "
        f"peer={'tle' if comm.use_tle else 'ipc'} rings={ring_kind}({len(comm.rings)}) "
        f"nchannels={comm.nchannels} SMs={comm.max_programs} buffer={comm.buf.numel() / 2**20:.0f} MiB")

    gw = torch.Generator("cuda").manual_seed(1)
    weight = torch.randn(H, dtype=dtype, device="cuda", generator=gw)
    qw = torch.randn(Q, dtype=dtype, device="cuda", generator=gw)
    kvw = torch.randn(KV, dtype=dtype, device="cuda", generator=gw)

    # odd sizes cover empty trailing ranks; growing then shrinking T changes the grid between
    # calls; the three ops interleave on the shared buffer, parity and flags
    sizes = sorted({1, W - 1, W + 1, 37, 513, T_max} - {0})
    for T in sizes + sizes[::-1]:
        _check_rs(rs, T, weight, eps, gather=True, seed=1)
        _check_qkv(qkv, T, qw, kvw, eps, seed=2)
        _check_rs(rs, T, weight, eps, gather=False, seed=3)
    errs = torch.tensor(comm.errors(), device="cuda")
    dist.all_reduce(errs)
    if errs.item():
        raise RuntimeError(f"{errs.item()} wait timeouts")
    log(f"correctness: rs+norm+ag / qkv norm+ag / rs+norm, T={sizes} up and down: pass")

    fg = None
    fg_path = os.environ.get("COMPARE_FLAGGEMS")
    if fg_path:
        fgm = _load_flaggems(fg_path)
        cpu = dist.new_group(backend="gloo")
        fg_rings = comm.rings if ring_kind == "ppu810e" else None
        fg = (fgm.fused_rs_add_rms_norm_ag(dist.group.WORLD, comm.device, T_max, H, dtype, cpu_group=cpu,
                                           rings=fg_rings, nchannels=nchannels),
              fgm.fused_qkv_norm_ag(dist.group.WORLD, comm.device, T_max, Q, KV, D, dtype, cpu_group=cpu,
                                    rings=fg_rings, nchannels=nchannels))

    results = []
    for T in tokens:
        b, e = comm.row_range(T)
        chunk = triton.cdiv(T, W)
        x = _rank_input(rank, T, H, dtype, 9)
        out = torch.empty_like(x)
        out_s = torch.empty(e - b, H, dtype=dtype, device="cuda")
        residual_full = _shared_input(T, H, dtype, 9)
        residual = residual_full[b:e].contiguous()
        xq = _shared_input(T, D, dtype, 10)[b:e].contiguous()
        qkv_out = torch.empty(T, D, dtype=dtype, device="cuda")
        # NCCL all_gather needs equal shards: pad to chunk rows
        xq_pad = torch.zeros(chunk, D, dtype=dtype, device="cuda")
        xq_pad[:e - b] = xq
        ag_out = torch.empty(W * chunk, D, dtype=dtype, device="cuda")
        xt = x.clone()  # all_reduce scratch: values blow up over the iterations, time does not

        def nccl_ar_norm():
            dist.all_reduce(xt)
            return _rms(xt.float() + residual_full.float(), weight, eps).to(dtype)

        def nccl_ag_norm():
            dist.all_gather_into_tensor(ag_out, xq_pad)
            # rank r's shard lands at rows [r * chunk, (r + 1) * chunk): the first T rows are the batch
            y = ag_out[:T].float()
            y[:, :Q] = _rms(y[:, :Q], qw, eps)
            y[:, Q:Q + KV] = _rms(y[:, Q:Q + KV], kvw, eps)
            return y.to(dtype)

        r = {"T": T}
        r["tle_rs_norm_ag_us"] = _bench(lambda: rs(x, residual, weight, eps, out=out), iters)
        r["tle_rs_norm_us"] = _bench(lambda: rs(x, residual, weight, eps, out=out_s, gather=False), iters)
        r["tle_qkv_norm_ag_us"] = _bench(lambda: qkv(xq, T, qw, eps, kvw, eps, out=qkv_out), iters)
        r["nccl_ar_us"] = _bench(lambda: dist.all_reduce(xt), iters)
        r["nccl_ar_norm_us"] = _bench(nccl_ar_norm, iters)
        r["nccl_ag_norm_us"] = _bench(nccl_ag_norm, iters)
        if fg:
            fg_rs, fg_qkv = fg
            r["fg_rs_norm_ag_us"] = _bench(lambda: fg_rs(x, residual, weight, eps, out=out), iters)
            r["fg_rs_norm_us"] = _bench(lambda: fg_rs(x, residual, weight, eps, out=out_s, gather=False), iters)
            r["fg_qkv_norm_ag_us"] = _bench(lambda: fg_qkv(xq, T, qw, eps, kvw, eps, out=qkv_out), iters)
        results.append(r)
        fgs = (f" FlagGems {r['fg_rs_norm_ag_us']:7.1f}" if fg else "")
        fgq = (f" FlagGems {r['fg_qkv_norm_ag_us']:7.1f}" if fg else "")
        log(f"T={T:5d} | rs+norm+ag: flagtree {r['tle_rs_norm_ag_us']:7.1f}{fgs} AR+norm {r['nccl_ar_norm_us']:7.1f}"
            f" (AR {r['nccl_ar_us']:7.1f}) | rs+norm: flagtree {r['tle_rs_norm_us']:7.1f}"
            + (f" FlagGems {r['fg_rs_norm_us']:7.1f}" if fg else "") +
            f" | qkv norm+ag: flagtree {r['tle_qkv_norm_ag_us']:7.1f}{fgq} AG+norm {r['nccl_ag_norm_us']:7.1f}  (us)")
    errs = torch.tensor(comm.errors() + (sum(o.errors() for o in fg) if fg else 0), device="cuda")
    dist.all_reduce(errs)
    if errs.item():
        raise RuntimeError(f"{errs.item()} wait timeouts during the benchmark")
    out_json = os.environ.get("OUT_JSON")
    if rank == 0 and out_json:
        meta = {"W": W, "dtype": str(dtype), "H": H, "Q": Q, "KV": KV, "D": D, "iters": iters,
                "gpu": platform.api.get_device_name(), "platform": platform.name, "peer": "tle" if comm.use_tle else "ipc",
                "rings": ring_kind, "nchannels": comm.nchannels}
        with open(out_json, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=1)

    if fg:
        for o in fg:
            o.close()
    comm.close()


if __name__ == "__main__":
    main()
