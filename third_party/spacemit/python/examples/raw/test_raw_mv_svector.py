"""smt_rvisa svector-level mv (feishu 3.3 example).

C = B @ A   with  B: [N, K] f16 row-major,  A: [K] f16,  C: [N] f32.

style2 (pure svector): vconfig/vzero/vload/vmacc/vreduce_sum/vstore, no packing.
style3 (svector + pack): the same, but B's 4-row block is pre-packed into a
        contiguous scratch buffer via srv.alloc + srv.pack before the K loop.

(Matrix-unit mv/mm goes through srv.vmadot -> vector_ext.cross_batch_matmul;
 see test_raw_mm_cbm.py / test_raw_mv_cbm.py.)

Fixed VL (f16 -> 64) this round: no dynamic vsetvl tail handling, so the tests
constrain K % 64 == 0 and N % 4 == 0 (full tiles only).

Memory access follows SPEC §6.2: vload(ptr, index)/vstore(ptr, index, value),
where index is a flat scalar element offset and 2D coordinates are flattened
by the user (e.g. B row ni column ki is written ni*K + ki). For ranked scratch
allocated via alloc (packed_B), the index is still a per-dimension index tuple
(natural ranked addressing).
"""

import torch
import triton
import triton.language as tl
from triton.backends.spacemit.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.smt_rvisa as srv
from triton.language.extra.smt_rvisa import call as srv_call

f16 = srv.f16
f32 = srv.f32


# ---------------------------------------------------------------------------
# style2 — pure svector
# ---------------------------------------------------------------------------
@srv.raw_kernel
def mv_block_style2(B: srv.mem(f16), A: srv.mem(f16), C: srv.mem(f32, out=True), K: srv.index, row_base: srv.index,
                    row_end: srv.index):
    # Grid parallelism: the host splits the N rows by program_id; this program
    # computes only rows [row_base, row_end).
    nvl = srv.vconfig(-1, 1)  # lmul=1 → VLMAX=64 (f16, SPEC §6.1)
    # strip-mine: the K loop is split into a main loop (full tiles) + a tail
    # loop (the K%VL remainder). Whether to pad is decided by codegen at
    # compile time based on "whether a narrowing vconfig was set" (plain
    # Python if; no runtime scf.if is emitted into the IR): the main loop sets
    # no narrowing → vload takes the fast direct-read path with zero padding
    # and zero branches; the tail loop sets vconfig(K-ki) → vload fill-0 pad.
    # The tail scf.for naturally runs 0 iterations (K%VL==0, evenly divisible
    # shape) or 1, replacing a branch with iteration count.
    Kfloor = (K // nvl) * nvl  # K range covered by full tiles (multiple of VL)
    for ni in srv.range(row_base, row_end, 4):
        acc0 = srv.vzero(f32)
        acc1 = srv.vzero(f32)
        acc2 = srv.vzero(f32)
        acc3 = srv.vzero(f32)
        for ki in srv.range(0, Kfloor, nvl):  # main loop: full tiles, fast direct read (no narrowing set)
            va = srv.vload(A, ki)
            vb0 = srv.vload(B, ni * K + ki)
            vb1 = srv.vload(B, (ni + 1) * K + ki)
            vb2 = srv.vload(B, (ni + 2) * K + ki)
            vb3 = srv.vload(B, (ni + 3) * K + ki)
            acc0 = srv.vmacc(acc0, vb0, va)
            acc1 = srv.vmacc(acc1, vb1, va)
            acc2 = srv.vmacc(acc2, vb2, va)
            acc3 = srv.vmacc(acc3, vb3, va)
        for ki in srv.range(Kfloor, K, nvl):  # tail loop: runs 0/1 times, narrowed → codegen uses fill-0 pad
            nvl = srv.vconfig(K - ki, 1)
            # Use distinct temporary names (ta/tb*): if they shared names with
            # the main loop's va/vb*, those would leak to the outer scope and
            # the tail loop's iter_arg detection (_find_reassigned) would
            # mistake these pure temporaries for loop-carried values →
            # generating bad iter_args that reference main-loop SSA values
            # already out of scope.
            ta = srv.vload(A, ki)
            tb0 = srv.vload(B, ni * K + ki)
            tb1 = srv.vload(B, (ni + 1) * K + ki)
            tb2 = srv.vload(B, (ni + 2) * K + ki)
            tb3 = srv.vload(B, (ni + 3) * K + ki)
            acc0 = srv.vmacc(acc0, tb0, ta)
            acc1 = srv.vmacc(acc1, tb1, ta)
            acc2 = srv.vmacc(acc2, tb2, ta)
            acc3 = srv.vmacc(acc3, tb3, ta)
        srv.sstore(C, ni, srv.vreduce_sum(acc0))
        srv.sstore(C, ni + 1, srv.vreduce_sum(acc1))
        srv.sstore(C, ni + 2, srv.vreduce_sum(acc2))
        srv.sstore(C, ni + 3, srv.vreduce_sum(acc3))


@triton.jit(do_not_specialize=["K", "N"])
def _mv_sv_host_style2(B, A, C, K, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    row_end = min(row_base + BLOCK, N)  # clamp rows: last program stays within N, no phantom B-row reads
    srv_call(mv_block_style2, outputs=[], inputs=[B, A, C, K, row_base, row_end])


# ---------------------------------------------------------------------------
# style3 — svector + srv.alloc/srv.pack pre-packing B
# ---------------------------------------------------------------------------
@srv.raw_kernel
def mv_block_style3(B: srv.mem(f16), A: srv.mem(f16), C: srv.mem(f32, out=True), K: srv.index, row_base: srv.index,
                    row_end: srv.index):
    # Grid parallelism: this program computes only rows [row_base, row_end).
    nvl = srv.vconfig(-1, 1)  # lmul=1 → VLMAX=64 (f16, SPEC §6.1)
    # The tile count must be ceil(K/nvl): pack's kb loop runs 0..K step nvl =
    # ceil(K/nvl) tiles; using floor(K//nvl) for alloc under-allocates when
    # K%64!=0 → pack writes dst[0,ceil-1,..] out of bounds on dim-1 → heap
    # buffer overflow (cross-kernel corruption, NaN). Use ceil to match the
    # number of tiles pack actually writes.
    kct = (K + nvl - 1) // nvl
    packed_B = srv.alloc((1, kct, 4, nvl), f16)
    # strip-mine (same as style2): the main loop reads full tiles via the
    # vload fast path (no narrowing set); the tail loop (the K%VL remainder)
    # sets vconfig(K-ki) → codegen uses fill-0 pad at compile time. Whether to
    # pad is decided by codegen based on narrowing, no runtime scf.if; the
    # tail scf.for runs 0/1 times instead of a branch. packed_B's tail tile is
    # already zero-padded by pack.
    Kfloor = (K // nvl) * nvl
    for ni in srv.range(row_base, row_end, 4):
        acc0 = srv.vzero(f32)
        acc1 = srv.vzero(f32)
        acc2 = srv.vzero(f32)
        acc3 = srv.vzero(f32)
        srv.pack(B, (ni, 0), packed_B, (1, kct, 4, nvl), K)
        for ki in srv.range(0, Kfloor, nvl):  # main loop: full tiles, fast path
            vb0 = srv.vload(packed_B, (0, ki // nvl, 0, 0))
            vb1 = srv.vload(packed_B, (0, ki // nvl, 1, 0))
            vb2 = srv.vload(packed_B, (0, ki // nvl, 2, 0))
            vb3 = srv.vload(packed_B, (0, ki // nvl, 3, 0))
            va = srv.vload(A, ki)
            acc0 = srv.vmacc(acc0, vb0, va)
            acc1 = srv.vmacc(acc1, vb1, va)
            acc2 = srv.vmacc(acc2, vb2, va)
            acc3 = srv.vmacc(acc3, vb3, va)
        for ki in srv.range(Kfloor, K,
                            nvl):  # tail loop: runs 0/1 times, narrowed → fill-0 (distinct temporaries tb*/ta)
            nvl = srv.vconfig(K - ki, 1)
            tb0 = srv.vload(packed_B, (0, ki // nvl, 0, 0))
            tb1 = srv.vload(packed_B, (0, ki // nvl, 1, 0))
            tb2 = srv.vload(packed_B, (0, ki // nvl, 2, 0))
            tb3 = srv.vload(packed_B, (0, ki // nvl, 3, 0))
            ta = srv.vload(A, ki)
            acc0 = srv.vmacc(acc0, tb0, ta)
            acc1 = srv.vmacc(acc1, tb1, ta)
            acc2 = srv.vmacc(acc2, tb2, ta)
            acc3 = srv.vmacc(acc3, tb3, ta)
        srv.sstore(C, ni, srv.vreduce_sum(acc0))
        srv.sstore(C, ni + 1, srv.vreduce_sum(acc1))
        srv.sstore(C, ni + 2, srv.vreduce_sum(acc2))
        srv.sstore(C, ni + 3, srv.vreduce_sum(acc3))


@triton.jit(do_not_specialize=["K", "N"])
def _mv_sv_host_style3(B, A, C, K, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    row_end = min(row_base + BLOCK, N)  # clamp rows: last program stays within N, pack reads no phantom B rows
    srv_call(mv_block_style3, outputs=[], inputs=[B, A, C, K, row_base, row_end])


def _run(host, N, K, BLOCK=4):
    # Grid parallelism: the N rows are split into ceil(N/BLOCK) programs (each
    # computes BLOCK rows, 4 rows per inner group). When N%BLOCK!=0 the last
    # program's row_end exceeds N and the inner unconditional vstore(C,
    # ni..ni+3) writes phantom rows C[N..Np-1]. C must be allocated with
    # Np=ceil(N/BLOCK)*BLOCK rows to absorb these writes (extra allocation
    # only, no data copy), otherwise the writes go out of bounds → heap
    # corruption (fix: N-tail phantom-row overflow). Slice C[:N] for the
    # result.
    Np = ((N + BLOCK - 1) // BLOCK) * BLOCK
    B = torch.randn(N, K, dtype=torch.float16)
    A = torch.randn(K, dtype=torch.float16)
    C = torch.empty(Np, dtype=torch.float32)
    grid = (Np // BLOCK, )
    host[grid](B.contiguous().reshape(-1), A.contiguous(), C, K, N, BLOCK=BLOCK)
    got = C[:N]
    ref = torch.mv(B.float(), A.float())
    max_diff = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, rtol=1e-2, atol=1e-2), \
        f"N={N} K={K} max_diff={max_diff:.4e}"


_SHAPES = [(4, 64), (8, 128), (16, 256), (32, 512), (64, 64), (128, 256)]


@pytest.mark.parametrize("N, K", _SHAPES)
def test_raw_mv_svector_style2(N, K):
    _run(_mv_sv_host_style2, N, K)


@pytest.mark.parametrize("N, K", _SHAPES)
def test_raw_mv_svector_style3(N, K):
    _run(_mv_sv_host_style3, N, K)


# ---------------------------------------------------------------------------
# Arbitrary shapes — all padding inside the kernel (zero host copy)
# ---------------------------------------------------------------------------
# Both constraints are resolved inside the kernel/host wrapper with no host
# copy of B/A:
#   (1) K%64: the K-tail vload is demoted by vconfig(K-ki,1) to a fill-0
#       scratch, tail lanes padded with 0 (confirmed by probe_svpad_fill);
#       vmacc/vreduce_sum are unaffected by the zero padding.
#   (2) N%BLOCK: grid=ceil(N/BLOCK), the last program's row_end>N and the
#       inner unconditional store writes phantom rows C[N..Np-1]; C is
#       allocated to Np=ceil(N/BLOCK)*BLOCK to absorb them (allocation only,
#       no copy), then sliced C[:N].
# _run already handles both (1) and (2), so arbitrary shapes reuse _run
# directly.
# Arbitrary shapes: K not a multiple of 64 / N not a multiple of BLOCK / both
_SHAPES_ARB = [(4, 60), (4, 100), (8, 65), (4, 63), (8, 127), (4, 200), (16, 130), (12, 50), (7, 64), (33, 65),
               (50, 130), (100, 100), (6, 60), (13, 200), (37, 130)]


@pytest.mark.parametrize("N, K", _SHAPES_ARB)
def test_raw_mv_svector_style2_arb(N, K):
    _run(_mv_sv_host_style2, N, K)


@pytest.mark.parametrize("N, K", _SHAPES_ARB)
def test_raw_mv_svector_style3_arb(N, K):
    _run(_mv_sv_host_style3, N, K)
