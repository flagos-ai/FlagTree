"""CPU compilation regressions for wave-private Padded LDS synchronization."""
import pytest
import triton
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler bindings are required", allow_module_level=True)
pytest.importorskip("triton.experimental.gluon.language.hcu")

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language.amd.cdna3 import buffer_load, buffer_store, mfma
from triton.experimental.gluon.language.hcu import warp_group_id
from triton.language import range as pipeline_range
from triton.runtime.jit import constexpr_function


@constexpr_function
def producer_layout(rows, cooperative=False):
    assert rows >= 8 and rows & (rows - 1) == 0
    registers = [[0, 0, 1], [0, 0, 2], [0, 0, 4]]
    bit = 8
    while bit < rows:
        registers.append([0, bit, 0]); bit *= 2
    if cooperative:
        # Same logical domain; exchange one row lane-bit with one batch wave-bit.
        # FullCTA remains numerically valid, private-wave proof MUST decline.
        return gl.DistributedLinearLayout(
            registers, [[0, 0, 8], [0, 0, 16], [0, 0, 32],
                        [1, 0, 0], [0, 2, 0], [0, 4, 0]],
            [[0, 1, 0], [2, 0, 0]], [], [4, rows, 64])
    # lane K8/K16/K32 then row1/2/4; wave bits select the K64 partition.
    return gl.DistributedLinearLayout(
        registers, [[0, 0, 8], [0, 0, 16], [0, 0, 32],
                    [0, 1, 0], [0, 2, 0], [0, 4, 0]],
        [[1, 0, 0], [2, 0, 0]], [], [4, rows, 64])


@constexpr_function
def partial_layout(bm, bn):
    # Actual rank3 HCU LEGACY, transposed=False, native four-accumulator group.
    # Physical offset bits: r0,r1,lane0..5,remaining r,split0,split1.
    bases = [[0, 0, 4], [0, 0, 8], [0, 1, 0], [0, 2, 0],
             [0, 4, 0], [0, 8, 0], [0, 0, 1], [0, 0, 2]]
    bit = 16
    while bit < bn:
        bases.append([0, 0, bit]); bit *= 2
    bit = 16
    while bit < bm:
        bases.append([0, bit, 0]); bit *= 2
    bases += [[1, 0, 0], [2, 0, 0]]
    # This interval is the whole allocation: no interior C padding.
    return gl.PaddedSharedLayout([[4 * bm * bn, 1]], bases, [], [4, bm, bn])


@g.jit
def input_coords(D: gl.constexpr, L: gl.constexpr):
    sp = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    row = gl.arange(0, D, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    k = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    return row[None, :, None], sp[:, None, None] * 64 + k[None, None, :]


@g.jit
def _wave_private_control(A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                           BM: gl.constexpr = 32, BN: gl.constexpr = 32, PROBE: gl.constexpr = 0):
    gl.static_assert(M > 0 and N > 0 and K >= 256 and K % 256 == 0)
    gl.static_assert(BM >= 16 and BN >= 16 and BM & (BM - 1) == 0 and BN & (BN - 1) == 0)
    gl.static_assert(BM + BN <= 112 and BM * BN <= 4096)
    gl.static_assert(2 * M * K < 2147483648 and 2 * N * K < 2147483648 and 2 * M * N < 2147483648)
    gl.static_assert(A.dtype.element_ty == gl.bfloat16 or A.dtype.element_ty == gl.float16)
    gl.static_assert(B.dtype.element_ty == A.dtype.element_ty and C.dtype.element_ty == A.dtype.element_ty)
    mma: gl.constexpr = AMDMFMALayout(3, [16, 16, 16], False, [4, 1, 1],
                                     tiles_per_warp=[1, 1, 1], mmac_layout=1)
    da: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    db: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    pa: gl.constexpr = producer_layout(BM, PROBE == 1)
    pb: gl.constexpr = producer_layout(BN, PROBE == 1)
    sa: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[64, 8]], [4, BM, 64], [0, 2, 1] if PROBE == 2 else [2, 1, 0])
    sb: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[64, 8]], [4, BN, 64], [0, 2, 1] if PROBE == 2 else [2, 1, 0])
    rm, ka = input_coords(BM, pa)
    rn, kb = input_coords(BN, pb)
    row_m = gl.program_id(0) * BM
    col_n = gl.program_id(1) * BN
    if PROBE == 3:
        # A deliberately opaque sideeffect asm must disable ALL private roots.
        gl.inline_asm_elementwise("", constraints="=s,0", args=[row_m], dtype=gl.int32, is_pure=False, pack=1)
    abase = A + row_m.to(gl.int64) * K
    bbase = B + col_n.to(gl.int64) * K
    ao = (rm * K + ka).to(gl.int32)
    bo = (rn * K + kb).to(gl.int32)
    if M % BM == 0:
        am = gl.full([4, BM, 64], True, gl.int1, pa)
    else:
        am = row_m + rm < M
    if N % BN == 0:
        bm = gl.full([4, BN, 64], True, gl.int1, pb)
    else:
        bm = col_n + rn < N
    acc = gl.full([4, BM, BN], 0, gl.float32, mma)
    aa = gl.allocate_shared_memory(A.dtype.element_ty, [4, BM, 64], sa)
    bb = gl.allocate_shared_memory(A.dtype.element_ty, [4, BN, 64], sb)
    av = buffer_load(abase, ao, am, other=0)
    bv = buffer_load(bbase, bo, bm, other=0)
    for q in pipeline_range(0, K // 256, loop_unroll_factor=1):
        aa.store(av)
        bb.store(bv)
        # No explicit A/B CTA: compiler must prove wave-private ownership
        # and supply per-wave LDS waits, or conservatively retain full CTA.
        ar = aa.load(da)
        br = bb.permute([0, 2, 1]).load(db)
        have_next = q + 1 < K // 256
        next_a = buffer_load(abase + (q + 1) * 256, ao, am & have_next, other=0)
        next_b = buffer_load(bbase + (q + 1) * 256, bo, bm & have_next, other=0)
        acc = mfma(ar, br, acc)
        av = next_a
        bv = next_b
    aa._keep_alive()
    bb._keep_alive()

    pc: gl.constexpr = partial_layout(BM, BN)
    cs = gl.allocate_shared_memory(gl.float32, [4, BM, BN], pc)
    cs.store(acc)
    gl.thread_barrier()
    wave = warp_group_id(1)
    if wave == 0:
        # batch=1 makes the MFMA warp bits redundant. All C coordinates live
        # in wave0, with exactly the same native register/lane mapping.
        c0 = cs.slice(0, 1, 0).load(mma)
        c1 = cs.slice(1, 1, 0).load(mma)
        c2 = cs.slice(2, 1, 0).load(mma)
        c3 = cs.slice(3, 1, 0).load(mma)
        value = ((c0 + c1) + (c2 + c3)).reshape([BM, BN])
        cl: gl.constexpr = gl.SliceLayout(0, mma)
        value = gl.convert_layout(value, cl)
        mi = gl.arange(0, BM, layout=gl.SliceLayout(1, cl))
        ni = gl.arange(0, BN, layout=gl.SliceLayout(0, cl))
        co = (mi[:, None] * N + ni[None, :]).to(gl.int32)
        cbase = C + row_m.to(gl.int64) * N + col_n.to(gl.int64)
        buffer_store(value.to(C.dtype.element_ty), cbase, co,
                     (row_m + mi[:, None] < M) & (col_n + ni[None, :] < N))


@pytest.mark.parametrize("probe,dtype,private", [
    (0, "bf16", True),
    (0, "fp16", True),
    (1, "bf16", False),  # Producer lanes cross wave ownership.
    (2, "bf16", False),  # Physical planes fail the current certificate.
    (3, "bf16", False),  # Opaque effects retain full synchronization.
])
def test_gluon_wave_private_padded_sync(probe, dtype, private):
    # Explicit target compilation never queries or initializes a GPU driver.
    # The kernel keeps an unrelated FP32 C allocation with sliced aliases:
    # declining those aliases must preserve an independent A/B certificate.
    fn = _wave_private_control
    constants = dict(M=32, N=32, K=256, BM=32, BN=32, PROBE=probe)
    signature = {name: ("*" + dtype if name in ("A", "B", "C") else "constexpr")
                 for name in fn.arg_names}
    source = GluonASTSource(fn, signature, constants)
    compiled = []
    for enabled in (False, True):
        compiled.append(triton.compile(
            source, target=GPUTarget("hip", "gfx936", 64),
            options=dict(num_warps=4, num_stages=1, waves_per_eu=1,
                         padded_slot_intervals=True, wave_private_sync=enabled,
                         loop_unroll_stage="post-pipeline",
                         disable_llvm_loop_unroll=True, kpack=2,
                         mmac_transposed=False),
        ))
    before, after = compiled
    def waves(kernel):
        return sum("call" in line and "@llvm.amdgcn.wave.barrier" in line
                   for line in kernel.asm["llir"].splitlines())
    assert waves(before) == 0
    if private:
        # Gluon bypasses make_ttgir: this fails if option forwarding is lost.
        assert waves(after) > 0
        assert before.asm["hsaco"] != after.asm["hsaco"]
    else:
        assert waves(after) == 0
        assert before.asm["hsaco"] == after.asm["hsaco"]
