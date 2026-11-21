# accumulator fence
"""Typed wave scheduling state, generic cohorts and effectful IR validation."""
import pytest
import triton
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)

import triton.language as tl
from triton.compiler import ASTSource
from triton.language.extra.hip import accumulator_fence as tl_accumulator_fence

from triton.backends.hcu.compiler import HIPBackend
from triton.experimental.gluon.language.hcu import accumulator_fence


@g.jit
def _accumulator_fence_fenced_copy(A, O, COHORT: gl.constexpr, UNUSED: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([COHORT], [64], [4], [0])
    i = gl.arange(0, COHORT * 256, layout=layout)
    value = gl.load(A + i)
    value = accumulator_fence(value, 1)
    value = accumulator_fence(value, 0)
    if not UNUSED:
        gl.store(O + i, value)
    else:
        gl.store(O, 0)


def _accumulator_fence_compile_copy(cohort, dtype="fp32", unused=False):
    source = GluonASTSource(
        _accumulator_fence_fenced_copy,
        {"A": "*" + dtype, "O": "*" + dtype, "COHORT": "constexpr", "UNUSED": "constexpr"},
        {"COHORT": cohort, "UNUSED": unused},
    )
    return triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options=dict(num_warps=4, num_stages=1))


@pytest.mark.parametrize("cohort", [1, 4, 16, 128])
@pytest.mark.parametrize("dtype", ["fp32", "i32"])
def test_priority_fence_derives_actual_cohort(cohort, dtype):
    kernel = _accumulator_fence_compile_copy(cohort, dtype)
    # The typed SSA operation survives until target lowering. Neither its
    # source expression nor TTGIR relies on elementwise inline assembly.
    assert kernel.asm["ttgir"].count("hcug.accumulator_fence") == 2
    assert "elementwise_inline_asm" not in kernel.asm["ttgir"]
    assert "s_setprio 1" in kernel.asm["amdgcn"]
    assert "s_setprio 0" in kernel.asm["amdgcn"]
    assert "~{memory}" in kernel.asm["llir"]


def test_unused_result_cannot_eliminate_or_merge_priority_state():
    kernel = _accumulator_fence_compile_copy(4, unused=True)
    assert kernel.asm["ttgir"].count("hcug.accumulator_fence") == 2
    assert "s_setprio 1" in kernel.asm["amdgcn"]
    assert "s_setprio 0" in kernel.asm["amdgcn"]


@pytest.mark.parametrize("dtype,priority,message", [
    ("f16", 1, "f32 or i32"),
    ("f32", 2, "priority must be 0"),
    ("i32", -1, "priority must be 0"),
])
def test_imported_invalid_fence_is_rejected(dtype, priority, message, tmp_path, capfd):
    text = f'''module {{
  tt.func public @invalid(%x: tensor<8x{dtype}>) {{
    %y = "hcug.accumulator_fence"(%x) {{priority = {priority} : i32}}
      : (tensor<8x{dtype}>) -> tensor<8x{dtype}>
    tt.return
  }}
}}'''
    path = tmp_path / "invalid.mlir"
    path.write_text(text)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    with pytest.raises(RuntimeError):
        libtriton.ir.parse_mlir_module(str(path), context)
    assert message in capfd.readouterr().err


# Ordinary TL must convert the target operation's tensor type and propagate
# encodings through it without treating the effectful fence as a pure op.


@triton.jit
def _accumulator_fence_tl_fenced_copy(A, O, COHORT: tl.constexpr, UNUSED: tl.constexpr):
    i = tl.arange(0, COHORT * 256)
    value = tl.load(A + i)
    value = tl_accumulator_fence(value, 1)
    value = tl_accumulator_fence(value, 0)
    if UNUSED:
        tl.store(O, 0)
    else:
        tl.store(O + i, value)


@pytest.mark.parametrize("cohort", [1, 16, 128])
@pytest.mark.parametrize("dtype", ["fp32", "i32"])
@pytest.mark.parametrize("unused", [False, True])
def test_ordinary_tl_fence_effects(cohort, dtype, unused):
    source = ASTSource(
        _accumulator_fence_tl_fenced_copy,
        {"A": "*" + dtype, "O": "*" + dtype, "COHORT": "constexpr", "UNUSED": "constexpr"},
        {"COHORT": cohort, "UNUSED": unused},
    )
    kernel = triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options=dict(num_warps=4, num_stages=1))
    assert kernel.asm["ttgir"].count("hcug.accumulator_fence") == 2
    assert "elementwise_inline_asm" not in kernel.asm["ttgir"]
    assert "s_setprio 1" in kernel.asm["amdgcn"]
    assert "s_setprio 0" in kernel.asm["amdgcn"]
    assert "~{memory}" in kernel.asm["llir"]


@triton.jit
def _accumulator_fence_tl_fenced_dot(A, B, C, BM: tl.constexpr, BN: tl.constexpr, FENCED: tl.constexpr):
    mi = tl.arange(0, BM)
    ni = tl.arange(0, BN)
    ki = tl.arange(0, 32)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(4):
        kk = start * 32 + ki
        av = tl.load(A + mi[:, None] * 128 + kk[None, :])
        bv = tl.load(B + kk[:, None] * BN + ni[None, :])
        if FENCED:
            acc = tl_accumulator_fence(acc, 1)
        acc = tl.dot(av, bv, acc, allow_tf32=False)
        if FENCED:
            acc = tl_accumulator_fence(acc, 0)
    tl.store(C + mi[:, None] * BN + ni[None, :], acc.to(C.dtype.element_ty))


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("bm,bn,warps", [(32, 32, 4), (64, 32, 4), (256, 256, 8)])
@pytest.mark.parametrize("delayed_priority", [False, True])
def test_ordinary_tl_dot_fence_keeps_mmac_layout(dtype, bm, bn, warps, delayed_priority):
    kernels = []
    for fenced in (False, True):
        source = ASTSource(
            _accumulator_fence_tl_fenced_dot,
            {
                "A": "*" + dtype, "B": "*" + dtype, "C": "*" + dtype, "BM": "constexpr", "BN": "constexpr", "FENCED":
                "constexpr"
            },
            {"BM": bm, "BN": bn, "FENCED": fenced},
        )
        kernels.append(
            triton.compile(
                source,
                target=GPUTarget("hip", "gfx936", 64),
                options=dict(num_warps=warps, num_stages=1, kpack=8, mmac_transposed=True,
                             disable_llvm_loop_unroll=True, priority_after_first_mmac=delayed_priority),
            ))
    plain, fenced = kernels
    lines = [line for line in fenced.asm["ttgir"].splitlines() if "hcug.accumulator_fence" in line]
    assert len(lines) == 2
    assert all("#mma" in line for line in lines), lines
    assert fenced.metadata.shared == plain.metadata.shared
    assert fenced.asm["amdgcn"].count("s_barrier") == plain.asm["amdgcn"].count("s_barrier")


@triton.jit
def _accumulator_fence_tl_multiple_fence_users(A, B, D, O, P):
    i = tl.arange(0, 32)
    a = tl.load(A + i[:, None] * 32 + i[None, :])
    b = tl.load(B + i[:, None] * 32 + i[None, :])
    d = tl.load(D + i[:, None] * 32 + i[None, :])
    value = tl.dot(a, b, allow_tf32=False)
    value = tl_accumulator_fence(value, 1)
    # These uses want different storage/dot-operand layouts. The effectful
    # fence may be re-encoded once, but must not be rematerialized per use.
    tl.store(O + i[:, None] * 32 + i[None, :], value)
    result = tl.dot(value.to(A.dtype.element_ty), d, allow_tf32=False)
    result = tl_accumulator_fence(result, 0)
    tl.store(P + i[:, None] * 32 + i[None, :], result)


def test_ordinary_tl_multiple_users_do_not_duplicate_priority_effect():
    source = ASTSource(
        _accumulator_fence_tl_multiple_fence_users,
        {"A": "*bf16", "B": "*bf16", "D": "*bf16", "O": "*fp32", "P": "*fp32"},
        {},
    )
    kernel = triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options=dict(num_warps=4, num_stages=1))
    assert kernel.asm["ttgir"].count("hcug.accumulator_fence") == 2
    assert kernel.asm["amdgcn"].count("s_setprio 1") == 1
    assert kernel.asm["amdgcn"].count("s_setprio 0") == 1


# priority schedule
import re

from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language.amd.cdna3 import mfma


@g.jit
def _priority_schedule_priority_dot(A, B, C, SIDE, M: gl.constexpr, EXTRA_USE: gl.constexpr):
    load: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [2, 2], [1, 0])
    mma: gl.constexpr = AMDMFMALayout(3, [16, 16, 16], True, [2, 2], tiles_per_warp=[1, 1], mmac_layout=1)
    x = gl.arange(0, M, layout=gl.SliceLayout(1, load))
    k = gl.arange(0, 32, layout=gl.SliceLayout(0, load))
    a = gl.load(A + x[:, None] * 32 + k[None, :])
    b = gl.permute(gl.load(B + x[:, None] * 32 + k[None, :]), (1, 0))
    a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
    b = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8))
    acc = gl.full((M, M), 0, gl.float32, mma)
    acc = accumulator_fence(acc, 1)
    if EXTRA_USE:
        gl.store(SIDE, gl.sum(gl.sum(acc, 0), 0))
    acc = mfma(a, b, acc)
    acc = accumulator_fence(acc, 0)
    out: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [2, 2], [1, 0])
    i = gl.arange(0, M, layout=gl.SliceLayout(1, out))
    j = gl.arange(0, M, layout=gl.SliceLayout(0, out))
    gl.store(C + i[:, None] * M + j[None, :], gl.convert_layout(acc, out))


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("tile", [32, 64])
@pytest.mark.parametrize("defer,extra_use", [(False, False), (True, False), (True, True)])
def test_first_mmac_priority_has_actual_ssa_cohort(dtype, tile, defer, extra_use):
    source = GluonASTSource(
        _priority_schedule_priority_dot,
        {"A": "*" + dtype, "B": "*" + dtype, "C": "*fp32", "SIDE": "*fp32", "M": "constexpr", "EXTRA_USE": "constexpr"},
        {"M": tile, "EXTRA_USE": extra_use},
    )
    kernel = triton.compile(
        source,
        target=GPUTarget("hip", "gfx936", 64),
        options=dict(num_warps=4, num_stages=1, priority_after_first_mmac=defer),
    )
    assembly = kernel.asm["amdgcn"]
    high = assembly.index("s_setprio 1")
    preceding = re.findall(r"^\s*v_mmac_", assembly[:high], re.M)
    assert len(preceding) == (1 if defer and not extra_use else 0)
    assert assembly.count("s_setprio 1") == assembly.count("s_setprio 0") == 1
    assert kernel.asm["ttgir"].count("hcug.accumulator_fence") == 2
    assert "elementwise_inline_asm" not in kernel.asm["ttgir"]
    assert "~{memory}" in kernel.asm["llir"]


# shared schedule model


def _shared_schedule_model_loop_source(structured=False, control="linear", overlap=False):
    name = "llvm.amdgcn." + ("struct" if structured else "raw") + ".ptr.buffer.load.lds"
    signature = "ptr addrspace(8),ptr addrspace(3),i32,i32,i32,i32,i32" + (",i32" if structured else "")
    lines = [
        'target datalayout = "e-p:64:64-p3:32:32"', "@global_smem = external addrspace(3) global [0 x i8], align 16",
        f"declare void @{name}({signature})", "declare i32 @llvm.amdgcn.workitem.id.x() #0",
        "declare void @llvm.amdgcn.s.waitcnt(i32)", "declare void @llvm.amdgcn.s.barrier()",
        "declare <4 x float> @llvm.hcu.mmac.test(<4 x float>,<4 x float>) #0",
        "define amdgpu_kernel void @ring(ptr addrspace(8) inreg %rsrc,i32 inreg %unknown) {", "entry:",
        " %tid = call i32 @llvm.amdgcn.workitem.id.x()", " %wave = lshr i32 %tid,6", " %waveoff = shl i32 %wave,10",
        " %laneoff_base = shl i32 %tid,4"
    ]
    if control in ["select", "select_unknown"]:
        operand = "%unknown" if control == "select_unknown" else "%tid"
        lines += [
            f" %parity = and i32 {operand},1", " %choice = icmp eq i32 %parity,0",
            " %swap = select i1 %choice,i32 0,i32 16", " %laneoff = xor i32 %laneoff_base,%swap"
        ]
    else:
        lines.append(" %laneoff = add i32 %laneoff_base,0")
    if control == "legacy":
        lines += [
            f" %c = and i32 %wave,{3 if structured else 1}", f" %hi = shl i32 %c,{16 if structured else 18}",
            " %copyoff = or i32 %waveoff,%hi"
        ]
    elif control == "unknown":
        lines += [" %copyoff = or i32 %waveoff,1048576"]
    elif control == "nonuniform":
        lines += [" %copyoff = add i32 %waveoff,%tid"]
    else:
        lines += [" %copyoff = add i32 %waveoff,0"]
    for slot in range(4):
        for side in ["a", "b"]:
            base = slot * 16384 + (8192 if side == "a" and not overlap else 0)
            key = f"{side}{slot}"
            lines += [
                f" %copyindex{key} = add i32 %copyoff,{base}",
                f" %copy{key} = getelementptr i8,ptr addrspace(3) @global_smem,i32 %copyindex{key}",
                f" %readindex{key} = add i32 %laneoff,{base}",
                f" %read{key} = getelementptr i8,ptr addrspace(3) @global_smem,i32 %readindex{key}"
            ]

    def copy(slot):
        for side in ["a", "b"]:
            args = f"ptr addrspace(8) %rsrc,ptr addrspace(3) %copy{side}{slot},i32 16,i32 0,i32 0,i32 0,i32 0" + (
                ",i32 0" if structured else "")
            lines.append(f" call void @{name}({args})")

    def fence(count):
        lines.extend([
            f" call void @llvm.amdgcn.s.waitcnt(i32 {0x3ff0|count})", " call void @llvm.amdgcn.s.waitcnt(i32 49279)",
            " call void @llvm.amdgcn.s.barrier()"
        ])

    def read(slot, label):
        for side in ["a", "b"]:
            lines.append(f" %value{side}{label} = load <8 x i16>,ptr addrspace(3) %read{side}{slot},align 16")

    for slot in [0, 1, 2]:
        copy(slot)
    fence(4)
    read(0, "prefix")
    copy(3)
    lines.extend([" br label %body", "body:", " %iv = phi i32 [0,%entry],[%next,%body]"])
    for group, slot in enumerate([1, 2, 3, 0]):
        fence(4)
        read(slot, "loop" + str(group))
        copy((slot + 3) % 4)
        lines.append(
            f" %mma{group} = call <4 x float> @llvm.hcu.mmac.test(<4 x float> zeroinitializer,<4 x float> zeroinitializer)"
        )
    lines.extend(
        [" %next = add i32 %iv,1", " %more = icmp ult i32 %next,3", " br i1 %more,label %body,label %exit", "exit:"])
    for slot, count in [(1, 4), (2, 2), (3, 0)]:
        fence(count)
        read(slot, "tail" + str(slot))
    lines.extend([" ret void", "}", "attributes #0 = { nounwind willreturn memory(none) }"])
    return "\n".join(lines)


@pytest.mark.parametrize("binding", ["prepare_hcu_post_mma_wait"])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("control", ["linear", "legacy", "select"])
def test_actual_complete_wave_domains_preserve_cross_wave_proof(binding, structured, control):
    source = _shared_schedule_model_loop_source(structured, control)
    output, loops, reason = getattr(libtriton.llvm, binding)(source, "gfx936", 256, 65536)
    assert loops == 1, reason
    assert output != source and "PASS_" in reason


@pytest.mark.parametrize("kind", ["unknown", "nonuniform", "overlap", "select_unknown"])
@pytest.mark.parametrize("binding", ["prepare_hcu_post_mma_wait"])
def test_unproven_placement_and_overlapping_dma_rejected(binding, kind):
    source = _shared_schedule_model_loop_source(control=kind, overlap=kind == "overlap")
    output, loops, reason = getattr(libtriton.llvm, binding)(source, "gfx936", 256, 65536)
    assert loops == 0 and output == source


@pytest.mark.parametrize("binding", ["prepare_hcu_post_mma_wait"])
@pytest.mark.parametrize("predicate,truth", [
    ("eq", False),
    ("ne", True),
    ("ugt", True),
    ("uge", True),
    ("ult", False),
    ("ule", False),
    ("sgt", False),
    ("sge", False),
    ("slt", True),
    ("sle", True),
])
@pytest.mark.parametrize("valid_read", [False, True])
def test_comparison_signedness_controls_real_lds_footprint(binding, predicate, truth, valid_read):
    source = _shared_schedule_model_loop_source(control="select")
    source = source.replace("%choice = icmp eq i32 %parity,0", f"%choice = icmp {predicate} i32 -2147483648,0")
    # The correct branch reads initialized bytes. Reversing the branches reads
    # the unwritten4KiB padding, which must fail the finite publication proof.
    zero_on_true = truth == valid_read
    true_value, false_value = (0, 4096) if zero_on_true else (4096, 0)
    source = source.replace("%swap = select i1 %choice,i32 0,i32 16",
                            f"%swap = select i1 %choice,i32 {true_value},i32 {false_value}")
    output, loops, reason = getattr(libtriton.llvm, binding)(source, "gfx936", 256, 65536)
    assert loops == int(valid_read), reason
    if not valid_read: assert output == source


# pack bf16 rounds

from triton._C.libtriton import llvm
from triton.experimental.gluon.language.amd.cdna3 import buffer_store


def _pack_bf16_rounds_rounding(prefix):
    return f"""
      %{prefix}.bits = bitcast float %{prefix} to i32
      %{prefix}.nan = fcmp uno float %{prefix}, 0.0
      %{prefix}.hi = lshr i32 %{prefix}.bits, 16
      %{prefix}.parity = and i32 %{prefix}.hi, 1
      %{prefix}.bias = add i32 %{prefix}.bits, 32767
      %{prefix}.rounded = add i32 %{prefix}.bias, %{prefix}.parity
      %{prefix}.shift = lshr i32 %{prefix}.rounded, 16
      %{prefix}.short = trunc i32 %{prefix}.shift to i16
      %{prefix}.bf = bitcast i16 %{prefix}.short to bfloat
      %{prefix}.res = select i1 %{prefix}.nan, bfloat 0xR7FFF, bfloat %{prefix}.bf
    """


def _pack_bf16_rounds_packed_ir(extra=""):
    return f"""
    target datalayout = "e"
    target triple = "amdgcn-amd-amdhsa"
    define i32 @test(float %x, float %y, ptr %out) {{
      {_pack_bf16_rounds_rounding("x")}
      {_pack_bf16_rounds_rounding("y")}
      {extra}
      %lo = insertelement <2 x bfloat> poison, bfloat %x.res, i64 0
      %hi = insertelement <2 x bfloat> %lo, bfloat %y.res, i64 1
      %packed = bitcast <2 x bfloat> %hi to i32
      ret i32 %packed
    }}
    """


def test_packs_two_rounded_high_halfwords():
    result, sites = llvm.pack_hcu_bf16_rounds(_pack_bf16_rounds_packed_ir())
    assert sites == 1
    assert "call i32 @llvm.amdgcn.perm" in result
    assert "i32 117834498)" in result  # Select high halfword from each input.
    assert result.count("i32 2147418112") == 2
    assert "%x.shift =" not in result and "%y.shift =" not in result
    assert result.index("select i1 %x.nan") < result.index("%y.bits =")
    assert llvm.pack_hcu_bf16_rounds(result) == (result, 0)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("bfloat 0xR7FFF", "bfloat 0xR7FC0"),
        ("i32 %x.bits, 32767", "i32 %x.bits, 32768"),
        ("fcmp uno float %x, 0.0", "fcmp uno float %y, 0.0"),
        ("lshr i32 %x.rounded", "lshr exact i32 %x.rounded"),
        ("bfloat %y.res, i64 1", "bfloat %y.res, i64 0"),
        ('datalayout = "e"', 'datalayout = "E"'),
    ],
)
def test_incompatible_pack_is_unchanged(original, replacement):
    source = _pack_bf16_rounds_packed_ir().replace(original, replacement, 1)
    assert llvm.pack_hcu_bf16_rounds(source) == (source, 0)


def test_keeps_conversion_with_another_user():
    source = _pack_bf16_rounds_packed_ir("store bfloat %x.res, ptr %out")
    assert llvm.pack_hcu_bf16_rounds(source) == (source, 0)


@g.jit
def _pack_bf16_rounds_cast_store(src, dst):
    layout: gl.constexpr = gl.DistributedLinearLayout([[1]], [[2], [4], [8], [16], [32], [64]], [], [], [128])
    index = gl.arange(0, 128, layout=layout)
    x = gl.load(src + index)
    buffer_store(x.to(gl.bfloat16), dst, index)


def test_packs_elementwise_conversion_without_matrix_layout():
    source = GluonASTSource(
        _pack_bf16_rounds_cast_store,
        {"src": "*fp32", "dst": "*bf16"},
        {},
        attrs={(0, ): [("tt.divisibility", 16)], (1, ): [("tt.divisibility", 16)]},
    )
    target = GPUTarget("hip", "gfx936", 64)
    before = triton.compile(source, target=target, options={"num_warps": 1})
    after = triton.compile(source, target=target, options={"num_warps": 1, "pack_bf16_rounds": True})
    assert after.metadata.hcu_packed_bf16_round_sites == 1
    shifts = r"^\s*v_lshrrev_b32(?:_e32|_e64)?\s"
    assert len(re.findall(shifts, before.asm["amdgcn"], re.M)) == 2
    assert not re.findall(shifts, after.asm["amdgcn"], re.M)


def test_combined_memory_rules_preserve_sequential_rounding_result():
    source = _pack_bf16_rounds_packed_ir()
    sequential, _ = llvm.factor_hcu_buffer_store_offsets(source, "gfx936")
    sequential, count = llvm.pack_hcu_bf16_rounds(sequential)
    sequential, _, _ = llvm.combine_hcu_shifted_disjoint_or(sequential, "gfx936")
    combined, stats = llvm.optimize_hcu_memory_ir(source, "gfx936", True, True, True)
    assert combined == sequential
    assert stats["packed_bf16_rounds"] == count == 1
