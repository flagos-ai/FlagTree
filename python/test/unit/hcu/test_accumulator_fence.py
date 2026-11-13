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

from triton.backends.hcu.compiler import HIPBackend
from triton.experimental.gluon.language.hcu import accumulator_fence


@g.jit
def _fenced_copy(A, O, COHORT: gl.constexpr, UNUSED: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([COHORT], [64], [4], [0])
    i = gl.arange(0, COHORT * 256, layout=layout)
    value = gl.load(A + i)
    value = accumulator_fence(value, 1)
    value = accumulator_fence(value, 0)
    if not UNUSED:
        gl.store(O + i, value)
    else:
        gl.store(O, 0)


def compile_copy(cohort, dtype="fp32", unused=False):
    source = GluonASTSource(
        _fenced_copy,
        {"A": "*" + dtype, "O": "*" + dtype,
         "COHORT": "constexpr", "UNUSED": "constexpr"},
        {"COHORT": cohort, "UNUSED": unused},
    )
    return triton.compile(source, target=GPUTarget("hip", "gfx936", 64),
                          options=dict(num_warps=4, num_stages=1))


@pytest.mark.parametrize("cohort", [1, 4, 16, 128])
@pytest.mark.parametrize("dtype", ["fp32", "i32"])
def test_priority_fence_derives_actual_cohort(cohort, dtype):
    kernel = compile_copy(cohort, dtype)
    # The typed SSA operation survives until target lowering. Neither its
    # source expression nor TTGIR relies on elementwise inline assembly.
    assert kernel.asm["ttgir"].count("hcug.accumulator_fence") == 2
    assert "elementwise_inline_asm" not in kernel.asm["ttgir"]
    assert "s_setprio 1" in kernel.asm["amdgcn"]
    assert "s_setprio 0" in kernel.asm["amdgcn"]
    assert "~{memory}" in kernel.asm["llir"]


def test_unused_result_cannot_eliminate_or_merge_priority_state():
    kernel = compile_copy(4, unused=True)
    assert kernel.asm["ttgir"].count("hcug.accumulator_fence") == 2
    assert "s_setprio 1" in kernel.asm["amdgcn"]
    assert "s_setprio 0" in kernel.asm["amdgcn"]


@pytest.mark.parametrize("dtype,priority,message", [
    ("f16", 1, "f32 or i32"),
    ("f32", 2, "priority must be 0"),
    ("i32", -1, "priority must be 0"),
])
def test_imported_invalid_fence_is_rejected(dtype, priority, message,
                                             tmp_path, capfd):
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
import triton.language as tl
from triton.compiler import ASTSource
from triton.language.extra.hip import accumulator_fence as tl_accumulator_fence


@triton.jit
def _tl_fenced_copy(A, O, COHORT: tl.constexpr, UNUSED: tl.constexpr):
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
        _tl_fenced_copy,
        {"A": "*" + dtype, "O": "*" + dtype,
         "COHORT": "constexpr", "UNUSED": "constexpr"},
        {"COHORT": cohort, "UNUSED": unused},
    )
    kernel = triton.compile(source, target=GPUTarget("hip", "gfx936", 64),
                            options=dict(num_warps=4, num_stages=1))
    assert kernel.asm["ttgir"].count("hcug.accumulator_fence") == 2
    assert "elementwise_inline_asm" not in kernel.asm["ttgir"]
    assert "s_setprio 1" in kernel.asm["amdgcn"]
    assert "s_setprio 0" in kernel.asm["amdgcn"]
    assert "~{memory}" in kernel.asm["llir"]


@triton.jit
def _tl_fenced_dot(A, B, C, BM: tl.constexpr, BN: tl.constexpr,
                   FENCED: tl.constexpr):
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
def test_ordinary_tl_dot_fence_keeps_mmac_layout(dtype, bm, bn, warps,
                                                delayed_priority):
    kernels = []
    for fenced in (False, True):
        source = ASTSource(
            _tl_fenced_dot,
            {"A": "*" + dtype, "B": "*" + dtype, "C": "*" + dtype,
             "BM": "constexpr", "BN": "constexpr", "FENCED": "constexpr"},
            {"BM": bm, "BN": bn, "FENCED": fenced},
        )
        kernels.append(triton.compile(
            source, target=GPUTarget("hip", "gfx936", 64),
            options=dict(num_warps=warps, num_stages=1, kpack=8,
                         mmac_transposed=True, disable_llvm_loop_unroll=True,
                         priority_after_first_mmac=delayed_priority),
        ))
    plain, fenced = kernels
    lines = [line for line in fenced.asm["ttgir"].splitlines()
             if "hcug.accumulator_fence" in line]
    assert len(lines) == 2
    assert all("#mma" in line for line in lines), lines
    assert fenced.metadata.shared == plain.metadata.shared
    assert fenced.asm["amdgcn"].count("s_barrier") == plain.asm["amdgcn"].count("s_barrier")


@triton.jit
def _tl_multiple_fence_users(A, B, D, O, P):
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
        _tl_multiple_fence_users,
        {"A": "*bf16", "B": "*bf16", "D": "*bf16",
         "O": "*fp32", "P": "*fp32"}, {},
    )
    kernel = triton.compile(source, target=GPUTarget("hip", "gfx936", 64),
                            options=dict(num_warps=4, num_stages=1))
    assert kernel.asm["ttgir"].count("hcug.accumulator_fence") == 2
    assert kernel.asm["amdgcn"].count("s_setprio 1") == 1
    assert kernel.asm["amdgcn"].count("s_setprio 0") == 1
