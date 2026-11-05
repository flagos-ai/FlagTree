"""Explicit blocked layouts must survive dot conversion and acceleration."""

import pytest
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from test_tle_utils import compile_musa, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _dot_block_layout(A, B, C, Out, M: tl.constexpr, K: tl.constexpr, LAYOUT: tl.constexpr, CAST: tl.constexpr,
                      CHAINED: tl.constexpr, INTEGER: tl.constexpr):
    rows = tl.arange(0, M)
    cols = tl.arange(0, 128)
    reduction = tl.arange(0, K)
    lhs = tl.load(A + rows[:, None] * K + reduction[None, :])
    rhs = tl.load(B + reduction[:, None] + cols[None, :] * K)
    offsets = rows[:, None] * 128 + cols[None, :]
    acc = tl.load(C + offsets)
    ACC_DTYPE: tl.constexpr = tl.int32 if INTEGER else tl.float32
    result = tl.dot(lhs, rhs, acc, out_dtype=ACC_DTYPE)
    if CHAINED:
        result = tle.gpu.set_layout(result, LAYOUT)
        result = tl.dot(lhs, rhs, result, out_dtype=ACC_DTYPE)
    if CAST == "before":
        result = result.to(tl.float32)
    result = tle.gpu.set_layout(result, LAYOUT)
    if CAST == "after":
        result = result.to(tl.float32)
    tl.store(Out + offsets, result, (rows[:, None] < M - 3) & (cols[None, :] < 125))


def _layout(vector_width, warps):
    return tle.gpu.BlockEncoding([1, vector_width], [4, 8], [warps, 1], [1, 0])


_DTYPE_CAST_CASES = [("i8", "none"), ("i8", "before"), ("i8", "after"), ("fp8e4nv", "none"), ("fp8e5", "none"),
                     ("bf16", "none")]


@pytest.mark.parametrize("m,warps", [(64, 4), (256, 16)])
@pytest.mark.parametrize("vector_width", [4, 16])
@pytest.mark.parametrize("dtype,cast", _DTYPE_CAST_CASES)
@pytest.mark.parametrize("chained", [False, True])
def test_dot_explicit_block_layout_compile(m, warps, vector_width, dtype, cast, chained):
    integer = dtype == "i8"
    acc_type = "i32" if integer else "fp32"
    # BF16 operands need twice the shared storage. Bound K so the explicit
    # result conversion and both operands fit MP31's 192 KiB shared budget.
    k = 64 if dtype == "bf16" else 128
    compiled = compile_musa(
        _dot_block_layout,
        signature={
            "A": "*" + dtype, "B": "*" + dtype, "C": "*" + acc_type, "Out":
            "*" + acc_type if cast == "none" else "*fp32"
        },
        constexprs={
            "M": m, "K": k, "LAYOUT": _layout(vector_width, warps), "CAST": cast, "CHAINED": chained, "INTEGER": integer
        },
        options={"num_warps": warps, "num_stages": 1},
    )
    ttgir = compiled.asm["ttgir"]
    assert "unrealized_conversion_cast" not in ttgir
    assert "ttmg.squad_dot" in ttgir
    assert f"sizePerThread = [1, {vector_width}]" in ttgir


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA device required")
@pytest.mark.parametrize("m,warps", [(64, 4), (256, 16)])
@pytest.mark.parametrize("vector_width", [4, 16])
@pytest.mark.parametrize("dtype,cast", _DTYPE_CAST_CASES)
@pytest.mark.parametrize("chained", [False, True])
def test_dot_explicit_block_layout_runtime(m, warps, vector_width, dtype, cast, chained):
    generator = torch.Generator().manual_seed(23)
    k = 64 if dtype == "bf16" else 128
    lhs_cpu = torch.randint(-8, 8, (m, k), generator=generator, dtype=torch.int8)
    rhs_cpu = torch.randint(-8, 8, (128, k), generator=generator, dtype=torch.int8)
    acc_cpu = torch.randint(-17, 17, (m, 128), generator=generator, dtype=torch.int32)
    integer = dtype == "i8"
    if not integer:
        input_dtype = {"fp8e4nv": torch.float8_e4m3fn, "fp8e5": torch.float8_e5m2, "bf16": torch.bfloat16}[dtype]
        # Dyadic values make every product and sum exact in FP32. This tests
        # layout correctness without introducing a tolerance for MMA rounding.
        lhs_cpu = (lhs_cpu.float() / 4).to(input_dtype)
        rhs_cpu = (rhs_cpu.float() / 4).to(input_dtype)
        acc_cpu = acc_cpu.float() / 16
    expected = acc_cpu.double() + (2 if chained else 1) * (lhs_cpu.double() @ rhs_cpu.double().T)
    output_dtype = torch.int32 if integer and cast == "none" else torch.float32
    valid = (torch.arange(m)[:, None] < m - 3) & (torch.arange(128)[None, :] < 125)
    expected = torch.where(valid, expected, -99999).to(output_dtype)
    lhs, rhs, acc = (tensor.to("musa") for tensor in (lhs_cpu, rhs_cpu, acc_cpu))
    output = torch.empty((m, 128), dtype=output_dtype, device="musa")
    for _ in range(3):
        output.fill_(-99999)
        _dot_block_layout[(1, )](lhs, rhs, acc, output, m, k, _layout(vector_width, warps), cast, chained, integer,
                                 num_warps=warps, num_stages=1)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@triton.jit
def _dot_shared_layout(A, B, V, Out, INTEGER: tl.constexpr, P_LAYOUT: tl.constexpr, K_LAYOUT: tl.constexpr,
                       SET_K: tl.constexpr):
    rows = tl.arange(0, 128)
    cols = tl.arange(0, 128)
    reduction = tl.arange(0, 128)
    lhs = tl.load(A + rows[:, None] * 128 + reduction[None, :])
    probabilities = tle.gpu.alloc((128, 128), tl.float8e4nv)
    values = tle.gpu.alloc((128, 128), tl.float8e4nv)
    acc = tl.zeros((128, 128), tl.float32)
    ACC_DTYPE: tl.constexpr = tl.int32 if INTEGER else tl.float32
    for block in range(2):
        tokens = block * 128 + cols
        rhs = tl.load(B + tokens[None, :] * 128 + reduction[:, None], tokens[None, :] < 253, 0.0)
        if SET_K:
            rhs = tle.gpu.set_layout(rhs, K_LAYOUT)
        scores = tl.dot(lhs, rhs, out_dtype=ACC_DTYPE).to(tl.float32)
        p = tle.gpu.set_layout((scores / 128).to(tl.float8e4nv), P_LAYOUT)
        v = tl.load(V + tokens[:, None] * 128 + reduction[None, :], tokens[:, None] < 253, 0.0)
        probabilities.store(p)
        values.store(v)
        tl.debug_barrier()
        part = tle.gpu.wgmma(probabilities, values)
        part = tle.gpu.wgmma_wait(0, part)
        acc += part
    tl.store(Out + rows[:, None] * 128 + cols[None, :], acc)


@pytest.mark.parametrize("dtype", ["i8", "fp8e4nv"])
@pytest.mark.parametrize("vector_width", [4, 16])
@pytest.mark.parametrize("set_k", [False, True])
def test_dot_shared_explicit_layout_compile(dtype, vector_width, set_k):
    compiled = compile_musa(
        _dot_shared_layout,
        signature={"A": "*" + dtype, "B": "*" + dtype, "V": "*fp8e4nv", "Out": "*fp32"},
        constexprs={
            "INTEGER": dtype == "i8", "P_LAYOUT": _layout(vector_width, 16), "K_LAYOUT": _layout(16, 16), "SET_K": set_k
        },
        options={"num_warps": 16, "num_stages": 1},
    )
    assert "unrealized_conversion_cast" not in compiled.asm["ttgir"]
    assert "ttmg.squad_dot" in compiled.asm["ttgir"]


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA device required")
@pytest.mark.parametrize("dtype", ["i8", "fp8e4nv"])
@pytest.mark.parametrize("vector_width", [4, 16])
@pytest.mark.parametrize("set_k", [False, True])
def test_dot_shared_explicit_layout_runtime(dtype, vector_width, set_k):
    generator = torch.Generator().manual_seed(41)
    a = torch.randint(-2, 3, (128, 128), generator=generator, dtype=torch.int8)
    b = torch.randint(-2, 3, (256, 128), generator=generator, dtype=torch.int8)
    v = (torch.randint(-8, 8, (256, 128), generator=generator).float() / 4).to(torch.float8_e4m3fn)
    if dtype != "i8":
        a, b = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
    expected = torch.zeros((128, 128), dtype=torch.float64)
    for start, end in [(0, 128), (128, 253)]:
        scores = a.double() @ b[start:end].double().T
        p = (scores.float() / 128).to(torch.float8_e4m3fn)
        expected += p.double() @ v[start:end].double()
    a, b, v = (x.to("musa") for x in (a, b, v))
    output = torch.empty((128, 128), dtype=torch.float32, device="musa")
    for _ in range(3):
        output.fill_(float("nan"))
        _dot_shared_layout[(1, )](a, b, v, output, dtype == "i8", _layout(vector_width, 16), _layout(16, 16), set_k,
                                  num_warps=16, num_stages=1)
        torch.testing.assert_close(output.cpu(), expected.float(), rtol=0, atol=0)
