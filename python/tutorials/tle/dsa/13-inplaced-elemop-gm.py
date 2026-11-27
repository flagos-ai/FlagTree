# Copyright 2026 FlagOS Contributors
# Copyright 2026- Xcoresigma Technology Co., Ltd

import torch
import torch_npu
import triton
import triton.language as tl
import triton.experimental.tle as tle
from triton.backends.ascend.testing import do_bench_npu

torch.manual_seed(20)
DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

M = N = K = 128
GRID = (1, )


@triton.jit
def addmm_ref_kernel(a_ptr, b_ptr, c_ptr):
    # Reference: load C, add, store.
    offs_m = tl.arange(0, 128)
    offs_n = tl.arange(0, 128)
    offs_k = tl.arange(0, 128)
    a = tl.load(a_ptr + offs_m[:, None] * 128 + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * 128 + offs_n[None, :])
    c = tl.load(c_ptr + offs_m[:, None] * 128 + offs_n[None, :])
    out = tl.dot(a, b) + c
    tl.store(c_ptr + offs_m[:, None] * 128 + offs_n[None, :], out)


@triton.jit
def addmm_gm_add_kernel(a_ptr, b_ptr, c_ptr):
    # C += A@B, never loading C
    offs_m = tl.arange(0, 128)
    offs_n = tl.arange(0, 128)
    offs_k = tl.arange(0, 128)
    a = tl.load(a_ptr + offs_m[:, None] * 128 + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * 128 + offs_n[None, :])
    out = tl.dot(a, b)

    tle.dsa.ascend.inplaced_elemop_gm("add", out.dtype)
    tl.store(c_ptr + offs_m[:, None] * 128 + offs_n[None, :], out)
    tle.dsa.ascend.inplaced_elemop_gm("none")


@triton.jit
def elem_maxmin_gm_kernel(x_ptr, c_ptr, IS_MAX: tl.constexpr):
    # C[i] = max/min(C[i], X[i]): the atomic read-modify-write
    # keeps the winning numeric value per element.
    offs = tl.arange(0, 1024)
    v = tl.load(x_ptr + offs)
    if IS_MAX:
        tle.dsa.ascend.inplaced_elemop_gm("max", v.dtype)
    else:
        tle.dsa.ascend.inplaced_elemop_gm("min", v.dtype)
    tl.store(c_ptr + offs, v)
    tle.dsa.ascend.inplaced_elemop_gm("none")


@triton.jit
def elem_maxmin_ref_kernel(x_ptr, c_ptr, IS_MAX: tl.constexpr):
    # Reference: load C, max/min, store.
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    v = tl.load(x_ptr + offs)
    c = tl.load(c_ptr + offs)
    if IS_MAX:
        out = tl.maximum(c, v)
    else:
        out = tl.minimum(c, v)
    tl.store(c_ptr + offs, out)


def test_addmm(torch_dtype, atol, rtol):
    # add (fixpipe): C += A@B, never loading C.
    a = torch.randn((M, K), dtype=torch_dtype, device="npu")
    b = torch.randn((K, N), dtype=torch_dtype, device="npu")
    c = torch.randn((M, N), dtype=torch_dtype, device="npu")
    # Accumulate in an f32 workspace, then quantize back to the original
    # dtype once (single rounding at the end).
    c_acc = c.float().clone()
    addmm_gm_add_kernel[GRID](a, b, c_acc)
    out = c_acc.to(c.dtype)
    # Reference: load C, add, store.
    c_ref = c.clone()
    addmm_ref_kernel[GRID](a, b, c_ref)
    # A/B: the only difference between the two kernels is the switch.
    torch.testing.assert_close(out.float(), c_ref.float(), rtol=rtol, atol=atol)
    # Anchor the reference kernel against torch matmul.
    torch.testing.assert_close(c_ref.float(), (a @ b).float() + c.float(), rtol=rtol, atol=atol)
    print("add(fixpipe): PASSED!")
    gm_time = do_bench_npu(lambda: addmm_gm_add_kernel[GRID](a, b, c_acc), clear_l2_cache=True)
    ref_time = do_bench_npu(lambda: addmm_ref_kernel[GRID](a, b, c_ref), clear_l2_cache=True)
    print(f"add(fixpipe): gm_add time: {gm_time * 1000:.2f} us, "
          f"ref time: {ref_time * 1000:.2f} us, speedup: {ref_time / gm_time:.2f}x")


def test_maxmin(torch_dtype):
    # max/min (Vector): selection involves no arithmetic, so the result
    # must be bit-identical to the reference.
    for kind in ("max", "min"):
        x = torch.randn(1024, dtype=torch_dtype, device="npu")
        c0 = torch.randn(1024, dtype=torch_dtype, device="npu")
        ref_fn = torch.maximum if kind == "max" else torch.minimum
        c_elem = c0.clone()
        elem_maxmin_gm_kernel[(1, )](x, c_elem, IS_MAX=(kind == "max"))
        # Reference: load C, max/min, store.
        c_ref = c0.clone()
        elem_maxmin_ref_kernel[(1, )](x, c_ref, IS_MAX=(kind == "max"))
        torch.npu.synchronize()
        # A/B: the only difference between the two kernels is the switch.
        exact = torch.equal(c_elem, c_ref)
        # Anchor the reference kernel against torch.maximum/minimum.
        assert exact and torch.equal(c_ref, ref_fn(c0, x)), f"atomic {kind} mismatch or bad reference kernel"
        print(f"{kind}(vector): PASSED!")
        gm_time = do_bench_npu(lambda: elem_maxmin_gm_kernel[(1, )](x, c_elem, IS_MAX=(kind == "max")),
                               clear_l2_cache=True)
        ref_time = do_bench_npu(lambda: elem_maxmin_ref_kernel[(1, )](x, c_ref, IS_MAX=(kind == "max")),
                                clear_l2_cache=True)
        print(f"{kind}(vector): gm time: {gm_time * 1000:.2f} us, "
              f"ref time: {ref_time * 1000:.2f} us, speedup: {ref_time / gm_time:.2f}x")


test_cases = [
    (torch.float32, 1e-3, 1e-3),
    (torch.float16, 1e-2, 1e-2),
    (torch.bfloat16, 2e-2, 2e-2),
]

if __name__ == "__main__":
    for i, (torch_dtype, atol, rtol) in enumerate(test_cases, start=1):
        print(f"====================第{i}次测试=================, dtype={torch_dtype}")
        test_addmm(torch_dtype, atol, rtol)
        test_maxmin(torch_dtype)
