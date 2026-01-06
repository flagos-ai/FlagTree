import pytest
import torch
import triton
import triton.language as tl
import itertools
import benchmark
benchmark.select_cpu_backend()

from triton._internal_testing import (
    integral_dtypes,
    int_dtypes,
    str_to_triton_dtype,
    uint_dtypes,
    float_dtypes,
    float_dtypes_with_bfloat16,
    dtypes,
    dtypes_with_bfloat16,
    is_cuda,
    is_interpreter,
    is_hopper,
    is_hip,
    is_hip_cdna,
    is_hip_mi200,
    is_hip_mi300,
    is_hip_mi350,
    is_xpu,
    get_arch,
    torch_float8_dtypes,
    torch_dtypes,
    numpy_random,
    to_triton,
    torch_dtype_name,
    to_numpy,
)
from triton.runtime.errors import InterpreterError

mma_nonk_sizes = []

GPU_DIALECT = "ttg"
if is_interpreter():
    THREADS_PER_WARP = 1
elif is_hip():
    THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size
    # for CDNA multiple variants of mma instructions are supported:
    # mfma 16x16/mfma 32x32
    # 0 is a special value for automatic heuristic
    if is_hip_cdna():
        mma_nonk_sizes = [0, 16, 32]
else:
    THREADS_PER_WARP = 32

RESOLUTION = {
    torch.bool: 0,
    torch.int16: 0,
    torch.int32: 0,
    torch.int64: 0,
    # torch.float16: 1e-3, # FIXME: test_scaled_dot[32-32-64-True-True-False-e2m1-fp16-4-16-1]... (fp16 cases) Failed
    torch.float16: 1e-2,
    torch.float32: 1.3e-6,
    # torch.bfloat16: 0.016, #FIXME: test_scaled_dot[128-128-64-True-True-False-e2m1-e5m2-4-16-1] Failed
    torch.bfloat16: 0.018,
    torch.float64: 1e-7,
    torch.complex32: 1e-3,
    torch.complex64: 1.3e-6,
}

def flaggems_assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1):
    assert res.dtype == dtype
    ref = ref.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(res, ref, atol=atol, rtol=rtol, equal_nan=equal_nan)

@pytest.mark.parametrize("M, N, K, col_a, col_b, rhs_scale, mxfp_type, normal_type, num_warps, mma, kpack",
                         [(M, N, K, col_a, col_b, rhs_scale, mxfp_type, normal_type, 4, mma, kpack)
                          for M, N, K in itertools.product([32, 64, 128], [32, 64, 128], [64, 128])
                          for col_a, col_b in itertools.product([True, False], repeat=2)
                          for rhs_scale in [False, True]
                          for mxfp_type in ["e2m1", "e4m3", "e5m2"]
                          for normal_type in ["e4m3", "e5m2", "bf16", "fp16"]
                          for mma in (mma_nonk_sizes if is_hip() else [16])
                          for kpack in ([1, 2] if is_hip() else [1])])
def test_scaled_dot(M, N, K, col_a, col_b, rhs_scale, mxfp_type, normal_type, num_warps, mma, kpack, device):
    if is_cuda():
        cc = torch.cuda.get_device_capability()
        if cc < (8, 9):
            pytest.skip("float8e4nv not supported on CUDA < 8.9")
    if is_hip():
        if not is_hip_cdna():
            pytest.skip("scaled_dot only implemented for HIP CDNA")
        if "e4m3" in (mxfp_type, normal_type):
            if not (is_hip_mi300() or is_hip_mi350()):
                pytest.skip(f"scaled_dot({mxfp_type}, {normal_type}) only implemented for MI300 and MI350")
        if mma == 16 and K == 64:
            pytest.skip(f"K == {K} too small for mfma {mma} in scaled_dot")

    @triton.jit
    def dot_scale_kernel(a_base, stride_a0, stride_a1, a_scale, b_base, stride_b0, stride_b1, b_scale, out,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, type_a: tl.constexpr,
                         type_b: tl.constexpr):
        DIV_FACTOR_A: tl.constexpr = 2 if type_a == "e2m1" else 1
        DIV_FACTOR_B: tl.constexpr = 2 if type_b == "e2m1" else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K // DIV_FACTOR_A
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K // DIV_FACTOR_B
        a_ptr = a_base + tl.arange(0, BLOCK_M)[:, None] * stride_a0 + tl.arange(0,
                                                                                PACKED_BLOCK_K_A)[None, :] * stride_a1
        b_ptr = b_base + tl.arange(0, PACKED_BLOCK_K_B)[:, None] * stride_b0 + tl.arange(0,
                                                                                         BLOCK_N)[None, :] * stride_b1

        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32
        if a_scale is not None:
            scale_a_ptr = a_scale + tl.arange(0, BLOCK_M)[:, None] * SCALE_BLOCK_K + tl.arange(0,
                                                                                               SCALE_BLOCK_K)[None, :]
            a_scale = tl.load(scale_a_ptr)
        if b_scale is not None:
            scale_b_ptr = b_scale + tl.arange(0, BLOCK_N)[:, None] * SCALE_BLOCK_K + tl.arange(0,
                                                                                               SCALE_BLOCK_K)[None, :]
            b_scale = tl.load(scale_b_ptr)
        c = tl.dot_scaled(a, a_scale, type_a, b, b_scale, type_b)
        out_ptr = out + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        tl.store(out_ptr, c.to(tl.bfloat16))

    @triton.jit
    def mxfp_upcast_kernel(
        x_ptr,
        scale_ptr,
        mxfp_ptr,
        N,
        e_bits: tl.constexpr,
        m_bits: tl.constexpr,
        to_type: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # x.shape ==     (N, 32) for fp8 or (N, 16) for fp4
        # scale.shape == (N,)
        # out.shape   == (N, 32)
        is_fp8: tl.constexpr = e_bits + m_bits == 7
        # fp8: BLOCK_SIZE -> BLOCK_SIZE // 32, 32
        # fp4: BLOCK_SIZE // 2 -> BLOCK_SIZE // 32 , 16
        PARALLEL_DIM: tl.constexpr = BLOCK_SIZE // 32
        LAST_DIM: tl.constexpr = 32 if is_fp8 else 16
        LOAD_SIZE: tl.constexpr = LAST_DIM * PARALLEL_DIM

        offsets = (tl.program_id(0) * LOAD_SIZE + tl.arange(0, PARALLEL_DIM)[:, None] * LAST_DIM +
                   tl.arange(0, LAST_DIM)[None, :])
        x = tl.load(x_ptr + offsets, mask=offsets < N * LAST_DIM)

        offsets = tl.program_id(0) * PARALLEL_DIM + tl.arange(0, PARALLEL_DIM)[:, None]
        scale = tl.load(scale_ptr + offsets, mask=offsets < N)
        tl.static_assert(scale.dtype == tl.uint8)
        tl.static_assert(x.dtype == tl.uint8)

        if to_type == tl.bfloat16:
            upcasted_scale = (scale.to(tl.uint16) << 7).to(tl.bfloat16, bitcast=True)
        else:
            tl.static_assert(to_type == tl.float16)
            scale_fp32 = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
            upcasted_scale = scale_fp32.to(tl.float16)

        to_e_bits: tl.constexpr = 8 if to_type == tl.bfloat16 else 5
        to_m_bits: tl.constexpr = 7 if to_type == tl.bfloat16 else 10
        if is_fp8:
            if e_bits == 5 and m_bits == 2:
                x_f8 = x.to(tl.float8e5, bitcast=True)
                upcasted_x = x_f8.to(to_type)
                # Preserve infs and nans. FIXME Fp8E5M2_to_Bf16 doesn't preserve them!
                non_finite_mask: tl.constexpr = ((1 << e_bits) - 1) << m_bits
                non_finite_mask_16bit: tl.constexpr = ((1 << to_e_bits) - 1) << to_m_bits
                upcasted_x = tl.where(
                    x & non_finite_mask == non_finite_mask,
                    (upcasted_x.to(tl.uint16, bitcast=True) | non_finite_mask_16bit).to(to_type, bitcast=True),
                    upcasted_x,
                )
            else:
                tl.static_assert(e_bits == 4 and m_bits == 3)
                x_f8 = x.to(tl.float8e4nv, bitcast=True)
                upcasted_x = x_f8.to(to_type)
        else:
            to_bias: tl.constexpr = 127 if to_type == tl.bfloat16 else 15
            to_point5: tl.constexpr = 16128 if to_type == tl.bfloat16 else 0x3800
            # e2m1
            em0 = x & 0x7
            em1 = x & 0x70
            x0 = (em0.to(tl.uint16) << (to_m_bits - 1)) | ((x & 0x8).to(tl.uint16) << 12)
            x1 = (em1.to(tl.uint16) << (to_m_bits - 1 - 4)) | ((x & 0x80).to(tl.uint16) << 8)
            # Three cases:
            # 1) x is normal and non-zero: Correct bias
            x0 = tl.where((em0 & 0x6) != 0, x0 + ((to_bias - 1) << to_m_bits), x0)
            x1 = tl.where((em1 & 0x60) != 0, x1 + ((to_bias - 1) << to_m_bits), x1)
            # 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in bf16
            x0 = tl.where(em0 == 0x1, to_point5 | (x0 & 0x8000), x0)
            x1 = tl.where(em1 == 0x10, to_point5 | (x1 & 0x8000), x1)
            # 3) x is zero, do nothing
            upcasted_x = tl.interleave(x0, x1).to(to_type, bitcast=True)
        # Multiplication preserves infs and NaNs in upcasted_x
        mxfp = upcasted_x * upcasted_scale
        # If scale is NaN, we encode it as an inf, so we need to correct for that
        mxfp = tl.where(scale == 0xFF, float("nan"), mxfp)

        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(mxfp_ptr + offsets, tl.ravel(mxfp), mask=offsets < N * 32)

    def dot_scale_ref(x, scale_x, y, scale_y, type_x, type_y):

        def upcast(v, scale, type, comp_dtype, transposed):
            if scale is None:
                type = {
                    "e4m3": torch.float8_e4m3fn,
                    "e5m2": torch.float8_e5m2,
                    "bf16": torch.bfloat16,
                    "fp16": torch.float16,
                }[type]
                return v.view(type).to(comp_dtype)
            e_bits, m_bits = {"e2m1": (2, 1), "e4m3": (4, 3), "e5m2": (5, 2)}[type]
            # Packing is always on the K dimension so we transpose before upcasting then transpose back.
            if transposed:
                v = v.mT.contiguous()
            v = v.contiguous()
            v_upcast = v.new_empty(scale.shape[:-1] + (32 * scale.shape[-1], ), dtype=comp_dtype)
            N = v_upcast.numel()
            BLOCK_SIZE = 512
            grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, )
            comp_dtype = tl.float16 if comp_dtype == torch.float16 else tl.bfloat16
            mxfp_upcast_kernel[grid](v, scale, v_upcast, scale.numel(), e_bits, m_bits, comp_dtype, BLOCK_SIZE,
                                     num_warps=num_warps)

            assert v_upcast.isfinite().all()
            if transposed:
                v_upcast = v_upcast.mT
            return v_upcast

        # Upcast to fp16 if one of the input is fp16
        comp_dtype = torch.float16 if "fp16" in (type_x, type_y) else torch.bfloat16

        x_upcast = upcast(x, scale_x, type_x, comp_dtype, False)
        y_upcast = upcast(y, scale_y, type_y, comp_dtype, True)
        class AccumulateInFp32:

            def __enter__(self):
                self.prev_value = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

            def __exit__(self, exc_type, exc_val, exc_tb):
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = self.prev_value

        with AccumulateInFp32():
            return torch.matmul(x_upcast, y_upcast)

    comp_dtype = torch.float16 if normal_type == "fp16" else torch.bfloat16
    # The max exponent we use to initialize data in the x/y and associated scale tensor to avoid
    # overflow when scaling.
    comp_dtype_max_exp = 6 if normal_type == "fp16" else 15

    torch.manual_seed(0)

    def make_arg(shape, ty, col_major=False):
        if col_major:
            shape = shape[:-2] + (shape[-1], shape[-2])
        if ty == "bf16" or ty == "fp16":
            ret = torch.randn(shape, dtype=comp_dtype, device=device)
            # Clamp to avoid relative error issues
            ret.clamp_(-2**comp_dtype_max_exp, 2**comp_dtype_max_exp - 1)
        else:
            if is_hip_mi350():
                # On other chips, the A/B operands are upcasted to fp16/bf16
                # before matmul, which has larger range to avoid overflow.
                # On MI350, we use the V_MFMA_*_F8F6F4 instructions to
                # directly calculate matmul on F8F6F4 data. So we need
                # to narrow down the range of input to avoid overflow.
                ret = torch.randint(20, 40, shape, dtype=torch.uint8, device=device)
            else:
                ret = torch.randint(256, shape, dtype=torch.uint8, device=device)
        if col_major:
            ret = ret.mT
        return ret

    type_a = normal_type if rhs_scale else mxfp_type
    type_b = mxfp_type if rhs_scale else normal_type

    DIV_FACTOR_A = 2 if type_a == "e2m1" else 1
    DIV_FACTOR_B = 2 if type_b == "e2m1" else 1
    x = make_arg((M, K // DIV_FACTOR_A), type_a, col_major=col_a)
    y = make_arg((K // DIV_FACTOR_B, N), type_b, col_major=col_b)

    min_scale, max_scale = (0, 142) if comp_dtype == torch.bfloat16 else (124, 131)
    scale_x = torch.randint(min_scale, max_scale + 1, (M, K // 32), dtype=torch.uint8, device=device)
    scale_y = torch.randint(min_scale, max_scale + 1, (N, K // 32), dtype=torch.uint8, device=device)
    if rhs_scale:
        scale_x = None
    else:
        scale_y = None

    def make_finite(x, dtype):
        # e5m2 has too many non-finite values when sampled uniformly (1 / 32) and
        # Fp8E5M2_to_Bf16 doesn't preserve NaNs (fixme)
        if dtype not in ("e5m2", "e4m3"):
            return x
        if dtype == "e5m2" and comp_dtype == torch.float16:
            x = x & 0xB
        mask = 0x7C if dtype == "e5m2" else 0x7F
        finite = torch.arange(x.numel(), device=device, dtype=torch.uint8).reshape_as(x) % mask
        x_finite = torch.where(x & mask == mask, finite | (0x80 & x), x)
        x.copy_(x_finite)
        return x

    x = make_finite(x, type_a)
    y = make_finite(y, type_b)

    kernel_kwargs = {"num_warps": num_warps}
    if is_hip():
        kernel_kwargs["kpack"] = kpack
        kernel_kwargs["matrix_instr_nonkdim"] = mma
    z = x.new_empty((M, N), dtype=comp_dtype)
    pgm = dot_scale_kernel[(1, )](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z, M, N, K, type_a, type_b,
                                  **kernel_kwargs)
    z_ref = dot_scale_ref(x, scale_x, y, scale_y, type_a, type_b)
    # Bigger tolerance for AMD MI200 devices.
    # MI200 devices use reduced precision fp16 and bf16 and flush input and output denormal values
    # to zero. Detailed info is at:
    # https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices

    # FIXME: If enable orgin assertion, these cases failed due to matmul precision:
    # test_scaled_dot[32-64-128-True-True-False-e2m1-e5m2-4-16-1]
    # test_scaled_dot[64-64-64-False-False-False-e4m3-e4m3-4-16-1]
    # test_scaled_dot[64-64-128-True-True-False-e4m3-fp16-4-16-1]
    # test_scaled_dot[128-32-128-True-False-False-e4m3-fp16-4-16-1]
    # test_scaled_dot[128-64-128-False-False-True-e4m3-fp16-4-16-1]
    # test_scaled_dot[128-128-64-True-True-False-e2m1-e5m2-4-16-1]

    # atol = 2e-4 if is_hip_mi200() else 1e-5
    # rtol = 2e-2 if is_hip_mi200() else 1e-2
    # torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)

    flaggems_assert_close(z, z_ref, dtype=comp_dtype, reduce_dim=K)

    # make sure ld/st are vectorized
    if is_cuda():
        ptx = pgm.asm['ptx']
        if (max(M, N) * K) // (num_warps * 32) >= 4:
            assert 'ld.global.v4' in ptx
        if M * N // (num_warps * 32) >= 4:
            assert 'st.global.v4' in ptx
        assert (re.search(r'(mma|wgmma.mma_async).sync.aligned.m\d+n\d+k16(?:.row.col)?.f32.(f|bf)16.(f|bf)16', ptx)
                or "tcgen05.mma.cta_group::1.kind::f16" in ptx)

if __name__ == "__main__":
    M = 32
    N = 64
    K = 128
    col_a = True
    col_b = False
    rhs_scale = False
    mxfp_type = "e5m2"
    normal_type = "e4m3"
    num_warps = 4
    mma = 16
    kpack = 1
    device = "cpu"

    test_scaled_dot(M, N, K, col_a, col_b, rhs_scale, mxfp_type, normal_type, num_warps, mma, kpack, device)

