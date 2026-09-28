import pytest
import torch
import triton
import triton.language as tl


@pytest.fixture(scope="module")
def nvidia_target():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    target = triton.runtime.driver.active.get_current_target()
    # Ordinary tl.dot uses MMA only on SM80+. SM75 instruction support is
    # covered separately by test/Conversion/mma_c_to_dot_b_movmatrix.mlir.
    if target.backend != "cuda" or int(target.arch) < 80:
        pytest.skip("the chained-dot test requires NVIDIA SM80 or newer")
    return target


@triton.jit
def _chained_dot(a_ptr, b_ptr, c_ptr, out_ptr, I: tl.constexpr, N: tl.constexpr):
    m = tl.arange(0, 16)
    i = tl.arange(0, I)
    k = tl.arange(0, 32)
    n = tl.arange(0, N)
    a = tl.load(a_ptr + i[:, None] * 32 + k[None, :])
    b = tl.load(b_ptr + k[:, None] * N + n[None, :])
    c = tl.load(c_ptr + m[:, None] * I + i[None, :])
    # The first dot's accumulator becomes operand B of the second dot,
    # exercising MMA C -> dot B layout conversion through ordinary Triton.
    ab = tl.dot(a, b).to(a.dtype)
    out = tl.dot(c, ab)
    tl.store(out_ptr + m[:, None] * N + n[None, :], out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inner,cols,num_warps,uses_movmatrix", [
    (16, 32, 4, True),
    (16, 128, 4, True),
    (32, 128, 4, True),
    (32, 16, 4, False),
    (16, 32, 1, True),
    (16, 32, 8, True),
])
def test_mma_c_to_dot_b_preserves_values(nvidia_target, dtype, inner, cols, num_warps, uses_movmatrix):
    torch.manual_seed(17)
    # Small integers keep both dot products exact in FP32 and the intermediate
    # cast exact in FP16/BF16, so register/lane mistakes cannot hide in tolerance.
    a = torch.randint(-2, 3, (inner, 32), device="cuda").to(dtype)
    b = torch.randint(-2, 3, (32, cols), device="cuda").to(dtype)
    c = torch.randint(-2, 3, (16, inner), device="cuda").to(dtype)
    out = torch.empty((16, cols), device="cuda", dtype=torch.float32)
    compiled = _chained_dot[(1, )](a, b, c, out, inner, cols, num_warps=num_warps)

    reference = c.float() @ (a.float() @ b.float()).to(dtype).float()
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
    assert ("movmatrix.sync.aligned.m8n8.trans.b16" in compiled.asm["ptx"]) == uses_movmatrix
