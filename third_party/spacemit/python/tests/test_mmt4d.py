import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

MICRO_M = 16
MICRO_N = 32
MICRO_K = 8
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 512


def pack_a_ref(a: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    assert m == BLOCK_SIZE_M and k == BLOCK_SIZE_K
    mb = BLOCK_SIZE_M // MICRO_M
    kb = BLOCK_SIZE_K // MICRO_K
    return a.view(mb, MICRO_M, kb, MICRO_K).permute(0, 2, 1, 3).contiguous()


def pack_b_ref(b: torch.Tensor) -> torch.Tensor:
    k, n = b.shape
    assert k == BLOCK_SIZE_K and n == BLOCK_SIZE_N
    kb = BLOCK_SIZE_K // MICRO_K
    nb = BLOCK_SIZE_N // MICRO_N
    return b.view(kb, MICRO_K, nb, MICRO_N).permute(2, 0, 3, 1).contiguous()


@triton.jit
def mmt4d_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am0,
    stride_am1,
    stride_am2,
    stride_am3,
    stride_bn0,
    stride_bn1,
    stride_bn2,
    stride_bn3,
    stride_cm,
    stride_cn,
    M,
    N,
    MB: tl.constexpr,
    NB: tl.constexpr,
    KB: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_N: tl.constexpr,
    MICRO_K: tl.constexpr,
):
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=[MB, KB, MICRO_M, MICRO_K],
        strides=[stride_am0, stride_am1, stride_am2, stride_am3],
        offsets=[0, 0, 0, 0],
        block_shape=[MB, KB, MICRO_M, MICRO_K],
        order=[3, 2, 1, 0],
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=[NB, KB, MICRO_N, MICRO_K],
        strides=[stride_bn0, stride_bn1, stride_bn2, stride_bn3],
        offsets=[0, 0, 0, 0],
        block_shape=[NB, KB, MICRO_N, MICRO_K],
        order=[3, 2, 1, 0],
    )

    a_packed = tl.load(a_block_ptr)
    b_packed = tl.load(b_block_ptr)
    c = smt.dot(a_packed, b_packed)
    c = smt.view(c, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1))

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        offsets=[0, 0],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        order=[1, 0],
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def run_case():
    torch.manual_seed(0)
    device = "cpu"

    a = torch.randn((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=torch.float16, device=device)
    b = torch.randn((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=torch.float16, device=device)
    a_packed = pack_a_ref(a)
    b_packed = pack_b_ref(b)
    c = torch.empty((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float16, device=device)
    mb = BLOCK_SIZE_M // MICRO_M
    nb = BLOCK_SIZE_N // MICRO_N
    kb = BLOCK_SIZE_K // MICRO_K

    mmt4d_kernel[(1, )](
        a_packed,
        b_packed,
        c,
        a_packed.stride(0),
        a_packed.stride(1),
        a_packed.stride(2),
        a_packed.stride(3),
        b_packed.stride(0),
        b_packed.stride(1),
        b_packed.stride(2),
        b_packed.stride(3),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        MB=mb,
        NB=nb,
        KB=kb,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        MICRO_M=MICRO_M,
        MICRO_N=MICRO_N,
        MICRO_K=MICRO_K,
    )

    ref = torch.matmul(a.float(), b.float()).half()
    return c, ref


def run_n_tail_case(n_value: int):
    assert 0 < n_value <= BLOCK_SIZE_N
    torch.manual_seed(0)
    device = "cpu"

    a = torch.randn((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=torch.float16, device=device)
    b_full = torch.randn((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=torch.float16, device=device)
    b_full[:, n_value:] = 0
    a_packed = pack_a_ref(a)
    b_packed = pack_b_ref(b_full)
    c = torch.empty((BLOCK_SIZE_M, n_value), dtype=torch.float16, device=device)

    mb = BLOCK_SIZE_M // MICRO_M
    nb = BLOCK_SIZE_N // MICRO_N
    kb = BLOCK_SIZE_K // MICRO_K

    mmt4d_kernel[(1, )](
        a_packed,
        b_packed,
        c,
        a_packed.stride(0),
        a_packed.stride(1),
        a_packed.stride(2),
        a_packed.stride(3),
        b_packed.stride(0),
        b_packed.stride(1),
        b_packed.stride(2),
        b_packed.stride(3),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M,
        n_value,
        MB=mb,
        NB=nb,
        KB=kb,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        MICRO_M=MICRO_M,
        MICRO_N=MICRO_N,
        MICRO_K=MICRO_K,
    )

    ref = torch.matmul(a.float(), b_full[:, :n_value].float()).half()
    return c, ref


if __name__ == "__main__":
    out, ref = run_case()
    diff = (out - ref).abs()

    print("Max error:", diff.max())
    print("Mean error:", diff.mean())
    print("out[:4, :8] =")
    print(out[:4, :8])
    print("ref[:4, :8] =")
    print(ref[:4, :8])

    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2)

    print("\n=== focused N-tail cases ===")
    for n_value in [1, 7, 15, 31, 47, 63]:
        out_tail, ref_tail = run_n_tail_case(n_value)
        tail_diff = (out_tail - ref_tail).abs()
        print(f"N={n_value}: max={tail_diff.max().item()} mean={tail_diff.mean().item()}")
        assert torch.allclose(out_tail, ref_tail, rtol=1e-2, atol=1e-2), f"tail case failed for N={n_value}"

    print("\nDirect packed-input mmt4d test passed.")
