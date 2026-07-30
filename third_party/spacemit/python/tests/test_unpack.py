import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

MICRO_M = 16
MICRO_N = 32
BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 128

MB = BLOCK_SIZE_M // MICRO_M
NB = BLOCK_SIZE_N // MICRO_N


def unpack_ref(x: torch.Tensor) -> torch.Tensor:
    # x: [MB, NB, MICRO_M, MICRO_N]
    assert x.shape == (MB, NB, MICRO_M, MICRO_N)
    return x.permute(0, 2, 1, 3).contiguous().view(BLOCK_SIZE_M, BLOCK_SIZE_N)


@triton.jit
def unpack_kernel(
    x_ptr,
    y_ptr,
    stride_x0,
    stride_x1,
    stride_x2,
    stride_x3,
    stride_y0,
    stride_y1,
    M,
    N,
    MB: tl.constexpr,
    NB: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_N: tl.constexpr,
):
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=[MB, NB, MICRO_M, MICRO_N],
        strides=[stride_x0, stride_x1, stride_x2, stride_x3],
        offsets=[0, 0, 0, 0],
        block_shape=[MB, NB, MICRO_M, MICRO_N],
        order=[3, 2, 1, 0],
    )

    x = tl.load(x_block_ptr)

    y = smt.view(x, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1))

    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=[M, N],
        strides=[stride_y0, stride_y1],
        offsets=[0, 0],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        order=[1, 0],
    )
    tl.store(y_block_ptr, y, boundary_check=(0, 1))


def run_unpack_case(n_value: int):
    assert 0 < n_value <= BLOCK_SIZE_N
    torch.manual_seed(0)
    device = "cpu"

    x = torch.randn((MB, NB, MICRO_M, MICRO_N), dtype=torch.float16, device=device)
    ref_full = unpack_ref(x)
    ref = ref_full[:, :n_value].contiguous()

    y = torch.empty((BLOCK_SIZE_M, n_value), dtype=torch.float16, device=device)

    unpack_kernel[(1, )](
        x,
        y,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        y.stride(0),
        y.stride(1),
        BLOCK_SIZE_M,
        n_value,
        MB=MB,
        NB=NB,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        MICRO_M=MICRO_M,
        MICRO_N=MICRO_N,
    )

    return y, ref


def _check_unpack_case(n_value: int):
    out, ref = run_unpack_case(n_value)
    diff = (out - ref).abs()

    print(f"\n=== unpack-only case: N={n_value} ===")
    print("out[:8, :min(8,N)] =")
    print(out[:8, :min(8, n_value)])
    print("ref[:8, :min(8,N)] =")
    print(ref[:8, :min(8, n_value)])
    print("diff[:8, :min(8,N)] =")
    print(diff[:8, :min(8, n_value)])
    print(f"max={diff.max().item()} mean={diff.mean().item()}")

    assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3), \
        f"unpack-only case failed for N={n_value}"


def test_unpack_n8():
    _check_unpack_case(8)


def test_unpack_n16():
    _check_unpack_case(16)


def test_unpack_n32():
    _check_unpack_case(32)


def test_unpack_n64():
    _check_unpack_case(64)


def test_unpack_n128():
    _check_unpack_case(128)


if __name__ == "__main__":
    for n in [8, 16, 32, 64, 128]:
        _check_unpack_case(n)
    print("\nAll unpack-only tests passed.")
