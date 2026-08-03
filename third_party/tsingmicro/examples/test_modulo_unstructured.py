import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def kernel_modulo_unstructured(
    in_ptr,
    idx_ptr,
    out_ptr,
    M,
    N,
    stride_m,
    stride_n,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Reproduce the assert(!lhsState.hasModulo() && !rhsState.hasModulo())
    in PtrAnalysisTS::addState (line 237 before fix).

    Pattern (matched to vllm flash attention tt_0.mlir):
      tt.load     → visitOperand fallback   → unstructured dim0
      arith.remsi → hasModulo on dim1
      arith.addi  → addState: dim0 unstructured branch,
                    rhsState.hasModulo() = true (dim1 has modulo) → ASSERT
    """
    pid = tl.program_id(0)

    # (A) Load indices → unstructured via fallback (visitOperand line 1132)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    indices = tl.load(idx_ptr + offs)

    # (B) Modulo column offset → hasModulo via arith.remsi
    #     (4 + arange) % N always in [0, N-1] → no out-of-bounds
    mod_offs = (4 + tl.arange(0, BLOCK_N)) % N

    # Build 2D: [BLOCK_M, 1] for dim0, [1, BLOCK_N] for dim1
    offs_a = indices[:, None] * stride_m  # dim0 unstructured, dim1 structured(0)
    offs_m = mod_offs[None, :] * stride_n  # dim0 structured(0), dim1 hasModulo

    # arith.addi → addState:
    #   dim0: lhs unstructured + rhs structured(offs_m has no offset on dim0)
    #         rhsState.hasModulo() = true (dim1!) → OLD ASSERT FIRES
    offs_final = offs_a + offs_m

    ptr = in_ptr + offs_final
    mask_m = (pid * BLOCK_M + tl.arange(0, BLOCK_M)) < M
    mask_n = tl.arange(0, BLOCK_N) < N
    mask = mask_m[:, None] & mask_n[None, :]
    val = tl.load(ptr, mask=mask, other=0)

    out_offs = (pid * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_cm \
               + tl.arange(0, BLOCK_N)[None, :] * stride_cn
    tl.store(out_ptr + out_offs, val, mask=mask)


@triton.jit
def kernel_modulo_unstructured_loop(
    in_ptr,
    idx_ptr,
    out_ptr,
    M,
    N,
    stride_m,
    stride_n,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_LOOPS: tl.constexpr,
):
    """
    Same pattern inside an scf.for loop (closer to vllm attention structure).
    """
    pid = tl.program_id(0)

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    indices = tl.load(idx_ptr + offs)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = (pid * BLOCK_M + tl.arange(0, BLOCK_M)) < M
    mask_n = tl.arange(0, BLOCK_N) < N
    mask = mask_m[:, None] & mask_n[None, :]

    for k in range(NUM_LOOPS):
        mod_offs = (4 + k * 8 + tl.arange(0, BLOCK_N)) % N

        offs_a = indices[:, None] * stride_m
        offs_m = mod_offs[None, :] * stride_n

        # arith.addi with unstructured + hasModulo → OLD assert
        offs_final = offs_a + offs_m
        ptr = in_ptr + offs_final

        val = tl.load(ptr, mask=mask, other=0)
        acc += val

    out_offs = (pid * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_cm \
               + tl.arange(0, BLOCK_N)[None, :] * stride_cn
    tl.store(out_ptr + out_offs, acc, mask=mask)


def test(device):
    M, N = 16, 32
    BLOCK_M, BLOCK_N = 8, 16

    in_data = torch.arange(M * N, device="cpu", dtype=torch.float32).reshape(M, N)
    idx_data = torch.arange(0, M, device="cpu", dtype=torch.int32)
    out = torch.full((M, N), -1.0, device="cpu", dtype=torch.float32)

    in_data = in_data.to(device)
    idx_data = idx_data.to(device)
    out = out.to(device)

    print("=== test_modulo_unstructured ===")
    grid = lambda meta: (1, )
    kernel_modulo_unstructured[grid](
        in_data,
        idx_data,
        out,
        M,
        N,
        in_data.stride(0),
        in_data.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    out_cpu = out.to("cpu")
    expected = torch.full((M, N), -1.0, dtype=torch.float32)
    for m in range(BLOCK_M):
        for n in range(BLOCK_N):
            src_m = idx_data[m].item()
            src_n = (4 + n) % N
            if 0 <= src_m < M and 0 <= src_n < N:
                expected[m, n] = in_data.cpu()[src_m, src_n].item()

    print("expected[:4, :4]:", expected[:4, :4])
    print("actual[:4, :4]:  ", out_cpu[:4, :4])
    max_diff = torch.max(torch.abs(expected - out_cpu))
    print(f"max diff = {max_diff}")
    assert torch.equal(expected.int(), out_cpu.int()), f"Mismatch! max diff = {max_diff}"
    print("PASSED")

    print()
    print("=== test_modulo_unstructured_loop ===")
    NUM_LOOPS = 3
    out.fill_(-1.0)

    kernel_modulo_unstructured_loop[grid](
        in_data,
        idx_data,
        out,
        M,
        N,
        in_data.stride(0),
        in_data.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NUM_LOOPS=NUM_LOOPS,
    )

    out_cpu = out.to("cpu")
    expected_loop = torch.full((M, N), -1.0, dtype=torch.float32)
    for m in range(BLOCK_M):
        for n in range(BLOCK_N):
            acc = 0.0
            for k in range(NUM_LOOPS):
                src_m = idx_data[m].item()
                src_n = (4 + k * 8 + n) % N
                if 0 <= src_m < M and 0 <= src_n < N:
                    acc += in_data.cpu()[src_m, src_n].item()
            if acc != 0:
                expected_loop[m, n] = acc

    print("expected[:4, :4]:", expected_loop[:4, :4])
    print("actual[:4, :4]:  ", out_cpu[:4, :4])
    max_diff = torch.max(torch.abs(expected_loop - out_cpu))
    print(f"max diff = {max_diff}")
    assert torch.equal(expected_loop.int(), out_cpu.int()), f"Mismatch! max diff = {max_diff}"
    print("PASSED")


if __name__ == "__main__":
    test(DEVICE)
