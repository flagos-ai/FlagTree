import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def kernel_rem_after_expand(
    x_ptr,        # [M, K]
    idx_ptr,      # [BLOCK_M] raw indices
    y_ptr,
    M, K,
    stride_xm, stride_xk,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Modulo AFTER expand_dims: hits visitOperandRem rank==2 path.

    tt_0.mlir pattern:
      %0 = tt.load idx_ptr  → tensor<8xi32>
      %1 = tt.expand_dims %0 {axis=1} → tensor<8x1xi32>
      %2 = tt.splat %M → tensor<8x1xi32>
      %3 = arith.remsi %1, %2 → tensor<8x1xi32>  (rank==2 in visitOperandRem)
    """
    pid_m = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)

    # Load indices, expand to 2D, THEN modulo
    rows_raw = tl.load(idx_ptr + offs_m)           # tensor<8xi32>
    rows_2d = rows_raw[:, None]                     # tensor<8x1xi32>
    rows = rows_2d % M                              # tensor<8x1xi32> — remsi on 2D!

    cols = tl.arange(0, BLOCK_K)[None, :]            # tensor<1x16xi32>

    ptrs = x_ptr + rows * stride_xm + cols * stride_xk
    vals = tl.load(ptrs)

    out_ptrs = y_ptr + offs_m[:, None] * BLOCK_K + cols
    tl.store(out_ptrs, vals)


def test_rem_after_expand(device):
    M, K = 16, 32
    BLOCK_M, BLOCK_K = 8, 16

    x = torch.arange(M * K, device="cpu", dtype=torch.float32).reshape(M, K)
    idx_raw = torch.tensor([0, 4, 18, 22, 5, 35, 12, 50],
                           device="cpu", dtype=torch.int32)
    y_out = torch.full((BLOCK_M, BLOCK_K), -1.0, device="cpu", dtype=torch.float32)

    x_gpu = x.to(device)
    idx_gpu = idx_raw.to(device)
    y_gpu = y_out.to(device)

    print("=== rem after expand_dims (rank==2 visitOperandRem) ===")
    kernel_rem_after_expand[(1,)](
        x_gpu, idx_gpu, y_gpu,
        M, K,
        x.stride(0), x.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
    )

    y_actual = y_gpu.to("cpu")
    expected = torch.full((BLOCK_M, BLOCK_K), -1.0, dtype=torch.float32)
    rows_mod = idx_raw % M
    for m in range(BLOCK_M):
        row = rows_mod[m].item()
        expected[m, :] = x[row, :BLOCK_K]

    print("rows_raw:", idx_raw.tolist())
    print("rows_mod:", rows_mod.tolist())
    print("expected[:2, :4]:", expected[:2, :4])
    print("actual[:2, :4]:  ", y_actual[:2, :4])
    max_diff = torch.max(torch.abs(expected - y_actual))
    print(f"max diff = {max_diff}")
    if max_diff > 0:
        print("FAILED — rank==2 visitOperandRem likely drops remsi for unstructured dim")
    else:
        print("PASSED")


def test(device):
    test_rem_after_expand(device)


if __name__ == "__main__":
    test(DEVICE)
