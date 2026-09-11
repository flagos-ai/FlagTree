import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def scatter_with_modulo_kernel(
    x_ptr,  # [M, K] output
    idx_ptr,  # [BLOCK_M] index tensor, values in [0, M), may repeat
    val_ptr,  # [BLOCK_M, BLOCK_K] values to store
    M,
    K,
    stride_xm,
    stride_xk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Modulo on the unstructured (scatter) dimension for stores.

    dim0 (scatter): rows_raw = tl.load(idx_ptr), rows = rows_raw % M
    dim1 (structured): cols = tl.arange(0, BLOCK_K)
    """
    pid_m = tl.program_id(0)

    # unstructured dim (dim 0): load indices and modulo
    row_offs = tl.arange(0, BLOCK_M)
    rows_raw = tl.load(idx_ptr + row_offs)  # tensor<BLOCK_M x i32>
    rows = rows_raw % M  # modulo on unstructured dim

    # structured dim (dim 1): contiguous range
    cols = tl.arange(0, BLOCK_K)

    ptrs = x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xk
    vals = tl.load(val_ptr + row_offs[:, None] * BLOCK_K + cols[None, :])
    tl.store(ptrs, vals)


def test_scatter_with_modulo(device):
    M, K = 16, 32
    BLOCK_M, BLOCK_K = 8, 16

    # data matrix [M, K] initialized to 0
    x = torch.zeros(M, K, device="cpu", dtype=torch.float32)
    # index tensor [BLOCK_M]: raw indices that may exceed M
    idx_raw = torch.tensor([0, 4, 18, 22, 5, 35, 12, 50], device="cpu", dtype=torch.int32)
    # values to store [BLOCK_M, BLOCK_K] = row index * 100 + col index
    val = torch.zeros(BLOCK_M, BLOCK_K, device="cpu", dtype=torch.float32)
    for m in range(BLOCK_M):
        for k in range(BLOCK_K):
            val[m, k] = idx_raw[m].item() * 100 + k

    x_gpu = x.to(device)
    idx_gpu = idx_raw.to(device)
    val_gpu = val.to(device)

    print("=== scatter_with_modulo ===")
    scatter_with_modulo_kernel[(1, )](
        x_gpu,
        idx_gpu,
        val_gpu,
        M,
        K,
        x.stride(0),
        x.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )

    x_actual = x_gpu.to("cpu")

    # expected: rows = idx_raw % M
    expected = torch.zeros(M, K, dtype=torch.float32)
    rows_mod = idx_raw % M  # [0, 4, 2, 6, 5, 3, 12, 2]
    for m in range(BLOCK_M):
        row = rows_mod[m].item()
        for k in range(BLOCK_K):
            expected[row, k] = val[m, k]

    print("rows_raw:", idx_raw.tolist())
    print("rows_mod:", rows_mod.tolist())
    print("expected[:7, :4]:")
    print(expected[:7, :4])
    print("actual[:7, :4]:")
    print(x_actual[:7, :4])
    max_diff = torch.max(torch.abs(expected - x_actual))
    print(f"max diff = {max_diff}")
    assert torch.equal(expected.int(), x_actual.int()), f"Mismatch! max diff = {max_diff}"
    print("PASSED")


def test(device):
    test_scatter_with_modulo(device)


if __name__ == "__main__":
    test(DEVICE)
