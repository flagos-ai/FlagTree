import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def gather_with_modulo_kernel(
    x_ptr,  # [M, K]
    idx_ptr,  # [BLOCK_M] index tensor, values in [0, M), may repeat
    y_ptr,  # [BLOCK_M, BLOCK_K] output
    M,
    K,
    stride_xm,
    stride_xk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Modulo on the unstructured (gather) dimension.

    dim0 (gather):   rows_raw = tl.load(idx_ptr), rows = rows_raw % M
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
    vals = tl.load(ptrs)

    out_ptrs = y_ptr + row_offs[:, None] * BLOCK_K + cols[None, :]
    tl.store(out_ptrs, vals)


def test_gather_with_modulo(device):
    M, K = 16, 32
    BLOCK_M, BLOCK_K = 8, 16

    # data matrix [M, K]
    x = torch.arange(M * K, device="cpu", dtype=torch.float32).reshape(M, K)
    # index tensor [BLOCK_M]: raw indices that may exceed M
    idx_raw = torch.tensor([0, 4, 18, 22, 5, 35, 12, 50], device="cpu", dtype=torch.int32)
    y_out = torch.full((BLOCK_M, BLOCK_K), -1.0, device="cpu", dtype=torch.float32)

    x_gpu = x.to(device)
    idx_gpu = idx_raw.to(device)
    y_gpu = y_out.to(device)

    print("=== gather_with_modulo ===")
    gather_with_modulo_kernel[(1, )](
        x_gpu,
        idx_gpu,
        y_gpu,
        M,
        K,
        x.stride(0),
        x.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )

    y_actual = y_gpu.to("cpu")

    # expected: rows = idx_raw % M
    expected = torch.full((BLOCK_M, BLOCK_K), -1.0, dtype=torch.float32)
    rows_mod = idx_raw % M  # [0, 4, 2, 6, 5, 3, 12, 2]
    for m in range(BLOCK_M):
        row = rows_mod[m].item()
        expected[m, :] = x[row, :BLOCK_K]

    print("rows_raw:", idx_raw.tolist())
    print("rows_mod:", rows_mod.tolist())
    print("expected[:4, :4]:", expected[:4, :4])
    print("actual[:4, :4]:  ", y_actual[:4, :4])
    max_diff = torch.max(torch.abs(expected - y_actual))
    print(f"max diff = {max_diff}")
    assert torch.equal(expected.int(), y_actual.int()), f"Mismatch! max diff = {max_diff}"
    print("PASSED")


def test(device):
    test_gather_with_modulo(device)


if __name__ == "__main__":
    test(DEVICE)
