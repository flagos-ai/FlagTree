import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle
# ------------------------------------------------------------
# Kernel
# ------------------------------------------------------------
@triton.jit
def dynamic_extract_kernel(
    x_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # load full tensor
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    # dynamic extract
    tile = tle.extract_tile(
        x,
        index=[pid_m, pid_n],
        tile_shape=[TM, TN]
    )
    # store tile
    tile_m = tl.arange(0, TM)
    tile_n = tl.arange(0, TN)
    out_ptr_tile = (out_ptr + pid_m * TM * N + pid_n * TN + tile_m[:, None] * N + tile_n[None, :])
    tl.store(out_ptr_tile, tile)
# ------------------------------------------------------------
# Reference
# ------------------------------------------------------------
def reference_extract(x, TM, TN):
    M, N = x.shape
    grid_m = M // TM
    grid_n = N // TN
    out = torch.zeros_like(x)
    for i in range(grid_m):
        for j in range(grid_n):
            out[i*TM:(i+1)*TM,j*TN:(j+1)*TN] = x[i*TM:(i+1)*TM,j*TN:(j+1)*TN]
    return out

# ------------------------------------------------------------
# Test Single Case
# ------------------------------------------------------------

def run_single_test(M, N, TM, TN, dtype):
    grid_m = M // TM
    grid_n = N // TN
    x_vals = torch.arange(M*N, device="cuda", dtype=torch.float32) % 10000
    x = x_vals.reshape(M, N).to(dtype)
    out = torch.zeros((M, N), device="cuda", dtype=dtype)
    dynamic_extract_kernel[(grid_m, grid_n)](x, out, M, N, TM, TN)
    expected = reference_extract(x, TM, TN)
    atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    Success = torch.allclose(out.float(), expected.float(), atol=atol)
    dtype_name = str(dtype).split(".")[-1]
    print(
        f"M={M:<4} N={N:<4} "
        f"TM={TM:<3} TN={TN:<3} "
        f"dtype={dtype_name:<8} "
        f"grid=({grid_m},{grid_n}) "
        f"{'PASS' if Success else 'FAIL'}"
    )
    return Success
# ------------------------------------------------------------
# Main Test
# ------------------------------------------------------------
def main():
    print("="*110)
    print("Triton extract_tile Dynamic Index Full Test")
    print("="*110)
    # Base shapes
    base_shapes = [(256,256),(128,128),(64,64),(32,32),]
    # Tile shapes
    tile_shapes = [(64,64),(32,32),(16,16),(32,16),(16,32),(8,64),(64,8),]
    # Dtypes
    dtypes = [torch.float32,torch.float16,torch.bfloat16,torch.int32]
    total = 0
    passed = 0
    for M, N in base_shapes:
        for TM, TN in tile_shapes:
            if TM > M or TN > N:
                continue
            if M % TM != 0 or N % TN != 0:
                continue
            for dtype in dtypes:
                Success = run_single_test(M, N, TM, TN, dtype)
                total += 1
                if Success:
                    passed += 1
    print(f"FINAL RESULT: {passed}/{total} PASSED")
if __name__ == "__main__":
    main()