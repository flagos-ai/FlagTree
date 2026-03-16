import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle

@triton.jit
def insert_tile_dynamic_kernel(
    x_ptr, y_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr,
    TM: tl.constexpr, TN: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = tl.arange(0, TM) + pid_m * TM
    offs_n = tl.arange(0, TN) + pid_n * TN

    # load tiles
    x_tile = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    y_tile = tl.load(y_ptr + tl.arange(0, TM)[:, None] * TN + tl.arange(0, TN)[None, :])

    z_tile = tle.insert_tile(x_tile, y_tile, index=[pid_m, pid_n])
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], z_tile)

# ------------------------------------------------------------
# CPU reference
# ------------------------------------------------------------
def reference_insert(x, y, TM, TN):
    M, N = x.shape
    grid_m = M // TM
    grid_n = N // TN
    ref = x.clone()
    for i in range(grid_m):
        for j in range(grid_n):
            ref[i*TM:(i+1)*TM, j*TN:(j+1)*TN] = y
    return ref

def run_single_test(M, N, TM, TN, dtype):
    grid_m = M // TM
    grid_n = N // TN

    x = torch.arange(M*N, device="cuda", dtype=torch.float32).reshape(M, N).to(dtype)
    y = 10000 + torch.arange(TM*TN, device="cuda", dtype=torch.float32).reshape(TM, TN).to(dtype)

    out_dynamic = torch.empty_like(x)
    insert_tile_dynamic_kernel[(grid_m, grid_n)](x, y, out_dynamic, M, N, TM, TN)
    expected_dynamic = reference_insert(x, y, TM, TN)
    atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    dynamic_pass = torch.allclose(out_dynamic.float(), expected_dynamic.float(), atol=atol)

    dtype_name = str(dtype).split(".")[-1]
    print(f"M={M:<4} N={N:<4} TM={TM:<3} TN={TN:<3} dtype={dtype_name:<8} "
          f"dynamic={'PASS' if dynamic_pass else 'FAIL'}")
    return dynamic_pass


def main():
    base_shapes = [(256,256),(128,128),(64,64),(32,32),]
    # Tile shapes
    tile_shapes = [(64,64),(32,32),(16,16),(32,16),(16,32),(8,64),(64,8),]
    # Dtypes
    dtypes = [torch.float32,torch.float16,torch.bfloat16,torch.int32]

    total = 0
    passed = 0
    print("Triton insert_tile Full Test: Dynamic Index only")

    for M, N in base_shapes:
        for TM, TN in tile_shapes:
            if TM > M or TN > N:
                continue
            if M % TM != 0 or N % TN != 0:
                continue
            for dtype in dtypes:
                dynamic_pass = run_single_test(M, N, TM, TN, dtype)
                total += 1
                if dynamic_pass:
                    passed += 1
    print("="*100)
    print(f"FINAL RESULT: {passed}/{total} PASSED")
if __name__ == "__main__":
    main()
