import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def insert_tile_dynamic_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
):
    # program id → dynamic_index
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = tl.arange(0, TM) + pid_m * TM
    offs_n = tl.arange(0, TN) + pid_n * TN
    # load base tile
    x_tile = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    y_tile = tl.load(y_ptr + tl.arange(0, TM)[:, None] * TN + tl.arange(0, TN)[None, :])
    z_tile = tle.insert_tile(x_tile, y_tile, index=[0, 0])
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], z_tile)


# ------------------------------------------------------------
# CPU Reference
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
    # base tensor
    x_vals = torch.arange(M*N, device="cuda", dtype=torch.float32) % 10000
    x = x_vals.reshape(M, N).to(dtype)
    # tile tensor
    y_vals = 10000 + torch.arange(TM*TN, device="cuda", dtype=torch.float32)
    y = y_vals.reshape(TM, TN).to(dtype)
    
    out = torch.empty_like(x)
    # launch kernel
    insert_tile_dynamic_kernel[(grid_m, grid_n)](x, y, out, M, N, TM, TN)
    # CPU reference
    expected = reference_insert(x, y, TM, TN)
    
    atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    success = torch.allclose(out.float(), expected.float(), atol=atol)
    dtype_name = str(dtype).split(".")[-1]
    print(
        f"M={M:<4} N={N:<4} "
        f"TM={TM:<3} TN={TN:<3} "
        f"dtype={dtype_name:<8} "
        f"grid=({grid_m},{grid_n}) "
        f"{'PASS' if success else 'FAIL'}"
    )
    return success

def main():
    print("="*110)
    print("Triton insert_tile program_id Dynamic Index Full Test")
    print("="*110)
    base_shapes = [(256,256), (128,128), (64,64), (32,32)]
    tile_shapes = [(64,64), (32,32), (16,16), (32,16), (16,32), (8,64), (64,8)]
    dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int32]
    total = 0
    passed = 0
    for M,N in base_shapes:
        for TM,TN in tile_shapes:
            if TM > M or TN > N:
                continue
            if M % TM != 0 or N % TN != 0:
                continue
            for dtype in dtypes:
                success = run_single_test(M,N,TM,TN,dtype)
                total += 1
                if success:
                    passed += 1
    print(f"FINAL RESULT : {passed}/{total} PASSED")
if __name__ == "__main__":
    main()