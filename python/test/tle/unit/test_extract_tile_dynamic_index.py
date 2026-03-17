import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle

@triton.jit
def simple_extract_kernel(
    x_ptr, out_ptr,
    stride_xb, stride_xm, stride_xn,
    stride_ob, stride_om, stride_on, # Strides for the output tensor
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr
):
    # 1. Get 3D coordinates: z (layer/batch)
    pid_z = tl.program_id(0)

    # 2. Read the full background slice of x for this layer (32x32)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + pid_z * stride_xb + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    bg_tile = tl.load(x_ptrs)

    # 3. Determine extraction position: 
    # Layer 0 extracts from top-left (0,0), Layer 1 extracts from bottom-right (1,1)
    tile_idx_m = pid_z % (BLOCK_M // TILE_M)
    tile_idx_n = pid_z % (BLOCK_N // TILE_N)

    # 4. Perform dynamic extraction
    extracted_tile = tle.extract_tile(
        bg_tile, 
        index=[tile_idx_m, tile_idx_n], 
        tile_shape=[TILE_M, TILE_N]
    )

    # 5. Store the extracted small tile into the corresponding Z layer of the output tensor
    offs_tm = tl.arange(0, TILE_M)
    offs_tn = tl.arange(0, TILE_N)
    out_ptrs = out_ptr + pid_z * stride_ob + offs_tm[:, None] * stride_om + offs_tn[None, :] * stride_on
    tl.store(out_ptrs, extracted_tile)

# ------------------------------------------------------------
# Minimal Test
# ------------------------------------------------------------
def main():
    B = 2             # 2 layers (Z dimension)
    M, N = 32, 32     # 32x32 size per layer
    TM, TN = 16, 16   # The extracted small tile is 16x16
    
    # x is the 3D background tensor
    x = torch.zeros((B, M, N), device="cuda", dtype=torch.float32)
    
    # Preset values in specific regions to easily observe the extraction results
    # Layer 0: Put 88.0 in the top-left corner
    x[0, 0:16, 0:16] = 88.0
    # Layer 1: Put 99.0 in the bottom-right corner
    x[1, 16:32, 16:32] = 99.0
    
    # 'out' stores the extracted 16x16 small tiles (one per layer)
    out = torch.zeros((B, TM, TN), device="cuda", dtype=torch.float32)
    
    # Launch Kernel: B layers, each needs exactly 1x1 block
    grid = (B, 1, 1)
    
    simple_extract_kernel[grid](
        x, out,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=32, BLOCK_N=32, TILE_M=16, TILE_N=16
    )

    # --- Visualizing the results ---
    print("=== Layer 0 (Z=0) Extraction Result ===")
    print("Expected: Should extract 88.0 from the top-left")
    print("Actual extracted mean:", out[0].mean().item(), "\n")

    print("=== Layer 1 (Z=1) Extraction Result ===")
    print("Expected: Should extract 99.0 from the bottom-right")
    print("Actual extracted mean:", out[1].mean().item())

if __name__ == "__main__":
    main()