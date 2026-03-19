import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit

def extract_tile_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    # Set M, N as input matrix dimensions
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    # Extract a 128x128 tile starting from index [1, 1]
    tile = tle.extract_tile(x, index=[1, 1], tile_shape=[128, 128])

    out_offs_m = tl.arange(0, 128)
    out_offs_n = tl.arange(0, 128)
    tl.store(out_ptr + out_offs_m[:, None] * 128 + out_offs_n[None, :], tile)

# Set matrix size
M, N = 512, 512
x = torch.arange(M * N, device='cuda', dtype=torch.float32).reshape(M, N)
# Output buffer for the 128x128 result
out = torch.zeros(128, 128, device='cuda', dtype=torch.float32)

print(f"Running kernel with size {M}x{N} (Target tile: 128x128)...")
extract_tile_kernel[(1, )](x, out, M, N)

print("☑ Kernel executed!\n")

# --- Print results (show first few rows) ---
print("--- Original data before extraction (512x512) ---")
print(x[:20, :].cpu().int())

print("\n--- Data after extraction (128x128) ---")
print(out.cpu().int())

# Validate result: the starting value should be 512 * 128 + 128 = 65536
expected = x[128:256, 128:256]  # 128x128 block
if torch.allclose(out, expected):
    print("\nTest passed! Successfully extracted a 128x128 data block.")
else:
    print("\nResult does not match.")
