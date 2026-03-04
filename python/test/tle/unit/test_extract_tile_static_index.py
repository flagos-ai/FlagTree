import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle

@triton.jit
def extract_tile_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    # 这里的 M, N 设为 
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    tile = tle.extract_tile(x, index=[1, 1], tile_shape=[128, 128])

    out_offs_m = tl.arange(0, 128)
    out_offs_n = tl.arange(0, 128)
    tl.store(out_ptr + out_offs_m[:, None] * 128 + out_offs_n[None, :], tile)

# 调整尺寸
M, N = 512,512
x = torch.arange(M * N, device='cuda', dtype=torch.float32).reshape(M, N)
# 接收 128x128 的结果
out = torch.zeros(128, 128, device='cuda', dtype=torch.float32)

print(f"Running kernel with size {M}x{N} (Target tile: 128x128)...")
extract_tile_kernel[(1,)](x, out, M, N)

print("☑ Kernel executed!\n")

# --- 打印结果 (展示前几行) ---
print("--- 提取前的原始数据 (512×512) ---")
print(x[:20, :].cpu().int())

print("\n--- 提取后的数据 (128x128) ---")
print(out.cpu().int())

# 校验结果 起始数据应为512 * 128 + 128 = 65536
expected = x[128:256, 128:256]  #  128x128 块
if torch.allclose(out, expected):
    print("\n 测试通过！成功提取了 128x128 的数据块。")
else:
    print("\n 结果不匹配。")