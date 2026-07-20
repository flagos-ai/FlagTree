"""Minimal E2E test: verify alloc alias shares the same physical smem."""
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

BLOCK = 128


@triton.jit
def alias_e2e_kernel(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    vals = tl.load(in_ptr + offs, mask=mask)

    # 1. 分配 v_smem，用 copy 写入数据
    v_smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, scope=tle.gpu.smem)
    tle.gpu.copy(in_ptr + offs, v_smem, [BLOCK])

    # 2. alias: o_smem 复用 v_smem 的物理内存
    o_smem = tle.gpu.alloc(
        [BLOCK], dtype=tl.float32, scope=tle.gpu.smem,
        alias=v_smem, alias_offset_bytes=0,
    )

    # 3. 从 o_smem 读回到 out —— 如果 alias 正确，值和输入一致
    tle.gpu.copy(o_smem, out_ptr + offs, [BLOCK])


if __name__ == "__main__":
    N = 1024
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.zeros(N, device="cuda", dtype=torch.float32)

    grid = (triton.cdiv(N, BLOCK),)
    alias_e2e_kernel[grid](x, y, N, BLOCK=BLOCK)
    torch.cuda.synchronize()

    max_diff = (x - y).abs().max().item()
    if max_diff < 1e-5:
        print("PASSED: alias E2E — o_smem reads back the same values written to v_smem")
    else:
        print(f"FAILED: max diff = {max_diff}")
