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

    # 1. 分配 v_smem 并写入输入数据
    v_smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, scope=tle.gpu.smem)
    vals = tl.load(in_ptr + offs, mask=mask)
    tle.gpu.copy(vals, v_smem, [BLOCK])

    # 2. alias: o_smem 复用 v_smem 的物理内存，偏移 0 字节
    o_smem = tle.gpu.alloc(
        [BLOCK], dtype=tl.float32, scope=tle.gpu.smem,
        alias=v_smem, alias_offset_bytes=0,
    )

    # 3. 从 o_smem 读回 —— 如果 alias 正确，应该和写入 v_smem 的值一致
    result = tl.zeros([BLOCK], dtype=tl.float32)
    tle.gpu.copy(o_smem, result, [BLOCK])
    tl.store(out_ptr + offs, result, mask=mask)


if __name__ == "__main__":
    N = 1024
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.empty(N, device="cuda", dtype=torch.float32)

    grid = (triton.cdiv(N, BLOCK),)
    alias_e2e_kernel[grid](x, y, N, BLOCK=BLOCK)
    torch.cuda.synchronize()

    max_diff = (x - y).abs().max().item()
    if max_diff < 1e-5:
        print("PASSED: alias E2E — o_smem reads back the same values written to v_smem")
    else:
        print(f"FAILED: max diff = {max_diff}")
