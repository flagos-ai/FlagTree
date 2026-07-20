"""E2E test: verify alloc alias shares the same physical smem via modify-and-observe."""
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

    # 阶段 1: 分配 v_smem，写入第一批数据 (原始输入)
    v_smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, scope=tle.gpu.smem)
    tle.gpu.copy(in_ptr + offs, v_smem, [BLOCK])

    # 阶段 2: alias — o_smem 复用 v_smem 的物理内存
    o_smem = tle.gpu.alloc(
        [BLOCK], dtype=tl.float32, scope=tle.gpu.smem,
        alias=v_smem, alias_offset_bytes=0,
    )

    # 阶段 3: 关键 — 通过 v_smem 写入第二批数据 (input + BLOCK 偏移)，
    #         覆盖同一块物理内存。如果 alias 正确，o_smem 应该看到覆盖后的值。
    offs2 = offs + BLOCK
    mask2 = offs2 < N
    tle.gpu.copy(in_ptr + offs2, v_smem, [BLOCK])

    # 阶段 4: 从 o_smem 读出
    tle.gpu.copy(o_smem, out_ptr + offs, [BLOCK])


if __name__ == "__main__":
    N = 2048  # enough for two BLOCK-sized chunks
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.zeros(1024, device="cuda", dtype=torch.float32)  # only first 1024

    grid = (triton.cdiv(1024, BLOCK),)
    alias_e2e_kernel[grid](x, y, 1024, BLOCK=BLOCK)  # 注意：N=1024，第二批数据在 idx 128..1151
    torch.cuda.synchronize()

    # 验证: y 应该等于 x[128:1152] (第二批)，而不是 x[0:1024] (第一批)
    # 这证明 v_smem 的覆盖写入被 o_smem 观察到了
    expected = x[128:1152]
    max_diff_alias = (expected - y).abs().max().item()

    # 对照组: y 不应该等于第一批数据
    first_batch = x[:1024]
    max_diff_original = (first_batch - y).abs().max().item()

    if max_diff_alias < 1e-5 and max_diff_original > 1e-5:
        print("PASSED: alias E2E verified")
        print(f"  - o_smem matches overwritten data (second batch): max diff = {max_diff_alias:.2e}")
        print(f"  - o_smem differs from original data (first batch):  max diff = {max_diff_original:.2e}")
    else:
        print(f"FAILED:")
        print(f"  diff vs overwritten (should be 0): {max_diff_alias:.2e}")
        print(f"  diff vs original (should be >0):   {max_diff_original:.2e}")
