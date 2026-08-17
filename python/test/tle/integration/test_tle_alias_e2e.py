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

    # Phase 1: allocate v_smem, write the first batch of data (original input)
    v_smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, scope=tle.gpu.smem)
    tle.gpu.copy(in_ptr + offs, v_smem, [BLOCK])

    # Phase 2: alias — o_smem reuses v_smem's physical memory
    o_smem = tle.gpu.alloc(
        [BLOCK],
        dtype=tl.float32,
        scope=tle.gpu.smem,
        alias=v_smem,
        alias_offset_bytes=0,
    )

    # Phase 3: critical — write second batch of data via v_smem (input + BLOCK offset),
    #          overwriting the same physical memory. If alias is correct, o_smem
    #          should see the overwritten values.
    offs2 = offs + BLOCK
    tle.gpu.copy(in_ptr + offs2, v_smem, [BLOCK])

    # Phase 4: read from o_smem
    tle.gpu.copy(o_smem, out_ptr + offs, [BLOCK])


if __name__ == "__main__":
    N = 2048  # enough for two BLOCK-sized chunks
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.zeros(1024, device="cuda", dtype=torch.float32)  # only first 1024

    grid = (triton.cdiv(1024, BLOCK), )
    alias_e2e_kernel[grid](x, y, 1024, BLOCK=BLOCK)
    torch.cuda.synchronize()

    # Verify: y should equal x[128:1152] (second batch), not x[0:1024] (first batch).
    # This confirms that v_smem's overwrite was observed by o_smem.
    expected = x[128:1152]
    max_diff_alias = (expected - y).abs().max().item()

    # Control: y should NOT equal the first batch
    first_batch = x[:1024]
    max_diff_original = (first_batch - y).abs().max().item()

    if max_diff_alias < 1e-5 and max_diff_original > 1e-5:
        print("PASSED: alias E2E verified")
        print(f"  - o_smem matches overwritten data (second batch): max diff = {max_diff_alias:.2e}")
        print(f"  - o_smem differs from original data (first batch):  max diff = {max_diff_original:.2e}")
    else:
        print("FAILED:")
        print(f"  diff vs overwritten (should be 0): {max_diff_alias:.2e}")
        print(f"  diff vs original (should be >0):   {max_diff_original:.2e}")
