import torch
import triton
import triton.language as tl


@triton.jit
def if_tensor_kernel(
        kv_start_idx,  # tensor
        output_ptr,
):
    pid = tl.program_id(0)
    if kv_start_idx:
        value = tl.load(kv_start_idx + pid)
        tl.store(output_ptr + pid, value)


# 测试函数
def test_kernel():
    n = 8
    device = 'npu'

    kv_start_idx = torch.arange(n, dtype=torch.float32, device=device)
    output1 = torch.zeros(n, dtype=torch.float32, device=device)
    if_tensor_kernel[(n,)](
        kv_start_idx, output1,
    )

    expected = torch.arange(n, dtype=torch.float32, device=device)
    assert torch.allclose(output1, expected), f"Output {output1} != Expected {expected}"
    print(f"RESULT: output1 = {output1}")
    print("✅ Test passed!")


if __name__ == "__main__":
    test_kernel()
