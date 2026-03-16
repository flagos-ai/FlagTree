# Copyright 2026- Xcoresigma Technology Co., Ltd
import torch
import torch_npu  # noqa
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def run_test(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    OP_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    a_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    b_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    c_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    tle.dsa.copy(x_ptr + offsets, a_ub, [BLOCK_SIZE])
    tle.dsa.copy(y_ptr + offsets, b_ub, [BLOCK_SIZE])

    if OP_ID == 0:  # add
        tle.dsa.add(a_ub, b_ub, c_ub)
    elif OP_ID == 1:  # sub
        tle.dsa.sub(a_ub, b_ub, c_ub)
    elif OP_ID == 2:  # mul
        tle.dsa.mul(a_ub, b_ub, c_ub)
    elif OP_ID == 3:  # div
        tle.dsa.div(a_ub, b_ub, c_ub)

    tle.dsa.copy(c_ub, output_ptr + offsets, [BLOCK_SIZE])


OP_REGISTRY = {
    'add': (0, torch.add),
    'sub': (1, torch.sub),
    'mul': (2, torch.mul),
    'div': (3, torch.div),
}


def common_test(op_name: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if op_name not in OP_REGISTRY:
        raise ValueError(f"Unsupported op: {op_name}")

    op_id, _ = OP_REGISTRY[op_name]
    output = torch.empty_like(x)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    run_test[grid](
        x,
        y,
        output,
        n_elements,
        OP_ID=op_id,
        BLOCK_SIZE=128,
    )
    return output


def test_binary_op(size: int = 1024, dtype=torch.float32):
    x = torch.rand(size, dtype=dtype).npu()
    y = torch.rand(size, dtype=dtype).npu()
    y = y + 0.1

    print(f"Testing {len(OP_REGISTRY)} operators with size={size}, dtype={dtype}")

    for op_name in OP_REGISTRY:
        torch_fn = OP_REGISTRY[op_name][1]
        triton_out = common_test(op_name, x, y)
        torch_out = torch_fn(x, y)

        max_diff = torch.max(torch.abs(torch_out - triton_out)).item()
        status = "SUCCESS" if max_diff < 1e-5 else "FAIL"
        print(f"{status} {op_name:8}: max diff = {max_diff:.2e}")


if __name__ == "__main__":
    test_binary_op(size=1024, dtype=torch.float32)
