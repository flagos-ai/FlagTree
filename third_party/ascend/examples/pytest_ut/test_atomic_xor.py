import triton
import triton.language as tl
import pytest
import test_common
import torch
import torch_npu


@triton.jit
def atomic_xor(in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_xor(out_ptr0 + (x1), tmp0, xmask)
    tl.store(out_ptr1 + (x1), tmp1, xmask)


@pytest.mark.parametrize('param_list',
                         [
                             ['int32', (32, 32), 2],
                             ['int16', (32, 32), 7],
                             ['int8', (32, 32), 10],
                         ]
                         )
def test_atomic_xor(param_list):
    dtype, shape, ncore = param_list
    block_size = shape[0] * shape[1] // ncore
    split_size = shape[0] // ncore

    # 初始化原始值 val，全为 0b0011（十进制 3）
    val_value = 3
    val = torch.full(shape, val_value, dtype=eval(f'torch.{dtype}')).npu()

    # 每个线程使用不同输入值 x1，全为 0b0101（十进制 5）
    pointer_value = 5
    pointer = torch.full((split_size, shape[1]), pointer_value, dtype=eval(f'torch.{dtype}')).npu()
    pointer_old = torch.full_like(pointer, -10)
    
    # 原子异或后：val ^= pointer 每个线程执行一次
    # 因为异或操作具有可逆性和对称性，参考更新次数
    # 所以参考值为 val_value ^ pointer_value ^ pointer_value ^ ...（ncore 次）
    pointer_result = pointer_value
    for _ in range(ncore):
        pointer_result ^= val_value

    pointer_ref = torch.full_like(pointer, pointer_result)

    n_elements = shape[0] * shape[1]
    atomic_xor[ncore, 1, 1](val, pointer, pointer_old, n_elements, BLOCK_SIZE=split_size * shape[1])
    test_common.validate_cmp(dtype, pointer, pointer_ref)
