# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest

import triton
import triton.language as tl
import test_common

import torch
import torch_npu


num_cube_core = 20
num_vector_core = 20

def foo(a, d ,shape ):
    y = a.reshape(shape)
    y = y.permute(0,2,1) + d
    return y


@triton.jit
def triton_gpu_revised(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, 
                       SHAPE0:tl.constexpr, SHAPE1:tl.constexpr,SHAPE2:tl.constexpr,
                       XBLOCK : tl.constexpr):
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    x2 = xindex[:, None]
    x2_1 = xindex[None, :]
    y3 = yindex[ :, None]
    y0 = (yindex % SHAPE1)[None,:]
    y1 = (yindex // SHAPE1)[None,:]
    tmp0 = tl.load(in_ptr0 + (x2_1 + (SHAPE2*y3)), (x2_1 < xnumel) & (y3 < ynumel) )
    tmp1 = tl.load(in_ptr1 + (y0 + (SHAPE1*x2) + (SHAPE1*SHAPE2*y1)), \
                   (xindex[:,None] < xnumel) & (yindex[None,:] < ynumel))
                   # (x2 < xnumel) & (y0 < SHAPE1))
                   # (xindex[:,None] < xnumel) & (yindex[None,:] < ynumel))
    tmp10 = tmp1.permute(1,0)
    tmp2 = tmp0 + tmp10
    tl.store(out_ptr0 + (x2_1 + (SHAPE2*y3)), tmp2, (x2_1 < xnumel) & (y3 < ynumel))

def biggest_divisor(num):
    for i in range(2,num):  
        if num % i == 0:  
            return num // i
    return num



def find_good_yblock(ynumel,xnumel, y_upper, dtype) :
    y = ynumel
    x = xnumel

    align_numel = 4 if dtype==torch.int64 else 8 
    ub_upper = 3900 if dtype==torch.int64 else 8000

 
    # optimize block_dim
    def get_block_dim(y,x) :
        return ((xnumel + x -1 )//x) * ((y_upper+ y -1 ) // y)
    
    count = 0
    while( get_block_dim(y,x) < num_vector_core and y > 8 and count < 20 ) :
        y_1 = biggest_divisor(y) 
        if get_block_dim(y_1, x) > num_vector_core :
            break
        y = y_1
        if  get_block_dim(y,x) < num_vector_core and x > align_numel :
            x = x // 2 
        count = count + 1

    # optimize block_size to avoid ub-overflow
    while( y * x > ub_upper) :
        y_1 = biggest_divisor(y)
        if y_1 == y or y_1 <=align_numel:
            break
        y = y_1

    while( y * x > ub_upper and x > align_numel) :
        x_1 = x // 2
        if x_1 <= align_numel :
            break
        x = x_1
            
    return (x,y)

def triton_foo(a, d ,shape, dtype) :
    z, y, x = shape
    out = torch.empty_strided((z, x, y), (x*y, 1, x), device='npu', dtype=dtype)
    XBLOCK, YBLOCK= find_good_yblock(y,x, y*z, dtype=dtype)
    print(f"XBLOCK={XBLOCK},YBLOCK={YBLOCK}, block_dim={((x + XBLOCK -1 )//XBLOCK) * (((y*z) + YBLOCK -1 ) // YBLOCK)}")
    grid = ((x + XBLOCK -1 )//XBLOCK, ((y*z) + YBLOCK -1 ) // YBLOCK, 1) 
    
    triton_gpu_revised[grid](
        a, d, out, y * z, x,
        SHAPE0=z, SHAPE1=y, SHAPE2=x,
        YBLOCK=YBLOCK, XBLOCK=XBLOCK,
        enable_linearize=True
    )
    return out 


types = [
    (torch.float32, 'float32'),
]


shapes = [(8,2048,4)]

@pytest.mark.parametrize('dtype,sigtype', types)
@pytest.mark.parametrize('Z,Y,X', shapes)
def test_linearize(Z,Y,X, dtype, sigtype) :
    shape = (Z,Y,X)
    print(f"start data validation on shape:{shape}")
    a = test_common.generate_tensor(shape=(Z, Y * X), dtype=sigtype).npu()
    d = test_common.generate_tensor(shape=(Z, X, Y), dtype=sigtype).npu()
    r = triton_foo( a,  d, shape, dtype)
    r1 = foo(a, d ,shape )
    test_common.validate_cmp(sigtype, r1, r)
    print(f"data validation passed")


# Test linearize offset handling with expert routing pattern
@triton.jit
def linearize_offset_kernel(
    bias_ptr,
    output_ptr,
    experts_ids_ptr,
    N: tl.constexpr,
    EM: tl.constexpr,
    stride_bias_e,
    stride_bias_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    off_experts = tl.load(experts_ids_ptr + pid_m).to(tl.int64)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N

    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + off_experts * stride_bias_e + offs_bn[None, :] * stride_bias_n
        )
        tl.store(output_ptr + pid*16 + tl.arange(0, 16), bias.reshape(16))


def torch_linearize_offset(bias_ptr, experts_ids_ptr, N, EM, stride_bias_e, stride_bias_n, 
                           BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M):
    """PyTorch reference implementation for offset handling"""
    output = torch.empty([16, 16], dtype=bias_ptr.dtype, device=bias_ptr.device)
    
    num_pid_m = (EM + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    for pid in range(16):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        
        off_experts = experts_ids_ptr[pid_m].to(torch.int64).item()
        offs_bn = torch.arange(pid_n * BLOCK_SIZE_N, 
                               (pid_n + 1) * BLOCK_SIZE_N, 
                               dtype=torch.int64, device=bias_ptr.device) % N
        
        bias = bias_ptr[off_experts, offs_bn]
        output[pid] = bias
    
    return output


@pytest.mark.parametrize('dtype,sigtype', [(torch.float32, 'float32')])
def test_linearize_offset_handling(dtype, sigtype):
    """
    Test linearization's handling of complex offset patterns.
    This test simulates expert routing scenarios where offsets are computed
    dynamically based on expert IDs, validating correct pointer arithmetic
    and linearization in the compiler.
    """
    print(f"Testing linearize offset handling with dtype={sigtype}")
    
    # Setup test data
    num_experts = 4
    hidden_dim = 64
    bias_ptr = torch.arange(0, num_experts * hidden_dim, dtype=dtype).npu().reshape(num_experts, hidden_dim)
    output_ptr = torch.empty([16, 16], dtype=dtype).npu()
    experts_ids_ptr = torch.tensor([1, 2, 3, 1], dtype=torch.int32).npu()
    
    # Kernel parameters
    N = 64
    EM = 64
    stride_bias_e = 64
    stride_bias_n = 1
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    GROUP_SIZE_M = 4
    
    # Run triton kernel
    linearize_offset_kernel[(16,)](
        bias_ptr=bias_ptr,
        output_ptr=output_ptr,
        experts_ids_ptr=experts_ids_ptr,
        N=N,
        EM=EM,
        stride_bias_e=stride_bias_e,
        stride_bias_n=stride_bias_n,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        GROUP_SIZE_M=GROUP_SIZE_M,
        enable_linearize=True
    )
    
    # Compute reference result
    expected = torch_linearize_offset(
        bias_ptr, experts_ids_ptr, N, EM, 
        stride_bias_e, stride_bias_n,
        BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )
    
    # Validate results
    test_common.validate_cmp(sigtype, expected, output_ptr)
    print(f"Linearize offset handling test passed")


def torch_expand_dims_and_add(buffer, cache, buffer_stride, BLOCK, NUMEL):
    for i in range(buffer.shape[0]):
        tmp = buffer[i, 0, :buffer_stride]
        accumulator = torch.zeros(2, buffer_stride).npu()
        accumulator += tmp[None, :]
        cache[i, 0, :] = accumulator.reshape(buffer_stride * 2)


@triton.jit
def expand_dims_and_add(
    buffer_ptr,
    cache_ptr,
    buffer_stride: tl.constexpr,
    BLOCK: tl.constexpr,
    NUMEL: tl.constexpr,
):
    pid = tl.program_id(0)
    buffer_offset = pid * buffer_stride
    buffer_index = (buffer_offset + tl.arange(0, buffer_stride)) % NUMEL

    cache_offset = pid * buffer_stride * 2
    cache_index = cache_offset + tl.arange(0, buffer_stride * 2)

    tmp = tl.load(buffer_ptr + buffer_index[None, :])
    accumulator = tl.zeros((2, buffer_stride), dtype=tl.float32)
    accumulator += tmp
    tl.store(cache_ptr + cache_index, accumulator.reshape(buffer_stride * 2))


cache_shapes = [
    (5, 15),
]


@pytest.mark.parametrize("dtype, sigtype", types)
@pytest.mark.parametrize("batch_size, buffer_len", cache_shapes)
def test_expand_dims_and_add(batch_size, buffer_len, dtype, sigtype):
    block = biggest_divisor(buffer_len)
    cache_len = buffer_len * 2
    buffer = test_common.generate_tensor(
        shape=(batch_size, 1, buffer_len), dtype=sigtype
    ).npu()
    cache = torch.zeros(batch_size, 1, cache_len, dtype=dtype).npu()
    cache_ref = torch.zeros(batch_size, 1, cache_len, dtype=dtype).npu()
    numel = batch_size * buffer_len * 2
    torch_expand_dims_and_add(buffer, cache_ref, buffer_len, block, numel)
    expand_dims_and_add[block, cache_len](
        buffer, cache, buffer_len, block, numel,
        enable_linearize=True
    )

    test_common.validate_cmp(sigtype, cache, cache_ref)


def torch_save_cache_to_buffer(
    buffer,
    cache1,
    cache2,
    buffer_stride,
    cache_stride,
    BLOCK
):
    idx = torch.arange(0, cache_stride)
    mask = ((idx // BLOCK) % 2 == 0)
    max_len = min(buffer_stride, mask.shape[0])
    for i in range(buffer.shape[0]):
        if i % 2 == 0:
            tmp = cache1[i, 0, mask]
            buffer[i, 0, :max_len] = tmp[:max_len]
        else:
            tmp = cache2[i, 0, mask]
            buffer[i, 0, :max_len] = tmp[:max_len]


def torch_save_cache_to_buffer_with_offset(
    buffer,
    cache1,
    cache2,
    buffer_stride,
    cache_stride,
    BLOCK
):
    idx = torch.arange(0, cache_stride)
    mask = ((idx // BLOCK) % 2 == 0)
    max_len = min(buffer_stride, mask.shape[0])
    for i in range(buffer.shape[0]):
        if i % 2 == 0:
            tmp = cache1[i, 0, mask]
            buffer[i, 0, :max_len] = tmp[:max_len]
        else:
            tmp = cache2[i, 0, ~mask]
            buffer[i, 0, :max_len] = tmp[:max_len]


def torch_rearrange_and_combine_two_buffer(
    buffer1,
    buffer2,
    cache,
    buffer_stride,
    NUM_BLOCK,
    BLOCK,
):
    for i in range(buffer1.shape[0]):
        tmp1 = buffer1[i, 0, :]
        tmp1 = tmp1.reshape(NUM_BLOCK, BLOCK)
        tmp1 = tmp1.permute(1, 0)
        tmp1 = tmp1.reshape(1, -1)

        tmp2 = buffer2[i, 0, :]
        tmp2 = tmp2.reshape(NUM_BLOCK, BLOCK)
        tmp2 = tmp2.permute(1, 0)
        tmp2 = tmp2.reshape(1, -1)

        cache[i, 0, :buffer_stride] = tmp1
        cache[i, 0, buffer_stride:] = tmp2


def torch_save_cache_to_buffer_with_mask(
    buffer,
    cache1,
    cache2,
    mask_int,
    buffer_stride,
    cache_stride,
    BLOCK,
    MASK_NUM
):
    idx = torch.arange(0, cache_stride)
    mask_idx = torch.arange(0, buffer_stride)
    mask = ((idx // BLOCK) % 2 == 0)
    max_len = min(buffer_stride, mask.shape[0])
    for i in range(buffer.shape[0]):
        if i % 2 == 0:
            tmp = cache1[i, 0, :]
            tmp[~(idx < MASK_NUM)] = 0
            tmp = tmp[mask]
            buffer[i, 0, :max_len] = tmp[:max_len]
        else:
            tmp = cache2[i, 0, mask]
            tmp[~((mask_idx < MASK_NUM) & (mask_idx < mask_int[i]))] = 0
            buffer[i, 0, :max_len] = tmp[:max_len]


def torch_rearrange_cache_with_mask(
    cache1,
    cache2,
    half_buffer_num,
    buffer_len,
    NUM_BLOCK,
    BLOCK,
):
    buffer_num = half_buffer_num * 2
    idx1 = torch.arange(0, half_buffer_num)
    idx2 = torch.arange(0, buffer_len)
    # all true
    mask1 = idx1 < buffer_num
    mask2 = idx2 < buffer_len

    for i in range(cache1.shape[0]):
        for j in range(2):
            mask = (mask1[j] & mask2)
            tmp1 = cache1[i, j, :]
            tmp1[~mask] = 0
            tmp1 = tmp1.reshape(NUM_BLOCK, BLOCK)
            tmp1 = tmp1.permute(1, 0)
            tmp1 = tmp1.reshape(1, -1)

            tmp2 = cache1[i, j+half_buffer_num, :]
            tmp2[~mask] = 0
            tmp2 = tmp2.reshape(NUM_BLOCK, BLOCK)
            tmp2 = tmp2.permute(1, 0)
            tmp2 = tmp2.reshape(1, -1)

            cache2[i, j, :] = -tmp1
            cache2[i, j+half_buffer_num, :] = tmp2


@triton.jit
def save_cache_to_buffer(
    buffer_ptr,
    cache_ptr1,
    cache_ptr2,
    buffer_stride: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid_loc = tl.program_id(0)
    buffer_offset = pid_loc * buffer_stride
    buffer_index = buffer_offset + tl.arange(0, buffer_stride)

    cache_offset = pid_loc * buffer_stride
    cache_index = cache_offset + tl.arange(0, buffer_stride)
    cache_index_0 = cache_index // BLOCK
    cache_index_1 = cache_index % BLOCK

    if pid_loc % 2 == 0:
        tmp = tl.load(cache_ptr1 + (2*BLOCK*cache_index_0 + cache_index_1))
        tl.store(buffer_ptr + buffer_index, tmp)
    if pid_loc % 2 == 1:
        tmp = tl.load(cache_ptr2 + (2*BLOCK*cache_index_0 + cache_index_1))
        tl.store(buffer_ptr + buffer_index, tmp)


@triton.jit
def save_cache_to_buffer_with_offset(
    buffer_ptr,
    cache_ptr1,
    cache_ptr2,
    buffer_stride: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid_loc = tl.program_id(0)
    buffer_offset = pid_loc * buffer_stride
    buffer_index = buffer_offset + tl.arange(0, buffer_stride)

    cache_offset = pid_loc * buffer_stride
    cache_index = cache_offset + tl.arange(0, buffer_stride)
    cache_index_0 = cache_index // BLOCK
    cache_index_1 = cache_index % BLOCK

    if pid_loc % 2 == 0:
        tmp = tl.load(cache_ptr1 + (2*BLOCK*cache_index_0 + cache_index_1))
        tl.store(buffer_ptr + buffer_index, tmp)
    if pid_loc % 2 == 1:
        tmp = tl.load(cache_ptr2 + BLOCK + (2*BLOCK*cache_index_0 + cache_index_1))
        tl.store(buffer_ptr + buffer_index, tmp)   


@triton.jit
def rearrange_and_combine_two_buffer(
    buffer_ptr1,
    buffer_ptr2,
    cache_ptr,
    buffer_stride: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    buffer_offset = pid_loc * buffer_stride
    buffer_index = buffer_offset + tl.arange(0, buffer_stride)

    tmp1 = tl.load(buffer_ptr1 + buffer_index)
    tmp2 = tl.load(buffer_ptr2 + buffer_index)

    x_offset = pid_loc * 2 * BLOCK * buffer_stride
    x_index = x_offset + tl.arange(0, buffer_stride)

    for i in range(2):
        n_index = (x_index // BLOCK)
        b_index = (x_index % BLOCK)

        if i % 2 == 0:
            tl.store(cache_ptr + NUM_BLOCK * b_index + n_index, tmp1)
        if i % 2 == 1:
            tl.store(cache_ptr + NUM_BLOCK * b_index + n_index, tmp2)

        x_index = x_index + buffer_stride * BLOCK



@triton.jit
def save_cache_to_buffer_with_mask(
    buffer_ptr,
    cache_ptr1,
    cache_ptr2,
    mask_int_ptr,
    buffer_stride: tl.constexpr,
    BLOCK: tl.constexpr,
    MASK_NUM: tl.constexpr
):
    pid_loc = tl.program_id(0)

    buffer_offset = pid_loc * buffer_stride

    buffer_index = tl.arange(0, buffer_stride)
    index = buffer_offset + buffer_index
    cache_index_0 = index // BLOCK
    cache_index_1 = index % BLOCK
    mask_int = tl.load(mask_int_ptr + pid_loc)
    if pid_loc % 2 == 0:
        tmp = tl.load(cache_ptr1 + (2*BLOCK*cache_index_0 + cache_index_1), \
            ((2*BLOCK*cache_index_0 + cache_index_1) < buffer_offset * 2 + MASK_NUM))
        tl.store(buffer_ptr + index, tmp)
    if pid_loc % 2 == 1:
        tmp = tl.load(cache_ptr2 + (2*BLOCK*cache_index_0 + cache_index_1), \
            (buffer_index < MASK_NUM) & (buffer_index < mask_int))
        tl.store(buffer_ptr + index, tmp)


@triton.jit
def rearrange_cache_with_mask(
    cache_ptr1,
    cache_ptr2,
    half_buffer_num: tl.constexpr,
    buffer_stride: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    buffer_num = half_buffer_num * 2
    buffer_offset = pid_loc * buffer_stride * buffer_num

    index = tl.arange(0, buffer_stride)
    buffer_index = buffer_offset + index
    mask_index = index[None, :]
    load_index = buffer_index[None, :]

    buffer_index2 = buffer_offset * BLOCK + index
    n_index = (buffer_index2 // BLOCK)
    b_index = (buffer_index2 % BLOCK)
    store_index = NUM_BLOCK * b_index + n_index
    store_index2 = store_index[None, :]

    buffer_num_offset = 0
    for i in range(2):
        buffer_num_index = buffer_num_offset + tl.arange(0, half_buffer_num)
        buffer_num_index2 = buffer_num_index[:, None]
        
        tmp1 = tl.load(cache_ptr1 + buffer_stride * buffer_num_index2 + load_index)
        tmp1 = tmp1 * (2 * i - 1)
        mask1 = buffer_num_index2 < buffer_num
        mask2 = mask_index < buffer_stride
        tl.store(cache_ptr2 + buffer_stride * buffer_num_index2 + store_index2, tmp1, mask1 & mask2)
        # tl.store(cache_ptr2 + buffer_stride * buffer_num_index2 + store_index2, tmp1, True)

        buffer_num_offset = buffer_num_offset + half_buffer_num


@pytest.mark.parametrize('dtype,sigtype', types)
@pytest.mark.parametrize('batch_size,buffer_len', cache_shapes)
def test_linearize_jump_load(batch_size, buffer_len, dtype, sigtype):
    block = biggest_divisor(buffer_len)
    cache_len = buffer_len * 2
    buffer_ref = torch.zeros(batch_size, 1, buffer_len, dtype=dtype)
    buffer = buffer_ref.npu()
    cache1_ref = test_common.generate_tensor(shape=(batch_size, 1, cache_len), dtype=sigtype)
    cache1 = cache1_ref.npu()
    cache2_ref = test_common.generate_tensor(shape=(batch_size, 1, cache_len), dtype=sigtype)
    cache2 = cache2_ref.npu()

    torch_save_cache_to_buffer(buffer_ref, cache1_ref, cache2_ref, buffer_len, cache_len, block)
    save_cache_to_buffer[(batch_size, 1, 1)](
        buffer, cache1, cache2, buffer_len, block,
        enable_linearize=True
    )
    test_common.validate_cmp(sigtype, buffer, buffer_ref)


@pytest.mark.parametrize('dtype,sigtype', types)
@pytest.mark.parametrize('batch_size,buffer_len', cache_shapes)
def test_linearize_jump_load_with_offset(batch_size, buffer_len, dtype, sigtype):
    block = biggest_divisor(buffer_len)
    cache_len = buffer_len * 2
    buffer_ref = torch.zeros(batch_size, 1, buffer_len, dtype=dtype)
    buffer = buffer_ref.npu()
    cache1_ref = test_common.generate_tensor(shape=(batch_size, 1, cache_len), dtype=sigtype)
    cache1 = cache1_ref.npu()
    cache2_ref = test_common.generate_tensor(shape=(batch_size, 1, cache_len), dtype=sigtype)
    cache2 = cache2_ref.npu()

    torch_save_cache_to_buffer_with_offset(buffer_ref, cache1_ref, cache2_ref, buffer_len, cache_len, block)
    save_cache_to_buffer_with_offset[(batch_size, 1, 1)](
        buffer, cache1, cache2, buffer_len, block,
        enable_linearize=True
    )
    test_common.validate_cmp(sigtype, buffer, buffer_ref)


@pytest.mark.parametrize('dtype,sigtype', types)
@pytest.mark.parametrize('batch_size,buffer_len', cache_shapes)
def test_linearize_rearrange(batch_size, buffer_len, dtype, sigtype):
    block = biggest_divisor(buffer_len)
    num_block = int(buffer_len / block)
    cache_len = buffer_len * 2
    buffer1_ref = test_common.generate_tensor(shape=(batch_size, 1, buffer_len), dtype=sigtype)
    buffer1 = buffer1_ref.npu()
    buffer2_ref = test_common.generate_tensor(shape=(batch_size, 1, buffer_len), dtype=sigtype)
    buffer2 = buffer2_ref.npu()
    cache_ref = torch.zeros(batch_size, 1, cache_len, dtype=dtype)
    cache = cache_ref.npu()

    torch_rearrange_and_combine_two_buffer(buffer1_ref, buffer2_ref, cache_ref, buffer_len, num_block, block)
    rearrange_and_combine_two_buffer[(batch_size, 1, 1)](
        buffer1, buffer2, cache, buffer_len, num_block, block,
        enable_linearize=True
    )
    test_common.validate_cmp(sigtype, cache, cache_ref)


@pytest.mark.skip(reason="mask load still has issues to be fixed by bisheng")
@pytest.mark.parametrize('dtype,sigtype', types)
@pytest.mark.parametrize('batch_size,buffer_len', cache_shapes)
def test_linearize_jump_load_with_mask(batch_size, buffer_len, dtype, sigtype):
    block = biggest_divisor(buffer_len)
    cache_len = buffer_len * 2
    buffer_ref = torch.zeros(batch_size, 1, buffer_len, dtype=dtype)
    buffer = buffer_ref.npu()
    cache1_ref = test_common.generate_tensor(shape=(batch_size, 1, cache_len), dtype=sigtype)
    cache1 = cache1_ref.npu()
    cache2_ref = test_common.generate_tensor(shape=(batch_size, 1, cache_len), dtype=sigtype)
    cache2 = cache2_ref.npu()
    mask_ref = torch.arange(0, batch_size, dtype=torch.int64)*2
    mask = mask_ref.npu()
    mask_num = 16
    torch_save_cache_to_buffer_with_mask(buffer_ref, cache1_ref, cache2_ref, mask_ref, buffer_len, cache_len, block, mask_num)
    save_cache_to_buffer_with_mask[(batch_size, 1, 1)](
        buffer, cache1, cache2, mask, buffer_len, block, mask_num,
        enable_linearize=True
    )
    test_common.validate_cmp(sigtype, buffer, buffer_ref)


@pytest.mark.skip(reason="mask still has issues to be fixed")
@pytest.mark.parametrize('dtype,sigtype', types)
@pytest.mark.parametrize('batch_size,buffer_len', cache_shapes)
def test_linearize_rearrange_with_mask(batch_size, buffer_len, dtype, sigtype):
    block = biggest_divisor(buffer_len)
    num_block = int(buffer_len/block)
    cache1_ref = test_common.generate_tensor(shape=(batch_size, 4, buffer_len), dtype=sigtype)
    cache1 = cache1_ref.npu()
    cache2_ref = torch.zeros(batch_size, 4, buffer_len, dtype=dtype)
    cache2 = cache2_ref.npu()

    torch_rearrange_cache_with_mask(cache1_ref, cache2_ref, 2, buffer_len, num_block, block)
    rearrange_cache_with_mask[(batch_size, 1, 1)](
        cache1, cache2, 2, buffer_len, num_block, block, 
        enable_linearize=True
    )
    test_common.validate_cmp(sigtype, cache2, cache2_ref)