# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl
import torch
import torch_npu

#number of block per aiv core
NBLOCKS = 32
#constants
TL_DTYPE_ATTN = tl.bfloat16

@triton.jit
def tl_fn_forward_update( output_ptr0, output_ptr1, output_ptr2,  prev_attn_out, prev_softmax_max, prev_softmax_sum,  cur_attn_out, cur_softmax_max,
                          cur_softmax_sum, S:tl.constexpr,B:tl.constexpr,H:tl.constexpr,N:tl.constexpr, E:tl.constexpr,                          
                          CUR_SM_N_STRIDE:tl.constexpr, PREV_SM_N_STRIDE:tl.constexpr,
                          S_NBLOCKS:tl.constexpr, S_SUB:tl.constexpr, BRC_SIZE:tl.constexpr):

    S_E_SUB_SIZE:tl.constexpr = S_SUB * E
    D:tl.constexpr = H // N
    S_BLOCK:tl.constexpr  = (S + S_NBLOCKS - 1) // S_NBLOCKS
    S_NSUB:tl.constexpr = (S_BLOCK + S_SUB - 1) // S_SUB

    D_SUB:tl.constexpr = E * BRC_SIZE
    D_NLOOP:tl.constexpr = (D + D_SUB - 1) // D_SUB

    LOOP_COUNT : tl.constexpr  = (S_BLOCK * B * N + S_SUB - 1) // S_SUB
    block_idx = tl.program_id(0)
    s_block_offset = block_idx % S_NBLOCKS * S_BLOCK

    for loop_index in range(LOOP_COUNT):
        b = loop_index // (N * S_NSUB)
        n = (loop_index // S_NSUB) % N
        s_loop_offset = loop_index % S_NSUB * S_SUB
        s = s_block_offset + s_loop_offset + tl.arange(0, S_SUB) #index on S axis
        se_offset = E * (s_block_offset + s_loop_offset) + tl.arange(0, S_E_SUB_SIZE)
        #assume no slice on N axis
        offsets_prev = PREV_SM_N_STRIDE * (b * N + n) + se_offset
        offsets_cur = CUR_SM_N_STRIDE * (b * N + n) + se_offset
        mask0 = None
        prev_softmax_max_local = tl.load(prev_softmax_max + offsets_prev, mask0)
        cur_softmax_max_local = tl.load(cur_softmax_max + offsets_cur, mask0)
        softmax_max = tl.maximum(prev_softmax_max_local, cur_softmax_max_local)
        prev_scale = tl.exp(prev_softmax_max_local - softmax_max)
        cur_scale = tl.exp(cur_softmax_max_local - softmax_max)
        prev_softmax_sum_local = tl.load(prev_softmax_sum + offsets_prev, mask0)
        cur_softmax_sum_local = tl.load(cur_softmax_sum + offsets_cur, mask0)
        prev_softmax_sum_scaled = prev_softmax_sum_local * prev_scale
        cur_softmax_sum_scaled = cur_softmax_sum_local * cur_scale
        softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled
        prev_out_scale_local = prev_softmax_sum_scaled / softmax_sum
        cur_out_scale_local = cur_softmax_sum_scaled / softmax_sum
        prev_out_scale_out = prev_out_scale_local
        cur_out_scale_out = cur_out_scale_local
        mask1 = None
        tl.store(output_ptr1 + offsets_cur, softmax_max, mask0)
        tl.store(output_ptr2 + offsets_cur, softmax_sum, mask0)
        prev_out_scale = prev_out_scale_local
        prev_out_scale = prev_out_scale.reshape(S_SUB,1,E)#(s_sub,8)->(1,1,s_sub,1,8)
        cur_out_scale = cur_out_scale_local
        cur_out_scale = cur_out_scale.reshape(S_SUB,1,E)
        for d_index in range(D_NLOOP):
            # (s,b,h) -> (s,b,n*d) -> (s,b,n,d)
            d = d_index * D_SUB + tl.arange(0, D_SUB)
            offsets2 = s[:,None] * B * H + b * H + n * D + d[None,:]
            mask2 = None
            prev_attn_out_local = tl.load(prev_attn_out + offsets2, mask2)
            cur_attn_out_local = tl.load(cur_attn_out + offsets2, mask2)
            prev_attn_out_local = prev_attn_out_local.to(tl.float32)
            cur_attn_out_local = cur_attn_out_local.to(tl.float32)
            prev_attn_out_local = prev_attn_out_local.reshape(S_SUB,BRC_SIZE,E)
            cur_attn_out_local = cur_attn_out_local.reshape(S_SUB,BRC_SIZE,E)
            prev_out_scale_brc = prev_out_scale
            prev_out_scale_brc = prev_out_scale_brc.broadcast_to(S_SUB,BRC_SIZE,E)
            cur_out_scale_brc = cur_out_scale
            cur_out_scale_brc = cur_out_scale_brc.broadcast_to(S_SUB,BRC_SIZE,E)
            attn_out_local = prev_attn_out_local * prev_out_scale_brc + cur_attn_out_local * cur_out_scale_brc
            attn_out_local = attn_out_local.reshape(S_SUB, D_SUB)
            attn_out_local = attn_out_local.to(TL_DTYPE_ATTN)
            tl.store(output_ptr0 + offsets2, attn_out_local, mask2)


#target_name = torch.npu.get_device_name(torch.npu.current_device())
guards = {"dummy":None}
def forward_update_triton(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out, cur_softmax_max, 
    cur_softmax_sum):
         
    # size of sub_block in one block
    S_SUB_SIZE = 64
    BROARDCAST_SIZE = 8 
        
    (S,B,H) = cur_attn_out.shape
    N = prev_softmax_max.shape[1]
    E = prev_softmax_max.shape[3]
    D = S//H
    D_SUB = E * BROARDCAST_SIZE
    #(b,n,s,8)
    PREV_SM_N_STRIDE = prev_softmax_max.stride()[1]
    CUR_SM_N_STRIDE = cur_softmax_max.stride()[1]

    GUARD = (S % NBLOCKS == 0 and H % N == 0  and (H // N) % (D_SUB) == 0)

    if (not GUARD ) :
        print(f"parameter does not meet compiling GUARD , fallback to eager foward_update \
            (S,H,N,D,D_SUB,NBLOCKS):{S},{H},{N},{D},{D_SUB},{NBLOCKS}")


    
    
    org_dtype = cur_attn_out.dtype
    device_id = cur_attn_out.device.index
    device = "npu:" +str(device_id)

    softmax_max = torch.empty_strided(cur_softmax_max.shape, cur_softmax_max.stride(),
            dtype=cur_softmax_max.dtype, device = device)
    softmax_sum = torch.empty_strided(cur_softmax_sum.shape, cur_softmax_sum.stride(),
            dtype=cur_softmax_max.dtype, device = device)
    attn_out = torch.empty_strided(cur_attn_out.shape, cur_attn_out.stride(), dtype=cur_attn_out.dtype, device = device)

    tl_fn_forward_update[NBLOCKS,1,1](attn_out, softmax_max, softmax_sum, prev_attn_out, prev_softmax_max,
            prev_softmax_sum, cur_attn_out, cur_softmax_max, cur_softmax_sum, S=S, B=B, H=H, N=N, E=E,
            CUR_SM_N_STRIDE=CUR_SM_N_STRIDE, PREV_SM_N_STRIDE=PREV_SM_N_STRIDE, S_NBLOCKS=NBLOCKS, S_SUB=S_SUB_SIZE, BRC_SIZE=BROARDCAST_SIZE, debug=True)
    
    return attn_out, softmax_max, softmax_sum
	

@triton.jit
def tl_fn_backward_update( dq, dk, dv, cur_dq, cur_dk, cur_dv, qnumel:tl.constexpr, knumel:tl.constexpr,
     XBLOCK:tl.constexpr, SIMD_SIZE:tl.constexpr):

    block_idx = tl.program_id(0)
    block_offset = block_idx * XBLOCK
    LOOP_COUNT : tl.constexpr  = (XBLOCK + SIMD_SIZE - 1) // SIMD_SIZE
    for loop_index in range(LOOP_COUNT):
        loop_offset = block_offset + loop_index * SIMD_SIZE + tl.arange(0, SIMD_SIZE)
        mask0 = loop_offset < qnumel
        tmp0 = tl.load(dq + loop_offset, mask0).to(tl.float32)
        tmp1 = tl.load(cur_dq + loop_offset, mask0).to(tl.float32)
        tmp0 = tmp1 + tmp0
        tl.store(dq + loop_offset, tmp0.to(tl.bfloat16), mask0)

        mask1 = loop_offset < knumel
        tmp2 = tl.load(dk + loop_offset, mask1).to(tl.float32)
        tmp3 = tl.load(cur_dk + loop_offset, mask1).to(tl.float32)
        tmp2 = tmp2 + tmp3

        tmp4 = tl.load(dv + loop_offset, mask1).to(tl.float32)
        tmp5 = tl.load(cur_dv + loop_offset, mask1).to(tl.float32)
        tmp4 = tmp4 + tmp5

        tl.store(dk + loop_offset, tmp2.to(tl.bfloat16), mask1)
        tl.store(dv + loop_offset, tmp4.to(tl.bfloat16), mask1)


def backward_update_triton( dq, dk, dv, cur_dq, cur_dk, cur_dv ) :
    # parameters need auto-tune.   
    SIMD_SIZE = 4*1024

    xblock = max(dq.numel(), dk.numel()) // NBLOCKS

    tl_fn_backward_update[40,1,1](dq, dk, dv, cur_dq, cur_dk, cur_dv, dq.numel(), dk.numel(), xblock, SIMD_SIZE, debug=True)

