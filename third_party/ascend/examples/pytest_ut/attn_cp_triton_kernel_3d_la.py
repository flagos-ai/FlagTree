# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import triton
import triton.language as tl
import torch
import torch_npu

#number of block per aiv core
NBLOCKS = 32
# size of sub_block in one block
# reduce the size from 64 to 32 due to UB overflow
S_SUB_SIZE = 32

@triton.jit
def tl_fn_forward_update_la( prev_attn_out, prev_softmax_log_max_sum, cur_attn_out, cur_softmax_log_max_sum,
        B:tl.constexpr,N:tl.constexpr,S:tl.constexpr,D:tl.constexpr, PREV_ATTN_NSTRIDE:tl.constexpr, PREV_SOFTMAX_NSTRIDE:tl.constexpr, 
        CUR_ATTN_NSTRIDE:tl.constexpr, CUR_SOFTMAX_NSTRIDE:tl.constexpr, S_NBLOCKS:tl.constexpr, S_SUB:tl.constexpr):

    S_BLOCK:tl.constexpr  = (S + S_NBLOCKS - 1) // S_NBLOCKS    
    S_NSUB:tl.constexpr = (S_BLOCK + S_SUB - 1) // S_SUB
    LOOP_COUNT : tl.constexpr  = (S_BLOCK * B * N  + S_SUB - 1) // S_SUB    
    block_idx = tl.program_id(0)

    s_block_start = block_idx * S_BLOCK
    # assuming S stride is D, if D is not contiguous, need to use 2-d offset
    SIMD_SIZE:tl.constexpr = S_SUB * D

    for loop_index in range(LOOP_COUNT):
        b = loop_index // (N * S_NSUB)
        n = (loop_index // S_NSUB) % N
        s_loop_start = (loop_index % S_NSUB) * S_SUB
        s = s_block_start + s_loop_start

        sd_offsets = D * s + tl.arange(0, SIMD_SIZE)
        s1_offsets = s + tl.arange(0, S_SUB)
        
        mask0 = None  
        softmax_offsets  =  PREV_SOFTMAX_NSTRIDE * (b * N + n) + s1_offsets
        prev_softmax_local = tl.load(prev_softmax_log_max_sum + softmax_offsets, mask0)
        offsets  =  CUR_SOFTMAX_NSTRIDE * (b * N + n) + s1_offsets
        cur_softmax_local = tl.load(cur_softmax_log_max_sum + offsets, mask0)

        attn_offsets  =  PREV_ATTN_NSTRIDE * (b * N + n) + sd_offsets
        prev_attn_local = tl.load(prev_attn_out + attn_offsets, mask0)
        offsets  =  CUR_ATTN_NSTRIDE * (b * N + n) + sd_offsets
        cur_attn_local = tl.load(cur_attn_out + offsets, mask0)

        tmp0 = tl.exp(cur_softmax_local)      
        tmp1 = tl.exp(prev_softmax_local)
        softmax_log_max_sum = tl.log(tmp0 + tmp1)
        tmp2 = (prev_softmax_local - softmax_log_max_sum).reshape(S_SUB,1).broadcast_to(S_SUB, D) 
        tmp3 = (cur_softmax_local- softmax_log_max_sum).reshape(S_SUB,1).broadcast_to(S_SUB,D)
 
        attn_out = tl.exp(tmp2) * prev_attn_local.reshape(S_SUB,D) + (tl.exp(tmp3) * cur_attn_local.reshape(S_SUB,D))
        mask1 = None
        tl.store(prev_softmax_log_max_sum + softmax_offsets, softmax_log_max_sum, mask1)
        tl.store(prev_attn_out + attn_offsets, attn_out.reshape(SIMD_SIZE,), mask1)

#target_name = torch.npu.get_device_name(torch.npu.current_device())
guards = {"dummy":None}
def forward_update_triton(prev_attn_out, prev_softmax_log_max_sum, cur_attn_out, cur_softmax_log_max_sum):
    (B,N,S,D) = cur_attn_out.shape
    #shape is (b,n,s,d)
    PREV_ATTN_NSTRIDE = prev_attn_out.stride()[1]
    PREV_SOFTMAX_NSTRIDE = prev_softmax_log_max_sum.stride()[1]
    CUR_ATTN_NSTRIDE = cur_attn_out.stride()[1]
    CUR_SOFTMAX_NSTRIDE = cur_softmax_log_max_sum.stride()[1]
    
    
    
    device_id = cur_attn_out.device.index
    device = "npu:" +str(device_id)
    
    tl_fn_forward_update_la[NBLOCKS,1,1](prev_attn_out, prev_softmax_log_max_sum, cur_attn_out, cur_softmax_log_max_sum, B=B, N=N, S=S, D=D,
            PREV_ATTN_NSTRIDE=PREV_ATTN_NSTRIDE, PREV_SOFTMAX_NSTRIDE=PREV_SOFTMAX_NSTRIDE, CUR_ATTN_NSTRIDE=CUR_ATTN_NSTRIDE, 
            CUR_SOFTMAX_NSTRIDE=CUR_SOFTMAX_NSTRIDE, S_NBLOCKS=NBLOCKS, S_SUB=S_SUB_SIZE, debug=True)
    
