"""Triton extern ABI for the unified two-warp D6-D9 dispatch tail."""

import triton.language as tl
import triton.language.core as core


@core.extern
def TleD8UnifiedDispatchPull(
    rsum_addr,
    recv_addr,
    queue_addr,
    metadata_addr,
    sf_table_addr,
    weight_table_addr,
    arrival_addr,
    l1_token_addr,
    l1_sf_addr,
    l1_weight_addr,
    stage0_addr,
    state0_addr,
    stage1_addr,
    state1_addr,
    token0,
    token1,
    token2,
    token3,
    token4,
    token5,
    token6,
    token7,
    num_ranks,
    num_sms,
    experts_per_rank,
    max_recv,
    block_m,
    hidden,
    num_sf,
    pool_tokens,
    topk,
    _semantic=None,
):
    i32 = core.dtype("int32")
    i64 = core.dtype("int64")
    addresses = [
        rsum_addr,
        recv_addr,
        queue_addr,
        metadata_addr,
        sf_table_addr,
        weight_table_addr,
        arrival_addr,
        l1_token_addr,
        l1_sf_addr,
        l1_weight_addr,
        stage0_addr,
        state0_addr,
        stage1_addr,
        state1_addr,
        token0,
        token1,
        token2,
        token3,
        token4,
        token5,
        token6,
        token7,
    ]
    constants = [
        num_ranks,
        num_sms,
        experts_per_rank,
        max_recv,
        block_m,
        hidden,
        num_sf,
        pool_tokens,
        topk,
    ]
    args = [tl.cast(x, tl.int64, _semantic=_semantic) for x in addresses]
    args += [tl.cast(x, tl.int32, _semantic=_semantic) for x in constants]
    signature = tuple([i64] * len(addresses) + [i32] * len(constants))
    return core.extern_call(
        "",
        "",
        args,
        {signature: ("TleD8UnifiedDispatchPull", ())},
        is_pure=False,
        _semantic=_semantic,
    )
