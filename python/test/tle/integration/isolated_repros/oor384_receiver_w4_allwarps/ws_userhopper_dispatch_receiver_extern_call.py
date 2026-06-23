import triton.language as tl
import triton.language.core as core


@core.extern
def userhopper_ws_dispatch_partition(symm_buffer, num_tokens, num_ranks, num_experts, num_max_tokens_per_rank,
                                     num_topk, hidden, _semantic=None):
    return core.extern_call(
        "",
        "",
        [
            symm_buffer,
            tl.cast(num_tokens, tl.int32, _semantic=_semantic),
            tl.cast(num_ranks, tl.int32, _semantic=_semantic),
            tl.cast(num_experts, tl.int32, _semantic=_semantic),
            tl.cast(num_max_tokens_per_rank, tl.int32, _semantic=_semantic),
            tl.cast(num_topk, tl.int32, _semantic=_semantic),
            tl.cast(hidden, tl.int32, _semantic=_semantic),
        ],
        {
            (
                core.pointer_type(core.dtype("uint8")),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
            ): ("userhopper_ws_dispatch_partition", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def userhopper_ws_dispatch_partition_cta_warp0(symm_buffer, num_tokens, num_ranks, num_experts,
                                               num_max_tokens_per_rank, num_topk, hidden, _semantic=None):
    return core.extern_call(
        "",
        "",
        [
            symm_buffer,
            tl.cast(num_tokens, tl.int32, _semantic=_semantic),
            tl.cast(num_ranks, tl.int32, _semantic=_semantic),
            tl.cast(num_experts, tl.int32, _semantic=_semantic),
            tl.cast(num_max_tokens_per_rank, tl.int32, _semantic=_semantic),
            tl.cast(num_topk, tl.int32, _semantic=_semantic),
            tl.cast(hidden, tl.int32, _semantic=_semantic),
        ],
        {
            (
                core.pointer_type(core.dtype("uint8")),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
            ): ("userhopper_ws_dispatch_partition_cta_warp0", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def userhopper_ws_dispatch_partition_cta_multiwarp(symm_buffer, num_tokens, num_ranks, num_experts,
                                                   num_max_tokens_per_rank, num_topk, hidden,
                                                   num_dispatch_warps, _semantic=None):
    return core.extern_call(
        "",
        "",
        [
            symm_buffer,
            tl.cast(num_tokens, tl.int32, _semantic=_semantic),
            tl.cast(num_ranks, tl.int32, _semantic=_semantic),
            tl.cast(num_experts, tl.int32, _semantic=_semantic),
            tl.cast(num_max_tokens_per_rank, tl.int32, _semantic=_semantic),
            tl.cast(num_topk, tl.int32, _semantic=_semantic),
            tl.cast(hidden, tl.int32, _semantic=_semantic),
            tl.cast(num_dispatch_warps, tl.int32, _semantic=_semantic),
        ],
        {
            (
                core.pointer_type(core.dtype("uint8")),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
            ): ("userhopper_ws_dispatch_partition_cta_multiwarp", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def userhopper_ws_receiver_partition(symm_buffer, expected_local_recv_tokens, num_ranks, num_experts,
                                     num_max_tokens_per_rank, num_topk, hidden, num_padded_sf_pool_tokens,
                                     _semantic=None):
    return core.extern_call(
        "",
        "",
        [
            symm_buffer,
            tl.cast(expected_local_recv_tokens, tl.int32, _semantic=_semantic),
            tl.cast(num_ranks, tl.int32, _semantic=_semantic),
            tl.cast(num_experts, tl.int32, _semantic=_semantic),
            tl.cast(num_max_tokens_per_rank, tl.int32, _semantic=_semantic),
            tl.cast(num_topk, tl.int32, _semantic=_semantic),
            tl.cast(hidden, tl.int32, _semantic=_semantic),
            tl.cast(num_padded_sf_pool_tokens, tl.int32, _semantic=_semantic),
        ],
        {
            (
                core.pointer_type(core.dtype("uint8")),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
            ): ("userhopper_ws_receiver_partition", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def userhopper_ws_receiver_partition_bounded(symm_buffer, expected_local_recv_tokens, num_ranks, num_experts,
                                             num_max_tokens_per_rank, num_topk, hidden,
                                             num_padded_sf_pool_tokens, _semantic=None):
    return core.extern_call(
        "",
        "",
        [
            symm_buffer,
            tl.cast(expected_local_recv_tokens, tl.int32, _semantic=_semantic),
            tl.cast(num_ranks, tl.int32, _semantic=_semantic),
            tl.cast(num_experts, tl.int32, _semantic=_semantic),
            tl.cast(num_max_tokens_per_rank, tl.int32, _semantic=_semantic),
            tl.cast(num_topk, tl.int32, _semantic=_semantic),
            tl.cast(hidden, tl.int32, _semantic=_semantic),
            tl.cast(num_padded_sf_pool_tokens, tl.int32, _semantic=_semantic),
        ],
        {
            (
                core.pointer_type(core.dtype("uint8")),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
            ): ("userhopper_ws_receiver_partition_bounded", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def userhopper_ws_compute_stub_partition(symm_buffer, l1_weights, l1_weights_sf, l2_weights, l2_weights_sf,
                                         num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
                                         hidden, intermediate_hidden, num_padded_sf_pool_tokens,
                                         compute_full_hidden,
                                         compute_parallel,
                                         compute_worker_warps,
                                         _semantic=None):
    return core.extern_call(
        "",
        "",
        [
            symm_buffer,
            l1_weights,
            l1_weights_sf,
            l2_weights,
            l2_weights_sf,
            tl.cast(num_ranks, tl.int32, _semantic=_semantic),
            tl.cast(num_experts, tl.int32, _semantic=_semantic),
            tl.cast(num_max_tokens_per_rank, tl.int32, _semantic=_semantic),
            tl.cast(num_topk, tl.int32, _semantic=_semantic),
            tl.cast(hidden, tl.int32, _semantic=_semantic),
            tl.cast(intermediate_hidden, tl.int32, _semantic=_semantic),
            tl.cast(num_padded_sf_pool_tokens, tl.int32, _semantic=_semantic),
            tl.cast(compute_full_hidden, tl.int32, _semantic=_semantic),
            tl.cast(compute_parallel, tl.int32, _semantic=_semantic),
            tl.cast(compute_worker_warps, tl.int32, _semantic=_semantic),
        ],
        {
            (
                core.pointer_type(core.dtype("uint8")),
                core.pointer_type(core.dtype("uint8")),
                core.pointer_type(core.dtype("fp32")),
                core.pointer_type(core.dtype("uint8")),
                core.pointer_type(core.dtype("fp32")),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
            ): ("userhopper_ws_compute_stub_partition", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def userhopper_ws_combine_reduce_partition(symm_buffer, y, num_ranks, num_experts, num_max_tokens_per_rank,
                                           num_topk, hidden, intermediate_hidden, num_padded_sf_pool_tokens,
                                           cleanup_workspace, _semantic=None):
    return core.extern_call(
        "",
        "",
        [
            symm_buffer,
            y,
            tl.cast(num_ranks, tl.int32, _semantic=_semantic),
            tl.cast(num_experts, tl.int32, _semantic=_semantic),
            tl.cast(num_max_tokens_per_rank, tl.int32, _semantic=_semantic),
            tl.cast(num_topk, tl.int32, _semantic=_semantic),
            tl.cast(hidden, tl.int32, _semantic=_semantic),
            tl.cast(intermediate_hidden, tl.int32, _semantic=_semantic),
            tl.cast(num_padded_sf_pool_tokens, tl.int32, _semantic=_semantic),
            tl.cast(cleanup_workspace, tl.int32, _semantic=_semantic),
        ],
        {
            (
                core.pointer_type(core.dtype("uint8")),
                core.pointer_type(core.dtype("uint8")),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
            ): ("userhopper_ws_combine_reduce_partition", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def userhopper_ws_tldot_combine_write_partition(symm_buffer, l2_out, num_ranks, num_experts,
                                                num_max_tokens_per_rank, num_topk, hidden,
                                                intermediate_hidden, num_padded_sf_pool_tokens,
                                                _semantic=None):
    return core.extern_call(
        "",
        "",
        [
            symm_buffer,
            l2_out,
            tl.cast(num_ranks, tl.int32, _semantic=_semantic),
            tl.cast(num_experts, tl.int32, _semantic=_semantic),
            tl.cast(num_max_tokens_per_rank, tl.int32, _semantic=_semantic),
            tl.cast(num_topk, tl.int32, _semantic=_semantic),
            tl.cast(hidden, tl.int32, _semantic=_semantic),
            tl.cast(intermediate_hidden, tl.int32, _semantic=_semantic),
            tl.cast(num_padded_sf_pool_tokens, tl.int32, _semantic=_semantic),
        ],
        {
            (
                core.pointer_type(core.dtype("uint8")),
                core.pointer_type(core.dtype("fp32")),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
                core.dtype("int32"),
            ): ("userhopper_ws_tldot_combine_write_partition", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )
