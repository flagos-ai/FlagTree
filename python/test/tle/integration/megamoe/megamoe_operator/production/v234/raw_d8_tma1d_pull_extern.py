"""Triton extern ABI for the nvcc-compiled live D8 TMA1D helper."""

import triton.language as tl
import triton.language.core as core


@core.extern
def TleD8Tma1d(
    token_stage,
    token_state,
    global_token,
    num_bytes,
    active,
    op,
    _semantic=None,
):
    i32 = core.dtype("int32")
    i64 = core.dtype("int64")
    return core.extern_call(
        "",
        "",
        [
            token_stage,
            token_state,
            global_token,
            tl.cast(num_bytes, tl.int32, _semantic=_semantic),
            tl.cast(active, tl.int32, _semantic=_semantic),
            tl.cast(op, tl.int32, _semantic=_semantic),
        ],
        {
            (
                i64,
                i64,
                i64,
                i32,
                i32,
                i32,
            ): ("TleD8Tma1d", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )
