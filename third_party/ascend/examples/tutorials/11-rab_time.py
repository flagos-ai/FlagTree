"""
Relative Attention Bias Timestamps
===============
"""

import math
import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver

NUM_BUCKETS = 128
BUCKET_DIVISOR = 0.301


# get device properties of npu
def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


def create_pos_w(train_len: int, num_layers: int) -> torch.Tensor:
    return torch.arange(0, 2 * train_len + 1).unsqueeze(1).repeat(1, num_layers)


def create_past_valid_lens(bs: int, past_len: int) -> torch.Tensor:
    return torch.randint(0, past_len, (bs,))


def create_timestamps(
    train_len: int, candidate_len: int, past_valid_lens: torch.Tensor
) -> torch.Tensor:
    bs = past_valid_lens.size(0)
    timestamps = torch.zeros(bs, train_len + candidate_len // 2)
    for i, valid_len in enumerate(past_valid_lens):
        if valid_len > 0:
            timestamps[i, :valid_len] = torch.arange(1, valid_len.int() + 1)

    if candidate_len <= 0:
        return timestamps
    timestamps[:, -candidate_len // 2:] = train_len + 1

    return timestamps


def create_timestamps_weights(num_layers: int):
    return (
        torch.arange(0, NUM_BUCKETS + 1)
        .repeat(num_layers)
        .reshape(NUM_BUCKETS + 1, num_layers)
    )


def create_rab_time_grad(num_layers: int, batchsize: int, s: int):
    return torch.rand(num_layers, batchsize, s, s) * 1e-4


def create_bucket_timestamps(batchsize: int, s: int):
    result = torch.arange(batchsize * s) % NUM_BUCKETS
    result = result.unsqueeze(-1).repeat(1, 1, s)
    return result


@triton.jit
def rab_time_forward_kernel(
    inp,
    out,
    index,
    index_len: tl.constexpr,
    inp_row_stride: tl.constexpr,
    clamp_max: tl.constexpr,
    bucketization_divisor: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    col_iter_num = tl.cdiv(BLOCK_SIZE, COL_BLOCK_SIZE)

    for col_idx in tl.range(0, col_iter_num):
        cols_offsets = (
            pid0 * BLOCK_SIZE + col_idx * COL_BLOCK_SIZE + tl.arange(0, COL_BLOCK_SIZE)
        )
        cols_mask = cols_offsets < index_len

        out_mask = cols_offsets < index_len

        index_val = tl.load(index + cols_offsets, mask=cols_mask, other=0.0)
        index_val = tl.abs(index_val)
        index_val = tl.minimum(tl.maximum(index_val, 1.0), clamp_max)
        index_val = tl.log(index_val)
        index_val = index_val / bucketization_divisor
        index_val = tl.cast(index_val, tl.int64)

        inp_val = tl.load(inp + pid1 * inp_row_stride + tl.arange(0, inp_row_stride))
        out_val = tl.gather(inp_val, index_val, 0)

        tl.store(out + pid1 * index_len + cols_offsets, out_val, mask=out_mask)


def get_outer_loop_num(num_layers, index_len):
    sub_num_layers = num_layers
    while sub_num_layers * index_len >= 2**31 - 1:
        sub_num_layers = sub_num_layers // 2
    outer_loop_num = (num_layers + sub_num_layers - 1) // sub_num_layers
    remain_layers = num_layers % sub_num_layers
    return outer_loop_num, sub_num_layers, remain_layers


def rab_time_forward_triton(ts_w, timestamps, bucketization_divisor):
    ts_w_trans = ts_w.t().contiguous()

    bs, seq_len = timestamps.shape
    infer_len = 2 * seq_len
    num_layers = ts_w.shape[1]
    num_buckets = ts_w.shape[0] - 1

    timestamps_expanded = timestamps.unsqueeze(-1).repeat(1, 1, 2)
    timestamps_expanded = timestamps_expanded.reshape(
        bs, infer_len, 1
    ) - timestamps_expanded.reshape(bs, 1, infer_len)

    timestamps_expanded = timestamps_expanded.view(-1)
    timestamps_expanded = timestamps_expanded.contiguous()

    clamp_max = torch.exp(torch.tensor(num_buckets * bucketization_divisor)).item()
    index_len = bs * infer_len * infer_len

    out = torch.empty((num_layers, index_len), dtype=ts_w.dtype, device=ts_w.device)
    outer_loop_num, sub_num_layers, remain_layers = get_outer_loop_num(
        num_layers, index_len
    )

    CORE_NUM = get_npu_properties()["num_vectorcore"]
    BLOCK_SIZE = math.ceil(index_len / CORE_NUM)
    COL_BLOCK_SIZE = 8 * 1024

    curr_layers = sub_num_layers
    for i in range(outer_loop_num):
        if i == outer_loop_num - 1 and remain_layers != 0:
            curr_layers = remain_layers
        grid = lambda meta: (triton.cdiv(index_len, meta["BLOCK_SIZE"]), curr_layers)

        rab_time_forward_kernel[grid](
            ts_w_trans[i * sub_num_layers],
            out[i * sub_num_layers],
            timestamps_expanded,
            index_len,
            num_buckets + 1,
            clamp_max,
            bucketization_divisor,
            BLOCK_SIZE,
            COL_BLOCK_SIZE,
        )

    out = out.view(num_layers, bs, infer_len, infer_len)

    return out


@triton.jit
def rab_time_backward_kernel(
    inp, src, index, index_len, BLOCK_SIZE: tl.constexpr, COL_BLOCK_SIZE: tl.constexpr
):
    pid0 = tl.program_id(axis=0)
    total_col_num = (
        BLOCK_SIZE
        if pid0 * BLOCK_SIZE + BLOCK_SIZE < index_len
        else index_len - pid0 * BLOCK_SIZE
    )
    COL_BLOCK_SIZE = min(COL_BLOCK_SIZE, total_col_num)
    col_iter_num = (total_col_num + COL_BLOCK_SIZE - 1) // COL_BLOCK_SIZE

    for col_idx in tl.range(0, col_iter_num):
        base_idx = 0
        base_idx = base_idx.to(index.dtype.element_ty)

        col_start_offset = col_idx * COL_BLOCK_SIZE

        acc_result = 0.0
        acc_result = acc_result.to(inp.dtype.element_ty)
        cur_col_num = (
            COL_BLOCK_SIZE
            if col_start_offset + COL_BLOCK_SIZE < total_col_num
            else total_col_num - col_start_offset
        )

        for cur_idx in range(0, cur_col_num):
            cur_offset = pid0 * BLOCK_SIZE + col_start_offset + cur_idx

            src_val = tl.load(src + cur_offset)
            new_idx = tl.load(index + cur_offset)

            if base_idx == new_idx:
                acc_result += src_val
            else:
                tl.atomic_add(inp + base_idx, acc_result)

                base_idx = new_idx
                acc_result = 0.0
                acc_result = acc_result.to(inp.dtype.element_ty)
                acc_result += src_val

        tl.atomic_add(inp + base_idx, acc_result)


def rab_time_backward_triton(
    rab_time_grad: torch.Tensor, bucket_timestamps: torch.Tensor
):
    num_layers, b, s, _ = rab_time_grad.shape
    tsw_grad = torch.zeros(num_layers, NUM_BUCKETS, dtype=torch.float32).to(
        rab_time_grad.device
    )

    bucket_timestamps_expand = (
        bucket_timestamps.reshape(b, s // 2, 1, s // 2, 1)
        .repeat(1, 1, 2, 1, 2)
        .reshape(b, s, s)
        .to(torch.int64)
    ).view(-1)

    index_len = bucket_timestamps_expand.numel()

    rab_time_grad_f32 = rab_time_grad.to(torch.float32)
    sorted_bucket_timestamps_expand, sorted_idx = torch.sort(
        bucket_timestamps_expand.view(-1)
    )

    torch.npu.synchronize()

    grid = lambda meta: (triton.cdiv(index_len, meta["BLOCK_SIZE"]),)

    CORE_NUM = get_npu_properties()["num_vectorcore"]
    BLOCK_SIZE = math.ceil(index_len / CORE_NUM)

    COL_BLOCK_SIZE = 8 * 1024

    for layer_idx in range(num_layers):
        curr_sorted_grad_f32 = rab_time_grad_f32[layer_idx].view(-1)[sorted_idx]
        rab_time_backward_kernel[grid](
            tsw_grad[layer_idx],
            curr_sorted_grad_f32,
            sorted_bucket_timestamps_expand,
            index_len,
            BLOCK_SIZE,
            COL_BLOCK_SIZE,
        )

    return tsw_grad


def rab_time_forward_golden(
    ts_w: torch.Tensor, timestamps: torch.Tensor, bucketization_divisor: float
) -> torch.Tensor:
    """
    torch realization of rab time forward for reference.
    """
    infer_len = timestamps.shape[1] * 2
    bs = timestamps.shape[0]
    num_layers = ts_w.shape[1]

    timestamps = timestamps.unsqueeze(-1).repeat(1, 1, 2)
    diff_timestamps = timestamps.reshape(bs, infer_len, 1) - timestamps.reshape(
        bs, 1, infer_len
    )

    clamp_max = torch.exp(torch.tensor(NUM_BUCKETS * BUCKET_DIVISOR))
    diff_timestamps = (
        torch.log(torch.abs(diff_timestamps).clamp(1, clamp_max))
        / bucketization_divisor
    )
    bucket_timestamps = diff_timestamps.long()
    bucket_timestamps = bucket_timestamps.view(-1)
    result = torch.index_select(ts_w, dim=0, index=bucket_timestamps)

    result = result.t()

    result = result.view(num_layers, bs, infer_len, infer_len)
    return result


def rab_time_backward_golden(
    rab_time_grad: torch.Tensor, bucket_timestamps: torch.Tensor
):
    """
    torch realization of rab time backward for reference.
    """
    num_layers, b, s, _ = rab_time_grad.shape
    tsw_grad = torch.zeros(num_layers, NUM_BUCKETS, dtype=torch.float32).to(
        rab_time_grad.device
    )

    bucket_timestamps_expand = (
        bucket_timestamps.reshape(b, s // 2, 1, s // 2, 1)
        .repeat(1, 1, 2, 1, 2)
        .reshape(b, s, s)
        .to(torch.int64)
    )
    for n, grad in enumerate(rab_time_grad.to(torch.float32)):
        tsw_grad[n] = tsw_grad[n].scatter_add(
            src=grad.view(-1), index=bucket_timestamps_expand.view(-1), dim=0
        )
    return tsw_grad


def rab_time_forward_test(num_layers, train_len, candidate_len, bs, dtype):
    past_valid_lens = create_past_valid_lens(bs, train_len).to(torch.int32)
    timestamps = create_timestamps(train_len, candidate_len, past_valid_lens).to(
        torch.int32
    )
    timestamps_weights = create_timestamps_weights(num_layers).to(dtype)
    timestamps = timestamps.npu()
    timestamps_weights = timestamps_weights.npu()

    torch_npu.npu.synchronize()

    # triton output
    rab_time_out_triton = rab_time_forward_triton(
        ts_w=timestamps_weights,
        timestamps=timestamps,
        bucketization_divisor=BUCKET_DIVISOR,
    )
    torch_npu.npu.synchronize()

    # pytorch output
    rab_time_out_golden = rab_time_forward_golden(
        ts_w=timestamps_weights,
        timestamps=timestamps,
        bucketization_divisor=BUCKET_DIVISOR,
    )
    torch_npu.npu.synchronize()

    torch.testing.assert_close(rab_time_out_triton, rab_time_out_golden)
    print(f"test pass!")


def rab_time_backward_test(num_layers: int, batchsize: int, s: int, dtype: torch.dtype):
    grad = create_rab_time_grad(num_layers, batchsize, s).to(dtype).npu()
    bucket_timestamps = (
        create_bucket_timestamps(batchsize, s // 2).to(torch.int32).npu()
    )

    torch_npu.npu.synchronize()

    golden_result = (
        rab_time_backward_golden(grad, bucket_timestamps).to(torch.float32).cpu()
    )
    op_result = (
        rab_time_backward_triton(grad, bucket_timestamps).to(torch.float32).cpu()
    )

    loss = 1e-4 if dtype == torch.float32 else 1e-3
    torch.testing.assert_close(op_result, golden_result, rtol=loss, atol=loss)
    print(f"test pass!")


if __name__ == "__main__":
    num_layers = 8
    train_len = 500
    candidate_len = 500
    batch_size = 4
    data_type = torch.float32
    print("running rab time forward test:")
    rab_time_forward_test(num_layers, train_len, candidate_len, batch_size, data_type)

    print("running rab time backward test:")
    rab_time_backward_test(
        num_layers, batch_size, 2 * train_len + candidate_len, data_type
    )
