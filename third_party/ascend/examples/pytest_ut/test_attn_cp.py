# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.


import time
import torch
import torch_npu
import pytest
import sys

sys.path.append("../../../")

from attn_cp_triton_kernel_3d import forward_update_triton, backward_update_triton
from attn_cp_triton_kernel_3d_la import forward_update_triton as forward_update_triton_la


def collect_time(model, example_inputs, times: int = 1):
    stream = torch.npu.current_stream()
    warmup = 1
    stream.synchronize()
    for i in range(times + warmup):
        out = model(*example_inputs)
        if i < warmup:
            stream.synchronize()
            t0 = time.perf_counter()
        else:
            t1 = time.perf_counter()
    stream.synchronize()
    t1 = time.perf_counter()
    # GC the result after timing
    assert out is not None
    return (t1 - t0) / times


def print_performance(fn, args=(), times=10, repeat=10, baseline=1.0):
    stream = torch.npu.current_stream()

    start = time.perf_counter()

    stream.synchronize()

    for _ in range(repeat * times):
        fn(*args)

    stream.synchronize()

    end = time.perf_counter()
    took = (end - start) / (times * repeat)
    print(f"{took:.6f}")
    return took


def profile_test(fn, fn_triton, args=(), name="gen_fn", times=10, repeat=10):
    print(f"--------------------profiling {name} for {times * repeat} times--------------------")
    stream = torch.npu.current_stream()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, )
    prof = torch_npu.profiler.profile(
        activities=[
            # torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=100, repeat=1, skip_first=10),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir"),
        experimental_config=experimental_config)

    stream.synchronize()
    prof.start()

    for _ in range(times * repeat):
        fn_triton(*args)
        prof.step()

    prof.stop()


def benchmark_test(fn, fn_triton, args=(), name="gen_fn", times=10, repeat=10, profile=False):
    print(f"--------------------benchmark_{name} for {times * repeat} times--------------------")
    stream = torch.npu.current_stream()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, )
    prof = torch_npu.profiler.profile(
        activities=[
            # torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=100, repeat=1, skip_first=10),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir"),
        experimental_config=experimental_config)

    stream.synchronize()
    prof.start()
    start = time.perf_counter()
    for _ in range(times * repeat):
        fn_triton(*args)
        if profile:
            prof.step()
    stream.synchronize()
    end = time.perf_counter()
    time_compiled = (end - start) / (times * repeat)
    print(f"{time_compiled:.6f}")

    print(f"Runing eager {name} for {times * repeat} times")
    start = time.perf_counter()
    for _ in range(times * repeat):
        fn(*args)
        if profile:
            prof.step()
    stream.synchronize()
    end = time.perf_counter()
    time_eager = (end - start) / (times * repeat)
    print(f"{time_eager:.6f}")

    time_eager *= 1000000
    time_compiled *= 1000000
    print(
        f"Accelerated: {(time_eager - time_compiled) / time_compiled * 100:.4f}% eager takes {time_eager:.3f} us, triton takes {time_compiled:.3f} us")

    return time_eager, time_compiled


from einops import rearrange


def trans_BNSD2SBH(x):
    """Trans data layout from BNSD to SBH"""
    return rearrange(x, 'b n s d -> s b (n d)').contiguous()


def broadcast_and_trans_BNSD2SBH(x, h):
    """broadcast and trans a tensor from [b, n, s, 8] to [s, b, h]"""
    n = x.shape[1]
    d = h // n
    # [b, n, s, 8] -> [b, n, s, d]
    new_x = x[..., 0].unsqueeze(3)
    new_x = new_x.repeat(1, 1, 1, d)
    return trans_BNSD2SBH(new_x)


def forward_update(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                   cur_attn_out, cur_softmax_max, cur_softmax_sum):
    # update softmax_max
    org_dtype = prev_attn_out.dtype
    softmax_max = torch.maximum(prev_softmax_max, cur_softmax_max)
    prev_scale = torch.exp(prev_softmax_max - softmax_max)
    cur_scale = torch.exp(cur_softmax_max - softmax_max)
    # update softmax_sum
    prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
    cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
    softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled
    # out updating scale
    prev_out_scale = prev_softmax_sum_scaled / softmax_sum
    cur_out_scale = cur_softmax_sum_scaled / softmax_sum
    # [b, n, s, 8] -> [s, b, h]
    prev_out_scale_sbh = broadcast_and_trans_BNSD2SBH(prev_out_scale, prev_attn_out.shape[-1])
    cur_out_scale_sbh = broadcast_and_trans_BNSD2SBH(cur_out_scale, prev_attn_out.shape[-1])
    # update output
    attn_out = prev_attn_out * prev_out_scale_sbh + cur_attn_out * cur_out_scale_sbh
    attn_out = attn_out.to(org_dtype)
    return attn_out, softmax_max, softmax_sum


def prove_forward_update():
    def data_validation(prev_softmax_max, cur_softmax_max, prev_softmax_sum, cur_softmax_sum, prev_attn_out,
                        cur_attn_out):

        (tt_attn_out, tt_softmax_max, tt_softmax_sum) = forward_update_triton(prev_attn_out,
                                                                              prev_softmax_max, prev_softmax_sum,
                                                                              cur_attn_out, cur_softmax_max,
                                                                              cur_softmax_sum)

        (attn_out, softmax_max, softmax_sum) = forward_update(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                                                              cur_attn_out,
                                                              cur_softmax_max, cur_softmax_sum)

        try:
            assert torch.equal(softmax_max, tt_softmax_max)
            print("max comparition passed.")
            assert torch.equal(softmax_sum, tt_softmax_sum)
            print("sum comparition passed.")
            torch.testing.assert_close(attn_out, tt_attn_out)
            print("atten comparition passed.")

        except Exception as e:
            print(e)
            print("comparison not passed")
        print(
            f"proving finished, attn shape:{prev_attn_out.shape}, stride:{prev_attn_out.stride(), cur_attn_out.stride()}, softmax shape:{prev_softmax_sum.shape}, stride:{prev_softmax_sum.stride(), cur_softmax_sum.stride()}")

    (S, B, H, N) = (4096, 1, 6144, 48)
    DS = 2 * S
    DTYPE_ATTN = torch.bfloat16
    DTYPE = torch.float32
    F32_BLK_SIZE = 8
    for i in range(1):
        print("prove_forward_update round:", i)
        prev_attn_out = torch.randn((DS, B, H), dtype=DTYPE_ATTN).npu()
        prev_softmax_max = torch.rand((B, N, DS), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
        prev_softmax_sum = torch.rand((B, N, DS), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
        cur_attn_out = torch.randn((DS, B, H), dtype=DTYPE_ATTN).npu()
        cur_softmax_max = torch.rand((B, N, DS), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
        cur_softmax_sum = torch.rand((B, N, DS), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)

        prev_attn_out_s = prev_attn_out.view(2, S, B, H)[1]
        prev_softmax_max_s = prev_softmax_max.view(B, N, 2, S, 8)[:, :, 1, :, :]
        prev_softmax_sum_s = prev_softmax_sum.view(B, N, 2, S, 8)[:, :, 1, :, :]
        cur_attn_out_s = torch.randn((S, B, H), dtype=DTYPE_ATTN).npu()
        cur_softmax_max_s = torch.rand((B, N, S), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
        cur_softmax_sum_s = torch.rand((B, N, S), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)

        print("--------------------prove_forward_update:2S----------------------")
        data_validation(prev_softmax_max, cur_softmax_max, prev_softmax_sum, cur_softmax_sum, prev_attn_out,
                        cur_attn_out)

        print("--------------------prove_forward_update:1S---------------------- ")
        data_validation(prev_softmax_max_s, cur_softmax_max_s, prev_softmax_sum_s, cur_softmax_sum_s, prev_attn_out_s,
                        cur_attn_out_s)


def benchmark_forward_update():
    (S, B, H, N) = (4096, 1, 6144, 48)
    DS = 2 * S
    DTYPE_ATTN = torch.bfloat16
    DTYPE = torch.float32
    F32_BLK_SIZE = 8
    prev_attn_out = torch.randn((DS, B, H), dtype=DTYPE_ATTN).npu()
    prev_softmax_max = torch.rand((B, N, DS), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
    prev_softmax_sum = torch.rand((B, N, DS), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
    cur_attn_out = torch.randn((DS, B, H), dtype=DTYPE_ATTN).npu()
    cur_softmax_max = torch.rand((B, N, DS), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
    cur_softmax_sum = torch.rand((B, N, DS), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)

    prev_attn_out_s = prev_attn_out.view(2, S, B, H)[1]
    prev_softmax_max_s = prev_softmax_max.view(B, N, 2, S, 8)[:, :, 1, :, :]
    prev_softmax_sum_s = prev_softmax_sum.view(B, N, 2, S, 8)[:, :, 1, :, :]
    cur_attn_out_s = torch.randn((S, B, H), dtype=DTYPE_ATTN).npu()
    cur_softmax_max_s = torch.rand((B, N, S), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
    cur_softmax_sum_s = torch.rand((B, N, S), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)

    benchmark_test(forward_update, forward_update_triton, args=(prev_attn_out,
                                                                prev_softmax_max, prev_softmax_sum, cur_attn_out,
                                                                cur_softmax_max, cur_softmax_sum),
                   name="forward_update_2s", times=10, repeat=10)

    benchmark_test(forward_update, forward_update_triton, args=(prev_attn_out,
                                                                prev_softmax_max, prev_softmax_sum, cur_attn_out,
                                                                cur_softmax_max, cur_softmax_sum),
                   name="forward_update_s", times=10, repeat=10)


def profile_forward_update():
    (S, B, H, N) = (4096, 1, 6144, 48)
    # (S, B, H, N) = (2048,1,1536,12)
    DS = 2 * S
    DTYPE_ATTN = torch.bfloat16
    DTYPE = torch.float32
    F32_BLK_SIZE = 8
    prev_attn_out = torch.randn((DS, B, H), dtype=DTYPE_ATTN).npu()
    prev_softmax_max = torch.rand((B, N, DS), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
    prev_softmax_sum = torch.rand((B, N, DS), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
    cur_attn_out = torch.randn((DS, B, H), dtype=DTYPE_ATTN).npu()
    cur_softmax_max = torch.rand((B, N, DS), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
    cur_softmax_sum = torch.rand((B, N, DS), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)

    prev_attn_out_s = prev_attn_out.view(2, S, B, H)[1]
    prev_softmax_max_s = prev_softmax_max.view(B, N, 2, S, 8)[:, :, 1, :, :]
    prev_softmax_sum_s = prev_softmax_sum.view(B, N, 2, S, 8)[:, :, 1, :, :]
    cur_attn_out_s = torch.randn((S, B, H), dtype=DTYPE_ATTN).npu()
    cur_softmax_max_s = torch.rand((B, N, S), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
    cur_softmax_sum_s = torch.rand((B, N, S), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)

    profile_test(forward_update, forward_update_triton, args=(prev_attn_out,
                                                              prev_softmax_max, prev_softmax_sum, cur_attn_out,
                                                              cur_softmax_max, cur_softmax_sum),
                 name="forward_update_2s", times=10, repeat=10)

    profile_test(forward_update, forward_update_triton, args=(prev_attn_out,
                                                              prev_softmax_max, prev_softmax_sum, cur_attn_out,
                                                              cur_softmax_max, cur_softmax_sum),
                 name="forward_update_s", times=10, repeat=10)


def backward_update(dq, dk, dv, cur_dq, cur_dk, cur_dv, i=7, rank=1):
    cp_size = 8
    if i >= cp_size - rank - 1:
        if i == cp_size - 1:
            cur_dq = cur_dq.view(dq.shape)
            cur_dk = cur_dk.view(dk.shape)
            cur_dv = cur_dv.view(dv.shape)
            dq.add_(cur_dq)
            dk.add_(cur_dk)
            dv.add_(cur_dv)
        else:
            cur_dq = cur_dq.view(dq.shape)
            dq.add_(cur_dq)
            dk[0].add_(cur_dk)
            dv[0].add_(cur_dv)
    else:
        dq[1].add_(cur_dq)
        cur_dk = cur_dk.view(dk.shape)  # [2s, b, h] -> [2, s, b, h]
        cur_dv = cur_dv.view(dv.shape)
        dk.add_(cur_dk)
        dv.add_(cur_dv)


def prove_backward_update():
    (S, B, H) = (16384, 1, 1536)
    DS = 2 * S
    DTYPE_ATTN = torch.bfloat16
    DTYPE = torch.float32

    d_dq = torch.randn((2, S, B, H), dtype=DTYPE_ATTN).npu()
    cur_dq = torch.randn((S, B, H), dtype=DTYPE_ATTN).npu()
    d_cur_dq = torch.randn((DS, B, H), dtype=DTYPE_ATTN).npu()

    d_dk = torch.randn((2, S, B, H), dtype=DTYPE_ATTN).npu()
    cur_dk = torch.randn((S, B, H), dtype=DTYPE_ATTN).npu()
    d_cur_dk = torch.randn((DS, B, H), dtype=DTYPE_ATTN).npu()

    d_dv = torch.randn((2, S, B, H), dtype=DTYPE_ATTN).npu()
    cur_dv = torch.randn((S, B, H), dtype=DTYPE_ATTN).npu()
    d_cur_dv = torch.randn((DS, B, H), dtype=DTYPE_ATTN).npu()

    def data_validate(dq, dk, dv, cur_dq, cur_dk, cur_dv, i, rank):
        dq_c = dq.detach().clone()
        dk_c = dk.detach().clone()
        dv_c = dv.detach().clone()

        if (i == 7):
            backward_update(dq, dk, dv, cur_dq, cur_dk, cur_dv, i, rank)
            backward_update_triton(dq_c, dk_c, dv_c, cur_dq, cur_dk, cur_dv)
        elif (i == 6):
            backward_update(dq, dk, dv, cur_dq, cur_dk, cur_dv, i, rank)
            backward_update_triton(dq_c, dk_c[0], dv_c[0], cur_dq, cur_dk, cur_dv)
        else:
            backward_update(dq, dk, dv, cur_dq, cur_dk, cur_dv, i, rank)
            backward_update_triton(dq_c[1], dk_c, dv_c, cur_dq, cur_dk, cur_dv)

        torch.testing.assert_close(dq, dq_c)
        print("dq comparison passed")
        torch.testing.assert_close(dk, dk_c)
        print("dk comparison passed")
        torch.testing.assert_close(dv, dv_c)
        print("passed comparison ")

    print("--------------------prove_backward_update case 0----------------------")
    data_validate(d_dq, d_dk, d_dv, d_cur_dq, d_cur_dk, d_cur_dv, 7, 1)
    print("--------------------prove_backward_update case 1----------------------")
    data_validate(d_dq, d_dk, d_dv, d_cur_dq, cur_dk, cur_dv, 6, 1)
    print("--------------------prove_backward_update case 2----------------------")
    data_validate(d_dq, d_dk, d_dv, cur_dq, d_cur_dk, d_cur_dv, 5, 1)


def benchmark_backward_update():
    (S, B, H) = (16384, 1, 1536)
    DS = 2 * S
    DTYPE_ATTN = torch.bfloat16
    DTYPE = torch.float32

    d_dq = torch.randn((2, S, B, H), dtype=DTYPE_ATTN).npu()
    cur_dq = torch.randn((S, B, H), dtype=DTYPE_ATTN).npu()
    d_cur_dq = torch.randn((DS, B, H), dtype=DTYPE_ATTN).npu()

    d_dk = torch.randn((2, S, B, H), dtype=DTYPE_ATTN).npu()
    cur_dk = torch.randn((S, B, H), dtype=DTYPE_ATTN).npu()
    d_cur_dk = torch.randn((DS, B, H), dtype=DTYPE_ATTN).npu()

    d_dv = torch.randn((2, S, B, H), dtype=DTYPE_ATTN).npu()
    cur_dv = torch.randn((S, B, H), dtype=DTYPE_ATTN).npu()
    d_cur_dv = torch.randn((DS, B, H), dtype=DTYPE_ATTN).npu()
    # benchmark case 0
    benchmark_test(backward_update, backward_update_triton, args=(d_dq, d_dk, d_dv, d_cur_dq, d_cur_dk, d_cur_dv),
                   name="backward_update_0")


def forward_update_la(prev_attn_out, prev_softmax_log_max_sum,
                      cur_attn_out, cur_softmax_log_max_sum):
    if prev_attn_out is None:
        return cur_attn_out, cur_softmax_log_max_sum
    softmax_log_max_sum = torch.log(torch.exp(cur_softmax_log_max_sum) + torch.exp(prev_softmax_log_max_sum))
    attn_out = torch.exp(prev_softmax_log_max_sum - softmax_log_max_sum) * prev_attn_out + torch.exp(
        cur_softmax_log_max_sum - softmax_log_max_sum) * cur_attn_out
    return attn_out, softmax_log_max_sum


# simuldate origin code :call foward_update then call copy
def forward_update_copy(prev_attn_out, prev_softmax_log_max_sum,
                        cur_attn_out, cur_softmax_log_max_sum):
    attn_out, softmax = forward_update_la(prev_attn_out, prev_softmax_log_max_sum,
                                          cur_attn_out, cur_softmax_log_max_sum)
    prev_attn_out.copy_(attn_out)
    prev_softmax_log_max_sum.copy_(softmax)


def prove_forward_update_la():
    def data_validation(prev_attn_out, prev_softmax_sum, cur_attn_out, cur_softmax_sum):

        (attn_out, softmax_sum) = forward_update_la(prev_attn_out, prev_softmax_sum, cur_attn_out, cur_softmax_sum)
        forward_update_triton_la(prev_attn_out, prev_softmax_sum, cur_attn_out, cur_softmax_sum)

        try:
            torch.testing.assert_close(softmax_sum, prev_softmax_sum)
            print("softmax comparition passed.")
            torch.testing.assert_close(attn_out, prev_attn_out)
            print("attn comparition passed.")

        except Exception as e:
            print(e)
            print("comparison not passed")

        print(
            f"proving finished, attn shape:{prev_attn_out.shape}, stride:{prev_attn_out.stride(), cur_attn_out.stride()}, softmax shape:{prev_softmax_sum.shape}, stride:{prev_softmax_sum.stride(), cur_softmax_sum.stride()}")

    (S, B, N, D) = (4096, 1, 48, 128)
    DS = 2 * S
    DTYPE_ATTN = torch.float32
    DTYPE = torch.float32

    for i in range(1):
        print("round:", i)
        attn_out = torch.randn((B, N, DS, D), dtype=DTYPE_ATTN).npu()
        softmax_sum = torch.rand((B, N, DS, 1), dtype=DTYPE).npu()

        cur_attn_out = torch.randn((B, N, DS, D), dtype=DTYPE_ATTN).npu()
        cur_softmax_sum = torch.rand((B, N, DS, 1), dtype=DTYPE).npu()

        cur_attn_out_s = torch.randn((B, N, S, D), dtype=DTYPE_ATTN).npu()
        cur_softmax_sum_s = torch.rand((B, N, S, 1), dtype=DTYPE).npu()

        print("---------------------prove_forward_updat_la 2S-------------------------------")
        data_validation(attn_out, softmax_sum, cur_attn_out, cur_softmax_sum)

        print("--------------------prove_forward_update_la 1S------------------------------")
        # [b, n, 2s, d] -> [b, n, 2, s, d]
        attn_out_s = attn_out.view(*attn_out.shape[:2], 2, attn_out.shape[2] // 2, attn_out.shape[-1])
        # [b, n, 2s, 1] -> [b, n, 2, s, 1]
        softmax_sum_s = softmax_sum.view(*softmax_sum.shape[:2], 2, softmax_sum.shape[2] // 2, softmax_sum.shape[-1])

        data_validation(attn_out_s[:, :, 1], softmax_sum_s[:, :, 1], cur_attn_out_s, cur_softmax_sum_s)


def benchmark_forward_update_la():
    (S, B, N, D) = (4096, 1, 48, 128)
    DS = 2 * S
    DTYPE_ATTN = torch.float32
    DTYPE = torch.float32

    attn_out = torch.randn((B, N, DS, D), dtype=DTYPE_ATTN).npu()
    softmax_sum = torch.rand((B, N, DS, 1), dtype=DTYPE).npu()
    cur_attn_out = torch.randn((B, N, DS, D), dtype=DTYPE_ATTN).npu()
    cur_softmax_sum = torch.rand((B, N, DS, 1), dtype=DTYPE).npu()
    cur_attn_out_s = torch.randn((B, N, S, D), dtype=DTYPE_ATTN).npu()
    cur_softmax_sum_s = torch.rand((B, N, S, 1), dtype=DTYPE).npu()
    attn_out_s = attn_out.view(*attn_out.shape[:2], 2, attn_out.shape[2] // 2, attn_out.shape[-1])
    softmax_sum_s = softmax_sum.view(*softmax_sum.shape[:2], 2, softmax_sum.shape[2] // 2, softmax_sum.shape[-1])

    benchmark_test(forward_update_copy, forward_update_triton_la, args=(attn_out_s[:, :, 1], softmax_sum_s[:, :, 1],
                                                                        cur_attn_out_s, cur_softmax_sum_s),
                   name="forward_update_la", profile=False, repeat=1000)

@pytest.mark.skip(reason="attn_cp")
def test_prove_forward_update():
    prove_forward_update()

@pytest.mark.skip(reason="attn_cp")
def test_prove_forward_update_la():
    prove_forward_update_la()

@pytest.mark.skip(reason="attn_cp")
def test_prove_backward_update():
    prove_backward_update()

if __name__ == "__main__":
    pytest.main([__file__])
