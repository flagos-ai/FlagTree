#!/usr/bin/env python3
"""Experimental TLE persistent warp-specialized GEMM.

This is a performance-oriented prototype that mirrors the simpler TLX
``hopper_gemm_ws.py`` structure:

* one producer partition issues TMA loads into staged shared-memory buffers
* two explicit consumer partitions split the output tile along M
* full/empty mbarriers coordinate producer/consumer ownership of each stage
* WGMMA computes each half tile and descriptor stores write C

It intentionally avoids the more advanced ping-pong epilogue scheduling in
``hopper-persistent-gemm-ws-pingpong.py`` so it can stay within today's TLE API.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import pathlib
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def alloc_fn(size: int, align: int, stream: Optional[int]):
    return torch.empty(size, dtype=torch.int8, device=DEVICE)


@triton.jit
def _compute_pid(tile_id, num_pid_n, num_pid_m, group_size_m: tl.constexpr):
    num_pid_in_group = group_size_m * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * group_size_m
    cur_group_size_m = min(num_pid_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + (tile_id % cur_group_size_m)
    pid_n = (tile_id % num_pid_in_group) // cur_group_size_m
    return pid_m, pid_n


@triton.jit
def _buf_phase(count, num_stages: tl.constexpr):
    buf = count % num_stages
    phase = (count // num_stages) & 1
    return buf, phase


@triton.jit
def _tle_ws_persistent_gemm_producer(
    a_desc,
    b_desc,
    a_smem,
    b_smem,
    empty_a,
    empty_b,
    full_a,
    full_b,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    BM_SPLIT: tl.constexpr,
    CID: tl.constexpr,
):
    sm_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_tiles = num_pid_m * num_pid_n

    tile_id = sm_id
    smem_count = 0
    while tile_id < num_tiles:
        pid_m, pid_n = _compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_SIZE_M)
        off_m = pid_m * BM
        off_n = pid_n * BN

        for k_iter in range(0, tl.cdiv(K, BK)):
            buf, phase = _buf_phase(smem_count, NUM_STAGES)
            off_k = k_iter * BK

            a0_idx = buf
            a1_idx = buf + NUM_STAGES

            tle.gpu.barrier_wait(empty_a[a0_idx], phaseIdx=phase)
            tle.gpu.copy(
                a_desc,
                a_smem.slot(a0_idx),
                [BM_SPLIT, BK],
                [off_m, off_k],
                barrier=full_a[a0_idx],
            )

            tle.gpu.barrier_wait(empty_b[buf], phaseIdx=phase)
            tle.gpu.copy(
                b_desc,
                b_smem.slot(buf),
                [BK, BN],
                [off_k, off_n],
                barrier=full_b[buf],
            )

            tle.gpu.barrier_wait(empty_a[a1_idx], phaseIdx=phase)
            tle.gpu.copy(
                a_desc,
                a_smem.slot(a1_idx),
                [BM_SPLIT, BK],
                [off_m + BM_SPLIT, off_k],
                barrier=full_a[a1_idx],
            )

            smem_count += 1

        tile_id += NUM_SMS


@triton.jit
def _tle_ws_persistent_gemm_consumer(
    a_smem,
    b_smem,
    empty_a,
    empty_b,
    full_a,
    full_b,
    c_desc,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    BM_SPLIT: tl.constexpr,
    WGMMA_PIPELINE: tl.constexpr,
    CID: tl.constexpr,
):
    sm_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_tiles = num_pid_m * num_pid_n
    consumer_idx: tl.constexpr = CID - 1

    tile_id = sm_id
    smem_count = 0
    while tile_id < num_tiles:
        pid_m, pid_n = _compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_SIZE_M)
        off_m = pid_m * BM + consumer_idx * BM_SPLIT
        off_n = pid_n * BN

        acc = tl.zeros((BM_SPLIT, BN), dtype=tl.float32)
        if WGMMA_PIPELINE:
            buf, phase = _buf_phase(smem_count, NUM_STAGES)
            a_idx = buf + consumer_idx * NUM_STAGES

            tle.gpu.barrier_wait(full_a[a_idx], phaseIdx=phase)
            tle.gpu.barrier_wait(full_b[buf], phaseIdx=phase)
            acc = tle.gpu.wgmma(a_smem.slot(a_idx), b_smem.slot(buf), acc)

            last_buf = buf
            last_phase = phase
            last_a_idx = a_idx
            smem_count += 1

            for k_iter in range(1, tl.cdiv(K, BK)):
                buf, phase = _buf_phase(smem_count, NUM_STAGES)
                a_idx = buf + consumer_idx * NUM_STAGES

                tle.gpu.barrier_wait(full_a[a_idx], phaseIdx=phase)
                tle.gpu.barrier_wait(full_b[buf], phaseIdx=phase)
                acc = tle.gpu.wgmma(a_smem.slot(a_idx), b_smem.slot(buf), acc)
                acc = tle.gpu.wgmma_wait(1, acc)

                tle.gpu.barrier_arrive(empty_a[last_a_idx], phaseIdx=last_phase)
                tle.gpu.barrier_arrive(empty_b[last_buf], phaseIdx=last_phase)
                last_buf = buf
                last_phase = phase
                last_a_idx = a_idx
                smem_count += 1

            acc = tle.gpu.wgmma_wait(0, acc)
            tle.gpu.barrier_arrive(empty_a[last_a_idx], phaseIdx=last_phase)
            tle.gpu.barrier_arrive(empty_b[last_buf], phaseIdx=last_phase)
        else:
            for k_iter in range(0, tl.cdiv(K, BK)):
                buf, phase = _buf_phase(smem_count, NUM_STAGES)
                a_idx = buf + consumer_idx * NUM_STAGES

                tle.gpu.barrier_wait(full_a[a_idx], phaseIdx=phase)
                tle.gpu.barrier_wait(full_b[buf], phaseIdx=phase)

                acc = tle.gpu.wgmma(a_smem.slot(a_idx), b_smem.slot(buf), acc)
                acc = tle.gpu.wgmma_wait(0, acc)

                tle.gpu.barrier_arrive(empty_a[a_idx], phaseIdx=phase)
                tle.gpu.barrier_arrive(empty_b[buf], phaseIdx=phase)
                smem_count += 1

        c_desc.store((off_m, off_n), acc.to(tl.float16))
        tile_id += NUM_SMS


@triton.jit
def _tle_ws_nonpersistent_gemm_producer(
    a_desc,
    b_desc,
    a_smem,
    b_smem,
    empty_a,
    empty_b,
    full_a,
    full_b,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    BM_SPLIT: tl.constexpr,
    CID: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    pid_m, pid_n = _compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_SIZE_M)
    off_m = pid_m * BM
    off_n = pid_n * BN

    for k_iter in range(0, tl.cdiv(K, BK)):
        buf, phase = _buf_phase(k_iter, NUM_STAGES)
        off_k = k_iter * BK

        a0_idx = buf
        a1_idx = buf + NUM_STAGES

        tle.gpu.barrier_wait(empty_a[a0_idx], phaseIdx=phase)
        tle.gpu.copy(
            a_desc,
            a_smem.slot(a0_idx),
            [BM_SPLIT, BK],
            [off_m, off_k],
            barrier=full_a[a0_idx],
        )

        tle.gpu.barrier_wait(empty_b[buf], phaseIdx=phase)
        tle.gpu.copy(
            b_desc,
            b_smem.slot(buf),
            [BK, BN],
            [off_k, off_n],
            barrier=full_b[buf],
        )

        tle.gpu.barrier_wait(empty_a[a1_idx], phaseIdx=phase)
        tle.gpu.copy(
            a_desc,
            a_smem.slot(a1_idx),
            [BM_SPLIT, BK],
            [off_m + BM_SPLIT, off_k],
            barrier=full_a[a1_idx],
        )


@triton.jit
def _tle_ws_nonpersistent_gemm_consumer(
    a_smem,
    b_smem,
    empty_a,
    empty_b,
    full_a,
    full_b,
    c_desc,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    BM_SPLIT: tl.constexpr,
    WGMMA_PIPELINE: tl.constexpr,
    CID: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    pid_m, pid_n = _compute_pid(tile_id, num_pid_n, num_pid_m, GROUP_SIZE_M)
    consumer_idx: tl.constexpr = CID - 1
    off_m = pid_m * BM + consumer_idx * BM_SPLIT
    off_n = pid_n * BN

    acc = tl.zeros((BM_SPLIT, BN), dtype=tl.float32)
    if WGMMA_PIPELINE:
        buf, phase = _buf_phase(0, NUM_STAGES)
        a_idx = buf + consumer_idx * NUM_STAGES

        tle.gpu.barrier_wait(full_a[a_idx], phaseIdx=phase)
        tle.gpu.barrier_wait(full_b[buf], phaseIdx=phase)
        acc = tle.gpu.wgmma(a_smem.slot(a_idx), b_smem.slot(buf), acc)

        last_buf = buf
        last_phase = phase
        last_a_idx = a_idx

        for k_iter in range(1, tl.cdiv(K, BK)):
            buf, phase = _buf_phase(k_iter, NUM_STAGES)
            a_idx = buf + consumer_idx * NUM_STAGES

            tle.gpu.barrier_wait(full_a[a_idx], phaseIdx=phase)
            tle.gpu.barrier_wait(full_b[buf], phaseIdx=phase)
            acc = tle.gpu.wgmma(a_smem.slot(a_idx), b_smem.slot(buf), acc)
            acc = tle.gpu.wgmma_wait(1, acc)

            tle.gpu.barrier_arrive(empty_a[last_a_idx], phaseIdx=last_phase)
            tle.gpu.barrier_arrive(empty_b[last_buf], phaseIdx=last_phase)
            last_buf = buf
            last_phase = phase
            last_a_idx = a_idx

        acc = tle.gpu.wgmma_wait(0, acc)
        tle.gpu.barrier_arrive(empty_a[last_a_idx], phaseIdx=last_phase)
        tle.gpu.barrier_arrive(empty_b[last_buf], phaseIdx=last_phase)
    else:
        for k_iter in range(0, tl.cdiv(K, BK)):
            buf, phase = _buf_phase(k_iter, NUM_STAGES)
            a_idx = buf + consumer_idx * NUM_STAGES

            tle.gpu.barrier_wait(full_a[a_idx], phaseIdx=phase)
            tle.gpu.barrier_wait(full_b[buf], phaseIdx=phase)

            acc = tle.gpu.wgmma(a_smem.slot(a_idx), b_smem.slot(buf), acc)
            acc = tle.gpu.wgmma_wait(0, acc)

            tle.gpu.barrier_arrive(empty_a[a_idx], phaseIdx=phase)
            tle.gpu.barrier_arrive(empty_b[buf], phaseIdx=phase)

    c_desc.store((off_m, off_n), acc.to(tl.float16))


@triton.jit
def _tle_ws_persistent_gemm_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    NUM_SMS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    A_STAGE_CAPACITY: tl.constexpr,
    B_STAGE_CAPACITY: tl.constexpr,
    WGMMA_PIPELINE: tl.constexpr,
):
    # v1 mirrors TLX's two-MMA-group split in M.
    BM_SPLIT: tl.constexpr = BM // 2

    a_smem = tle.gpu.alloc(
        [A_STAGE_CAPACITY, BM_SPLIT, BK],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
    )
    b_smem = tle.gpu.alloc(
        [B_STAGE_CAPACITY, BK, BN],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
    )

    empty_a = tle.gpu.alloc_barriers(num_barriers=A_STAGE_CAPACITY, arrive_count=1, init=tle.gpu.READY)
    empty_b = tle.gpu.alloc_barriers(num_barriers=B_STAGE_CAPACITY, arrive_count=2, init=tle.gpu.READY)
    full_a = tle.gpu.alloc_barriers(
        num_barriers=A_STAGE_CAPACITY,
        arrive_count=1,
        expect_bytes=BM_SPLIT * BK * 2,
    )
    full_b = tle.gpu.alloc_barriers(
        num_barriers=B_STAGE_CAPACITY,
        arrive_count=1,
        expect_bytes=BK * BN * 2,
    )

    # Pass partition CIDs explicitly: producer is 0, consumers start at 1.
    tle.gpu.warp_specialize(
        [
            (
                _tle_ws_persistent_gemm_producer,
                (
                    a_desc,
                    b_desc,
                    a_smem,
                    b_smem,
                    empty_a,
                    empty_b,
                    full_a,
                    full_b,
                    M,
                    N,
                    K,
                    NUM_SMS,
                    BM,
                    BN,
                    BK,
                    GROUP_SIZE_M,
                    NUM_STAGES,
                    BM_SPLIT,
                    0,
                ),
            ),
            (
                _tle_ws_persistent_gemm_consumer,
                (
                    a_smem,
                    b_smem,
                    empty_a,
                    empty_b,
                    full_a,
                    full_b,
                    c_desc,
                    M,
                    N,
                    K,
                    NUM_SMS,
                    BM,
                    BN,
                    BK,
                    GROUP_SIZE_M,
                    NUM_STAGES,
                    BM_SPLIT,
                    WGMMA_PIPELINE,
                    1,
                ),
            ),
            (
                _tle_ws_persistent_gemm_consumer,
                (
                    a_smem,
                    b_smem,
                    empty_a,
                    empty_b,
                    full_a,
                    full_b,
                    c_desc,
                    M,
                    N,
                    K,
                    NUM_SMS,
                    BM,
                    BN,
                    BK,
                    GROUP_SIZE_M,
                    NUM_STAGES,
                    BM_SPLIT,
                    WGMMA_PIPELINE,
                    2,
                ),
            ),
        ],
        [4, 4],
        [168, 168],
    )


@triton.jit
def _tle_ws_nonpersistent_gemm_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    A_STAGE_CAPACITY: tl.constexpr,
    B_STAGE_CAPACITY: tl.constexpr,
    WGMMA_PIPELINE: tl.constexpr,
):
    BM_SPLIT: tl.constexpr = BM // 2

    a_smem = tle.gpu.alloc(
        [A_STAGE_CAPACITY, BM_SPLIT, BK],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
    )
    b_smem = tle.gpu.alloc(
        [B_STAGE_CAPACITY, BK, BN],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
    )

    empty_a = tle.gpu.alloc_barriers(num_barriers=A_STAGE_CAPACITY, arrive_count=1, init=tle.gpu.READY)
    empty_b = tle.gpu.alloc_barriers(num_barriers=B_STAGE_CAPACITY, arrive_count=2, init=tle.gpu.READY)
    full_a = tle.gpu.alloc_barriers(
        num_barriers=A_STAGE_CAPACITY,
        arrive_count=1,
        expect_bytes=BM_SPLIT * BK * 2,
    )
    full_b = tle.gpu.alloc_barriers(
        num_barriers=B_STAGE_CAPACITY,
        arrive_count=1,
        expect_bytes=BK * BN * 2,
    )

    # Pass partition CIDs explicitly: producer is 0, consumers start at 1.
    tle.gpu.warp_specialize(
        [
            (
                _tle_ws_nonpersistent_gemm_producer,
                (
                    a_desc,
                    b_desc,
                    a_smem,
                    b_smem,
                    empty_a,
                    empty_b,
                    full_a,
                    full_b,
                    M,
                    N,
                    K,
                    BM,
                    BN,
                    BK,
                    GROUP_SIZE_M,
                    NUM_STAGES,
                    BM_SPLIT,
                    0,
                ),
            ),
            (
                _tle_ws_nonpersistent_gemm_consumer,
                (
                    a_smem,
                    b_smem,
                    empty_a,
                    empty_b,
                    full_a,
                    full_b,
                    c_desc,
                    M,
                    N,
                    K,
                    BM,
                    BN,
                    BK,
                    GROUP_SIZE_M,
                    NUM_STAGES,
                    BM_SPLIT,
                    WGMMA_PIPELINE,
                    1,
                ),
            ),
            (
                _tle_ws_nonpersistent_gemm_consumer,
                (
                    a_smem,
                    b_smem,
                    empty_a,
                    empty_b,
                    full_a,
                    full_b,
                    c_desc,
                    M,
                    N,
                    K,
                    BM,
                    BN,
                    BK,
                    GROUP_SIZE_M,
                    NUM_STAGES,
                    BM_SPLIT,
                    WGMMA_PIPELINE,
                    2,
                ),
            ),
        ],
        [4, 4],
        [168, 168],
    )


def tle_ws_persistent_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    bm: int = 128,
    bn: int = 128,
    bk: int = 64,
    group_size_m: int = 8,
    num_stages: int = 3,
    wgmma_pipeline: bool = True,
    producer_num_warps: int = 4,
    out: torch.Tensor | None = None,
):
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape[1] == b.shape[0]
    assert bm % 2 == 0
    if wgmma_pipeline and num_stages < 2:
        raise ValueError("wgmma_pipeline requires at least two logical smem stages")

    triton.set_allocator(alloc_fn)

    m, k = a.shape
    _, n = b.shape
    if out is None:
        c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    else:
        assert out.shape == (m, n)
        assert out.device == a.device and out.dtype == torch.float16
        assert out.is_contiguous()
        c = out

    a_desc = TensorDescriptor.from_tensor(a, block_shape=[bm // 2, bk])
    b_desc = TensorDescriptor.from_tensor(b, block_shape=[bk, bn])
    c_desc = TensorDescriptor.from_tensor(c, block_shape=[bm // 2, bn])
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = (min(num_sms, triton.cdiv(m, bm) * triton.cdiv(n, bn)), )
    a_stage_capacity = _next_power_of_2(num_stages * 2)
    b_stage_capacity = _next_power_of_2(num_stages)

    kernel = _tle_ws_persistent_gemm_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        m,
        n,
        k,
        NUM_SMS=num_sms,
        BM=bm,
        BN=bn,
        BK=bk,
        GROUP_SIZE_M=group_size_m,
        NUM_STAGES=num_stages,
        A_STAGE_CAPACITY=a_stage_capacity,
        B_STAGE_CAPACITY=b_stage_capacity,
        WGMMA_PIPELINE=wgmma_pipeline,
        num_warps=producer_num_warps,
    )
    return c, kernel


def tle_ws_nonpersistent_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    bm: int = 128,
    bn: int = 128,
    bk: int = 64,
    group_size_m: int = 8,
    num_stages: int = 3,
    wgmma_pipeline: bool = True,
    producer_num_warps: int = 4,
    out: torch.Tensor | None = None,
):
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape[1] == b.shape[0]
    assert bm % 2 == 0
    if wgmma_pipeline and num_stages < 2:
        raise ValueError("wgmma_pipeline requires at least two logical smem stages")

    triton.set_allocator(alloc_fn)

    m, k = a.shape
    _, n = b.shape
    if m % bm != 0 or n % bn != 0 or k % bk != 0:
        raise ValueError("non-persistent TLE benchmark currently expects M, N, K to be exact tile multiples")
    if out is None:
        c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    else:
        assert out.shape == (m, n)
        assert out.device == a.device and out.dtype == torch.float16
        assert out.is_contiguous()
        c = out

    a_desc = TensorDescriptor.from_tensor(a, block_shape=[bm // 2, bk])
    b_desc = TensorDescriptor.from_tensor(b, block_shape=[bk, bn])
    c_desc = TensorDescriptor.from_tensor(c, block_shape=[bm // 2, bn])
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), )
    a_stage_capacity = _next_power_of_2(num_stages * 2)
    b_stage_capacity = _next_power_of_2(num_stages)

    kernel = _tle_ws_nonpersistent_gemm_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        m,
        n,
        k,
        BM=bm,
        BN=bn,
        BK=bk,
        GROUP_SIZE_M=group_size_m,
        NUM_STAGES=num_stages,
        A_STAGE_CAPACITY=a_stage_capacity,
        B_STAGE_CAPACITY=b_stage_capacity,
        WGMMA_PIPELINE=wgmma_pipeline,
        num_warps=producer_num_warps,
    )
    return c, kernel


def _next_power_of_2(value: int) -> int:
    assert value > 0
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class Problem:
    m: int
    n: int
    k: int

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k


@dataclass(frozen=True)
class GemmConfig:
    bm: int
    bn: int
    bk: int
    num_stages: int
    group_size_m: int
    wgmma_pipeline: bool
    producer_num_warps: int

    def label(self) -> str:
        wait = "pipe" if self.wgmma_pipeline else "sync"
        return (f"BM{self.bm}.BN{self.bn}.BK{self.bk}.S{self.num_stages}."
                f"G{self.group_size_m}.P{self.producer_num_warps}.{wait}")


def parse_problem(text: str) -> Problem:
    m, n, k = [int(x) for x in text.lower().replace(",", "x").split("x")]
    return Problem(m, n, k)


def parse_gemm_config(text: str) -> GemmConfig:
    dims_text, _, mode_text = text.partition(":")
    try:
        dims = [int(x) for x in dims_text.lower().replace(",", "x").split("x")]
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"invalid config '{text}', expected BMxBNxBKxSTAGES[xGROUP_SIZE_M[xPRODUCER_WARPS]][:pipe|sync]") from exc
    if len(dims) not in (4, 5, 6):
        raise argparse.ArgumentTypeError(
            f"invalid config '{text}', expected BMxBNxBKxSTAGES[xGROUP_SIZE_M[xPRODUCER_WARPS]][:pipe|sync]")
    mode = mode_text.lower() or "pipe"
    if mode in ("pipe", "pipeline", "pipelined", "wait1"):
        wgmma_pipeline = True
    elif mode in ("sync", "serial", "wait0", "nopipe", "no-pipe"):
        wgmma_pipeline = False
    else:
        raise argparse.ArgumentTypeError("config mode must be pipe or sync")

    group_size_m = dims[4] if len(dims) >= 5 else 8
    producer_num_warps = dims[5] if len(dims) == 6 else 4
    if producer_num_warps % 4 != 0:
        raise argparse.ArgumentTypeError("PRODUCER_WARPS must be a multiple of 4")
    return GemmConfig(dims[0], dims[1], dims[2], dims[3], group_size_m, wgmma_pipeline, producer_num_warps)


def default_compare_configs(args: argparse.Namespace) -> list[GemmConfig]:
    configs = [
        GemmConfig(args.bm, args.bn, args.bk, args.num_stages, args.group_size_m, False, args.producer_num_warps),
        GemmConfig(args.bm, args.bn, args.bk, args.num_stages, args.group_size_m, True, args.producer_num_warps),
    ]
    aligned_bn = 256 if args.bn == 128 else args.bn
    aligned_stages = 2 if args.num_stages != 2 else args.num_stages
    configs.append(
        GemmConfig(args.bm, aligned_bn, args.bk, aligned_stages, args.group_size_m, True, args.producer_num_warps))

    unique: list[GemmConfig] = []
    for cfg in configs:
        if cfg not in unique:
            unique.append(cfg)
    return unique


def bench_ms(fn: Callable[[], object], warmup: int, rep: int, *,
             cuda_graph: bool = False) -> tuple[float, float, float]:
    if cuda_graph:
        result = triton.testing.do_bench_cudagraph(fn, rep=rep, quantiles=(0.5, 0.2, 0.8))
    else:
        result = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=(0.5, 0.2, 0.8))
    if not isinstance(result, (tuple, list)):
        ms = float(result)
        return ms, ms, ms
    return float(result[0]), float(result[1]), float(result[2])


def make_row(
    variant: str,
    problem: Problem,
    ms: float,
    p20: float,
    p80: float,
    bm: int,
    bn: int,
    bk: int,
    num_stages: int,
    group_size_m: int,
    wgmma_pipeline: bool,
    producer_num_warps: int,
    *,
    has_warp_specialize: bool | None = None,
    has_wgmma: bool | None = None,
    cuda_graph: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "variant": variant,
        "M": problem.m,
        "N": problem.n,
        "K": problem.k,
        "BM": bm,
        "BN": bn,
        "BK": bk,
        "NUM_STAGES": num_stages,
        "GROUP_SIZE_M": group_size_m,
        "PRODUCER_NUM_WARPS": producer_num_warps,
        "wgmma_pipeline": wgmma_pipeline,
        "producer_num_warps": producer_num_warps,
        "cuda_graph": cuda_graph,
        "ms": f"{ms:.6f}",
        "p20_ms": f"{p20:.6f}",
        "p80_ms": f"{p80:.6f}",
        "tflops": f"{problem.flops / (ms * 1e-3) / 1e12:.3f}",
    }
    if has_warp_specialize is not None:
        row["has_warp_specialize"] = has_warp_specialize
    if has_wgmma is not None:
        row["has_wgmma"] = has_wgmma
    return row


def make_baseline_row(
    variant: str,
    problem: Problem,
    ms: float,
    p20: float,
    p80: float,
    *,
    baseline_source: str | None = None,
    cuda_graph: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "variant": variant,
        "M": problem.m,
        "N": problem.n,
        "K": problem.k,
        "cuda_graph": cuda_graph,
        "ms": f"{ms:.6f}",
        "p20_ms": f"{p20:.6f}",
        "p80_ms": f"{p80:.6f}",
        "tflops": f"{problem.flops / (ms * 1e-3) / 1e12:.3f}",
    }
    if baseline_source is not None:
        row["baseline_source"] = baseline_source
    return row


def make_error_row(variant: str, problem: Problem, note: str,
                   extra: dict[str, object] | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "variant": variant,
        "M": problem.m,
        "N": problem.n,
        "K": problem.k,
        "ms": "",
        "p20_ms": "",
        "p80_ms": "",
        "tflops": "",
        "note": note,
    }
    if extra:
        row.update(extra)
    return row


def make_native_row(
    variant: str,
    problem: Problem,
    ms: float,
    p20: float,
    p80: float,
    cfg: dict[str, int],
    *,
    has_warp_specialize: bool,
    has_wgmma: bool,
    cuda_graph: bool = False,
) -> dict[str, object]:
    return {
        "variant": variant,
        "M": problem.m,
        "N": problem.n,
        "K": problem.k,
        "ms": f"{ms:.6f}",
        "p20_ms": f"{p20:.6f}",
        "p80_ms": f"{p80:.6f}",
        "tflops": f"{problem.flops / (ms * 1e-3) / 1e12:.3f}",
        **cfg,
        "cuda_graph": cuda_graph,
        "has_warp_specialize": has_warp_specialize,
        "has_wgmma": has_wgmma,
    }


def run_torch_matmul_baseline(
    variant: str,
    problem: Problem,
    a: torch.Tensor,
    b: torch.Tensor,
    warmup: int,
    rep: int,
    *,
    use_out: bool,
    cuda_graph: bool = False,
) -> dict[str, object]:
    if use_out or cuda_graph:
        c = torch.empty((problem.m, problem.n), device=a.device, dtype=torch.float16)

        def run():
            torch.matmul(a, b, out=c)
    else:

        def run():
            torch.matmul(a, b)

    ms, p20, p80 = bench_ms(run, warmup, rep, cuda_graph=cuda_graph)
    baseline_source = "tlx.torch.matmul" if variant == "cuBLAS" else None
    return make_baseline_row(variant, problem, ms, p20, p80, baseline_source=baseline_source, cuda_graph=cuda_graph)


def add_cublas_speedups(rows: list[dict[str, object]]) -> None:
    cublas_ms: dict[tuple[int, int, int], float] = {}
    for row in rows:
        if row.get("variant") == "cuBLAS" and row.get("ms"):
            key = (int(row["M"]), int(row["N"]), int(row["K"]))
            cublas_ms[key] = float(row["ms"])

    for row in rows:
        key = (int(row["M"]), int(row["N"]), int(row["K"]))
        if key not in cublas_ms or not row.get("ms"):
            continue
        row["speedup_vs_cublas"] = f"{cublas_ms[key] / float(row['ms']):.3f}"


def import_from_path(module_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def find_flagtree_repo_root() -> pathlib.Path | None:
    marker = pathlib.Path("python") / "test" / "unit" / "language" / "test_warp_specialization.py"
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    return None


def cfg_extra(cfg: GemmConfig) -> dict[str, object]:
    return {
        "BM": cfg.bm,
        "BN": cfg.bn,
        "BK": cfg.bk,
        "NUM_STAGES": cfg.num_stages,
        "GROUP_SIZE_M": cfg.group_size_m,
        "PRODUCER_NUM_WARPS": cfg.producer_num_warps,
        "wgmma_pipeline": cfg.wgmma_pipeline,
    }


def run_tle_compare_variant(
    problem: Problem,
    a: torch.Tensor,
    b: torch.Tensor,
    ref: torch.Tensor | None,
    warmup: int,
    rep: int,
    cfg: GemmConfig,
    *,
    persistent: bool,
    dump_summary: bool,
    cuda_graph: bool = False,
) -> dict[str, object]:
    split = "persistent_split_m" if persistent else "nonpersistent_split_m"
    variant = f"flagtree.tle.warp_specialize.{split}.{cfg.label()}"
    matmul_fn = tle_ws_persistent_matmul if persistent else tle_ws_nonpersistent_matmul
    try:
        c, kernel = matmul_fn(
            a,
            b,
            bm=cfg.bm,
            bn=cfg.bn,
            bk=cfg.bk,
            group_size_m=cfg.group_size_m,
            num_stages=cfg.num_stages,
            wgmma_pipeline=cfg.wgmma_pipeline,
            producer_num_warps=cfg.producer_num_warps,
        )
        torch.cuda.synchronize()
        ttgir = kernel.asm["ttgir"]
        has_warp_specialize = "ttg.warp_specialize" in ttgir
        has_wgmma = "ttng.warp_group_dot" in ttgir
        if ref is not None:
            torch.testing.assert_close(c, ref, atol=2e-2, rtol=2e-2)
        if dump_summary:
            print(
                f"{problem.m}x{problem.n}x{problem.k} {variant}: "
                f"has_warp_specialize={has_warp_specialize} has_wgmma={has_wgmma} "
                f"asm_keys={','.join(kernel.asm.keys())}",
                file=sys.stderr,
            )

        bench_out = torch.empty((problem.m, problem.n), device=a.device, dtype=torch.float16)

        def run():
            matmul_fn(
                a,
                b,
                bm=cfg.bm,
                bn=cfg.bn,
                bk=cfg.bk,
                group_size_m=cfg.group_size_m,
                num_stages=cfg.num_stages,
                wgmma_pipeline=cfg.wgmma_pipeline,
                producer_num_warps=cfg.producer_num_warps,
                out=bench_out,
            )[0]

        ms, p20, p80 = bench_ms(run, warmup, rep, cuda_graph=cuda_graph)
        return make_row(
            variant,
            problem,
            ms,
            p20,
            p80,
            cfg.bm,
            cfg.bn,
            cfg.bk,
            cfg.num_stages,
            cfg.group_size_m,
            cfg.wgmma_pipeline,
            cfg.producer_num_warps,
            has_warp_specialize=has_warp_specialize,
            has_wgmma=has_wgmma,
            cuda_graph=cuda_graph,
        )
    except Exception as exc:
        return make_error_row(
            variant,
            problem,
            f"compile/run failed: {type(exc).__name__}: {exc}",
            cfg_extra(cfg),
        )


def run_native_ws_tma_variants(problem: Problem, warmup: int, rep: int, *,
                               cuda_graph: bool = False) -> list[dict[str, object]]:
    repo_root = find_flagtree_repo_root()
    if repo_root is None:
        return [
            make_error_row(
                "flagtree.native_tl_range_ws_tma.skip",
                problem,
                "cannot find FlagTree python/test/unit/language/test_warp_specialization.py",
            )
        ]

    sys.path.insert(0, str(repo_root / "python"))
    try:
        module = import_from_path(
            "flagtree_native_ws_test",
            repo_root / "python" / "test" / "unit" / "language" / "test_warp_specialization.py",
        )
    except Exception as exc:
        return [
            make_error_row(
                "flagtree.native_tl_range_ws_tma.skip",
                problem,
                f"cannot import native WS test kernel: {type(exc).__name__}: {exc}",
            )
        ]

    triton.set_allocator(alloc_fn)
    rows: list[dict[str, object]] = []
    configs = [
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "num_stages": 3, "num_warps": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "num_stages": 3, "num_warps": 4},
    ]
    for cfg in configs:
        smem_bytes = (cfg["num_stages"] * cfg["BLOCK_SIZE_K"] *
                      (cfg["BLOCK_SIZE_M"] + cfg["BLOCK_SIZE_N"]) + cfg["BLOCK_SIZE_M"] * cfg["BLOCK_SIZE_N"]) * 2
        if smem_bytes > 228 * 1024:
            continue

        try:
            a = torch.randn((problem.m, problem.k), device=DEVICE, dtype=torch.float16).contiguous()
            b = torch.randn((problem.n, problem.k), device=DEVICE, dtype=torch.float16).contiguous()
            c = torch.empty((problem.m, problem.n), device=DEVICE, dtype=torch.float16)
            grid = lambda meta: (triton.cdiv(problem.m, meta["BLOCK_SIZE_M"]) * triton.cdiv(
                problem.n, meta["BLOCK_SIZE_N"]), )

            def launch():
                return module.matmul_tma_ws_kernel[grid](
                    a,
                    b,
                    c,
                    *a.stride(),
                    *b.stride(),
                    *c.stride(),
                    problem.m,
                    problem.n,
                    problem.k,
                    cfg["num_stages"],
                    cfg["BLOCK_SIZE_M"],
                    cfg["BLOCK_SIZE_N"],
                    cfg["BLOCK_SIZE_K"],
                    8,
                    num_warps=cfg["num_warps"],
                    USE_FP8=False,
                )

            kernel = launch()
            torch.cuda.synchronize()
            ttgir = kernel.asm["ttgir"]

            def run():
                launch()

            ms, p20, p80 = bench_ms(run, warmup, rep, cuda_graph=cuda_graph)
            rows.append(
                make_native_row(
                    "flagtree.native_tl_range_ws_tma",
                    problem,
                    ms,
                    p20,
                    p80,
                    cfg,
                    has_warp_specialize="ttg.warp_specialize" in ttgir,
                    has_wgmma="ttng.warp_group_dot" in ttgir,
                    cuda_graph=cuda_graph,
                ))
        except Exception as exc:
            rows.append(
                make_error_row(
                    "flagtree.native_tl_range_ws_tma",
                    problem,
                    f"compile/run failed: {type(exc).__name__}: {exc}",
                    cfg,
                ))
    return rows


def run_compare(args: argparse.Namespace, problems: list[Problem]) -> list[dict[str, object]]:
    configs = args.compare_config or default_compare_configs(args)
    rows: list[dict[str, object]] = []
    for problem in problems:
        a = torch.randn((problem.m, problem.k), device=DEVICE, dtype=torch.float16).contiguous()
        b = torch.randn((problem.k, problem.n), device=DEVICE, dtype=torch.float16).contiguous()
        problem_rows: list[dict[str, object]] = []

        if not args.no_cublas:
            problem_rows.append(
                run_torch_matmul_baseline(
                    "cuBLAS",
                    problem,
                    a,
                    b,
                    args.warmup,
                    args.rep,
                    use_out=False,
                    cuda_graph=args.cuda_graph,
                ))
        if args.include_torch:
            problem_rows.append(
                run_torch_matmul_baseline(
                    "torch.matmul.out",
                    problem,
                    a,
                    b,
                    args.warmup,
                    args.rep,
                    use_out=True,
                    cuda_graph=args.cuda_graph,
                ))

        ref = torch.matmul(a, b) if args.check else None
        for cfg in configs:
            problem_rows.append(
                run_tle_compare_variant(
                    problem,
                    a,
                    b,
                    ref,
                    args.warmup,
                    args.rep,
                    cfg,
                    persistent=True,
                    dump_summary=args.dump_summary,
                    cuda_graph=args.cuda_graph,
                ))
        if not args.no_nonpersistent:
            for cfg in configs:
                problem_rows.append(
                    run_tle_compare_variant(
                        problem,
                        a,
                        b,
                        ref,
                        args.warmup,
                        args.rep,
                        cfg,
                        persistent=False,
                        dump_summary=args.dump_summary,
                        cuda_graph=args.cuda_graph,
                    ))

        if not args.no_native:
            problem_rows.extend(run_native_ws_tma_variants(problem, args.warmup, args.rep, cuda_graph=args.cuda_graph))

        if not args.no_cublas:
            add_cublas_speedups(problem_rows)
        rows.extend(problem_rows)
    return rows


def write_rows(rows: Iterable[dict[str, object]], out: str | None) -> None:
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if out:
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=parse_problem, default=[])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--cuda-graph", action="store_true",
                        help="benchmark by capturing the workload once and timing CUDA Graph replay")
    parser.add_argument("--out", default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="run cuBLAS, TLE persistent/non-persistent configs, and native WS TMA configs")
    parser.add_argument("--compare-config", action="append", type=parse_gemm_config, default=[],
                        help="compare config BMxBNxBKxSTAGES[xGROUP_SIZE_M[xPRODUCER_WARPS]][:pipe|sync]")
    parser.add_argument("--include-cublas", action="store_true",
                        help="include torch.matmul(a, b), matching the TLX cuBLAS baseline")
    parser.add_argument("--include-torch", action="store_true",
                        help="include a preallocated torch.matmul(a, b, out=c) baseline")
    parser.add_argument("--no-cublas", action="store_true", help="omit cuBLAS baseline in --compare mode")
    parser.add_argument("--no-native", action="store_true", help="omit native tl.range WS TMA rows in --compare mode")
    parser.add_argument("--no-nonpersistent", action="store_true",
                        help="omit TLE non-persistent split-M rows in --compare mode")
    parser.add_argument("--dump-summary", action="store_true",
                        help="print a compact kernel TTGIR feature summary to stderr")
    parser.add_argument("--bm", type=int, default=128)
    parser.add_argument("--bn", type=int, default=128)
    parser.add_argument("--bk", type=int, default=64)
    parser.add_argument("--group-size-m", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--no-wgmma-pipeline", action="store_true")
    parser.add_argument("--producer-num-warps", type=int, default=4)
    parser.add_argument("--non-persistent", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("Hopper or newer CUDA GPU is required")
    if args.producer_num_warps % 4 != 0:
        parser.error("--producer-num-warps must be a multiple of 4 for TLE warp_specialize")

    problems = args.shape or [Problem(4096, 4096, 4096), Problem(8192, 8192, 512)]
    if args.compare:
        write_rows(run_compare(args, problems), args.out)
        return

    rows = []
    for problem in problems:
        a = torch.randn((problem.m, problem.k), device=DEVICE, dtype=torch.float16).contiguous()
        b = torch.randn((problem.k, problem.n), device=DEVICE, dtype=torch.float16).contiguous()
        problem_rows: list[dict[str, object]] = []

        if args.compare or args.include_cublas:
            problem_rows.append(
                run_torch_matmul_baseline(
                    "cuBLAS",
                    problem,
                    a,
                    b,
                    args.warmup,
                    args.rep,
                    use_out=False,
                    cuda_graph=args.cuda_graph,
                ))
        if args.include_torch:
            problem_rows.append(
                run_torch_matmul_baseline(
                    "torch.matmul.out",
                    problem,
                    a,
                    b,
                    args.warmup,
                    args.rep,
                    use_out=True,
                    cuda_graph=args.cuda_graph,
                ))

        wgmma_pipeline = not args.no_wgmma_pipeline
        matmul_fn = tle_ws_nonpersistent_matmul if args.non_persistent else tle_ws_persistent_matmul
        c, kernel = matmul_fn(
            a,
            b,
            bm=args.bm,
            bn=args.bn,
            bk=args.bk,
            group_size_m=args.group_size_m,
            num_stages=args.num_stages,
            wgmma_pipeline=wgmma_pipeline,
            producer_num_warps=args.producer_num_warps,
        )
        torch.cuda.synchronize()
        ttgir = kernel.asm["ttgir"]
        has_warp_specialize = "ttg.warp_specialize" in ttgir
        has_wgmma = "ttng.warp_group_dot" in ttgir
        assert has_warp_specialize
        assert has_wgmma

        if args.dump_summary:
            variant = "nonpersistent" if args.non_persistent else "persistent"
            print(
                f"{problem.m}x{problem.n}x{problem.k} {variant}: "
                f"has_warp_specialize={has_warp_specialize} has_wgmma={has_wgmma} "
                f"asm_keys={','.join(kernel.asm.keys())}",
                file=sys.stderr,
            )

        if args.check:
            torch.testing.assert_close(c, torch.matmul(a, b), atol=2e-2, rtol=2e-2)

        bench_out = torch.empty((problem.m, problem.n), device=a.device, dtype=torch.float16)

        def run():
            matmul_fn(
                a,
                b,
                bm=args.bm,
                bn=args.bn,
                bk=args.bk,
                group_size_m=args.group_size_m,
                num_stages=args.num_stages,
                wgmma_pipeline=wgmma_pipeline,
                producer_num_warps=args.producer_num_warps,
                out=bench_out,
            )[0]

        ms, p20, p80 = bench_ms(run, args.warmup, args.rep, cuda_graph=args.cuda_graph)
        variant = "flagtree.tle.warp_specialize.nonpersistent_split_m" if args.non_persistent else \
            "flagtree.tle.warp_specialize.persistent_split_m"
        problem_rows.append(
            make_row(
                variant,
                problem,
                ms,
                p20,
                p80,
                args.bm,
                args.bn,
                args.bk,
                args.num_stages,
                args.group_size_m,
                wgmma_pipeline,
                args.producer_num_warps,
                has_warp_specialize=has_warp_specialize,
                has_wgmma=has_wgmma,
                cuda_graph=args.cuda_graph,
            ))
        if args.compare or args.include_cublas:
            add_cublas_speedups(problem_rows)
        rows.extend(problem_rows)

    write_rows(rows, args.out)


if __name__ == "__main__":
    main()
