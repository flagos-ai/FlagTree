#!/usr/bin/env python3
"""Experimental TLE FA3-style forward attention.

* one producer partition issues TMA loads for Q/K/V into shared memory
* two explicit consumer partitions split the query block along M
* mbarriers coordinate producer/consumer ownership of TMA buffers
* named barriers serialize the two consumer WGMMA issue streams in a ping-pong pattern
* K uses WGMMA descriptor transposition via ``wgmma(..., trans_b=True)``
* PV uses a register/tensor A operand, avoiding an intermediate P shared tile
"""

from __future__ import annotations

import argparse
import csv
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
def _buf_phase(count, num_buffers: tl.constexpr):
    buf = count % num_buffers
    phase_idx = count // num_buffers
    return buf, phase_idx


@triton.jit
def _compute_offsets(tile_idx, H, N_CTX, BLOCK_M: tl.constexpr):
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    off_hz = tile_idx // num_pid_m
    start_m = tile_idx % num_pid_m
    off_z = off_hz // H
    off_h = off_hz % H
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    kv_offset_y = offset_y
    return start_m, off_hz, qo_offset_y, kv_offset_y


@triton.jit
def _attn_fwd_tle_ws_pipelined_pingpong_producer(
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    N_CTX,
    q_smem,
    k_smem,
    v_smem,
    q_empties,
    q_fulls,
    k_empties,
    k_fulls,
    v_empties,
    v_fulls,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    BM_SPLIT: tl.constexpr,
    CID: tl.constexpr,
):
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    total_tiles = num_pid_m * Z * H

    tile_idx = prog_id
    tile_count = 0
    accum_cnt_kv = 0
    while tile_idx < total_tiles:
        start_m, off_hz, qo_offset_y, kv_offset_y = _compute_offsets(tile_idx, H, N_CTX, BLOCK_M)

        q_buf, q_phase_idx = _buf_phase(tile_count, NUM_BUFFERS_Q)
        q0_idx = q_buf
        q1_idx = q_buf + NUM_BUFFERS_Q

        tle.gpu.barrier_wait(q_empties[q0_idx], phaseIdx=q_phase_idx)
        tle.gpu.copy(
            desc_q,
            q_smem.slot(q0_idx),
            [BM_SPLIT, HEAD_DIM],
            [qo_offset_y, 0],
            barrier=q_fulls[q0_idx],
        )

        kv_buf, kv_phase_idx = _buf_phase(accum_cnt_kv, NUM_BUFFERS_KV)
        tle.gpu.barrier_wait(k_empties[kv_buf], phaseIdx=kv_phase_idx)
        tle.gpu.copy(
            desc_k,
            k_smem.slot(kv_buf),
            [BLOCK_N, HEAD_DIM],
            [kv_offset_y, 0],
            barrier=k_fulls[kv_buf],
        )

        tle.gpu.barrier_wait(q_empties[q1_idx], phaseIdx=q_phase_idx)
        tle.gpu.copy(
            desc_q,
            q_smem.slot(q1_idx),
            [BM_SPLIT, HEAD_DIM],
            [qo_offset_y + BM_SPLIT, 0],
            barrier=q_fulls[q1_idx],
        )

        tle.gpu.barrier_wait(v_empties[kv_buf], phaseIdx=kv_phase_idx)
        tle.gpu.copy(
            desc_v,
            v_smem.slot(kv_buf),
            [BLOCK_N, HEAD_DIM],
            [kv_offset_y, 0],
            barrier=v_fulls[kv_buf],
        )
        accum_cnt_kv += 1

        for kv_idx in range(BLOCK_N, N_CTX, BLOCK_N):
            kv_buf, kv_phase_idx = _buf_phase(accum_cnt_kv, NUM_BUFFERS_KV)
            kv_offset = kv_offset_y + kv_idx

            tle.gpu.barrier_wait(k_empties[kv_buf], phaseIdx=kv_phase_idx)
            tle.gpu.copy(
                desc_k,
                k_smem.slot(kv_buf),
                [BLOCK_N, HEAD_DIM],
                [kv_offset, 0],
                barrier=k_fulls[kv_buf],
            )

            tle.gpu.barrier_wait(v_empties[kv_buf], phaseIdx=kv_phase_idx)
            tle.gpu.copy(
                desc_v,
                v_smem.slot(kv_buf),
                [BLOCK_N, HEAD_DIM],
                [kv_offset, 0],
                barrier=v_fulls[kv_buf],
            )
            accum_cnt_kv += 1

        tile_idx += num_progs
        tile_count += 1


@triton.jit
def _attn_fwd_tle_ws_pipelined_pingpong_consumer(
    sm_scale,
    m_ptr,
    Z,
    H,
    desc_o,
    N_CTX,
    q_smem,
    k_smem,
    v_smem,
    q_empties,
    q_fulls,
    k_empties,
    k_fulls,
    v_empties,
    v_fulls,
    ping_to_c0,
    ping_to_c1,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    BM_SPLIT: tl.constexpr,
    CID: tl.constexpr,
):
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    total_tiles = num_pid_m * Z * H
    consumer_idx: tl.constexpr = CID - 1

    if consumer_idx == 1:
        tle.gpu.barrier_arrive(ping_to_c0)

    tile_idx = prog_id
    tile_count = 0
    accum_cnt_kv = 0
    while tile_idx < total_tiles:
        start_m, off_hz, qo_offset_y, kv_offset_y = _compute_offsets(tile_idx, H, N_CTX, BLOCK_M)

        m_i = tl.zeros([BM_SPLIT], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BM_SPLIT], dtype=tl.float32) + 1.0
        acc = tl.zeros([BM_SPLIT, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale * 1.44269504

        q_buf, q_phase_idx = _buf_phase(tile_count, NUM_BUFFERS_Q)
        q_idx = q_buf + consumer_idx * NUM_BUFFERS_Q
        tle.gpu.barrier_wait(q_fulls[q_idx], phaseIdx=q_phase_idx)

        kv_buf, kv_phase_idx = _buf_phase(accum_cnt_kv, NUM_BUFFERS_KV)
        tle.gpu.barrier_wait(k_fulls[kv_buf], phaseIdx=kv_phase_idx)

        if consumer_idx == 0:
            tle.gpu.barrier_wait(ping_to_c0)
        else:
            tle.gpu.barrier_wait(ping_to_c1)
        qk = tle.gpu.wgmma(
            q_smem.slot(q_idx),
            k_smem.slot(kv_buf),
            out_dtype=tl.float32,
            trans_b=True,
        )
        if consumer_idx == 0:
            tle.gpu.barrier_arrive(ping_to_c1)
        else:
            tle.gpu.barrier_arrive(ping_to_c0)
        qk = tle.gpu.wgmma_wait(0, qk)
        tle.gpu.barrier_arrive(k_empties[kv_buf], phaseIdx=kv_phase_idx)

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        accum_cnt_kv += 1

        for _ in range(BLOCK_N, N_CTX, BLOCK_N):
            kv_buf, kv_phase_idx = _buf_phase(accum_cnt_kv, NUM_BUFFERS_KV)
            tle.gpu.barrier_wait(k_fulls[kv_buf], phaseIdx=kv_phase_idx)

            if consumer_idx == 0:
                tle.gpu.barrier_wait(ping_to_c0)
            else:
                tle.gpu.barrier_wait(ping_to_c1)
            qk = tle.gpu.wgmma(
                q_smem.slot(q_idx),
                k_smem.slot(kv_buf),
                out_dtype=tl.float32,
                trans_b=True,
            )
            if consumer_idx == 0:
                tle.gpu.barrier_arrive(ping_to_c1)
            else:
                tle.gpu.barrier_arrive(ping_to_c0)

            v_buf, v_phase_idx = _buf_phase(accum_cnt_kv - 1, NUM_BUFFERS_KV)
            tle.gpu.barrier_wait(v_fulls[v_buf], phaseIdx=v_phase_idx)
            acc = tle.gpu.wgmma(p.to(tl.float16), v_smem.slot(v_buf), acc)

            qk = tle.gpu.wgmma_wait(1, qk)
            tle.gpu.barrier_arrive(k_empties[kv_buf], phaseIdx=kv_phase_idx)

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            alpha = tl.math.exp2(m_i - m_ij)
            l_ij = tl.sum(p, 1)
            l_i = l_i * alpha + l_ij
            m_i = m_ij

            acc = tle.gpu.wgmma_wait(0, acc)
            tle.gpu.barrier_arrive(v_empties[v_buf], phaseIdx=v_phase_idx)
            acc = acc * alpha[:, None]
            accum_cnt_kv += 1

        v_buf, v_phase_idx = _buf_phase(accum_cnt_kv - 1, NUM_BUFFERS_KV)
        tle.gpu.barrier_wait(v_fulls[v_buf], phaseIdx=v_phase_idx)
        acc = tle.gpu.wgmma(p.to(tl.float16), v_smem.slot(v_buf), acc)

        acc = tle.gpu.wgmma_wait(1, acc)
        tle.gpu.barrier_arrive(q_empties[q_idx], phaseIdx=q_phase_idx)

        acc = tle.gpu.wgmma_wait(0, acc)
        tle.gpu.barrier_arrive(v_empties[v_buf], phaseIdx=v_phase_idx)

        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]

        offs_m = start_m * BLOCK_M + consumer_idx * BM_SPLIT + tl.arange(0, BM_SPLIT)
        m_ptrs = m_ptr + off_hz * N_CTX + offs_m
        tl.store(m_ptrs, m_i)

        desc_o.store((qo_offset_y + consumer_idx * BM_SPLIT, 0), acc.to(tl.float16))

        tile_idx += num_progs
        tile_count += 1


@triton.jit
def _attn_fwd_tle_ws_pipelined_pingpong_persistent(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_MMA_WARPS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    Q_STAGE_CAPACITY: tl.constexpr,
    KV_STAGE_CAPACITY: tl.constexpr,
):
    BM_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    THREADS_IN_MMA_GROUPS: tl.constexpr = NUM_MMA_WARPS * 32

    q_smem = tle.gpu.alloc(
        [Q_STAGE_CAPACITY, BM_SPLIT, HEAD_DIM],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
    )
    k_smem = tle.gpu.alloc(
        [KV_STAGE_CAPACITY, BLOCK_N, HEAD_DIM],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
    )
    v_smem = tle.gpu.alloc(
        [KV_STAGE_CAPACITY, BLOCK_N, HEAD_DIM],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
    )
    q_empties = tle.gpu.alloc_barriers(num_barriers=Q_STAGE_CAPACITY, arrive_count=1, init=tle.gpu.READY)
    q_fulls = tle.gpu.alloc_barriers(
        num_barriers=Q_STAGE_CAPACITY,
        arrive_count=1,
        expect_bytes=BM_SPLIT * HEAD_DIM * 2,
    )
    k_empties = tle.gpu.alloc_barriers(
        num_barriers=KV_STAGE_CAPACITY,
        arrive_count=NUM_MMA_GROUPS,
        init=tle.gpu.READY,
    )
    k_fulls = tle.gpu.alloc_barriers(
        num_barriers=KV_STAGE_CAPACITY,
        arrive_count=1,
        expect_bytes=BLOCK_N * HEAD_DIM * 2,
    )
    v_empties = tle.gpu.alloc_barriers(
        num_barriers=KV_STAGE_CAPACITY,
        arrive_count=NUM_MMA_GROUPS,
        init=tle.gpu.READY,
    )
    v_fulls = tle.gpu.alloc_barriers(
        num_barriers=KV_STAGE_CAPACITY,
        arrive_count=1,
        expect_bytes=BLOCK_N * HEAD_DIM * 2,
    )

    # Named barriers for the consumer ping-pong issue protocol. They are used
    # without phaseIdx, selecting TLE's named-barrier backend.
    pingpong = tle.gpu.alloc_barriers(num_barriers=2, arrive_count=THREADS_IN_MMA_GROUPS)
    ping_to_c0 = pingpong[0]
    ping_to_c1 = pingpong[1]

    mma_warps: tl.constexpr = NUM_MMA_WARPS // NUM_MMA_GROUPS
    # Pass partition CIDs explicitly: producer is 0, consumers start at 1.
    tle.gpu.warp_specialize(
        [
            (
                _attn_fwd_tle_ws_pipelined_pingpong_producer,
                (
                    Z,
                    H,
                    desc_q,
                    desc_k,
                    desc_v,
                    N_CTX,
                    q_smem,
                    k_smem,
                    v_smem,
                    q_empties,
                    q_fulls,
                    k_empties,
                    k_fulls,
                    v_empties,
                    v_fulls,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    NUM_BUFFERS_Q,
                    NUM_BUFFERS_KV,
                    BM_SPLIT,
                    0,
                ),
            ),
            (
                _attn_fwd_tle_ws_pipelined_pingpong_consumer,
                (
                    sm_scale,
                    M,
                    Z,
                    H,
                    desc_o,
                    N_CTX,
                    q_smem,
                    k_smem,
                    v_smem,
                    q_empties,
                    q_fulls,
                    k_empties,
                    k_fulls,
                    v_empties,
                    v_fulls,
                    ping_to_c0,
                    ping_to_c1,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    NUM_BUFFERS_Q,
                    NUM_BUFFERS_KV,
                    BM_SPLIT,
                    1,
                ),
            ),
            (
                _attn_fwd_tle_ws_pipelined_pingpong_consumer,
                (
                    sm_scale,
                    M,
                    Z,
                    H,
                    desc_o,
                    N_CTX,
                    q_smem,
                    k_smem,
                    v_smem,
                    q_empties,
                    q_fulls,
                    k_empties,
                    k_fulls,
                    v_empties,
                    v_fulls,
                    ping_to_c0,
                    ping_to_c1,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    NUM_BUFFERS_Q,
                    NUM_BUFFERS_KV,
                    BM_SPLIT,
                    2,
                ),
            ),
        ],
        [mma_warps, mma_warps],
        [232, 232],
    )


def tle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    *,
    block_m: int = 128,
    block_n: int = 128,
    num_buffers_q: int = 1,
    num_buffers_kv: int = 2,
    num_mma_warps: int = 8,
    num_mma_groups: int = 2,
    producer_num_warps: int = 4,
    return_kernel: bool = False,
    out: torch.Tensor | None = None,
    m_out: torch.Tensor | None = None,
):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == torch.float16 and k.dtype == torch.float16 and v.dtype == torch.float16
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q.shape == k.shape == v.shape
    assert q.ndim == 4
    assert num_mma_groups == 2
    assert block_m % num_mma_groups == 0
    assert producer_num_warps % 4 == 0

    z, h, n_ctx, head_dim = q.shape
    if head_dim not in (16, 32, 64, 128, 256):
        raise ValueError("HEAD_DIM must be one of 16, 32, 64, 128, 256")
    if n_ctx % block_m != 0 or n_ctx % block_n != 0:
        raise ValueError("this prototype expects N_CTX to be a multiple of BLOCK_M and BLOCK_N")

    triton.set_allocator(alloc_fn)

    if out is None:
        o = torch.empty_like(q)
    else:
        assert out.shape == q.shape
        assert out.device == q.device and out.dtype == q.dtype
        assert out.is_contiguous()
        o = out
    if m_out is None:
        m = torch.empty((z, h, n_ctx), device=q.device, dtype=torch.float32)
    else:
        assert m_out.shape == (z, h, n_ctx)
        assert m_out.device == q.device and m_out.dtype == torch.float32
        assert m_out.is_contiguous()
        m = m_out
    y_dim = z * h * n_ctx
    block_m_split = block_m // num_mma_groups

    desc_q = TensorDescriptor(q, shape=[y_dim, head_dim], strides=[head_dim, 1], block_shape=[block_m_split, head_dim])
    desc_k = TensorDescriptor(k, shape=[y_dim, head_dim], strides=[head_dim, 1], block_shape=[block_n, head_dim])
    desc_v = TensorDescriptor(v, shape=[y_dim, head_dim], strides=[head_dim, 1], block_shape=[block_n, head_dim])
    desc_o = TensorDescriptor(o, shape=[y_dim, head_dim], strides=[head_dim, 1], block_shape=[block_m_split, head_dim])

    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    total_tiles = triton.cdiv(n_ctx, block_m) * z * h
    grid = (min(num_sms, total_tiles), )
    q_stage_capacity = _next_power_of_2(num_buffers_q * num_mma_groups)
    kv_stage_capacity = _next_power_of_2(num_buffers_kv)

    kernel = _attn_fwd_tle_ws_pipelined_pingpong_persistent[grid](
        sm_scale,
        m,
        z,
        h,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        n_ctx,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NUM_BUFFERS_Q=num_buffers_q,
        NUM_BUFFERS_KV=num_buffers_kv,
        NUM_MMA_WARPS=num_mma_warps,
        NUM_MMA_GROUPS=num_mma_groups,
        Q_STAGE_CAPACITY=q_stage_capacity,
        KV_STAGE_CAPACITY=kv_stage_capacity,
        num_warps=producer_num_warps,
    )
    if return_kernel:
        return o, m, kernel
    return o


def _next_power_of_2(value: int) -> int:
    assert value > 0
    return 1 << (value - 1).bit_length()


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.transpose(-2, -1).float()) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=sm_scale,
        is_causal=False,
    )


def bench_ms(fn: Callable[[], object], warmup: int, rep: int, *,
             cuda_graph: bool = False) -> tuple[float, float, float]:
    if cuda_graph:
        result = triton.testing.do_bench_cudagraph(fn, rep=rep, quantiles=(0.5, 0.2, 0.8))
    else:
        result = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=(0.5, 0.2, 0.8))
    if isinstance(result, (tuple, list)):
        return float(result[0]), float(result[1]), float(result[2])
    ms = float(result)
    return ms, ms, ms


@dataclass(frozen=True)
class AttentionProblem:
    z: int
    h: int
    n_ctx: int
    head_dim: int

    @property
    def flops(self) -> int:
        return 4 * self.z * self.h * self.n_ctx * self.n_ctx * self.head_dim


def parse_problem(text: str) -> AttentionProblem:
    z, h, n_ctx, head_dim = [int(x) for x in text.lower().replace(",", "x").split("x")]
    if min(z, h, n_ctx, head_dim) <= 0:
        raise argparse.ArgumentTypeError("Z, H, N_CTX, and HEAD_DIM must be positive")
    return AttentionProblem(z, h, n_ctx, head_dim)


def make_row(
    variant: str,
    problem: AttentionProblem,
    ms: float,
    p20: float,
    p80: float,
    block_m: int,
    block_n: int,
    extra: dict[str, object] | None = None,
    cuda_graph: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "variant": variant,
        "Z": problem.z,
        "H": problem.h,
        "N_CTX": problem.n_ctx,
        "HEAD_DIM": problem.head_dim,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "cuda_graph": cuda_graph,
        "ms": f"{ms:.6f}",
        "p20_ms": f"{p20:.6f}",
        "p80_ms": f"{p80:.6f}",
        "tflops_approx": f"{problem.flops / (ms * 1e-3) / 1e12:.3f}",
    }
    if extra:
        row.update(extra)
    return row


def make_error_row(
    variant: str,
    problem: AttentionProblem,
    block_m: int,
    block_n: int,
    exc: Exception,
) -> dict[str, object]:
    return {
        "variant": variant,
        "Z": problem.z,
        "H": problem.h,
        "N_CTX": problem.n_ctx,
        "HEAD_DIM": problem.head_dim,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "ms": "",
        "p20_ms": "",
        "p80_ms": "",
        "tflops_approx": "",
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def write_rows(rows: Iterable[dict[str, object]], out: str | None) -> None:
    rows = list(rows)
    fields: list[str] = []
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
    parser.add_argument("--problem", action="append", type=parse_problem, default=[],
                        help="attention problem ZxHxN_CTXxHEAD_DIM; may be repeated")
    parser.add_argument("--sm-scale", type=float, default=None)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--cuda-graph", action="store_true",
                        help="benchmark by capturing the workload once and timing CUDA Graph replay")
    parser.add_argument("--out", default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--include-sdpa", action="store_true", help="also benchmark torch SDPA on the same inputs")
    parser.add_argument("--sdpa-requires-grad", action="store_true",
                        help="set q/k/v.requires_grad_() to match the TLX SDPA perf test")
    parser.add_argument("--continue-on-tle-error", action="store_true",
                        help="record a CSV error row instead of aborting if the TLE kernel fails")
    parser.add_argument("--dump-summary", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("Hopper or newer CUDA GPU is required")

    problems = args.problem or [AttentionProblem(1, 1, 1024, 64)]
    rows = []
    for problem in problems:
        q = torch.randn((problem.z, problem.h, problem.n_ctx, problem.head_dim), device=DEVICE, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        if args.sdpa_requires_grad:
            q.requires_grad_()
            k.requires_grad_()
            v.requires_grad_()
        sm_scale = args.sm_scale
        if sm_scale is None:
            sm_scale = problem.head_dim**-0.5

        if args.include_sdpa:
            sdpa_out = sdpa_attention(q, k, v, sm_scale)
            torch.cuda.synchronize()

            def run_sdpa():
                sdpa_attention(q, k, v, sm_scale)

            ms, p20, p80 = bench_ms(run_sdpa, args.warmup, args.rep, cuda_graph=args.cuda_graph)
            rows.append(
                make_row(
                    "SDPA",
                    problem,
                    ms,
                    p20,
                    p80,
                    args.block_m,
                    args.block_n,
                    {
                        "output_dtype": str(sdpa_out.dtype).replace("torch.", ""),
                        "has_warp_specialize": False,
                        "has_wgmma": False,
                        "baseline_source": "torch.nn.functional.scaled_dot_product_attention",
                        "requires_grad": args.sdpa_requires_grad,
                        "status": "ok",
                    },
                    cuda_graph=args.cuda_graph,
                ))

        try:
            o, m, kernel = tle_attention(
                q,
                k,
                v,
                sm_scale,
                block_m=args.block_m,
                block_n=args.block_n,
                return_kernel=True,
            )
            torch.cuda.synchronize()

            if args.check:
                ref = reference_attention(q, k, v, sm_scale)
                torch.testing.assert_close(o, ref, atol=5e-2, rtol=5e-2)

            bench_o = torch.empty_like(q)
            bench_m = torch.empty((problem.z, problem.h, problem.n_ctx), device=q.device, dtype=torch.float32)

            def run():
                tle_attention(
                    q,
                    k,
                    v,
                    sm_scale,
                    block_m=args.block_m,
                    block_n=args.block_n,
                    out=bench_o,
                    m_out=bench_m,
                )

            ms, p20, p80 = bench_ms(run, args.warmup, args.rep, cuda_graph=args.cuda_graph)
            extra = {
                "output_dtype": str(o.dtype).replace("torch.", ""),
                "has_warp_specialize": "ttg.warp_specialize" in kernel.asm["ttgir"],
                "has_wgmma": "ttng.warp_group_dot" in kernel.asm["ttgir"],
                "status": "ok",
            }
            if args.dump_summary:
                extra["ttgir_len"] = len(kernel.asm.get("ttgir", ""))
                extra["ptx_len"] = len(kernel.asm.get("ptx", ""))
            rows.append(
                make_row(
                    "flagtree.tle.fa3.ws_pipelined_pingpong_persistent",
                    problem,
                    ms,
                    p20,
                    p80,
                    args.block_m,
                    args.block_n,
                    extra,
                    cuda_graph=args.cuda_graph,
                ))
        except Exception as exc:
            if not args.continue_on_tle_error:
                raise
            rows.append(
                make_error_row(
                    "flagtree.tle.fa3.ws_pipelined_pingpong_persistent",
                    problem,
                    args.block_m,
                    args.block_n,
                    exc,
                ))

    write_rows(rows, args.out)


if __name__ == "__main__":
    main()
