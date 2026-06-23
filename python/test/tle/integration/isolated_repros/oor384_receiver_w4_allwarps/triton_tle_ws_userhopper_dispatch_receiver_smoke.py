"""TLE warp-specialized raw-NVSHMEM UserHopper dispatch/receiver smoke.

This is a capability test, not a performance implementation.  It keeps the
public symmetric-buffer byte layout aligned with UserHopper MegaMoE and runs
explicit TLE WS worker partitions:

* dispatch: build UserHopper-style per-expert send counts, write remote
  src_token_topk queues, and publish remote recv counts through raw NVSHMEM.
* receiver: wait on local recv counts, select source ranks with the same
  round-robin policy, and pull token bytes/scale/weight into the local L1 pool.

The optional ``single_pipe_compute_stub`` mode adds compute and combine roles
after receiver.  Those roles do not implement GEMM yet; they verify that later
compute/combine partitions can observe receiver-produced L1 arrival counters,
consume receiver-produced L1 token/weight data, write remote combine staging,
and reduce local combine slots through additional TLE pipes.
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.knobs as knobs
import triton.language as tl
import triton.experimental.tle.language as tle
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect


HERE = Path(__file__).resolve().parent
NVSHMEM_HOME = os.environ.get("NVSHMEM_HOME")
if not NVSHMEM_HOME:
    raise RuntimeError("NVSHMEM_HOME must be set for the raw NVSHMEM repro")

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


NUM_RANKS = _env_int("USERHOPPER_WS_NUM_RANKS", 2)
NUM_EXPERTS = _env_int("USERHOPPER_WS_NUM_EXPERTS", NUM_RANKS)
NUM_TOPK = _env_int("USERHOPPER_WS_NUM_TOPK", 1)
NUM_TOKENS = _env_int("USERHOPPER_WS_NUM_TOKENS", 4)
NUM_MAX_TOKENS_PER_RANK = _env_int("USERHOPPER_WS_NUM_MAX_TOKENS_PER_RANK", 384)
HIDDEN = _env_int("USERHOPPER_WS_HIDDEN", 128)
INTERMEDIATE_HIDDEN = _env_int("USERHOPPER_WS_INTERMEDIATE_HIDDEN", 128)
ROUTE_MODE = os.environ.get("USERHOPPER_WS_ROUTE_MODE", "uniform")
PRINT_LIMIT = _env_int("USERHOPPER_WS_PRINT_LIMIT", 32)
MAXNREG = _env_int("USERHOPPER_WS_MAXNREG", 240)
CLEANUP_WORKSPACE = _env_int("USERHOPPER_WS_CLEANUP", 0)
REPEAT_LAUNCHES = _env_int("USERHOPPER_WS_REPEAT_LAUNCHES", 1)
NUM_WARPS = _env_int("USERHOPPER_WS_NUM_WARPS", 4)
NUM_DISPATCH_WARPS = _env_int("USERHOPPER_WS_NUM_DISPATCH_WARPS", 1)
COMPUTE_FULL_HIDDEN = _env_int("USERHOPPER_WS_COMPUTE_FULL_HIDDEN", 0)
COMPUTE_PARALLEL = _env_int("USERHOPPER_WS_COMPUTE_PARALLEL", 0)
COMPUTE_WORKER_WARPS = _env_int("USERHOPPER_WS_COMPUTE_WORKER_WARPS", 1)
MAX_DISPATCH_WARPS = 8

if NUM_EXPERTS % NUM_RANKS != 0:
    raise ValueError(f"NUM_EXPERTS must be divisible by NUM_RANKS, got {NUM_EXPERTS}/{NUM_RANKS}")
if NUM_TOPK <= 0 or NUM_TOPK > 32:
    raise ValueError(f"NUM_TOPK must be in [1, 32], got {NUM_TOPK}")
if NUM_TOPK > NUM_EXPERTS:
    raise ValueError(f"NUM_TOPK must be <= NUM_EXPERTS for unique topk experts, got {NUM_TOPK}>{NUM_EXPERTS}")
if ROUTE_MODE not in {"uniform", "skew", "masked"}:
    raise ValueError(f"unknown USERHOPPER_WS_ROUTE_MODE={ROUTE_MODE!r}; expected uniform, skew, or masked")
if HIDDEN % 128 != 0:
    raise ValueError(f"HIDDEN must be divisible by 128 for SM90 SF layout, got {HIDDEN}")
if INTERMEDIATE_HIDDEN % 128 != 0:
    raise ValueError(
        f"INTERMEDIATE_HIDDEN must be divisible by 128 for UserHopper layout, got {INTERMEDIATE_HIDDEN}"
    )
if NUM_TOKENS > NUM_MAX_TOKENS_PER_RANK:
    raise ValueError(
        f"NUM_TOKENS must be <= NUM_MAX_TOKENS_PER_RANK, got {NUM_TOKENS}>{NUM_MAX_TOKENS_PER_RANK}"
    )
if PRINT_LIMIT <= 0:
    raise ValueError(f"USERHOPPER_WS_PRINT_LIMIT must be positive, got {PRINT_LIMIT}")
if MAXNREG <= 0:
    raise ValueError(f"USERHOPPER_WS_MAXNREG must be positive, got {MAXNREG}")
if CLEANUP_WORKSPACE not in (0, 1):
    raise ValueError(f"USERHOPPER_WS_CLEANUP must be 0 or 1, got {CLEANUP_WORKSPACE}")
if REPEAT_LAUNCHES <= 0:
    raise ValueError(f"USERHOPPER_WS_REPEAT_LAUNCHES must be positive, got {REPEAT_LAUNCHES}")
if NUM_WARPS <= 0 or NUM_WARPS % 4 != 0:
    raise ValueError(f"USERHOPPER_WS_NUM_WARPS must be a positive multiple of 4, got {NUM_WARPS}")
if NUM_DISPATCH_WARPS <= 0:
    raise ValueError(f"USERHOPPER_WS_NUM_DISPATCH_WARPS must be positive, got {NUM_DISPATCH_WARPS}")
if NUM_DISPATCH_WARPS > MAX_DISPATCH_WARPS:
    raise ValueError(
        f"USERHOPPER_WS_NUM_DISPATCH_WARPS must be <= {MAX_DISPATCH_WARPS}, got {NUM_DISPATCH_WARPS}"
    )
if NUM_DISPATCH_WARPS > NUM_WARPS:
    raise ValueError(
        f"USERHOPPER_WS_NUM_DISPATCH_WARPS must be <= USERHOPPER_WS_NUM_WARPS, "
        f"got {NUM_DISPATCH_WARPS}>{NUM_WARPS}"
    )
if COMPUTE_FULL_HIDDEN not in (0, 1):
    raise ValueError(f"USERHOPPER_WS_COMPUTE_FULL_HIDDEN must be 0 or 1, got {COMPUTE_FULL_HIDDEN}")
if COMPUTE_PARALLEL not in (0, 1):
    raise ValueError(f"USERHOPPER_WS_COMPUTE_PARALLEL must be 0 or 1, got {COMPUTE_PARALLEL}")
if COMPUTE_WORKER_WARPS <= 0:
    raise ValueError(f"USERHOPPER_WS_COMPUTE_WORKER_WARPS must be positive, got {COMPUTE_WORKER_WARPS}")
EXPECTED_LOCAL_RECV_TOKENS = NUM_TOKENS * NUM_TOPK
NUM_EXPERTS_PER_RANK = NUM_EXPERTS // NUM_RANKS

K_LCM_CANDIDATE_BLOCK_M = 384
K_MAX_CANDIDATE_BLOCK_M = 192
K_MIN_CANDIDATE_BLOCK_M = 8
K_BLOCK_M = 64
K_CANDIDATE_BLOCK_M = (8, 16, 32, 64, 96, 128, 192)


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _num_max_pool_tokens(num_ranks: int, num_max_tokens_per_rank: int, num_topk: int,
                         num_experts_per_rank: int) -> int:
    num_max_recv_tokens = num_ranks * num_max_tokens_per_rank
    num_max_experts_per_token = min(num_topk, num_experts_per_rank)
    return _align(
        num_max_recv_tokens * num_max_experts_per_token
        + num_experts_per_rank * (K_MAX_CANDIDATE_BLOCK_M - 1),
        K_LCM_CANDIDATE_BLOCK_M,
    )


def _layout() -> dict[str, int]:
    num_experts_per_rank = NUM_EXPERTS // NUM_RANKS
    num_max_recv_tokens_per_expert = NUM_RANKS * NUM_MAX_TOKENS_PER_RANK
    num_max_pool_tokens = _num_max_pool_tokens(
        NUM_RANKS, NUM_MAX_TOKENS_PER_RANK, NUM_TOPK, num_experts_per_rank
    )
    num_max_pool_blocks = num_max_pool_tokens // K_MIN_CANDIDATE_BLOCK_M
    workspace = 0
    workspace += 32
    workspace += NUM_EXPERTS * 8 * 2
    workspace += num_experts_per_rank * 8
    workspace += _align(num_max_pool_blocks, 2) * 4
    workspace += num_max_pool_blocks * 8
    workspace += num_experts_per_rank * NUM_RANKS * num_max_recv_tokens_per_expert * 4
    workspace += num_max_pool_tokens * 12
    workspace = _align(workspace, 16)

    input_token = workspace
    input_sf = input_token + NUM_MAX_TOKENS_PER_RANK * HIDDEN
    input_topk_idx = input_sf + NUM_MAX_TOKENS_PER_RANK * (HIDDEN // 32)
    input_topk_weight = input_topk_idx + NUM_MAX_TOKENS_PER_RANK * NUM_TOPK * 8
    l1_token = input_topk_weight + NUM_MAX_TOKENS_PER_RANK * NUM_TOPK * 4

    num_max_padded_sf_pool_tokens = max(
        (num_max_pool_tokens // block_m) * _align(block_m, 128)
        for block_m in K_CANDIDATE_BLOCK_M
    )
    l1_sf = l1_token + num_max_pool_tokens * HIDDEN
    l1_topk_weight = l1_sf + num_max_padded_sf_pool_tokens * (HIDDEN // 32)
    l2_token = l1_topk_weight + num_max_pool_tokens * 4
    l2_sf = l2_token + num_max_pool_tokens * INTERMEDIATE_HIDDEN
    l2_sf_bytes_per_token = INTERMEDIATE_HIDDEN // 16
    combine_token = l2_sf + num_max_padded_sf_pool_tokens * l2_sf_bytes_per_token
    total_bytes = combine_token + NUM_TOPK * NUM_MAX_TOKENS_PER_RANK * HIDDEN * 2

    return {
        "workspace": workspace,
        "input_token": input_token,
        "input_sf": input_sf,
        "input_topk_idx": input_topk_idx,
        "input_topk_weight": input_topk_weight,
        "l1_token": l1_token,
        "l1_sf": l1_sf,
        "l1_topk_weight": l1_topk_weight,
        "l2_token": l2_token,
        "l2_sf": l2_sf,
        "combine_token": combine_token,
        "num_max_pool_tokens": num_max_pool_tokens,
        "num_max_pool_blocks": num_max_pool_blocks,
        "num_max_padded_sf_pool_tokens": num_max_padded_sf_pool_tokens,
        "total_bytes": total_bytes,
    }


LAYOUT = _layout()


def _uniform_route_expert(rank: int, topk: int) -> int:
    dst_rank = (rank + 1 + topk) % NUM_RANKS
    dst_local_expert = topk % NUM_EXPERTS_PER_RANK
    return dst_rank * NUM_EXPERTS_PER_RANK + dst_local_expert


def _route_experts_for_rank(rank: int) -> list[int]:
    if ROUTE_MODE == "uniform":
        candidates = [_uniform_route_expert(rank, topk) for topk in range(NUM_TOPK)]
    elif ROUTE_MODE == "skew":
        rank_offsets = (1, 1, 2, 4, 4, 7, 3, 6, 0, 5, 2, 7, 1, 4, 6, 3)
        local_offsets = (0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1)
        candidates = []
        for topk in range(NUM_TOPK):
            dst_rank = (rank + rank_offsets[topk % len(rank_offsets)]) % NUM_RANKS
            dst_local = (local_offsets[topk % len(local_offsets)] + topk // len(local_offsets)) % NUM_EXPERTS_PER_RANK
            candidates.append(dst_rank * NUM_EXPERTS_PER_RANK + dst_local)
    else:
        candidates = []
        for topk in range(NUM_TOPK):
            candidates.append(-1 if topk % 3 == 2 else _uniform_route_expert(rank, topk))

    seen = set()
    unique = []
    for expert in candidates:
        if expert < 0:
            unique.append(expert)
            continue
        while expert in seen:
            expert = (expert + 1) % NUM_EXPERTS
        seen.add(expert)
        unique.append(expert)
    return unique


def _route_expert(rank: int, topk: int) -> int:
    return _route_experts_for_rank(rank)[topk]


def _topks_for_destination(src_rank: int, dst_rank: int, dst_local_expert: int) -> list[int]:
    topks = []
    for topk in range(NUM_TOPK):
        expert = _route_expert(src_rank, topk)
        if expert >= 0 and expert // NUM_EXPERTS_PER_RANK == dst_rank and expert % NUM_EXPERTS_PER_RANK == dst_local_expert:
            topks.append(topk)
    return topks


def _rank_counts_for_destination(dst_rank: int, dst_local_expert: int) -> list[int]:
    return [
        NUM_TOKENS * len(_topks_for_destination(src_rank, dst_rank, dst_local_expert))
        for src_rank in range(NUM_RANKS)
    ]


def _expected_counts_for_rank(rank: int) -> list[int]:
    return [
        sum(_rank_counts_for_destination(rank, local_expert))
        for local_expert in range(NUM_EXPERTS_PER_RANK)
    ]


def _choose_rank_round_robin(token_idx_in_expert: int, rank_counts: list[int]) -> tuple[int, int]:
    remaining = rank_counts.copy()
    offset = 0
    slot_idx = token_idx_in_expert
    while True:
        active_ranks = [rank for rank, count in enumerate(remaining) if count > 0]
        if not active_ranks:
            break
        length = min(remaining[rank] for rank in active_ranks)
        num_round_tokens = length * len(active_ranks)
        if slot_idx < num_round_tokens:
            return active_ranks[slot_idx % len(active_ranks)], offset + slot_idx // len(active_ranks)
        slot_idx -= num_round_tokens
        offset += length
        for rank in active_ranks:
            remaining[rank] -= length
    return 0, 0


def _input_fp8_value(rank: int, token: int) -> float:
    return 0.25 + 0.25 * rank + 0.015625 * token


@lru_cache(maxsize=None)
def _input_fp8_byte(rank: int, token: int) -> int:
    value = torch.tensor([_input_fp8_value(rank, token)], dtype=torch.float32)
    return int(value.to(torch.float8_e4m3fn).view(torch.uint8)[0].item())


def _expected_send_counts_for_rank(rank: int) -> list[int]:
    counts = [0 for _ in range(NUM_EXPERTS)]
    for _token in range(NUM_TOKENS):
        for topk in range(NUM_TOPK):
            expert = _route_expert(rank, topk)
            if expert >= 0:
                counts[expert] += 1
    return counts


def _expected_queue_for_destination(dst_rank: int, dst_local_expert: int, src_rank: int) -> list[int]:
    entries: list[int] = []
    for token in range(NUM_TOKENS):
        for topk in range(NUM_TOPK):
            expert = _route_expert(src_rank, topk)
            if expert >= 0 and expert // NUM_EXPERTS_PER_RANK == dst_rank and expert % NUM_EXPERTS_PER_RANK == dst_local_expert:
                entries.append(token * NUM_TOPK + topk)
    return entries


def _validate_dispatch_queue(rank: int, queue: torch.Tensor) -> None:
    for local_expert in range(NUM_EXPERTS_PER_RANK):
        for src_rank in range(NUM_RANKS):
            expected_entries = _expected_queue_for_destination(rank, local_expert, src_rank)
            if not expected_entries:
                continue
            got_entries = queue[local_expert, src_rank, :len(expected_entries)]
            expected_tensor = torch.tensor(expected_entries, dtype=torch.uint32)
            if not torch.equal(got_entries, expected_tensor):
                raise SystemExit(
                    "dispatch queue mismatch: local_expert={} src_rank={} got={} expected={}".format(
                        local_expert,
                        src_rank,
                        got_entries.tolist(),
                        expected_tensor.tolist(),
                    )
                )


def _validate_dispatch_workspace(
    rank: int,
    send_and_recv_count: torch.Tensor,
    recv_sum: torch.Tensor,
    queue: torch.Tensor,
) -> None:
    send_counts = _expected_send_counts_for_rank(rank)
    expected_send = torch.tensor(
        [(1 << 32) | count for count in send_counts],
        dtype=torch.uint64,
    )
    got_send = send_and_recv_count[:NUM_EXPERTS]
    if not torch.equal(got_send, expected_send):
        raise SystemExit(
            "dispatch send_count mismatch: got={} expected={}".format(
                got_send.tolist(), expected_send.tolist()
            )
        )

    expected_recv_rows = torch.tensor(
        [
            _rank_counts_for_destination(rank, local_expert)
            for local_expert in range(NUM_EXPERTS_PER_RANK)
        ],
        dtype=torch.uint64,
    ).T.contiguous()
    got_recv_rows = send_and_recv_count[NUM_EXPERTS:NUM_EXPERTS * 2].reshape(NUM_RANKS, NUM_EXPERTS_PER_RANK)
    if not torch.equal(got_recv_rows, expected_recv_rows):
        raise SystemExit(
            "dispatch recv_count mismatch: got={} expected={}".format(
                got_recv_rows.tolist(), expected_recv_rows.tolist()
            )
        )

    expected_recv_sum = torch.tensor(
        [
            (NUM_RANKS << 32) | sum(_rank_counts_for_destination(rank, local_expert))
            for local_expert in range(NUM_EXPERTS_PER_RANK)
        ],
        dtype=torch.uint64,
    )
    if not torch.equal(recv_sum, expected_recv_sum):
        raise SystemExit(
            "dispatch recv_sum mismatch: got={} expected={}".format(
                recv_sum.tolist(), expected_recv_sum.tolist()
            )
        )

    _validate_dispatch_queue(rank, queue)


def _validate_workspace_cleanup(
    send_and_recv_count: torch.Tensor,
    recv_sum: torch.Tensor,
    arrival: torch.Tensor,
    l2_arrival_mask: torch.Tensor,
) -> None:
    if bool(torch.any(send_and_recv_count != 0).item()):
        raise SystemExit(f"cleanup send/recv count not zero: {send_and_recv_count.tolist()}")
    if bool(torch.any(recv_sum != 0).item()):
        raise SystemExit(f"cleanup recv_sum not zero: {recv_sum.tolist()}")
    if bool(torch.any(arrival != 0).item()):
        raise SystemExit(f"cleanup arrival not zero: nonzero={torch.nonzero(arrival).flatten().tolist()}")
    if bool(torch.any(l2_arrival_mask != 0).item()):
        raise SystemExit(f"cleanup l2_arrival_mask not zero: nonzero={torch.nonzero(l2_arrival_mask).flatten().tolist()}")


def _expected_receive(rank: int) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int]]:
    indices: list[int] = []
    rows: list[torch.Tensor] = []
    sf_rows: list[torch.Tensor] = []
    weight_values: list[float] = []
    meta_values: list[list[int]] = []
    arrival_counts: dict[int, int] = {}
    pool_token_base = 0

    for local_expert in range(NUM_EXPERTS_PER_RANK):
        rank_counts = _rank_counts_for_destination(rank, local_expert)
        total = sum(rank_counts)
        block_offset = pool_token_base // K_BLOCK_M
        for token_idx_in_expert in range(total):
            src_rank, token_idx_in_rank = _choose_rank_round_robin(token_idx_in_expert, rank_counts)
            topks = _topks_for_destination(src_rank, rank, local_expert)
            topk = topks[token_idx_in_rank % len(topks)]
            src_token = token_idx_in_rank // len(topks)
            pool_idx = pool_token_base + token_idx_in_expert
            row = torch.empty((HIDDEN,), dtype=torch.uint8)
            row.fill_(_input_fp8_byte(src_rank, src_token))
            indices.append(pool_idx)
            rows.append(row)
            sf_base = src_rank + 0.125 * (src_token + 1)
            sf_row = torch.tensor(
                [sf_base + 0.01 * sf_idx for sf_idx in range(HIDDEN // 128)],
                dtype=torch.float32,
            )
            sf_rows.append(sf_row)
            weight_values.append(src_rank + 0.25 * (src_token + 1) + 0.03125 * topk)
            meta_values.append([src_rank, src_token, topk])
            arrival_idx = block_offset + token_idx_in_expert // K_BLOCK_M
            arrival_counts[arrival_idx] = arrival_counts.get(arrival_idx, 0) + 1
        pool_token_base += _align(total, K_BLOCK_M)

    expected_rows = torch.stack(rows) if rows else torch.empty((0, HIDDEN), dtype=torch.uint8)
    expected_sf = torch.stack(sf_rows) if sf_rows else torch.empty((0, HIDDEN // 128), dtype=torch.float32)
    expected_weight = torch.tensor(weight_values, dtype=torch.float32)
    expected_meta = torch.tensor(meta_values, dtype=torch.uint32)
    return indices, expected_rows, expected_sf, expected_weight, expected_meta, arrival_counts


def _expected_l1_checksum(expected_rows: torch.Tensor, expected_weight: torch.Tensor) -> int:
    checksum = 2166136261
    first_bytes = expected_rows[:, 0].to(torch.int64).tolist()
    weight_bits = expected_weight.view(torch.int32).to(torch.int64).tolist()
    for byte, bits in zip(first_bytes, weight_bits):
        checksum ^= int(byte) & 0xffffffff
        checksum = (checksum * 16777619) & 0xffffffff
        checksum ^= int(bits) & 0xffffffff
        checksum = (checksum * 16777619) & 0xffffffff
    return checksum


def _expected_l1_weight_checksum(
    rank: int,
    l1_compute_w: torch.Tensor,
    l1_compute_sf: torch.Tensor,
) -> int:
    checksum = 2166136261
    compute_h = _compute_h()
    for local_expert in range(NUM_EXPERTS_PER_RANK):
        rank_counts = _rank_counts_for_destination(rank, local_expert)
        total = sum(rank_counts)
        scale_bits = int(l1_compute_sf[local_expert, 0, 0].view(torch.int32).item()) & 0xffffffff
        for token_idx_in_expert in range(total):
            src_rank, token_idx_in_rank = _choose_rank_round_robin(token_idx_in_expert, rank_counts)
            topks = _topks_for_destination(src_rank, rank, local_expert)
            src_token = token_idx_in_rank // len(topks)
            token_byte = _input_fp8_byte(src_rank, src_token)
            weighted = 0
            for h in range(compute_h):
                weighted = (weighted + token_byte * int(l1_compute_w[local_expert, 0, h].item())) & 0xffffffff
            checksum ^= weighted
            checksum = (checksum * 16777619) & 0xffffffff
            checksum ^= scale_bits
            checksum = (checksum * 16777619) & 0xffffffff
    return checksum


def _fp8_e4m3fn_byte_to_float(value: int) -> float:
    value &= 0xff
    if value == 0:
        return 0.0
    sign = -1.0 if value & 0x80 else 1.0
    exp = (value >> 3) & 0x0f
    mant = value & 0x07
    if exp == 0:
        mag = math.ldexp(mant / 8.0, -6)
    else:
        mag = math.ldexp(1.0 + mant / 8.0, exp - 7)
    return sign * mag


def _topk_weight_value(src_rank: int, src_token: int, src_topk: int) -> float:
    return src_rank + 0.25 * (src_token + 1) + 0.03125 * src_topk


def _compute_h() -> int:
    return HIDDEN if COMPUTE_FULL_HIDDEN else min(HIDDEN, 32)


def _input_sf_value(src_rank: int, src_token: int, h: int) -> float:
    return src_rank + 0.125 * (src_token + 1) + 0.01 * (h // 128)


def _l1_weight_sf_value(l1_compute_sf: torch.Tensor, local_expert: int, row: int, h: int) -> float:
    return float(l1_compute_sf[local_expert, row // 128, h // 128].item())


def _expected_l1_scalar_sum(
    rank: int,
    l1_compute_w: torch.Tensor,
    l1_compute_sf: torch.Tensor,
) -> float:
    scalar_sum = 0.0
    compute_h = _compute_h()
    compute_i = INTERMEDIATE_HIDDEN
    for local_expert in range(NUM_EXPERTS_PER_RANK):
        rank_counts = _rank_counts_for_destination(rank, local_expert)
        total = sum(rank_counts)
        for token_idx_in_expert in range(total):
            src_rank, token_idx_in_rank = _choose_rank_round_robin(token_idx_in_expert, rank_counts)
            topks = _topks_for_destination(src_rank, rank, local_expert)
            src_token = token_idx_in_rank // len(topks)
            src_topk = topks[token_idx_in_rank % len(topks)]
            topk_weight = _topk_weight_value(src_rank, src_token, src_topk)
            token_raw = _input_fp8_byte(src_rank, src_token)
            for ii in range(compute_i):
                group = ii // 8
                lane = ii - group * 8
                gate_row = group * 16 + lane
                up_row = gate_row + 8
                gate_sf = _l1_weight_sf_value(l1_compute_sf, local_expert, ii, 0)
                up_sf = _l1_weight_sf_value(l1_compute_sf, local_expert, INTERMEDIATE_HIDDEN + ii, 0)
                gate_acc = 0.0
                up_acc = 0.0
                for h in range(compute_h):
                    token_value = _fp8_e4m3fn_byte_to_float(token_raw) * _input_sf_value(src_rank, src_token, h)
                    gate_sf_h = (
                        _l1_weight_sf_value(l1_compute_sf, local_expert, ii, h)
                        if COMPUTE_FULL_HIDDEN
                        else gate_sf
                    )
                    up_sf_h = (
                        _l1_weight_sf_value(l1_compute_sf, local_expert, INTERMEDIATE_HIDDEN + ii, h)
                        if COMPUTE_FULL_HIDDEN
                        else up_sf
                    )
                    gate_acc += (
                        token_value
                        * _fp8_e4m3fn_byte_to_float(int(l1_compute_w[local_expert, gate_row, h].item()))
                        * gate_sf_h
                    )
                    up_acc += (
                        token_value
                        * _fp8_e4m3fn_byte_to_float(int(l1_compute_w[local_expert, up_row, h].item()))
                        * up_sf_h
                    )
                scalar_sum += gate_acc * (1.0 / (1.0 + math.exp(-gate_acc))) * up_acc * topk_weight
    return scalar_sum


def _expected_l2_token_floats(
    rank: int,
    l1_compute_w: torch.Tensor,
    l1_compute_sf: torch.Tensor,
) -> torch.Tensor:
    rows: list[list[float]] = []
    compute_h = _compute_h()
    compute_i = INTERMEDIATE_HIDDEN
    for local_expert in range(NUM_EXPERTS_PER_RANK):
        rank_counts = _rank_counts_for_destination(rank, local_expert)
        total = sum(rank_counts)
        for token_idx_in_expert in range(total):
            src_rank, token_idx_in_rank = _choose_rank_round_robin(token_idx_in_expert, rank_counts)
            topks = _topks_for_destination(src_rank, rank, local_expert)
            src_token = token_idx_in_rank // len(topks)
            src_topk = topks[token_idx_in_rank % len(topks)]
            topk_weight = _topk_weight_value(src_rank, src_token, src_topk)
            token_raw = _input_fp8_byte(src_rank, src_token)
            row: list[float] = []
            for ii in range(compute_i):
                group = ii // 8
                lane = ii - group * 8
                gate_row = group * 16 + lane
                up_row = gate_row + 8
                gate_sf = _l1_weight_sf_value(l1_compute_sf, local_expert, ii, 0)
                up_sf = _l1_weight_sf_value(l1_compute_sf, local_expert, INTERMEDIATE_HIDDEN + ii, 0)
                gate_acc = 0.0
                up_acc = 0.0
                for h in range(compute_h):
                    token_value = _fp8_e4m3fn_byte_to_float(token_raw) * _input_sf_value(src_rank, src_token, h)
                    gate_sf_h = (
                        _l1_weight_sf_value(l1_compute_sf, local_expert, ii, h)
                        if COMPUTE_FULL_HIDDEN
                        else gate_sf
                    )
                    up_sf_h = (
                        _l1_weight_sf_value(l1_compute_sf, local_expert, INTERMEDIATE_HIDDEN + ii, h)
                        if COMPUTE_FULL_HIDDEN
                        else up_sf
                    )
                    gate_acc += (
                        token_value
                        * _fp8_e4m3fn_byte_to_float(int(l1_compute_w[local_expert, gate_row, h].item()))
                        * gate_sf_h
                    )
                    up_acc += (
                        token_value
                        * _fp8_e4m3fn_byte_to_float(int(l1_compute_w[local_expert, up_row, h].item()))
                        * up_sf_h
                    )
                row.append(gate_acc * (1.0 / (1.0 + math.exp(-gate_acc))) * up_acc * topk_weight)
            rows.append(row)
    if not rows:
        return torch.empty((0, INTERMEDIATE_HIDDEN), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def _expected_l2_sf_and_scaled_floats(l2_float: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if l2_float.numel() == 0:
        return (
            torch.empty((0, INTERMEDIATE_HIDDEN // 64), dtype=torch.float32),
            torch.empty((0, INTERMEDIATE_HIDDEN), dtype=torch.float32),
        )
    grouped = l2_float.reshape(l2_float.shape[0], INTERMEDIATE_HIDDEN // 64, 64)
    max_abs = grouped.abs().amax(dim=2)
    scale = torch.where(max_abs > 0.0, max_abs / 448.0, torch.ones_like(max_abs))
    scaled = (grouped / scale[:, :, None]).reshape_as(l2_float)
    return scale.contiguous(), scaled.contiguous()


def _float_to_cuda_satfinite_e4m3_bytes(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    bytes_ = values.to(device=device).to(torch.float8_e4m3fn).view(torch.uint8).detach().cpu()
    bytes_[bytes_ == 0x7F] = 0x7E
    bytes_[bytes_ == 0xFF] = 0xFE
    return bytes_


def _expected_l2_float_for_route(
    src_rank: int,
    src_token: int,
    src_topk: int,
    local_expert: int,
    l1_compute_w: torch.Tensor,
    l1_compute_sf: torch.Tensor,
) -> torch.Tensor:
    compute_h = _compute_h()
    topk_weight = _topk_weight_value(src_rank, src_token, src_topk)
    token_raw = _input_fp8_byte(src_rank, src_token)
    row: list[float] = []
    for ii in range(INTERMEDIATE_HIDDEN):
        group = ii // 8
        lane = ii - group * 8
        gate_row = group * 16 + lane
        up_row = gate_row + 8
        gate_sf = _l1_weight_sf_value(l1_compute_sf, local_expert, ii, 0)
        up_sf = _l1_weight_sf_value(l1_compute_sf, local_expert, INTERMEDIATE_HIDDEN + ii, 0)
        gate_acc = 0.0
        up_acc = 0.0
        for h in range(compute_h):
            token_value = _fp8_e4m3fn_byte_to_float(token_raw) * _input_sf_value(src_rank, src_token, h)
            gate_sf_h = (
                _l1_weight_sf_value(l1_compute_sf, local_expert, ii, h)
                if COMPUTE_FULL_HIDDEN
                else gate_sf
            )
            up_sf_h = (
                _l1_weight_sf_value(l1_compute_sf, local_expert, INTERMEDIATE_HIDDEN + ii, h)
                if COMPUTE_FULL_HIDDEN
                else up_sf
            )
            gate_acc += (
                token_value
                * _fp8_e4m3fn_byte_to_float(int(l1_compute_w[local_expert, gate_row, h].item()))
                * gate_sf_h
            )
            up_acc += (
                token_value
                * _fp8_e4m3fn_byte_to_float(int(l1_compute_w[local_expert, up_row, h].item()))
                * up_sf_h
            )
        row.append(gate_acc * (1.0 / (1.0 + math.exp(-gate_acc))) * up_acc * topk_weight)
    return torch.tensor(row, dtype=torch.float32)


def _expected_combine_float(
    rank: int,
    l1_compute_w: torch.Tensor,
    l1_compute_sf: torch.Tensor,
    l2_compute_w: torch.Tensor,
    l2_compute_sf: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    expected = torch.zeros((NUM_TOPK, NUM_MAX_TOKENS_PER_RANK, HIDDEN), dtype=torch.float32)
    for token in range(NUM_TOKENS):
        for topk in range(NUM_TOPK):
            expert = _route_expert(rank, topk)
            if expert < 0:
                continue
            local_expert = expert % NUM_EXPERTS_PER_RANK
            l2_float = _expected_l2_float_for_route(
                rank,
                token,
                topk,
                local_expert,
                l1_compute_w,
                l1_compute_sf,
            ).unsqueeze(0)
            l2_sf_row, l2_scaled = _expected_l2_sf_and_scaled_floats(l2_float)
            l2_bytes = _float_to_cuda_satfinite_e4m3_bytes(l2_scaled, device)[0]
            l2_sf_values = l2_sf_row[0]
            for h in range(HIDDEN):
                acc = 0.0
                for ii in range(INTERMEDIATE_HIDDEN):
                    act = _fp8_e4m3fn_byte_to_float(int(l2_bytes[ii].item())) * float(
                        l2_sf_values[ii // 64].item()
                    )
                    w = _fp8_e4m3fn_byte_to_float(int(l2_compute_w[local_expert, h, ii].item()))
                    sf = float(l2_compute_sf[local_expert, h // 128, ii // 128].item())
                    acc += act * w * sf
                expected[topk, token, h] = acc
    return expected


@dialect(
    name="cuda",
    compiler="nvcc",
    file=HERE / "ws_userhopper_dispatch_receiver_device.cu",
    extern=HERE / "ws_userhopper_dispatch_receiver_extern_call.py",
    extern_func_name="userhopper_ws_dispatch_partition",
    libs={"nvshmem": NVSHMEM_HOME},
    links=["nvshmem_device"],
)
def edsl_userhopper_ws_dispatch(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    compiler="nvcc",
    file=HERE / "ws_userhopper_dispatch_receiver_device.cu",
    extern=HERE / "ws_userhopper_dispatch_receiver_extern_call.py",
    extern_func_name="userhopper_ws_dispatch_partition_cta_warp0",
    libs={"nvshmem": NVSHMEM_HOME},
    links=["nvshmem_device"],
)
def edsl_userhopper_ws_dispatch_cta_warp0(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    compiler="nvcc",
    file=HERE / "ws_userhopper_dispatch_receiver_device.cu",
    extern=HERE / "ws_userhopper_dispatch_receiver_extern_call.py",
    extern_func_name="userhopper_ws_dispatch_partition_cta_multiwarp",
    libs={"nvshmem": NVSHMEM_HOME},
    links=["nvshmem_device"],
)
def edsl_userhopper_ws_dispatch_cta_multiwarp(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    compiler="nvcc",
    file=HERE / "ws_userhopper_dispatch_receiver_device.cu",
    extern=HERE / "ws_userhopper_dispatch_receiver_extern_call.py",
    extern_func_name="userhopper_ws_receiver_partition",
    libs={"nvshmem": NVSHMEM_HOME},
    links=["nvshmem_device"],
)
def edsl_userhopper_ws_receiver(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    compiler="nvcc",
    file=HERE / "ws_userhopper_dispatch_receiver_device.cu",
    extern=HERE / "ws_userhopper_dispatch_receiver_extern_call.py",
    extern_func_name="userhopper_ws_receiver_partition_bounded",
    libs={"nvshmem": NVSHMEM_HOME},
    links=["nvshmem_device"],
)
def edsl_userhopper_ws_receiver_bounded(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    compiler="nvcc",
    file=HERE / "ws_userhopper_dispatch_receiver_device.cu",
    extern=HERE / "ws_userhopper_dispatch_receiver_extern_call.py",
    extern_func_name="userhopper_ws_compute_stub_partition",
    libs={"nvshmem": NVSHMEM_HOME},
    links=["nvshmem_device"],
)
def edsl_userhopper_ws_compute_stub(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    compiler="nvcc",
    file=HERE / "ws_userhopper_dispatch_receiver_device.cu",
    extern=HERE / "ws_userhopper_dispatch_receiver_extern_call.py",
    extern_func_name="userhopper_ws_combine_reduce_partition",
    libs={"nvshmem": NVSHMEM_HOME},
    links=["nvshmem_device"],
)
def edsl_userhopper_ws_combine_reduce(*args, **kwargs):
    ...


@triton.jit
def _stage_default_partition(marker, VALUE: tl.constexpr):
    tl.store(marker, VALUE)


@triton.jit
def _dispatch_default_partition(
    symm_buffer,
    marker,
    NUM_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
):
    tle_raw.call(
        edsl_userhopper_ws_dispatch,
        [
            symm_buffer,
            NUM_TOKENS_C,
            NUM_RANKS_C,
            NUM_EXPERTS_C,
            NUM_MAX_TOKENS_PER_RANK_C,
            NUM_TOPK_C,
            HIDDEN_C,
        ],
    )
    tl.store(marker, 0x4452)


@triton.jit
def _dispatch_pipe_partition(
    sync_writer,
    symm_buffer,
    marker,
    NUM_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    NUM_DISPATCH_WARPS_C: tl.constexpr,
):
    sync_slot = sync_writer.acquire(0)
    if NUM_DISPATCH_WARPS_C == 1:
        tle_raw.call(
            edsl_userhopper_ws_dispatch_cta_warp0,
            [
                symm_buffer,
                NUM_TOKENS_C,
                NUM_RANKS_C,
                NUM_EXPERTS_C,
                NUM_MAX_TOKENS_PER_RANK_C,
                NUM_TOPK_C,
                HIDDEN_C,
            ],
        )
    else:
        tle_raw.call(
            edsl_userhopper_ws_dispatch_cta_multiwarp,
            [
                symm_buffer,
                NUM_TOKENS_C,
                NUM_RANKS_C,
                NUM_EXPERTS_C,
                NUM_MAX_TOKENS_PER_RANK_C,
                NUM_TOPK_C,
                HIDDEN_C,
                NUM_DISPATCH_WARPS_C,
            ],
        )
    tl.store(tle.gpu.local_ptr(sync_slot.done, (0,)), 1)
    tl.store(marker, 0x5100)
    sync_writer.commit(0)


@triton.jit
def _receiver_partition(
    symm_buffer,
    NUM_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
):
    tle_raw.call(
        edsl_userhopper_ws_receiver,
        [
            symm_buffer,
            NUM_TOKENS_C,
            NUM_RANKS_C,
            NUM_EXPERTS_C,
            NUM_MAX_TOKENS_PER_RANK_C,
            NUM_TOPK_C,
            HIDDEN_C,
            NUM_PADDED_SF_POOL_TOKENS_C,
        ],
    )


@triton.jit
def _receiver_pipe_partition(
    sync_reader,
    symm_buffer,
    NUM_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
):
    wait_result = sync_reader.wait(0)
    _ = tl.load(tle.gpu.local_ptr(wait_result.slot.done, (0,)))
    tle_raw.call(
        edsl_userhopper_ws_receiver,
        [
            symm_buffer,
            NUM_TOKENS_C,
            NUM_RANKS_C,
            NUM_EXPERTS_C,
            NUM_MAX_TOKENS_PER_RANK_C,
            NUM_TOPK_C,
            HIDDEN_C,
            NUM_PADDED_SF_POOL_TOKENS_C,
        ],
    )
    sync_reader.release(0)


@triton.jit
def _receiver_pipe_to_compute_partition(
    dispatch_reader,
    compute_writer,
    symm_buffer,
    NUM_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
):
    wait_result = dispatch_reader.wait(0)
    _ = tl.load(tle.gpu.local_ptr(wait_result.slot.done, (0,)))
    tle_raw.call(
        edsl_userhopper_ws_receiver,
        [
            symm_buffer,
            NUM_TOKENS_C,
            NUM_RANKS_C,
            NUM_EXPERTS_C,
            NUM_MAX_TOKENS_PER_RANK_C,
            NUM_TOPK_C,
            HIDDEN_C,
            NUM_PADDED_SF_POOL_TOKENS_C,
        ],
    )
    compute_slot = compute_writer.acquire(0)
    tl.store(tle.gpu.local_ptr(compute_slot.done, (0,)), 1)
    compute_writer.commit(0)
    dispatch_reader.release(0)


@triton.jit
def _compute_stub_pipe_partition(
    compute_reader,
    combine_writer,
    symm_buffer,
    l1_weights,
    l1_weights_sf,
    l2_weights,
    l2_weights_sf,
    marker,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    INTERMEDIATE_HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
    COMPUTE_FULL_HIDDEN_C: tl.constexpr,
    COMPUTE_PARALLEL_C: tl.constexpr,
    COMPUTE_WORKER_WARPS_C: tl.constexpr,
):
    wait_result = compute_reader.wait(0)
    _ = tl.load(tle.gpu.local_ptr(wait_result.slot.done, (0,)))
    tle_raw.call(
        edsl_userhopper_ws_compute_stub,
        [
            symm_buffer,
            l1_weights,
            l1_weights_sf,
            l2_weights,
            l2_weights_sf,
            NUM_RANKS_C,
            NUM_EXPERTS_C,
            NUM_MAX_TOKENS_PER_RANK_C,
            NUM_TOPK_C,
            HIDDEN_C,
            INTERMEDIATE_HIDDEN_C,
            NUM_PADDED_SF_POOL_TOKENS_C,
            COMPUTE_FULL_HIDDEN_C,
            COMPUTE_PARALLEL_C,
            COMPUTE_WORKER_WARPS_C,
        ],
    )
    combine_slot = combine_writer.acquire(0)
    tl.store(tle.gpu.local_ptr(combine_slot.done, (0,)), 1)
    combine_writer.commit(0)
    tl.store(marker, 0x5300)
    compute_reader.release(0)


@triton.jit
def _combine_reduce_pipe_partition(
    combine_reader,
    symm_buffer,
    y,
    marker,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    INTERMEDIATE_HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
    CLEANUP_WORKSPACE_C: tl.constexpr,
):
    wait_result = combine_reader.wait(0)
    _ = tl.load(tle.gpu.local_ptr(wait_result.slot.done, (0,)))
    tle_raw.call(
        edsl_userhopper_ws_combine_reduce,
        [
            symm_buffer,
            y,
            NUM_RANKS_C,
            NUM_EXPERTS_C,
            NUM_MAX_TOKENS_PER_RANK_C,
            NUM_TOPK_C,
            HIDDEN_C,
            INTERMEDIATE_HIDDEN_C,
            NUM_PADDED_SF_POOL_TOKENS_C,
            CLEANUP_WORKSPACE_C,
        ],
    )
    tl.store(marker, 0x5400)
    combine_reader.release(0)


@triton.jit
def _receiver_pipe_debug_partition(
    sync_reader,
    symm_buffer,
    marker,
    NUM_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
):
    wait_result = sync_reader.wait(0)
    _ = tl.load(tle.gpu.local_ptr(wait_result.slot.done, (0,)))
    tle_raw.call(
        edsl_userhopper_ws_receiver_bounded,
        [
            symm_buffer,
            NUM_TOKENS_C,
            NUM_RANKS_C,
            NUM_EXPERTS_C,
            NUM_MAX_TOKENS_PER_RANK_C,
            NUM_TOPK_C,
            HIDDEN_C,
            NUM_PADDED_SF_POOL_TOKENS_C,
        ],
    )
    tl.store(marker, 0x510D)
    sync_reader.release(0)


@triton.jit
def _ws_dispatch_stage_kernel(
    symm_buffer,
    marker,
    NUM_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
):
    tle.gpu.warp_specialize(
        [
            (_stage_default_partition, (marker, 0xD100)),
            (
                _dispatch_default_partition,
                (
                    symm_buffer,
                    marker,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_DISPATCH_WARPS_C,
                ),
            ),
        ],
        [1],
        [40],
    )


@triton.jit
def _ws_receiver_stage_kernel(
    symm_buffer,
    marker,
    NUM_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
):
    tle.gpu.warp_specialize(
        [
            (_stage_default_partition, (marker, 0xD200)),
            (
                _receiver_partition,
                (
                    symm_buffer,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                ),
            ),
        ],
        [1],
        [40],
    )


@triton.jit
def _ws_single_pipe_dispatch_receiver_kernel(
    symm_buffer,
    marker,
    NUM_TOKENS_C: tl.constexpr,
    EXPECTED_LOCAL_RECV_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
    NUM_DISPATCH_WARPS_C: tl.constexpr,
):
    sync_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    sync_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="userhopper_dispatch_receiver_sync",
        done=sync_done,
    )
    sync_writer = sync_pipe.writer()
    sync_reader = sync_pipe.reader()
    tle.gpu.warp_specialize(
        [
            (
                _dispatch_pipe_partition,
                (
                    sync_writer,
                    symm_buffer,
                    marker,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_DISPATCH_WARPS_C,
                ),
            ),
            (
                _receiver_pipe_partition,
                (
                    sync_reader,
                    symm_buffer,
                    EXPECTED_LOCAL_RECV_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                ),
            ),
        ],
        [1],
        [40],
    )


@triton.jit
def _ws_single_pipe_compute_stub_dispatch_receiver_kernel(
    symm_buffer,
    y,
    l1_weights,
    l1_weights_sf,
    l2_weights,
    l2_weights_sf,
    marker,
    NUM_TOKENS_C: tl.constexpr,
    EXPECTED_LOCAL_RECV_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    INTERMEDIATE_HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
    CLEANUP_WORKSPACE_C: tl.constexpr,
    NUM_DISPATCH_WARPS_C: tl.constexpr,
    COMPUTE_FULL_HIDDEN_C: tl.constexpr,
    COMPUTE_PARALLEL_C: tl.constexpr,
    COMPUTE_WORKER_WARPS_C: tl.constexpr,
):
    dispatch_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    compute_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    combine_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    dispatch_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="userhopper_dispatch_receiver_compute_dispatch_sync",
        done=dispatch_done,
    )
    compute_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="userhopper_dispatch_receiver_compute_stub_sync",
        done=compute_done,
    )
    combine_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="userhopper_dispatch_receiver_combine_sync",
        done=combine_done,
    )
    dispatch_writer = dispatch_pipe.writer()
    dispatch_reader = dispatch_pipe.reader()
    compute_writer = compute_pipe.writer()
    compute_reader = compute_pipe.reader()
    combine_writer = combine_pipe.writer()
    combine_reader = combine_pipe.reader()
    tle.gpu.warp_specialize(
        [
            (
                _dispatch_pipe_partition,
                (
                    dispatch_writer,
                    symm_buffer,
                    marker,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_DISPATCH_WARPS_C,
                ),
            ),
            (
                _receiver_pipe_to_compute_partition,
                (
                    dispatch_reader,
                    compute_writer,
                    symm_buffer,
                    EXPECTED_LOCAL_RECV_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                ),
            ),
            (
                _compute_stub_pipe_partition,
                (
                    compute_reader,
                    combine_writer,
                    symm_buffer,
                    l1_weights,
                    l1_weights_sf,
                    l2_weights,
                    l2_weights_sf,
                    marker,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    INTERMEDIATE_HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                    COMPUTE_FULL_HIDDEN_C,
                    COMPUTE_PARALLEL_C,
                    COMPUTE_WORKER_WARPS_C,
                ),
            ),
            (
                _combine_reduce_pipe_partition,
                (
                    combine_reader,
                    symm_buffer,
                    y,
                    marker,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    INTERMEDIATE_HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                    CLEANUP_WORKSPACE_C,
                ),
            ),
        ],
        [1, COMPUTE_WORKER_WARPS_C, 1],
        [40, 40, 40],
    )


@triton.jit
def _ws_single_pipe_debug_dispatch_receiver_kernel(
    symm_buffer,
    marker,
    NUM_TOKENS_C: tl.constexpr,
    EXPECTED_LOCAL_RECV_TOKENS_C: tl.constexpr,
    NUM_RANKS_C: tl.constexpr,
    NUM_EXPERTS_C: tl.constexpr,
    NUM_MAX_TOKENS_PER_RANK_C: tl.constexpr,
    NUM_TOPK_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    NUM_PADDED_SF_POOL_TOKENS_C: tl.constexpr,
    NUM_DISPATCH_WARPS_C: tl.constexpr,
):
    sync_done = tle.gpu.alloc(
        [1, 1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    sync_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="userhopper_dispatch_receiver_debug_sync",
        done=sync_done,
    )
    sync_writer = sync_pipe.writer()
    sync_reader = sync_pipe.reader()
    tle.gpu.warp_specialize(
        [
            (
                _dispatch_pipe_partition,
                (
                    sync_writer,
                    symm_buffer,
                    marker,
                    NUM_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_DISPATCH_WARPS_C,
                ),
            ),
            (
                _receiver_pipe_debug_partition,
                (
                    sync_reader,
                    symm_buffer,
                    marker,
                    EXPECTED_LOCAL_RECV_TOKENS_C,
                    NUM_RANKS_C,
                    NUM_EXPERTS_C,
                    NUM_MAX_TOKENS_PER_RANK_C,
                    NUM_TOPK_C,
                    HIDDEN_C,
                    NUM_PADDED_SF_POOL_TOKENS_C,
                ),
            ),
        ],
        [1],
        [40],
    )


def _nvcc_path() -> str:
    explicit = os.environ.get("NVCC")
    if explicit:
        return explicit
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8")
    candidate = Path(cuda_home) / "bin" / "nvcc"
    return str(candidate if candidate.exists() else "nvcc")


def _compile_host_library(arch: str) -> Path:
    src = HERE / "ws_userhopper_dispatch_receiver_host.cu"
    out = src.with_suffix(".so")
    if out.exists() and out.stat().st_mtime_ns >= src.stat().st_mtime_ns:
        return out
    cmd = [
        _nvcc_path(),
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-rdc=true",
        f"-arch={arch}",
        f"-I{NVSHMEM_HOME}/include",
        f"-L{NVSHMEM_HOME}/lib",
        "-lnvshmem_host",
        "-lnvshmem_device",
        "-o",
        str(out),
        str(src),
    ]
    subprocess.run(cmd, check=True)
    return out


def _tensor_from_pointer(ptr: int, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device):
    elem_size = torch.empty((), dtype=dtype).element_size()
    numel = 1
    for dim in shape:
        numel *= dim
    storage = torch._C._construct_storage_from_data_pointer(ptr, device, elem_size * numel)
    return torch.empty(0, dtype=dtype, device=device).set_(storage).view(*shape)


def _view(ptr: ctypes.c_void_p, offset: int, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device):
    return _tensor_from_pointer(ptr.value + offset, shape, dtype, device)


def _view_strided(
    ptr: ctypes.c_void_p,
    offset: int,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
):
    if len(shape) != len(stride):
        raise ValueError(f"shape/stride rank mismatch: shape={shape}, stride={stride}")
    storage_numel = 1
    if shape:
        storage_numel = 1 + sum((dim - 1) * step for dim, step in zip(shape, stride))
    base = _tensor_from_pointer(ptr.value + offset, (storage_numel,), dtype, device)
    return torch.as_strided(base, shape, stride)


def _install_cumodule_hook(lib) -> None:
    def cumodule_init_hook(*args, **kwargs):
        key = kwargs["key"]
        jit_function = kwargs["fn"].jit_function
        dev = kwargs["compile"]["device"]
        kernel_cache = jit_function.device_caches[dev][0]
        kernel = kernel_cache.get(key, None)
        assert kernel is not None
        kernel._init_handles()
        ret = lib.userhopper_ws_nvshmemx_cumodule_init_wrapper(ctypes.c_void_p(kernel.module))
        assert ret == 0, f"nvshmemx_cumodule_init_wrapper failed: {ret}"

    knobs.runtime.jit_post_compile_hook = cumodule_init_hook


def _setup_lib():
    lib_path = _compile_host_library(os.environ.get("TRITON_WS_NVSHMEM_ARCH", "sm_90"))
    lib = ctypes.CDLL(str(lib_path))
    lib.userhopper_ws_nvshmem_init_wrapper.argtypes = []
    lib.userhopper_ws_nvshmem_init_wrapper.restype = None
    lib.userhopper_ws_nvshmem_team_mype_wrapper.argtypes = []
    lib.userhopper_ws_nvshmem_team_mype_wrapper.restype = ctypes.c_int
    lib.userhopper_ws_nvshmem_n_pes_wrapper.argtypes = []
    lib.userhopper_ws_nvshmem_n_pes_wrapper.restype = ctypes.c_int
    lib.userhopper_ws_nvshmem_alloc_bytes_wrapper.argtypes = [ctypes.c_longlong]
    lib.userhopper_ws_nvshmem_alloc_bytes_wrapper.restype = ctypes.c_void_p
    lib.userhopper_ws_nvshmem_barrier_all_wrapper.argtypes = []
    lib.userhopper_ws_nvshmem_barrier_all_wrapper.restype = None
    lib.userhopper_ws_nvshmem_finalize_wrapper.argtypes = [ctypes.c_void_p]
    lib.userhopper_ws_nvshmem_finalize_wrapper.restype = None
    return lib


def main() -> None:
    lib = _setup_lib()
    lib.userhopper_ws_nvshmem_init_wrapper()
    rank = lib.userhopper_ws_nvshmem_team_mype_wrapper()
    npes = lib.userhopper_ws_nvshmem_n_pes_wrapper()
    if npes != NUM_RANKS:
        raise RuntimeError(f"this smoke expects exactly {NUM_RANKS} PEs, got {npes}")

    torch.cuda.set_device(rank)
    device = triton.runtime.driver.active.get_active_torch_device()
    stream = torch.cuda.Stream(device=device)
    ptr = ctypes.c_void_p(lib.userhopper_ws_nvshmem_alloc_bytes_wrapper(LAYOUT["total_bytes"]))

    whole = _view(ptr, 0, (LAYOUT["total_bytes"],), torch.uint8, device)
    x = _view(ptr, LAYOUT["input_token"], (NUM_MAX_TOKENS_PER_RANK, HIDDEN), torch.uint8, device)
    x_fp8 = _view(ptr, LAYOUT["input_token"], (NUM_MAX_TOKENS_PER_RANK, HIDDEN), torch.float8_e4m3fn, device)
    x_sf = _view(ptr, LAYOUT["input_sf"], (NUM_MAX_TOKENS_PER_RANK, HIDDEN // 128), torch.float32, device)
    topk_idx = _view(ptr, LAYOUT["input_topk_idx"], (NUM_MAX_TOKENS_PER_RANK, NUM_TOPK), torch.int64, device)
    topk_weights = _view(
        ptr, LAYOUT["input_topk_weight"], (NUM_MAX_TOKENS_PER_RANK, NUM_TOPK), torch.float32, device
    )
    l1 = _view(ptr, LAYOUT["l1_token"], (LAYOUT["num_max_pool_tokens"], HIDDEN), torch.uint8, device)
    l1_fp8 = _view(ptr, LAYOUT["l1_token"], (LAYOUT["num_max_pool_tokens"], HIDDEN), torch.float8_e4m3fn, device)
    l1_sf = _view_strided(
        ptr,
        LAYOUT["l1_sf"],
        (LAYOUT["num_max_padded_sf_pool_tokens"], HIDDEN // 128),
        (1, LAYOUT["num_max_padded_sf_pool_tokens"]),
        torch.float32,
        device,
    )
    l1_weight = _view(ptr, LAYOUT["l1_topk_weight"], (LAYOUT["num_max_pool_tokens"],), torch.float32, device)
    l2_token = _view(
        ptr,
        LAYOUT["l2_token"],
        (LAYOUT["num_max_pool_tokens"], INTERMEDIATE_HIDDEN),
        torch.uint8,
        device,
    )
    l2_sf = _view_strided(
        ptr,
        LAYOUT["l2_sf"],
        (LAYOUT["num_max_padded_sf_pool_tokens"], INTERMEDIATE_HIDDEN // 64),
        (1, LAYOUT["num_max_padded_sf_pool_tokens"]),
        torch.float32,
        device,
    )
    combine = _view(
        ptr,
        LAYOUT["combine_token"],
        (NUM_TOPK, NUM_MAX_TOKENS_PER_RANK, HIDDEN),
        torch.uint16,
        device,
    )
    l2_debug_offset = LAYOUT["l2_token"] + LAYOUT["num_max_pool_tokens"] * INTERMEDIATE_HIDDEN - 32
    l2_debug = _view(ptr, l2_debug_offset, (8,), torch.uint32, device)
    l1_compute_w = torch.empty(
        (NUM_EXPERTS_PER_RANK, 2 * INTERMEDIATE_HIDDEN, HIDDEN),
        dtype=torch.uint8,
        device=device,
    )
    l1_compute_w_fp8 = l1_compute_w.view(torch.float8_e4m3fn)
    l1_compute_sf = torch.empty(
        (NUM_EXPERTS_PER_RANK, 2 * INTERMEDIATE_HIDDEN // 128, HIDDEN // 128),
        dtype=torch.float32,
        device=device,
    )
    l2_compute_w = torch.empty(
        (NUM_EXPERTS_PER_RANK, HIDDEN, INTERMEDIATE_HIDDEN),
        dtype=torch.uint8,
        device=device,
    )
    l2_compute_w_fp8 = l2_compute_w.view(torch.float8_e4m3fn)
    l2_compute_sf = torch.empty(
        (NUM_EXPERTS_PER_RANK, HIDDEN // 128, INTERMEDIATE_HIDDEN // 128),
        dtype=torch.float32,
        device=device,
    )
    y = torch.empty((NUM_MAX_TOKENS_PER_RANK, HIDDEN), dtype=torch.bfloat16, device=device)
    y_u8 = y.view(torch.uint8)
    marker = torch.empty((1,), dtype=torch.int32, device=device)

    metadata_offset = (
        32
        + NUM_EXPERTS * 8 * 2
        + (NUM_EXPERTS // NUM_RANKS) * 8
        + _align(LAYOUT["num_max_pool_blocks"], 2) * 4
        + LAYOUT["num_max_pool_blocks"] * 8
        + (NUM_EXPERTS // NUM_RANKS) * NUM_RANKS * (NUM_RANKS * NUM_MAX_TOKENS_PER_RANK) * 4
    )
    metadata = _view(ptr, metadata_offset, (LAYOUT["num_max_pool_tokens"], 3), torch.uint32, device)
    arrival_offset = 32 + NUM_EXPERTS * 8 * 2 + (NUM_EXPERTS // NUM_RANKS) * 8
    arrival = _view(ptr, arrival_offset, (_align(LAYOUT["num_max_pool_blocks"], 2),), torch.uint32, device)
    l2_arrival_mask_offset = arrival_offset + _align(LAYOUT["num_max_pool_blocks"], 2) * 4
    l2_arrival_mask = _view(ptr, l2_arrival_mask_offset, (LAYOUT["num_max_pool_blocks"],), torch.uint64, device)
    queue_offset = (
        32
        + NUM_EXPERTS * 8 * 2
        + (NUM_EXPERTS // NUM_RANKS) * 8
        + _align(LAYOUT["num_max_pool_blocks"], 2) * 4
        + LAYOUT["num_max_pool_blocks"] * 8
    )
    queue = _view(
        ptr,
        queue_offset,
        (NUM_EXPERTS // NUM_RANKS, NUM_RANKS, NUM_RANKS * NUM_MAX_TOKENS_PER_RANK),
        torch.uint32,
        device,
    )
    send_count = _view(ptr, 32, (NUM_EXPERTS * 2,), torch.uint64, device)
    recv_sum = _view(ptr, 32 + NUM_EXPERTS * 8 * 2, (NUM_EXPERTS // NUM_RANKS,), torch.uint64, device)
    debug_words = _view(ptr, 0, (8,), torch.uint32, device)
    expected_count_words = _view(ptr, 0, (NUM_EXPERTS_PER_RANK,), torch.uint32, device)

    try:
        with torch.cuda.stream(stream):
            whole.zero_()
            marker.zero_()
            y.zero_()
            l1_weight_values = (
                0.03125
                * (
                    1.0
                    + (
                        torch.arange(l1_compute_w.numel(), dtype=torch.float32, device=device)
                        % 7.0
                    )
                )
            ).reshape_as(l1_compute_w_fp8)
            l1_compute_w_fp8.copy_(l1_weight_values)
            l1_compute_sf.copy_(
                (
                    1.0
                    + 0.125
                    * torch.arange(l1_compute_sf.numel(), dtype=torch.float32, device=device)
                ).reshape_as(l1_compute_sf)
            )
            l2_weight_values = (
                0.015625
                * (
                    1.0
                    + (
                        torch.arange(l2_compute_w.numel(), dtype=torch.float32, device=device)
                        % 11.0
                    )
                )
            ).reshape_as(l2_compute_w_fp8)
            l2_compute_w_fp8.copy_(l2_weight_values)
            l2_compute_sf.copy_(
                (
                    0.5
                    + 0.0625
                    * torch.arange(l2_compute_sf.numel(), dtype=torch.float32, device=device)
                ).reshape_as(l2_compute_sf)
            )
            expected_count_words.copy_(
                torch.tensor(_expected_counts_for_rank(rank), dtype=torch.uint32, device=device)
            )
            debug_words[7] = 0x45585043
            for token in range(NUM_TOKENS):
                x_fp8[token].fill_(_input_fp8_value(rank, token))
                for sf_idx in range(HIDDEN // 128):
                    x_sf[token, sf_idx] = rank + 0.125 * (token + 1) + 0.01 * sf_idx
                for topk in range(NUM_TOPK):
                    topk_idx[token, topk] = _route_expert(rank, topk)
                    topk_weights[token, topk] = rank + 0.25 * (token + 1) + 0.03125 * topk
        stream.synchronize()
        lib.userhopper_ws_nvshmem_barrier_all_wrapper()

        _install_cumodule_hook(lib)
        mode = os.environ.get("USERHOPPER_WS_MODE", "staged")
        if REPEAT_LAUNCHES > 1 and (mode != "single_pipe_compute_stub" or CLEANUP_WORKSPACE == 0):
            raise RuntimeError(
                "USERHOPPER_WS_REPEAT_LAUNCHES>1 requires "
                "USERHOPPER_WS_MODE=single_pipe_compute_stub and USERHOPPER_WS_CLEANUP=1"
            )
        debug_only = False
        compute_stub = False
        if mode == "single_pipe":
            with torch.cuda.stream(stream):
                compiled_single = _ws_single_pipe_dispatch_receiver_kernel[(1,)](
                    whole,
                    marker,
                    NUM_TOKENS,
                    EXPECTED_LOCAL_RECV_TOKENS,
                    NUM_RANKS,
                    NUM_EXPERTS,
                    NUM_MAX_TOKENS_PER_RANK,
                    NUM_TOPK,
                    HIDDEN,
                    LAYOUT["num_max_padded_sf_pool_tokens"],
                    NUM_DISPATCH_WARPS,
                    num_warps=NUM_WARPS,
                    maxnreg=MAXNREG,
                )
            stream.synchronize()
            lib.userhopper_ws_nvshmem_barrier_all_wrapper()
            has_ws = "ttg.warp_specialize" in compiled_single.asm.get("ttgir", "")
            if not has_ws:
                raise SystemExit("generated TTGIR missing ttg.warp_specialize for single_pipe")
            expected_marker = 0x5100
        elif mode == "single_pipe_compute_stub":
            compute_stub = True
            with torch.cuda.stream(stream):
                compiled_single = _ws_single_pipe_compute_stub_dispatch_receiver_kernel[(1,)](
                    whole,
                    y_u8,
                    l1_compute_w,
                    l1_compute_sf,
                    l2_compute_w,
                    l2_compute_sf,
                    marker,
                    NUM_TOKENS,
                    EXPECTED_LOCAL_RECV_TOKENS,
                    NUM_RANKS,
                    NUM_EXPERTS,
                    NUM_MAX_TOKENS_PER_RANK,
                    NUM_TOPK,
                    HIDDEN,
                    INTERMEDIATE_HIDDEN,
                    LAYOUT["num_max_padded_sf_pool_tokens"],
                    CLEANUP_WORKSPACE,
                    NUM_DISPATCH_WARPS,
                    COMPUTE_FULL_HIDDEN,
                    COMPUTE_PARALLEL,
                    COMPUTE_WORKER_WARPS,
                    num_warps=NUM_WARPS,
                    maxnreg=MAXNREG,
                )
            stream.synchronize()
            lib.userhopper_ws_nvshmem_barrier_all_wrapper()
            has_ws = "ttg.warp_specialize" in compiled_single.asm.get("ttgir", "")
            if not has_ws:
                raise SystemExit("generated TTGIR missing ttg.warp_specialize for single_pipe_compute_stub")
            expected_marker = 0x5400
        elif mode == "single_pipe_debug":
            debug_only = True
            with torch.cuda.stream(stream):
                compiled_single = _ws_single_pipe_debug_dispatch_receiver_kernel[(1,)](
                    whole,
                    marker,
                    NUM_TOKENS,
                    EXPECTED_LOCAL_RECV_TOKENS,
                    NUM_RANKS,
                    NUM_EXPERTS,
                    NUM_MAX_TOKENS_PER_RANK,
                    NUM_TOPK,
                    HIDDEN,
                    LAYOUT["num_max_padded_sf_pool_tokens"],
                    NUM_DISPATCH_WARPS,
                    num_warps=NUM_WARPS,
                    maxnreg=MAXNREG,
                )
            stream.synchronize()
            lib.userhopper_ws_nvshmem_barrier_all_wrapper()
            has_ws = "ttg.warp_specialize" in compiled_single.asm.get("ttgir", "")
            if not has_ws:
                raise SystemExit("generated TTGIR missing ttg.warp_specialize for single_pipe_debug")
            expected_marker = 0x510D
        elif mode == "staged":
            with torch.cuda.stream(stream):
                compiled_dispatch = _ws_dispatch_stage_kernel[(1,)](
                    whole,
                    marker,
                    NUM_TOKENS,
                    NUM_RANKS,
                    NUM_EXPERTS,
                    NUM_MAX_TOKENS_PER_RANK,
                    NUM_TOPK,
                    HIDDEN,
                    num_warps=NUM_WARPS,
                    maxnreg=MAXNREG,
                )
            stream.synchronize()
            lib.userhopper_ws_nvshmem_barrier_all_wrapper()

            with torch.cuda.stream(stream):
                compiled_receiver = _ws_receiver_stage_kernel[(1,)](
                    whole,
                    marker,
                    EXPECTED_LOCAL_RECV_TOKENS,
                    NUM_RANKS,
                    NUM_EXPERTS,
                    NUM_MAX_TOKENS_PER_RANK,
                    NUM_TOPK,
                    HIDDEN,
                    LAYOUT["num_max_padded_sf_pool_tokens"],
                    NUM_DISPATCH_WARPS,
                    num_warps=NUM_WARPS,
                    maxnreg=MAXNREG,
                )
            stream.synchronize()
            lib.userhopper_ws_nvshmem_barrier_all_wrapper()

            has_dispatch_ws = "ttg.warp_specialize" in compiled_dispatch.asm.get("ttgir", "")
            has_receiver_ws = "ttg.warp_specialize" in compiled_receiver.asm.get("ttgir", "")
            if not has_dispatch_ws or not has_receiver_ws:
                raise SystemExit(
                    f"generated TTGIR missing ttg.warp_specialize: dispatch={has_dispatch_ws} receiver={has_receiver_ws}"
            )
            expected_marker = 0xD200
        else:
            raise RuntimeError(
                "unknown USERHOPPER_WS_MODE={!r}; expected staged, single_pipe, "
                "single_pipe_compute_stub, or single_pipe_debug".format(mode)
            )

        if int(marker.cpu()[0]) != expected_marker:
            raise SystemExit(f"default partition marker mismatch: {int(marker.cpu()[0])}")

        expected_indices, expected_rows, expected_sf, expected_weight, expected_meta, expected_arrival = (
            _expected_receive(rank)
        )

        got_l1 = l1.detach().cpu()[expected_indices]
        got_l1_fp8_bytes = l1_fp8.view(torch.uint8).detach().cpu()[expected_indices]
        got_sf = l1_sf.detach().cpu()[expected_indices, :]
        got_weight = l1_weight.detach().cpu()[expected_indices]
        got_l2_tokens = l2_token.detach().cpu()[expected_indices]
        got_l2_sf = l2_sf.detach().cpu()[expected_indices, :]
        got_combine = combine.detach().cpu()
        got_y = y.detach().cpu()
        got_meta = metadata.detach().cpu()[expected_indices]
        got_arrival = arrival.detach().cpu()
        got_l2_arrival_mask = l2_arrival_mask.detach().cpu()
        expected_total = len(expected_indices)
        got_queue = queue.detach().cpu()
        got_send_count = send_count.detach().cpu()
        got_recv_sum = recv_sum.detach().cpu()
        got_debug = debug_words.detach().cpu()
        got_l2_debug = l2_debug.detach().cpu()

        if debug_only:
            preview = min(PRINT_LIMIT, got_l1.shape[0])
            queue_preview = got_queue[:, :, :max(preview, 1)]
            print(
                "rank={} ws_userhopper_dispatch_receiver_single_pipe_debug marker={} debug={} "
                "checked={} preview={} l1_first_bytes={} sf={} weights={} meta={} arrival={} "
                "queue={} send_count={} recv_sum={}".format(
                    rank,
                    int(marker.cpu()[0]),
                    got_debug.tolist(),
                    got_l1.shape[0],
                    preview,
                    got_l1[:preview, 0].tolist(),
                    [round(float(v), 4) for v in got_sf[:preview, 0].tolist()],
                    [round(float(v), 4) for v in got_weight[:preview].tolist()],
                    got_meta[:preview].tolist(),
                    {int(k): int(got_arrival[k].item()) for k in sorted(expected_arrival)},
                    queue_preview.tolist(),
                    got_send_count.tolist(),
                    got_recv_sum.tolist(),
                ),
                flush=True,
            )
            return

        if x.data_ptr() != x_fp8.data_ptr() or l1.data_ptr() != l1_fp8.data_ptr():
            raise SystemExit(
                "FP8 view pointer mismatch: x={} x_fp8={} l1={} l1_fp8={}".format(
                    x.data_ptr(), x_fp8.data_ptr(), l1.data_ptr(), l1_fp8.data_ptr()
                )
            )
        cleanup_active = compute_stub and CLEANUP_WORKSPACE != 0
        if cleanup_active:
            _validate_dispatch_queue(rank, got_queue)
            _validate_workspace_cleanup(got_send_count, got_recv_sum, got_arrival, got_l2_arrival_mask)
        else:
            _validate_dispatch_workspace(rank, got_send_count, got_recv_sum, got_queue)
        if compute_stub:
            observed_total = int(got_debug[4].item())
            expected_total_from_stub = int(got_debug[5].item())
            stub_status = int(got_debug[6].item())
            expected_checksum = _expected_l1_checksum(expected_rows, expected_weight)
            expected_weight_checksum = _expected_l1_weight_checksum(
                rank,
                l1_compute_w.detach().cpu(),
                l1_compute_sf.detach().cpu(),
            )
            expected_scalar_sum = _expected_l1_scalar_sum(
                rank,
                l1_compute_w.detach().cpu(),
                l1_compute_sf.detach().cpu(),
            )
            expected_l2_float = _expected_l2_token_floats(
                rank,
                l1_compute_w.detach().cpu(),
                l1_compute_sf.detach().cpu(),
            )
            expected_l2_sf, expected_l2_scaled = _expected_l2_sf_and_scaled_floats(expected_l2_float)
            expected_l2_tokens = _float_to_cuda_satfinite_e4m3_bytes(expected_l2_scaled, device)
            expected_combine = _expected_combine_float(
                rank,
                l1_compute_w.detach().cpu(),
                l1_compute_sf.detach().cpu(),
                l2_compute_w.detach().cpu(),
                l2_compute_sf.detach().cpu(),
                device,
            )
            got_combine_f32 = got_combine.contiguous().view(torch.bfloat16).float()
            expected_combine_bf16_f32 = expected_combine.to(torch.bfloat16).float()
            expected_y = expected_combine.sum(dim=0).to(torch.bfloat16)
            got_y_valid = got_y[:NUM_TOKENS].float()
            expected_y_valid = expected_y[:NUM_TOKENS].float()
            got_y_invalid = got_y[NUM_TOKENS:]
            combine_expected_mask = torch.zeros((NUM_TOPK, NUM_MAX_TOKENS_PER_RANK), dtype=torch.bool)
            for token in range(NUM_TOKENS):
                for topk in range(NUM_TOPK):
                    if _route_expert(rank, topk) >= 0:
                        combine_expected_mask[topk, token] = True
            got_combine_valid = got_combine_f32[combine_expected_mask]
            expected_combine_valid = expected_combine_bf16_f32[combine_expected_mask]
            got_combine_invalid = got_combine[~combine_expected_mask]
            l2_sf_tol = max(1e-5, 1e-4 * float(expected_l2_sf.abs().max().item() if expected_l2_sf.numel() else 1.0))
            combine_tol = max(1e-1, 5e-3 * float(expected_combine_valid.abs().max().item() if expected_combine_valid.numel() else 1.0))
            y_tol = max(1e-1, 5e-3 * float(expected_y_valid.abs().max().item() if expected_y_valid.numel() else 1.0))
            l2_checksum = int(got_l2_debug[0].item())
            l2_observed_total = int(got_l2_debug[1].item())
            l2_expected_total = int(got_l2_debug[2].item())
            l2_status = int(got_l2_debug[3].item())
            l2_weight_checksum = int(got_l2_debug[4].item())
            l2_weight_status = int(got_l2_debug[5].item())
            l2_scalar_sum = float(got_l2_debug[6:7].view(torch.float32)[0].item())
            l2_scalar_status = int(got_l2_debug[7].item())
            scalar_tol = max(1e-2, 1e-4 * abs(expected_scalar_sum))
            if (
                observed_total != expected_total
                or expected_total_from_stub != expected_total
                or stub_status != 0xC0DEC0DE
                or l2_checksum != expected_checksum
                or l2_observed_total != expected_total
                or l2_expected_total != expected_total
                or l2_status != 0xC0DEC0DE
                or l2_weight_checksum != expected_weight_checksum
                or l2_weight_status != 0x1A10C001
                or abs(l2_scalar_sum - expected_scalar_sum) > scalar_tol
                or l2_scalar_status != 0x51A10F32
                or not torch.equal(got_l2_tokens, expected_l2_tokens)
                or not torch.allclose(got_l2_sf, expected_l2_sf, atol=l2_sf_tol, rtol=1e-4)
                or not torch.allclose(got_combine_valid, expected_combine_valid, atol=combine_tol, rtol=5e-3)
                or bool(torch.any(got_combine_invalid != 0).item())
                or not torch.allclose(got_y_valid, expected_y_valid, atol=y_tol, rtol=5e-3)
                or bool(torch.any(got_y_invalid != 0).item())
            ):
                raise SystemExit(
                    "compute stub L1 checksum mismatch: observed={} stub_expected={} expected={} "
                    "status={} checksum={} expected_checksum={} weight_checksum={} "
                    "expected_weight_checksum={} scalar_sum={} expected_scalar_sum={} "
                    "scalar_tol={} l2_token_preview={} expected_l2_token_preview={} "
                    "l2_sf_preview={} expected_l2_sf_preview={} l2_sf_tol={} "
                    "combine_preview={} expected_combine_preview={} combine_tol={} "
                    "combine_invalid_nonzero={} "
                    "y_preview={} expected_y_preview={} y_tol={} y_invalid_nonzero={} "
                    "l2_debug={} debug={}".format(
                        observed_total,
                        expected_total_from_stub,
                        expected_total,
                        hex(stub_status),
                        l2_checksum,
                        expected_checksum,
                        l2_weight_checksum,
                        expected_weight_checksum,
                        l2_scalar_sum,
                        expected_scalar_sum,
                        scalar_tol,
                        got_l2_tokens[: min(2, got_l2_tokens.shape[0]), : min(16, INTERMEDIATE_HIDDEN)].tolist(),
                        expected_l2_tokens[: min(2, expected_l2_tokens.shape[0]), : min(16, INTERMEDIATE_HIDDEN)].tolist(),
                        got_l2_sf[: min(2, got_l2_sf.shape[0]), : min(4, INTERMEDIATE_HIDDEN // 64)].tolist(),
                        expected_l2_sf[: min(2, expected_l2_sf.shape[0]), : min(4, INTERMEDIATE_HIDDEN // 64)].tolist(),
                        l2_sf_tol,
                        got_combine_valid[: min(2, got_combine_valid.shape[0]), : min(8, HIDDEN)].tolist(),
                        expected_combine_valid[: min(2, expected_combine_valid.shape[0]), : min(8, HIDDEN)].tolist(),
                        combine_tol,
                        int(torch.count_nonzero(got_combine_invalid.to(torch.int32)).item()),
                        got_y_valid[: min(2, got_y_valid.shape[0]), : min(8, HIDDEN)].tolist(),
                        expected_y_valid[: min(2, expected_y_valid.shape[0]), : min(8, HIDDEN)].tolist(),
                        y_tol,
                        int(torch.count_nonzero(got_y_invalid.to(torch.int32)).item()),
                        got_l2_debug.tolist(),
                        got_debug.tolist(),
                    )
                )
            for repeat_idx in range(1, REPEAT_LAUNCHES):
                with torch.cuda.stream(stream):
                    marker.zero_()
                    _ws_single_pipe_compute_stub_dispatch_receiver_kernel[(1,)](
                        whole,
                        y_u8,
                        l1_compute_w,
                        l1_compute_sf,
                        l2_compute_w,
                        l2_compute_sf,
                        marker,
                        NUM_TOKENS,
                        EXPECTED_LOCAL_RECV_TOKENS,
                        NUM_RANKS,
                        NUM_EXPERTS,
                        NUM_MAX_TOKENS_PER_RANK,
                        NUM_TOPK,
                        HIDDEN,
                        INTERMEDIATE_HIDDEN,
                        LAYOUT["num_max_padded_sf_pool_tokens"],
                        CLEANUP_WORKSPACE,
                        NUM_DISPATCH_WARPS,
                        COMPUTE_FULL_HIDDEN,
                        COMPUTE_PARALLEL,
                        COMPUTE_WORKER_WARPS,
                        num_warps=NUM_WARPS,
                        maxnreg=MAXNREG,
                    )
                stream.synchronize()
                lib.userhopper_ws_nvshmem_barrier_all_wrapper()
                if int(marker.cpu()[0]) != expected_marker:
                    raise SystemExit(
                        f"repeat launch {repeat_idx} marker mismatch: {int(marker.cpu()[0])}"
                    )

                repeat_l1 = l1.detach().cpu()[expected_indices]
                repeat_l1_fp8_bytes = l1_fp8.view(torch.uint8).detach().cpu()[expected_indices]
                repeat_sf = l1_sf.detach().cpu()[expected_indices, :]
                repeat_weight = l1_weight.detach().cpu()[expected_indices]
                repeat_l2_tokens = l2_token.detach().cpu()[expected_indices]
                repeat_l2_sf = l2_sf.detach().cpu()[expected_indices, :]
                repeat_combine = combine.detach().cpu()
                repeat_y = y.detach().cpu()
                repeat_meta = metadata.detach().cpu()[expected_indices]
                repeat_arrival = arrival.detach().cpu()
                repeat_l2_arrival_mask = l2_arrival_mask.detach().cpu()
                repeat_queue = queue.detach().cpu()
                repeat_send_count = send_count.detach().cpu()
                repeat_recv_sum = recv_sum.detach().cpu()
                repeat_debug = debug_words.detach().cpu()
                repeat_l2_debug = l2_debug.detach().cpu()
                _validate_dispatch_queue(rank, repeat_queue)
                _validate_workspace_cleanup(
                    repeat_send_count, repeat_recv_sum, repeat_arrival, repeat_l2_arrival_mask
                )
                repeat_combine_f32 = repeat_combine.contiguous().view(torch.bfloat16).float()
                repeat_combine_valid = repeat_combine_f32[combine_expected_mask]
                repeat_combine_invalid = repeat_combine[~combine_expected_mask]
                repeat_y_valid = repeat_y[:NUM_TOKENS].float()
                repeat_y_invalid = repeat_y[NUM_TOKENS:]
                repeat_l2_checksum = int(repeat_l2_debug[0].item())
                repeat_l2_observed_total = int(repeat_l2_debug[1].item())
                repeat_l2_expected_total = int(repeat_l2_debug[2].item())
                repeat_l2_status = int(repeat_l2_debug[3].item())
                repeat_l2_weight_checksum = int(repeat_l2_debug[4].item())
                repeat_l2_weight_status = int(repeat_l2_debug[5].item())
                repeat_l2_scalar_sum = float(repeat_l2_debug[6:7].view(torch.float32)[0].item())
                repeat_l2_scalar_status = int(repeat_l2_debug[7].item())
                if (
                    int(repeat_debug[4].item()) != expected_total
                    or int(repeat_debug[5].item()) != expected_total
                    or int(repeat_debug[6].item()) != 0xC0DEC0DE
                    or repeat_l2_checksum != expected_checksum
                    or repeat_l2_observed_total != expected_total
                    or repeat_l2_expected_total != expected_total
                    or repeat_l2_status != 0xC0DEC0DE
                    or repeat_l2_weight_checksum != expected_weight_checksum
                    or repeat_l2_weight_status != 0x1A10C001
                    or abs(repeat_l2_scalar_sum - expected_scalar_sum) > scalar_tol
                    or repeat_l2_scalar_status != 0x51A10F32
                    or not torch.equal(repeat_l1, expected_rows)
                    or not torch.equal(repeat_l1_fp8_bytes, expected_rows)
                    or not torch.allclose(repeat_sf, expected_sf)
                    or not torch.allclose(repeat_weight, expected_weight)
                    or not torch.equal(repeat_meta, expected_meta)
                    or not torch.equal(repeat_l2_tokens, expected_l2_tokens)
                    or not torch.allclose(repeat_l2_sf, expected_l2_sf, atol=l2_sf_tol, rtol=1e-4)
                    or not torch.allclose(repeat_combine_valid, expected_combine_valid, atol=combine_tol, rtol=5e-3)
                    or bool(torch.any(repeat_combine_invalid != 0).item())
                    or not torch.allclose(repeat_y_valid, expected_y_valid, atol=y_tol, rtol=5e-3)
                    or bool(torch.any(repeat_y_invalid != 0).item())
                ):
                    raise SystemExit(
                        "repeat launch {} mismatch: debug={} l2_debug={} "
                        "l1_preview={} expected_l1_preview={} y_preview={} expected_y_preview={}".format(
                            repeat_idx,
                            repeat_debug.tolist(),
                            repeat_l2_debug.tolist(),
                            repeat_l1[: min(2, repeat_l1.shape[0]), : min(8, HIDDEN)].tolist(),
                            expected_rows[: min(2, expected_rows.shape[0]), : min(8, HIDDEN)].tolist(),
                            repeat_y_valid[: min(2, repeat_y_valid.shape[0]), : min(8, HIDDEN)].tolist(),
                            expected_y_valid[: min(2, expected_y_valid.shape[0]), : min(8, HIDDEN)].tolist(),
                        )
                    )

        if not torch.equal(got_l1, expected_rows):
            raise SystemExit(
                "l1 token bytes mismatch: got={} expected={} queue={} send_count={} recv_sum={} meta={}".format(
                    got_l1[:, 0].tolist(),
                    expected_rows[:, 0].tolist(),
                    got_queue[:, :, :max(PRINT_LIMIT, 1)].tolist(),
                    got_send_count.tolist(),
                    got_recv_sum.tolist(),
                    got_meta.tolist(),
                )
            )
        if not torch.equal(got_l1_fp8_bytes, expected_rows):
            raise SystemExit(
                "l1 fp8 byte view mismatch: got={} expected={}".format(
                    got_l1_fp8_bytes[:, 0].tolist(),
                    expected_rows[:, 0].tolist(),
                )
            )
        if not torch.allclose(got_sf, expected_sf):
            raise SystemExit(f"l1 sf mismatch: got={got_sf.tolist()} expected={expected_sf.tolist()}")
        if not torch.allclose(got_weight, expected_weight):
            raise SystemExit(f"l1 weight mismatch: got={got_weight.tolist()} expected={expected_weight.tolist()}")
        if not torch.equal(got_meta, expected_meta):
            raise SystemExit(f"metadata mismatch: got={got_meta.tolist()} expected={expected_meta.tolist()}")
        if not cleanup_active:
            got_arrival_subset = {int(k): int(got_arrival[k].item()) for k in sorted(expected_arrival)}
            if got_arrival_subset != expected_arrival:
                raise SystemExit(f"arrival counter mismatch: got={got_arrival_subset} expected={expected_arrival}")

        preview = min(PRINT_LIMIT, got_l1.shape[0])
        print(
            "rank={} ws_userhopper_dispatch_receiver_{}=PASS dispatch=checked fp8=checked cleanup={} repeats={} checked={} preview={} "
            "l1_first_bytes={} sf={} weights={} meta={}".format(
                rank,
                mode,
                int(cleanup_active),
                REPEAT_LAUNCHES,
                got_l1.shape[0],
                preview,
                got_l1[:preview, 0].tolist(),
                [round(float(v), 4) for v in got_sf[:preview, 0].tolist()],
                [round(float(v), 4) for v in got_weight[:preview].tolist()],
                got_meta[:preview].tolist(),
            ),
            flush=True,
        )
    finally:
        lib.userhopper_ws_nvshmem_barrier_all_wrapper()
        lib.userhopper_ws_nvshmem_finalize_wrapper(ptr)


if __name__ == "__main__":
    main()
