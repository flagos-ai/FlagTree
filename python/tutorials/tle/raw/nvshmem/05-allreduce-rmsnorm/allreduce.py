#!/usr/bin/env python3
# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect

from triton.experimental.tle.raw.nvshmem.utils import (
    init_nvshmem_by_torch_pg,
    init_torch_distributed,
    load_common_host,
    load_host,
    tensor_from_pointer,
)

CODE_DIR = Path(__file__).parent.resolve()
HIDDEN_SIZE = 2048
THREADS_PER_BLOCK = 256
SUPPORTED_DTYPES = (torch.bfloat16,)
ABS_TOL = 1.25e-1
REL_TOL = 4.0


@dialect(
    name="cuda",
    compiler="clang",
    file=CODE_DIR / "allreduce-device.cu",
    extern_func_name="fused_allreduce_bf16",
)
def fused_allreduce_bf16(*args, **kwargs):
    ...


@triton.jit
def residual_rmsnorm_kernel(
    reduced_ptr,
    residual_ptr,
    weight_ptr,
    norm_out_ptr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < BLOCK
    base = row * BLOCK + offs

    reduced = tl.load(reduced_ptr + base, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
    z = reduced + residual

    tl.store(reduced_ptr + base, z, mask=mask)

    mean_square = tl.sum(z * z, axis=0) / BLOCK
    rstd = tl.rsqrt(mean_square + eps)
    weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(norm_out_ptr + base, z * rstd * weight, mask=mask)


@triton.jit(do_not_specialize=["rank", "world_size", "num_tokens"])
def fused_allreduce_kernel(
    reduced_ptr,
    input_ptrs,
    rank,
    world_size,
    num_tokens,
):
    tle_raw.call(
        fused_allreduce_bf16,
        [reduced_ptr, input_ptrs, rank, world_size, num_tokens],
        output_indices=[0],
    )


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _seeded_batch(device: torch.device, tokens: int, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(20260904 + tokens * 1009 + rank * 10007)
    x = torch.randn(tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device, generator=generator)
    residual = torch.randn(tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device, generator=generator)
    weight = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device=device, generator=generator)
    return x, residual, weight


def _configure_host(host) -> None:
    host.tle_ar_rmsnorm_workspace_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    host.tle_ar_rmsnorm_workspace_create.restype = ctypes.c_int

    host.tle_ar_rmsnorm_peer_workspace_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    host.tle_ar_rmsnorm_peer_workspace_ptr.restype = ctypes.c_void_p
    host.tle_ar_rmsnorm_sync_ready.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    host.tle_ar_rmsnorm_sync_ready.restype = ctypes.c_int
    host.tle_ar_rmsnorm_workspace_destroy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    host.tle_ar_rmsnorm_workspace_destroy.restype = ctypes.c_int


@dataclass
class _Runtime:
    host: Any
    common: Any
    group: Any


class FusedAllReduceResidualRMSNorm:
    def __init__(self, max_tokens: int, group=None) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        self.group = group if group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        self.max_tokens = int(_round_up(max_tokens, 1))
        self._closed = False
        self._epoch = 0

        self.runtime = self._load_runtime()
        self._create_workspace()

    def _load_runtime(self) -> _Runtime:
        host = load_host(CODE_DIR / "allreduce-host.cu")
        _configure_host(host)
        common = load_common_host()
        init_nvshmem_by_torch_pg(common, self.group)
        return _Runtime(host=host, common=common, group=self.group)

    def _create_workspace(self) -> None:
        input_ptr = ctypes.c_void_p()
        ready_ptr = ctypes.c_void_p()
        stream_ptr = ctypes.c_void_p()
        rank = ctypes.c_int()
        world = ctypes.c_int()
        local_rank = ctypes.c_int()
        local_world = ctypes.c_int()

        rc = self.runtime.host.tle_ar_rmsnorm_workspace_create(
            self.max_tokens,
            HIDDEN_SIZE,
            ctypes.byref(input_ptr),
            ctypes.byref(ready_ptr),
            ctypes.byref(stream_ptr),
            ctypes.byref(rank),
            ctypes.byref(world),
            ctypes.byref(local_rank),
            ctypes.byref(local_world),
        )
        if rc != 0:
            raise RuntimeError(f"tle_ar_rmsnorm_workspace_create failed: {rc}")

        if rank.value != self.rank or world.value != self.world_size:
            raise RuntimeError(
                "NVSHMEM/Torch rank mismatch: "
                f"nvshmem={rank.value}/{world.value} torch={self.rank}/{self.world_size}"
            )
        if world.value != local_world.value:
            raise RuntimeError("tutorial supports one node only")

        self.input_ptr = input_ptr
        self.ready_ptr = ready_ptr
        self.stream_ptr = stream_ptr
        self.comm_stream = torch.cuda.ExternalStream(stream_ptr.value, device=self.device)
        self.input_workspace = tensor_from_pointer(
            input_ptr, (self.max_tokens, HIDDEN_SIZE), torch.bfloat16, self.device
        )
        self.reduced_workspace = torch.empty(
            (self.max_tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device=self.device
        )
        self.norm_workspace = torch.empty_like(self.reduced_workspace)

        input_ptrs = []
        for peer in range(self.world_size):
            input_ptrs.append(
                int(self.runtime.host.tle_ar_rmsnorm_peer_workspace_ptr(self.input_ptr, peer))
            )

        self.input_ptrs = torch.tensor(input_ptrs, dtype=torch.int64, device=self.device)

    def _next_epoch(self) -> int:
        self._epoch += 1
        return self._epoch

    def _validate(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> int:
        if self._closed:
            raise RuntimeError("operator is closed")
        if x.device != self.device or residual.device != self.device or weight.device != self.device:
            raise ValueError("all tensors must be on the local CUDA device")
        if x.dtype not in SUPPORTED_DTYPES or residual.dtype not in SUPPORTED_DTYPES or weight.dtype not in SUPPORTED_DTYPES:
            raise ValueError("tutorial supports bfloat16 only")
        if not x.is_contiguous() or not residual.is_contiguous() or not weight.is_contiguous():
            raise ValueError("all tensors must be contiguous")
        if x.ndim != 2 or x.shape[1] != HIDDEN_SIZE:
            raise ValueError(f"input must have shape (tokens, {HIDDEN_SIZE})")
        if residual.shape != x.shape:
            raise ValueError("residual must match input shape")
        if weight.ndim != 1 or weight.shape[0] != HIDDEN_SIZE:
            raise ValueError(f"weight must have shape ({HIDDEN_SIZE},)")
        if x.shape[0] > self.max_tokens:
            raise ValueError("input exceeds configured capacity")
        return int(x.shape[0])

    def _validate_epilogue(
        self, tokens: int, residual: torch.Tensor, weight: torch.Tensor
    ) -> None:
        if residual.device != self.device or weight.device != self.device:
            raise ValueError("epilogue tensors must be on the local CUDA device")
        if residual.dtype not in SUPPORTED_DTYPES or weight.dtype not in SUPPORTED_DTYPES:
            raise ValueError("tutorial supports bfloat16 only")
        if not residual.is_contiguous() or not weight.is_contiguous():
            raise ValueError("epilogue tensors must be contiguous")
        if residual.shape != (tokens, HIDDEN_SIZE):
            raise ValueError("residual shape mismatch")
        if weight.shape != (HIDDEN_SIZE,):
            raise ValueError("weight shape mismatch")

    def _validate_input_only(self, x: torch.Tensor) -> int:
        if x.device != self.device:
            raise ValueError(f"input must be on {self.device}, got {x.device}")
        if x.dtype not in SUPPORTED_DTYPES:
            raise ValueError("tutorial supports bfloat16 only")
        if not x.is_contiguous():
            raise ValueError("input must be contiguous")
        if x.ndim != 2 or x.shape[1] != HIDDEN_SIZE:
            raise ValueError(f"input must have shape (tokens, {HIDDEN_SIZE})")
        if x.shape[0] > self.max_tokens:
            raise ValueError("input exceeds configured capacity")
        return int(x.shape[0])

    def prepare_input(self, x: torch.Tensor) -> int:
        tokens = self._validate_input_only(x)
        self.input_workspace[:tokens].copy_(x, non_blocking=True)
        return tokens

    def run_prepared(
        self,
        tokens: int,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_epilogue(tokens, residual, weight)

        current_stream = torch.cuda.current_stream(self.device)
        self.comm_stream.wait_stream(current_stream)
        epoch = self._next_epoch()
        rc = self.runtime.host.tle_ar_rmsnorm_sync_ready(
            self.ready_ptr,
            ctypes.c_uint64(epoch),
            self.rank,
            self.world_size,
            self.stream_ptr,
        )
        if rc != 0:
            raise RuntimeError(f"tle_ar_rmsnorm_sync_ready failed: {rc}")

        with torch.cuda.stream(self.comm_stream):
            fused_allreduce_kernel[(1,)](
                self.reduced_workspace,
                self.input_ptrs,
                self.rank,
                self.world_size,
                tokens,
                num_warps=THREADS_PER_BLOCK // 32,
            )
            residual_rmsnorm_kernel[(tokens,)](
                self.reduced_workspace,
                residual,
                weight,
                self.norm_workspace,
                float(eps),
                BLOCK=HIDDEN_SIZE,
                num_warps=THREADS_PER_BLOCK // 32,
            )

        current_stream.wait_stream(self.comm_stream)
        return self.reduced_workspace[:tokens], self.norm_workspace[:tokens]

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.prepare_input(x)
        return self.run_prepared(tokens, residual, weight, eps=eps)

    def reference(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(x, residual, weight)
        ref = x.to(torch.float32).clone()
        dist.all_reduce(ref, op=dist.ReduceOp.SUM, group=self.group)
        ref = ref + residual.to(torch.float32)
        rstd = torch.rsqrt(ref.pow(2).mean(dim=-1, keepdim=True) + float(eps))
        norm = ref * rstd * weight.to(torch.float32)
        return ref.to(torch.bfloat16), norm.to(torch.bfloat16)

    def close(self) -> None:
        if self._closed:
            return
        self.comm_stream.synchronize()
        rc = self.runtime.host.tle_ar_rmsnorm_workspace_destroy(
            self.input_ptr, self.ready_ptr, self.stream_ptr
        )
        if rc != 0:
            raise RuntimeError(f"tle_ar_rmsnorm_workspace_destroy failed: {rc}")
        finalize = getattr(self.runtime.common, "nvshmem_finalize_from_torch_distributed", None)
        if finalize is not None:
            finalize_rc = finalize()
            if finalize_rc != 0:
                raise RuntimeError(
                    f"nvshmem_finalize_from_torch_distributed failed: {finalize_rc}"
                )
        self._closed = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NVSHMEM fused AllReduce + Residual + RMSNorm tutorial")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 64, 128, 256, 1024, 4096])
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--eps", type=float, default=1.0e-6)
    return parser.parse_args()


def run_correctness(tokens_list: list[int], eps: float) -> None:
    group = init_torch_distributed()
    rank = dist.get_rank(group)

    operator = FusedAllReduceResidualRMSNorm(max(tokens_list), group=group)
    try:
        if rank == 0:
            print(
                f"world_size={operator.world_size} hidden={HIDDEN_SIZE} dtype=bfloat16",
                flush=True,
            )

        for tokens in tokens_list:
            x, residual, weight = _seeded_batch(operator.device, tokens, rank)
            reduced, norm = operator.forward(x, residual, weight, eps=eps)
            dist.barrier(group=operator.group)
            ref_reduced, ref_norm = operator.reference(x, residual, weight, eps=eps)
            abs_reduced = (reduced - ref_reduced).abs().max().item()
            abs_norm = (norm - ref_norm).abs().max().item()
            rel_reduced = (
                (reduced - ref_reduced).abs()
                / torch.maximum(
                    ref_reduced.abs(),
                    torch.tensor(1e-2, device=operator.device, dtype=ref_reduced.dtype),
                )
            ).max().item()
            rel_norm = (
                (norm - ref_norm).abs()
                / torch.maximum(
                    ref_norm.abs(),
                    torch.tensor(1e-2, device=operator.device, dtype=ref_norm.dtype),
                )
            ).max().item()
            max_abs = max(abs_reduced, abs_norm)
            max_rel = max(rel_reduced, rel_norm)
            ok = max_abs <= ABS_TOL
            if rank == 0:
                print(
                    f"tokens={tokens} max_abs={max_abs:.6f} max_rel={max_rel:.6f} "
                    f"status={'PASS' if ok else 'FAIL'}",
                    flush=True,
                )
            if not ok:
                raise RuntimeError(f"correctness failed for tokens={tokens}")
    finally:
        operator.close()
        dist.destroy_process_group(group)
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    run_correctness(args.tokens, args.eps)


if __name__ == "__main__":
    main()
