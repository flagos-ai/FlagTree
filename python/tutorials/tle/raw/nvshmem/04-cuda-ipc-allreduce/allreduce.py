from __future__ import annotations

import argparse
import ctypes
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import triton

import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect
from triton.experimental.tle.raw.nvshmem.utils import (
    load_host,
    tensor_from_pointer,
)

CODE_DIR = Path(__file__).parent.resolve()
MAX_BLOCKS = 36
THREADS_PER_BLOCK = 512
SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

ALGORITHM_AUTO = "auto"
ALGORITHM_ONESHOT = "oneshot"
ALGORITHM_TWOSHOT = "twoshot"


def _device_function(extern_func_name: str):

    @dialect(
        name="cuda",
        file=CODE_DIR / "allreduce-device.cu",
        extern_func_name=extern_func_name,
    )
    def device_function(*args, **kwargs):
        ...

    return device_function


IPC_ONESHOT_FP16_2 = _device_function("ipc_allreduce_oneshot_fp16_2")
IPC_ONESHOT_FP16_4 = _device_function("ipc_allreduce_oneshot_fp16_4")
IPC_ONESHOT_FP16_6 = _device_function("ipc_allreduce_oneshot_fp16_6")
IPC_ONESHOT_FP16_8 = _device_function("ipc_allreduce_oneshot_fp16_8")
IPC_TWOSHOT_FP16_2 = _device_function("ipc_allreduce_twoshot_fp16_2")
IPC_TWOSHOT_FP16_4 = _device_function("ipc_allreduce_twoshot_fp16_4")
IPC_TWOSHOT_FP16_6 = _device_function("ipc_allreduce_twoshot_fp16_6")
IPC_TWOSHOT_FP16_8 = _device_function("ipc_allreduce_twoshot_fp16_8")

IPC_ONESHOT_BF16_2 = _device_function("ipc_allreduce_oneshot_bf16_2")
IPC_ONESHOT_BF16_4 = _device_function("ipc_allreduce_oneshot_bf16_4")
IPC_ONESHOT_BF16_6 = _device_function("ipc_allreduce_oneshot_bf16_6")
IPC_ONESHOT_BF16_8 = _device_function("ipc_allreduce_oneshot_bf16_8")
IPC_TWOSHOT_BF16_2 = _device_function("ipc_allreduce_twoshot_bf16_2")
IPC_TWOSHOT_BF16_4 = _device_function("ipc_allreduce_twoshot_bf16_4")
IPC_TWOSHOT_BF16_6 = _device_function("ipc_allreduce_twoshot_bf16_6")
IPC_TWOSHOT_BF16_8 = _device_function("ipc_allreduce_twoshot_bf16_8")

IPC_ONESHOT_FP32_2 = _device_function("ipc_allreduce_oneshot_fp32_2")
IPC_ONESHOT_FP32_4 = _device_function("ipc_allreduce_oneshot_fp32_4")
IPC_ONESHOT_FP32_6 = _device_function("ipc_allreduce_oneshot_fp32_6")
IPC_ONESHOT_FP32_8 = _device_function("ipc_allreduce_oneshot_fp32_8")
IPC_TWOSHOT_FP32_2 = _device_function("ipc_allreduce_twoshot_fp32_2")
IPC_TWOSHOT_FP32_4 = _device_function("ipc_allreduce_twoshot_fp32_4")
IPC_TWOSHOT_FP32_6 = _device_function("ipc_allreduce_twoshot_fp32_6")
IPC_TWOSHOT_FP32_8 = _device_function("ipc_allreduce_twoshot_fp32_8")


@triton.jit(do_not_specialize=["rank"])
def ipc_oneshot_fp16_2_kernel(
    output,
    input_pointer_table,
    signal_pointer_table,
    rank,
    numel,
):
    tle_raw.call(
        IPC_ONESHOT_FP16_2,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_oneshot_fp16_4_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_ONESHOT_FP16_4,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_oneshot_fp16_6_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_ONESHOT_FP16_6,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_oneshot_fp16_8_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_ONESHOT_FP16_8,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_twoshot_fp16_2_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_TWOSHOT_FP16_2,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_twoshot_fp16_4_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_TWOSHOT_FP16_4,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_twoshot_fp16_6_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_TWOSHOT_FP16_6,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_twoshot_fp16_8_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_TWOSHOT_FP16_8,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_oneshot_bf16_2_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_ONESHOT_BF16_2,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_oneshot_bf16_4_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_ONESHOT_BF16_4,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_oneshot_bf16_6_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_ONESHOT_BF16_6,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_oneshot_bf16_8_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_ONESHOT_BF16_8,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_twoshot_bf16_2_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_TWOSHOT_BF16_2,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_twoshot_bf16_4_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_TWOSHOT_BF16_4,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_twoshot_bf16_6_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_TWOSHOT_BF16_6,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_twoshot_bf16_8_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_TWOSHOT_BF16_8,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_oneshot_fp32_2_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_ONESHOT_FP32_2,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_oneshot_fp32_4_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_ONESHOT_FP32_4,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_oneshot_fp32_6_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_ONESHOT_FP32_6,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_oneshot_fp32_8_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_ONESHOT_FP32_8,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_twoshot_fp32_2_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_TWOSHOT_FP32_2,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_twoshot_fp32_4_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_TWOSHOT_FP32_4,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_twoshot_fp32_6_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_TWOSHOT_FP32_6,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


@triton.jit(do_not_specialize=["rank"])
def ipc_twoshot_fp32_8_kernel(output, input_pointer_table, signal_pointer_table, rank, numel):
    tle_raw.call(
        IPC_TWOSHOT_FP32_8,
        [output, input_pointer_table, signal_pointer_table, rank, numel],
        output_indices=[0],
    )


IPC_ALLREDUCE_KERNELS = {
    (torch.float16, ALGORITHM_ONESHOT, 2): ipc_oneshot_fp16_2_kernel,
    (torch.float16, ALGORITHM_ONESHOT, 4): ipc_oneshot_fp16_4_kernel,
    (torch.float16, ALGORITHM_ONESHOT, 6): ipc_oneshot_fp16_6_kernel,
    (torch.float16, ALGORITHM_ONESHOT, 8): ipc_oneshot_fp16_8_kernel,
    (torch.float16, ALGORITHM_TWOSHOT, 2): ipc_twoshot_fp16_2_kernel,
    (torch.float16, ALGORITHM_TWOSHOT, 4): ipc_twoshot_fp16_4_kernel,
    (torch.float16, ALGORITHM_TWOSHOT, 6): ipc_twoshot_fp16_6_kernel,
    (torch.float16, ALGORITHM_TWOSHOT, 8): ipc_twoshot_fp16_8_kernel,
    (torch.bfloat16, ALGORITHM_ONESHOT, 2): ipc_oneshot_bf16_2_kernel,
    (torch.bfloat16, ALGORITHM_ONESHOT, 4): ipc_oneshot_bf16_4_kernel,
    (torch.bfloat16, ALGORITHM_ONESHOT, 6): ipc_oneshot_bf16_6_kernel,
    (torch.bfloat16, ALGORITHM_ONESHOT, 8): ipc_oneshot_bf16_8_kernel,
    (torch.bfloat16, ALGORITHM_TWOSHOT, 2): ipc_twoshot_bf16_2_kernel,
    (torch.bfloat16, ALGORITHM_TWOSHOT, 4): ipc_twoshot_bf16_4_kernel,
    (torch.bfloat16, ALGORITHM_TWOSHOT, 6): ipc_twoshot_bf16_6_kernel,
    (torch.bfloat16, ALGORITHM_TWOSHOT, 8): ipc_twoshot_bf16_8_kernel,
    (torch.float32, ALGORITHM_ONESHOT, 2): ipc_oneshot_fp32_2_kernel,
    (torch.float32, ALGORITHM_ONESHOT, 4): ipc_oneshot_fp32_4_kernel,
    (torch.float32, ALGORITHM_ONESHOT, 6): ipc_oneshot_fp32_6_kernel,
    (torch.float32, ALGORITHM_ONESHOT, 8): ipc_oneshot_fp32_8_kernel,
    (torch.float32, ALGORITHM_TWOSHOT, 2): ipc_twoshot_fp32_2_kernel,
    (torch.float32, ALGORITHM_TWOSHOT, 4): ipc_twoshot_fp32_4_kernel,
    (torch.float32, ALGORITHM_TWOSHOT, 6): ipc_twoshot_fp32_6_kernel,
    (torch.float32, ALGORITHM_TWOSHOT, 8): ipc_twoshot_fp32_8_kernel,
}


class CudaIpcRuntime:

    def __init__(self) -> None:
        self.lib = load_host(CODE_DIR / "allreduce-host.cu")
        self.lib.tle_ipc_handle_size.argtypes = []
        self.lib.tle_ipc_handle_size.restype = ctypes.c_size_t
        self.lib.tle_ipc_signal_size.argtypes = []
        self.lib.tle_ipc_signal_size.restype = ctypes.c_size_t
        self.lib.tle_ipc_allocate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
        ]
        self.lib.tle_ipc_allocate.restype = ctypes.c_int
        self.lib.tle_ipc_export_pointer.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.tle_ipc_export_pointer.restype = ctypes.c_int
        self.lib.tle_ipc_open.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.tle_ipc_open.restype = ctypes.c_int
        self.lib.tle_ipc_close.argtypes = [ctypes.c_void_p]
        self.lib.tle_ipc_close.restype = ctypes.c_int
        self.lib.tle_ipc_free.argtypes = [ctypes.c_void_p]
        self.lib.tle_ipc_free.restype = ctypes.c_int
        self.lib.tle_ipc_error_string.argtypes = [ctypes.c_int]
        self.lib.tle_ipc_error_string.restype = ctypes.c_char_p

        self.handle_size = self.lib.tle_ipc_handle_size()
        self.signal_size = self.lib.tle_ipc_signal_size()

    def check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        message = self.lib.tle_ipc_error_string(result).decode()
        raise RuntimeError(f"{operation} failed: CUDA error {result}: {message}")

    def allocate(self, size: int) -> tuple[int, bytes]:
        pointer = ctypes.c_void_p()
        handle = (ctypes.c_ubyte * self.handle_size)()
        self.check(
            self.lib.tle_ipc_allocate(ctypes.byref(pointer), handle, ctypes.c_size_t(size)),
            "cudaMalloc/cudaIpcGetMemHandle",
        )
        assert pointer.value is not None
        return pointer.value, bytes(handle)

    def export_pointer(self, pointer: int) -> tuple[bytes, int]:
        handle = (ctypes.c_ubyte * self.handle_size)()
        offset = ctypes.c_size_t()
        self.check(
            self.lib.tle_ipc_export_pointer(ctypes.c_void_p(pointer), handle, ctypes.byref(offset)),
            "cuPointerGetAttribute/cudaIpcGetMemHandle",
        )
        return bytes(handle), offset.value

    def open(self, serialized_handle: bytes) -> int:
        if len(serialized_handle) != self.handle_size:
            raise ValueError(f"invalid CUDA IPC handle size {len(serialized_handle)}; "
                             f"expected {self.handle_size}")
        handle = (ctypes.c_ubyte * self.handle_size).from_buffer_copy(serialized_handle)
        pointer = ctypes.c_void_p()
        self.check(
            self.lib.tle_ipc_open(handle, ctypes.byref(pointer)),
            "cudaIpcOpenMemHandle",
        )
        assert pointer.value is not None
        return pointer.value

    def close(self, pointer: int) -> None:
        self.check(
            self.lib.tle_ipc_close(ctypes.c_void_p(pointer)),
            "cudaIpcCloseMemHandle",
        )

    def free(self, pointer: int) -> None:
        self.check(self.lib.tle_ipc_free(ctypes.c_void_p(pointer)), "cudaFree")


class CudaIpcGraphRegistration:

    def __init__(
        self,
        runtime: CudaIpcRuntime,
        remote_base_pointers: list[int],
        group,
        device: torch.device,
    ) -> None:
        self.runtime = runtime
        self.remote_base_pointers = remote_base_pointers
        self.group = group
        self.device = device
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        for pointer in self.remote_base_pointers:
            self.runtime.close(pointer)
        dist.barrier(group=self.group)
        self._closed = True


class CudaIpcAllReduce:
    """vLLM-style single-node CUDA IPC custom AllReduce."""

    def __init__(self, max_numel: int, group=None) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        if max_numel <= 0:
            raise ValueError("max_numel must be positive")
        if dist.get_backend(group) != dist.Backend.GLOO:
            raise ValueError("CUDA IPC AllReduce requires a pure Gloo process group "
                             "for control-plane communication")

        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError("CUDA IPC AllReduce supports world sizes 2, 4, 6, and 8")

        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        self.max_numel = max_numel
        self.runtime = CudaIpcRuntime()
        self._closed = False

        self._validate_single_node_p2p()

        # Four bytes per element allows the same allocation to stage FP16,
        # BF16, and FP32 tensors up to max_numel.
        buffer_bytes = max_numel * torch.float32.itemsize
        self.local_input_pointer, input_handle = self.runtime.allocate(buffer_bytes)

        # Signal is immediately followed by the two-shot reduce-scatter
        # temporary buffer, matching vLLM's allocation layout.
        metadata_bytes = self.runtime.signal_size + buffer_bytes
        self.local_signal_pointer, signal_handle = self.runtime.allocate(metadata_bytes)

        all_handles: list[tuple[bytes, bytes] | None] = [None] * self.world_size
        dist.all_gather_object(all_handles, (input_handle, signal_handle), group=self.group)

        self.input_pointers: list[int] = []
        self.signal_pointers: list[int] = []
        self.remote_pointers: list[int] = []
        for peer, handles in enumerate(all_handles):
            assert handles is not None
            if peer == self.rank:
                input_pointer = self.local_input_pointer
                signal_pointer = self.local_signal_pointer
            else:
                input_pointer = self.runtime.open(handles[0])
                signal_pointer = self.runtime.open(handles[1])
                self.remote_pointers.extend((input_pointer, signal_pointer))
            self.input_pointers.append(input_pointer)
            self.signal_pointers.append(signal_pointer)

        self.input_pointer_table = torch.tensor(self.input_pointers, dtype=torch.int64, device=self.device)
        self.signal_pointer_table = torch.tensor(self.signal_pointers, dtype=torch.int64, device=self.device)
        self.staging = {
            dtype: tensor_from_pointer(
                self.local_input_pointer,
                (max_numel, ),
                dtype,
                self.device,
            )
            for dtype in SUPPORTED_DTYPES
        }
        dist.barrier(group=self.group)

    def _validate_single_node_p2p(self) -> None:
        hostnames: list[str | None] = [None] * self.world_size
        dist.all_gather_object(hostnames, socket.gethostname(), group=self.group)
        if len(set(hostnames)) != 1:
            raise RuntimeError("CUDA IPC AllReduce only supports ranks on one host")

        local_ranks: list[int | None] = [None] * self.world_size
        dist.all_gather_object(local_ranks, self.local_rank, group=self.group)
        for peer_rank in local_ranks:
            assert peer_rank is not None
            if peer_rank == self.local_rank:
                continue
            if not torch.cuda.can_device_access_peer(self.local_rank, peer_rank):
                raise RuntimeError(f"CUDA device {self.local_rank} cannot access "
                                   f"peer {peer_rank}")

    def _validate_tensor(self, tensor: torch.Tensor) -> None:
        if self._closed:
            raise RuntimeError("CudaIpcAllReduce is closed")
        if tensor.device != self.device:
            raise ValueError(f"input must be on {self.device}, got {tensor.device}")
        if tensor.dtype not in SUPPORTED_DTYPES:
            raise ValueError("CUDA IPC AllReduce supports float16, bfloat16, and float32")
        if not tensor.is_contiguous():
            raise ValueError("input must be contiguous")
        if tensor.numel() > self.max_numel:
            raise ValueError(f"input has {tensor.numel()} elements, "
                             f"maximum is {self.max_numel}")

        packed_elements = 16 // tensor.element_size()
        if tensor.numel() == 0 or tensor.numel() % packed_elements:
            raise ValueError(f"input numel must be positive and divisible by "
                             f"{packed_elements} for {tensor.dtype}")

    def select_algorithm(self, tensor: torch.Tensor) -> str:
        """Apply the same size-based selection policy as vLLM."""
        self._validate_tensor(tensor)
        nbytes = tensor.numel() * tensor.element_size()
        if self.world_size == 2:
            return ALGORITHM_ONESHOT
        if self.world_size <= 4:
            return (ALGORITHM_ONESHOT if nbytes < 512 * 1024 else ALGORITHM_TWOSHOT)
        return (ALGORITHM_ONESHOT if nbytes < 256 * 1024 else ALGORITHM_TWOSHOT)

    def _resolve_algorithm(self, tensor: torch.Tensor, algorithm: str) -> str:
        normalized = algorithm.lower().replace("_", "").replace("-", "")
        aliases = {
            "auto": ALGORITHM_AUTO,
            "oneshot": ALGORITHM_ONESHOT,
            "twoshot": ALGORITHM_TWOSHOT,
        }
        if normalized not in aliases:
            raise ValueError("algorithm must be auto, oneshot, or twoshot")
        resolved = aliases[normalized]
        if resolved == ALGORITHM_AUTO:
            return self.select_algorithm(tensor)
        return resolved

    def _launch(
        self,
        tensor: torch.Tensor,
        input_pointer_table: torch.Tensor,
        algorithm: str,
    ) -> torch.Tensor:
        flat_input = tensor.view(-1)
        output = torch.empty_like(flat_input)
        packed_elements = 16 // tensor.element_size()
        packed_count = flat_input.numel() // packed_elements
        blocks = min(
            MAX_BLOCKS,
            triton.cdiv(packed_count, THREADS_PER_BLOCK),
        )
        kernel = IPC_ALLREDUCE_KERNELS[(tensor.dtype, algorithm, self.world_size)]
        kernel[(blocks, )](
            output,
            input_pointer_table,
            self.signal_pointer_table,
            self.rank,
            flat_input.numel(),
            num_warps=THREADS_PER_BLOCK // 32,
        )
        return output.view_as(tensor)

    def all_reduce(
        self,
        tensor: torch.Tensor,
        algorithm: str = ALGORITHM_AUTO,
    ) -> torch.Tensor:
        self._validate_tensor(tensor)
        resolved_algorithm = self._resolve_algorithm(tensor, algorithm)
        flat_input = tensor.view(-1)
        self.staging[tensor.dtype][:flat_input.numel()].copy_(flat_input, non_blocking=True)
        return self._launch(tensor, self.input_pointer_table, resolved_algorithm)

    def create_graph_pointer_table(self) -> torch.Tensor:
        """Create stable pointer storage for one captured AllReduce node."""
        return self.input_pointer_table.clone()

    def all_reduce_registered(
        self,
        tensor: torch.Tensor,
        graph_pointer_table: torch.Tensor,
        algorithm: str = ALGORITHM_AUTO,
    ) -> torch.Tensor:
        """Launch from an input whose IPC handles were registered after capture."""
        self._validate_tensor(tensor)
        self._validate_pointer_table(graph_pointer_table)
        resolved_algorithm = self._resolve_algorithm(tensor, algorithm)
        return self._launch(tensor, graph_pointer_table, resolved_algorithm)

    def _validate_pointer_table(self, graph_pointer_table: torch.Tensor) -> None:
        if graph_pointer_table.device != self.device:
            raise ValueError("graph pointer table must be on the local CUDA device")
        if graph_pointer_table.dtype != torch.int64:
            raise ValueError("graph pointer table must use torch.int64")
        if graph_pointer_table.numel() != self.world_size:
            raise ValueError("graph pointer table must contain one pointer for every rank")

    def register_graph_input(
        self,
        tensor: torch.Tensor,
        graph_pointer_tables: torch.Tensor | list[torch.Tensor],
    ) -> CudaIpcGraphRegistration:
        """Exchange graph-input IPC metadata and populate captured tables."""
        self._validate_tensor(tensor)
        if isinstance(graph_pointer_tables, torch.Tensor):
            graph_pointer_tables = [graph_pointer_tables]
        if not graph_pointer_tables:
            raise ValueError("at least one graph pointer table is required")
        for graph_pointer_table in graph_pointer_tables:
            self._validate_pointer_table(graph_pointer_table)

        handle, offset = self.runtime.export_pointer(tensor.data_ptr())
        all_metadata: list[tuple[bytes, int] | None] = [None] * self.world_size
        dist.all_gather_object(all_metadata, (handle, offset), group=self.group)

        input_pointers: list[int] = []
        remote_base_pointers: list[int] = []
        try:
            for peer, metadata in enumerate(all_metadata):
                assert metadata is not None
                peer_handle, peer_offset = metadata
                if peer == self.rank:
                    input_pointer = tensor.data_ptr()
                else:
                    base_pointer = self.runtime.open(peer_handle)
                    remote_base_pointers.append(base_pointer)
                    input_pointer = base_pointer + peer_offset
                input_pointers.append(input_pointer)
        except Exception:
            for pointer in remote_base_pointers:
                self.runtime.close(pointer)
            raise

        pointer_values = torch.tensor(input_pointers, dtype=torch.int64, device=self.device)
        for graph_pointer_table in graph_pointer_tables:
            graph_pointer_table.copy_(pointer_values)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        return CudaIpcGraphRegistration(
            self.runtime,
            remote_base_pointers,
            self.group,
            self.device,
        )

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)

        self.staging = {}
        self.input_pointer_table = None
        self.signal_pointer_table = None
        for pointer in self.remote_pointers:
            self.runtime.close(pointer)

        dist.barrier(group=self.group)
        self.runtime.free(self.local_input_pointer)
        self.runtime.free(self.local_signal_pointer)
        self._closed = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified vLLM-style CUDA IPC AllReduce validation")
    parser.add_argument("--numel", type=int, default=256 * 1024)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--algorithm",
        choices=("auto", "oneshot", "twoshot"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_by_name[args.dtype]

    communicator = CudaIpcAllReduce(args.numel)
    try:
        rank_sum = (communicator.world_size * (communicator.world_size - 1) // 2)
        base = torch.arange(args.numel, dtype=torch.int32, device=device)
        base = (base % 16).to(dtype)
        local_input = base + rank
        expected = (base.float() * communicator.world_size + rank_sum).to(dtype)
        output = communicator.all_reduce(local_input, algorithm=args.algorithm)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.cuda.synchronize(device)
        dist.barrier(group=communicator.group)
        if rank == 0:
            resolved = communicator._resolve_algorithm(local_input, args.algorithm)
            print(
                f"passed: dtype={dtype}, requested={args.algorithm}, "
                f"selected={resolved}, numel={args.numel}",
                flush=True,
            )
    finally:
        communicator.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
