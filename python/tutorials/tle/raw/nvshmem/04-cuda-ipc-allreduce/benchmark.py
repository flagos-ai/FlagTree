from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

CODE_DIR = Path(__file__).parent.resolve()
DEFAULT_SEQUENCE_LENGTHS = [1, 2, 4, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096, 8192]
HIDDEN_SIZE = 8192
CUDA_GRAPH_CAPTURE_CYCLES = 10

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
BENCHMARK_ALGORITHMS = {
    "ca_1stage": "oneshot",
    "ca_2stage": "twoshot",
}


def load_communicator_class():
    source = CODE_DIR / "allreduce.py"
    module_name = "_tle_cuda_ipc_allreduce"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.CudaIpcAllReduce


CudaIpcAllReduce = load_communicator_class()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("vLLM-aligned CUDA Graph benchmark for the unified CUDA IPC "
                                                  "AllReduce"))
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEQUENCE_LENGTHS,
        help="tensor sequence lengths; hidden size is fixed at 8192",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="bfloat16",
    )
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, dtype: torch.dtype) -> None:
    if not args.sequence_lengths or min(args.sequence_lengths) <= 0:
        raise ValueError("--sequence-lengths must contain positive values")
    if args.num_warmup < 0:
        raise ValueError("--num-warmup must be non-negative")
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive")

    packed_elements = 16 // dtype.itemsize
    for sequence_length in args.sequence_lengths:
        numel = sequence_length * HIDDEN_SIZE
        if numel % packed_elements:
            raise ValueError(f"shape ({sequence_length}, {HIDDEN_SIZE}) has {numel} "
                             f"elements; {dtype} requires a multiple of "
                             f"{packed_elements}")


def benchmark_allreduce_single(
    communicator,
    sequence_length: int,
    dtype: torch.dtype,
    algorithm: str,
    num_warmup: int,
    num_trials: int,
) -> float:
    tensor = torch.randn(
        sequence_length,
        HIDDEN_SIZE,
        dtype=dtype,
        device=communicator.device,
    )
    stream = torch.cuda.Stream(device=communicator.device)
    registration = None
    try:
        torch.cuda.synchronize(communicator.device)
        with torch.cuda.stream(stream):
            graph_input = tensor.clone()
            graph_pointer_tables = [communicator.create_graph_pointer_table() for _ in range(CUDA_GRAPH_CAPTURE_CYCLES)]

            # Compile the selected specialization and advance every rank's
            # persistent signal state before capture.
            for _ in range(3):
                communicator.all_reduce(graph_input, algorithm=algorithm)

            graph = torch.cuda.CUDAGraph()
            graph_pool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(graph, pool=graph_pool, stream=stream):
                for pointer_table in graph_pointer_tables:
                    communicator.all_reduce_registered(
                        graph_input,
                        pointer_table,
                        algorithm=algorithm,
                    )
        stream.synchronize()

        # CUDA Graph nodes retain stable pointer-table addresses. Exchange the
        # captured input allocation once, then fill all ten tables.
        registration = communicator.register_graph_input(graph_input, graph_pointer_tables)

        torch.cuda.synchronize(communicator.device)
        dist.barrier(group=communicator.group)
        for _ in range(num_warmup):
            graph.replay()
        torch.cuda.synchronize(communicator.device)
        dist.barrier(group=communicator.group)

        start_time = time.perf_counter()
        for _ in range(num_trials):
            graph.replay()
        torch.cuda.synchronize(communicator.device)
        end_time = time.perf_counter()

        latency_ms = ((end_time - start_time) / num_trials / CUDA_GRAPH_CAPTURE_CYCLES * 1000.0)
        dist.barrier(group=communicator.group)
        return latency_ms
    finally:
        if registration is not None:
            registration.close()


def print_results(
    results: dict[int, dict[str, float]],
    sequence_lengths: list[int],
    world_size: int,
    dtype: torch.dtype,
) -> None:
    algorithms = sorted({algorithm for shape_results in results.values() for algorithm in shape_results})
    width = 42 + 20 * len(algorithms)
    print(f"\n{'=' * width}")
    print("CUDA IPC Device Communicator Benchmark Results")
    print(f"World Size: {world_size}, Data Type: {dtype}, "
          f"Hidden Size: {HIDDEN_SIZE}")
    print(f"{'=' * width}")

    header = f"{'Tensor Shape':<22}{'Tensor Size':<20}"
    for algorithm in algorithms:
        header += f"{algorithm:<20}"
    print(header)
    print("-" * len(header))

    for sequence_length in sequence_lengths:
        numel = sequence_length * HIDDEN_SIZE
        size_mib = numel * dtype.itemsize / (1024 * 1024)
        row = (f"{f'({sequence_length}, {HIDDEN_SIZE})':<22}"
               f"{f'{size_mib:.2f} MiB':<20}")
        for algorithm in algorithms:
            latency = results[sequence_length].get(algorithm)
            row += (f"{latency:<20.3f}" if latency is not None else f"{'N/A':<20}")
        print(row)

    print(f"{'=' * width}")
    print("Times are milliseconds per AllReduce operation; each timed graph "
          f"contains {CUDA_GRAPH_CAPTURE_CYCLES} operations.")


def main() -> None:
    args = parse_args()
    dtype = DTYPES[args.dtype]
    validate_args(args, dtype)

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    communicator = CudaIpcAllReduce(
        max(args.sequence_lengths) * HIDDEN_SIZE,
        group=cpu_group,
    )
    algorithms = dict(BENCHMARK_ALGORITHMS)

    results: dict[int, dict[str, float]] = {}
    try:
        for sequence_length in args.sequence_lengths:
            shape_results: dict[str, float] = {}
            for result_name, algorithm in algorithms.items():
                if rank == 0:
                    print(
                        f"Benchmarking {result_name}: "
                        f"shape=({sequence_length}, {HIDDEN_SIZE}), "
                        f"dtype={dtype}",
                        flush=True,
                    )
                shape_results[result_name] = benchmark_allreduce_single(
                    communicator,
                    sequence_length,
                    dtype,
                    algorithm,
                    args.num_warmup,
                    args.num_trials,
                )
            results[sequence_length] = shape_results

        if rank == 0:
            print_results(results, args.sequence_lengths, world_size, dtype)
            if args.output_json is not None:
                output = {
                    "world_size": world_size,
                    "dtype": str(dtype),
                    "hidden_size": HIDDEN_SIZE,
                    "sequence_lengths": args.sequence_lengths,
                    "num_warmup": args.num_warmup,
                    "num_trials": args.num_trials,
                    "cuda_graph_capture_cycles": CUDA_GRAPH_CAPTURE_CYCLES,
                    "results": {
                        str(sequence_length): {"timings": shape_results}
                        for sequence_length, shape_results in results.items()
                    },
                }
                args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
                print(f"Results saved to {args.output_json}", flush=True)
    finally:
        communicator.close()
        dist.destroy_process_group(cpu_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
