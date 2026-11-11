"""One case/process. Exact checks precede timing; incorrect runs get no timing.

Baseline rows contain 16-bit or 32-bit elements, Q >= 32, a power of two. Physical strided
layouts are exercised separately by compiler regressions: tl.split's logical
axis must not be confused with HBM strides or UB strides chosen downstream.
"""
import argparse
import ctypes
import json
import os
import statistics
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def split_kernel(X, Z, Y, Q: tl.constexpr):
    r = tl.program_id(0)
    v = tl.load(X + r * 2 * Q + tl.arange(0, 2 * Q))
    a, b = tl.split(tl.reshape(v, (Q, 2)))
    i = tl.arange(0, Q)
    tl.store(Y + r * 2 * Q + i, a)
    tl.store(Y + r * 2 * Q + Q + i, b)


@triton.jit
def join_kernel(X, Z, Y, Q: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, Q)
    a = tl.load(X + r * Q + i)
    b = tl.load(Z + r * Q + i)
    v = tl.reshape(tl.join(a, b), (2 * Q, ))
    tl.store(Y + r * 2 * Q + tl.arange(0, 2 * Q), v)


@triton.jit
def nested_kernel(X, Z, Y, T, J, Q: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, 4 * Q)
    v = tl.load(X + r * 4 * Q + i)
    t = tl.trans(tl.reshape(v, (4, Q)))
    tl.store(T + r * 4 * Q + i, tl.reshape(t, (4 * Q, )))
    lo, hi = tl.split(tl.reshape(t, (Q, 2, 2)))
    a, b = tl.split(lo)
    c, d = tl.split(hi)
    j = tl.arange(0, Q)
    tl.store(Y + r * 4 * Q + j, a)
    tl.store(Y + r * 4 * Q + Q + j, b)
    tl.store(Y + r * 4 * Q + 2 * Q + j, c)
    tl.store(Y + r * 4 * Q + 3 * Q + j, d)
    joined = tl.join(tl.join(a, b), tl.join(c, d))
    tl.store(J + r * 4 * Q + i, tl.reshape(joined, (4 * Q, )))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["triton", "aiv"], required=True)
    p.add_argument("--case", choices=["split", "join", "nested"], required=True)
    p.add_argument("--join-lowering", choices=["auto", "native", "packed"], default="auto")
    p.add_argument("--address-hoisting", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--store-sync-optimization", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--q", type=int, default=256)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16", "int16", "int32"], default="float32")
    p.add_argument("--library", type=Path)
    p.add_argument("--dump-dir", type=Path, help="save compiler artifacts before timing")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--graph-steps", type=int, default=64,
                   help="kernel calls per captured graph, to amortize host replay gaps")
    p.add_argument("--aiv-blocks", type=int, default=0,
                   help="0: persistent 40-block reference; use batch for matching logical grid")
    args = p.parse_args()
    if args.batch < 1 or args.aiv_blocks < 0 or args.graph_steps < 1:
        p.error("batch must be positive and aiv-blocks nonnegative")
    if args.q < 32 or args.q > 4096 or args.q & (args.q - 1):
        p.error("Q must be a power of two in [32,4096]")
    if args.warmup < 3 or args.iters < 30 or args.rounds < 3:
        p.error("require warmup>=3, iters>=30, rounds>=3")
    torch.npu.set_device(0)
    torch.manual_seed(20261027)
    q, batch = args.q, args.batch
    width = (4 if args.case == "nested" else 2) * q
    dtype = getattr(torch, args.dtype)
    xcpu = torch.randn(batch, width).to(dtype) if dtype.is_floating_point else torch.randint(
        -100, 100, (batch, width), dtype=dtype)
    zcpu = torch.randn(batch, q).to(dtype) if dtype.is_floating_point else torch.randint(
        -100, 100, (batch, q), dtype=dtype)
    x, z = xcpu.npu(), zcpu.npu()
    y = torch.full((batch, width), float("nan") if dtype.is_floating_point else -111, device="npu", dtype=dtype)
    t, j = torch.empty_like(y), torch.empty_like(y)
    if args.impl == "aiv":
        if not args.library or (args.case == "nested" and q > 1024):
            p.error("AIV reference needs --library; nested requires Q<=1024")
        lib = ctypes.CDLL(str(args.library.resolve()))
        call = lib.launch_permute
        call.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 6 + [ctypes.c_uint32] * 4
        call.restype = ctypes.c_uint32
        idx = ((torch.arange(2 * q) // 2) + (torch.arange(2 * q) % 2) * q).to(torch.int32) * x.element_size()
        if args.case == "nested":
            i = torch.arange(4 * q)
            idx = ((i // 4) + (i % 4) * q).to(torch.int32) * x.element_size()
        idx = idx.npu()
        blocks = args.aiv_blocks or min(batch, 40)

        def launch():
            # torch.npu.graph may switch streams on entry. Resolve it here,
            # otherwise the native kernel can escape graph capture entirely.
            stream = torch.npu.current_stream().npu_stream
            rc = call(blocks, stream, x.data_ptr(), z.data_ptr(), y.data_ptr(), idx.data_ptr(), t.data_ptr(),
                      j.data_ptr(), q, batch, {"split": 0, "join": 1, "nested": 2}[args.case], x.element_size())
            if rc:
                raise RuntimeError(f"AIV launch failed: {rc}")
    elif args.case == "nested":

        def launch():
            return nested_kernel[(batch, )](x, z, y, t, j, Q=q, structured_join_lowering=args.join_lowering,
                                            structured_address_hoisting=args.address_hoisting,
                                            structured_store_sync_optimization=args.store_sync_optimization)
    else:
        kernel = split_kernel if args.case == "split" else join_kernel

        def launch():
            return kernel[(batch, )](x, z, y, Q=q, structured_join_lowering=args.join_lowering,
                                     structured_address_hoisting=args.address_hoisting,
                                     structured_store_sync_optimization=args.store_sync_optimization)

    compiled = launch()
    if args.impl == "triton":
        # Match archived-binary controls exactly. Compile through JIT once,
        # then capture the CompiledKernel launcher for every measured call.
        runner = compiled[(batch, 1, 1)]
        values = (x, z, y, t, j, q) if args.case == "nested" else (x, z, y, q)

        def launch():
            runner(*values)
            return compiled

    torch.npu.synchronize()
    if args.impl == "triton" and args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        for name, data in compiled.asm.items():
            dest = args.dump_dir / ("kernel." + name)
            if isinstance(data, (bytes, bytearray)):
                dest.write_bytes(data)
            elif isinstance(data, str):
                dest.write_text(data)
    if args.case == "split":
        expected = xcpu.reshape(batch, q, 2).transpose(1, 2).reshape(batch, 2 * q)
    elif args.case == "join":
        # X is allocated with width 2Q; both implementations read the same
        # contiguous first batch*Q values, not an alternative row stride.
        a = xcpu.reshape(-1)[:batch * q].reshape(batch, q)
        expected = torch.stack((a, zcpu), dim=-1).reshape(batch, 2 * q)
    else:
        expected = xcpu.reshape(batch, 4, q)[:, [0, 2, 1, 3], :].reshape(batch, 4 * q)
        transposed = xcpu.reshape(batch, 4, q).transpose(1, 2).reshape(batch, 4 * q)
        torch.testing.assert_close(t.cpu(), transposed, rtol=0, atol=0)
        torch.testing.assert_close(j.cpu(), transposed, rtol=0, atol=0)
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    for _ in range(args.warmup):
        launch()
    torch.npu.synchronize()
    # Device graph makes the timing independent of Python vs ctypes launch cost.
    graph = torch.npu.NPUGraph()
    initial_stream = torch.npu.current_stream().npu_stream
    with torch.npu.graph(graph):
        capture_stream = torch.npu.current_stream().npu_stream
        for _ in range(args.graph_steps):
            launch()
    torch.npu.synchronize()
    # Prove that graph replay executes the kernel, including every side output.
    sentinel = float("nan") if dtype.is_floating_point else -111
    y.fill_(sentinel)
    if args.case == "nested":
        t.fill_(sentinel)
        j.fill_(sentinel)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    if args.case == "nested":
        torch.testing.assert_close(t.cpu(), transposed, rtol=0, atol=0)
        torch.testing.assert_close(j.cpu(), transposed, rtol=0, atol=0)
    times = []
    for _ in range(args.rounds):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (args.iters * args.graph_steps))
    result = vars(args).copy()
    launch_blocks = blocks if args.impl == "aiv" else batch
    if args.impl == "triton":
        from triton.backends.ascend.driver import NPUUtils
        from triton.backends.ascend.utils import _is_auto_map_parallel_blocks_enabled
        automatic = getattr(compiled.metadata, "enable_auto_blockify", None)
        if automatic is None:
            automatic = _is_auto_map_parallel_blocks_enabled()
        if automatic:
            launch_blocks = min(batch, NPUUtils().get_aivector_core_num())
    result.update(status="pass", kernel_us=statistics.median(times), rounds_us=times,
                  us_per_row=statistics.median(times) / batch, triton=triton.__version__,
                  cache=os.environ.get("TRITON_CACHE_DIR"), timing="device_graph_events_amortized",
                  logical_grid=blocks if args.impl == "aiv" else batch, launch_blocks=launch_blocks,
                  auto_blockify_env=os.environ.get("TRITON_ALL_BLOCKS_PARALLEL"), replay_outputs_verified=True,
                  launch_api="compiled_kernel" if args.impl == "triton" else "native_aiv",
                  capture_stream_changed=capture_stream != initial_stream,
                  compiler_metadata=compiled.metadata._asdict() if args.impl == "triton" else None,
                  aiv_join="vector_gather_including_index_load")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()
