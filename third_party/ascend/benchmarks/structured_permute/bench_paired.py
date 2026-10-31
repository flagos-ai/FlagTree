"""Compare compiler optimizations and vector AIV on the same buffers and stream."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
import torch_npu
import triton

from bench import nested_kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', type=int, default=256)
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16', 'int16', 'int32'], default='float32')
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.batch < 1 or args.q < 32 or args.q > 1024 or args.q & (args.q - 1):
        parser.error('batch>0 and power-of-two Q in [32,1024] required')
    torch.npu.set_device(0)
    q, batch, steps, iters, rounds = args.q, args.batch, 64, 100, 5
    mode = int(os.environ.get('TRITON_ALL_BLOCKS_PARALLEL', '0'))
    blocks = min(batch, 40) if mode else batch
    dtype = getattr(torch, args.dtype)
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    cpu = torch.arange(batch * 4 * q, dtype=torch.int32).remainder(97).to(dtype)
    x = cpu.npu()
    z = torch.zeros(batch * q, device='npu', dtype=dtype)
    y, t, j = [torch.full_like(x, -111) for _ in range(3)]
    ref = cpu.reshape(batch, 4, q)
    expected_y = ref[:, [0, 2, 1, 3], :].reshape(-1)
    expected_t = ref.transpose(1, 2).contiguous().reshape(-1)

    def verify():
        assert torch.equal(y.cpu(), expected_y)
        assert torch.equal(t.cpu(), expected_t)
        assert torch.equal(j.cpu(), expected_t)

    rows, graphs = {}, {}
    for name in ['baseline', 'addresses', 'stores', 'optimized', 'aiv']:
        if name != 'aiv':
            dump = out.with_suffix('') / name
            os.environ['FLAGTREE_PERMUTE_DUMP_DIR'] = str(dump)
            kernel = nested_kernel[(batch, )](x, z, y, t, j, Q=q, structured_address_hoisting=name
                                              in ('addresses', 'optimized'), structured_store_sync_optimization=name
                                              in ('stores', 'optimized'))
            run = kernel[(batch, 1, 1)]

            def launch():
                run(x, z, y, t, j, q)

            metadata = kernel.metadata._asdict()
            dump.mkdir(parents=True, exist_ok=True)
            for ext, data in kernel.asm.items():
                dest = dump / ('kernel.' + ext)
                if isinstance(data, str):
                    dest.write_text(data)
                elif isinstance(data, (bytes, bytearray)):
                    dest.write_bytes(data)
        else:
            library = args.library.resolve()
            lib = ctypes.CDLL(str(library))
            call = lib.launch_permute
            call.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 6 + [ctypes.c_uint32] * 4
            call.restype = ctypes.c_uint32
            i = torch.arange(4 * q)
            index = (((i // 4) + (i % 4) * q) * x.element_size()).to(torch.int32).npu()

            def launch():
                rc = call(blocks,
                          torch.npu.current_stream().npu_stream, x.data_ptr(), z.data_ptr(), y.data_ptr(),
                          index.data_ptr(), t.data_ptr(), j.data_ptr(), q, batch, 2, x.element_size())
                if rc:
                    raise RuntimeError(f'AIV launch failed: {rc}')

            metadata = {'library': str(library), 'sha256': hashlib.sha256(library.read_bytes()).hexdigest()}
        launch()
        torch.npu.synchronize()
        verify()
        for _ in range(10):
            launch()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(steps):
                launch()
        torch.npu.synchronize()
        for value in [y, t, j]:
            value.fill_(-111)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        verify()
        graphs[name] = graph
        rows[name] = {'metadata': metadata, 'rounds_us': []}

    policies = list(rows)
    for round_id in range(rounds):
        shift = round_id % len(policies)
        for name in policies[shift:] + policies[:shift]:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                graphs[name].replay()
            end.record()
            end.synchronize()
            rows[name]['rounds_us'].append(start.elapsed_time(end) * 1000 / (iters * steps))
            verify()
    for row in rows.values():
        row['us_per_row'] = statistics.median(row['rounds_us']) / batch
    result = dict(status='pass', q=q, batch=batch, dtype=args.dtype, mode=mode, launch_blocks=blocks, graph_steps=steps,
                  iters=iters, rounds=rounds, warmup=10, same_inputs_and_outputs=True, replay_outputs_verified=True,
                  rotated_order=True, policies=rows, package=triton.__file__)
    out.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print({key: row['us_per_row'] for key, row in rows.items()}, flush=True)


if __name__ == '__main__':
    main()
