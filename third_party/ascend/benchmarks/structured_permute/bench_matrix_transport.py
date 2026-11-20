"""Paired matrix transport timing with exact checks and an independent AIV."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
import triton
import triton.language as tl


@triton.jit
def _transport(X, Y, JOIN: tl.constexpr, TRANSPOSE: tl.constexpr):
    if TRANSPOSE:
        base = tl.program_id(0) * 4
        r = tl.arange(0, 16)[:, None]
        c = base * 2 + tl.arange(0, 8)[None, :]
        raw = tl.load(X + r * 442 + c, (r < 11) & (c < 442), other=0.)
        if JOIN:
            a, b = tl.split(tl.reshape(raw, (64, 2)))
            value = tl.reshape(tl.join(a + 1., b + 1.), (16, 4, 2))
        else:
            value = tl.reshape(raw + 1., (16, 4, 2))
        value = tl.reshape(tl.permute(value, 1, 0, 2), (4, 32))
        rr = base + tl.arange(0, 4)[:, None]
        cc = tl.arange(0, 32)[None, :]
        tl.store(Y + rr * 418 + cc, value, (rr < 221) & (cc < 22))
    else:
        r = tl.arange(0, 1024)[:, None]
        c = tl.program_id(0) * 4 + tl.arange(0, 4)[None, :]
        offsets = r * 1024 + c
        raw = tl.load(X + offsets)
        if JOIN:
            a, b = tl.split(tl.reshape(raw, (2048, 2)))
            value = tl.reshape(tl.join(a + 1., b + 1.), (1024, 4))
        else:
            value = raw + 1.
        tl.store(Y + offsets, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', choices=['transpose', 'store'], required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('TRITON_ALL_BLOCKS_PARALLEL') != '1':
        parser.error('TRITON_ALL_BLOCKS_PARALLEL=1 is required for the paired 40-block comparison')
    torch.npu.set_device(0)
    transpose = args.scene == 'transpose'
    programs = 56 if transpose else 256
    count = 209 * 221 * 2 if transpose else 1024 * 1024
    cpu = torch.arange(count, dtype=torch.float32)
    x = cpu.npu()
    # Guards expose accidental burst rounding and cross-program overwrites.
    storage = torch.full((count + 64, ), -17., device='npu')
    y = storage[32:-32]
    expected = torch.full_like(cpu, -17.)
    if transpose:
        view = cpu.reshape(209, 221, 2)
        expected = torch.full_like(view, -17.)
        expected[:11] = view[:11] + 1.
        expected = expected.transpose(0, 1).contiguous().flatten()
    else:
        expected = cpu + 1.
    i = torch.arange(128)
    indices = (((i % 32 // 2) * 8 + (i // 32) * 2 + i % 2) * 4).to(torch.int32).npu()
    library = args.library.resolve()
    lib = ctypes.CDLL(str(library))
    call = lib.launch_matrix_transport
    call.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32]
    call.restype = ctypes.c_uint32

    def verify():
        got = storage.cpu()
        assert torch.equal(got[32:-32], expected)
        assert torch.all(got[:32] == -17.) and torch.all(got[-32:] == -17.)

    graphs, results = {}, {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in ['plain', 'computed_join', 'aiv']:
        if name == 'aiv':

            def launch():
                rc = call(torch.npu.current_stream().npu_stream, x.data_ptr(), y.data_ptr(), indices.data_ptr(),
                          int(transpose))
                assert rc == 0, rc

            metadata = {'library': str(library), 'sha256': hashlib.sha256(library.read_bytes()).hexdigest()}
        else:
            dump = args.output.with_suffix('') / name
            dump.mkdir(parents=True, exist_ok=True)
            os.environ['FLAGTREE_PERMUTE_DUMP_DIR'] = str(dump)
            joined = name == 'computed_join'
            compiled = _transport[(programs, )](x, y, joined, transpose, num_warps=8, num_stages=1, multibuffer=False)
            metadata = compiled.metadata._asdict()
            for ext, data in compiled.asm.items():
                path = dump / ('kernel.' + ext)
                if isinstance(data, str):
                    path.write_text(data)
                elif isinstance(data, (bytes, bytearray)):
                    path.write_bytes(data)

            def launch():
                _transport[(programs, )](x, y, joined, transpose, num_warps=8, num_stages=1, multibuffer=False)

        launch()
        torch.npu.synchronize()
        verify()
        for _ in range(10):
            launch()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(64):
                launch()
        torch.npu.synchronize()
        storage.fill_(-17.)
        graph.replay()
        torch.npu.synchronize()
        verify()
        graphs[name] = graph
        results[name] = {'metadata': metadata, 'rounds_us': []}
    policies = list(graphs)
    for round_id in range(5):
        shift = round_id % len(policies)
        for name in policies[shift:] + policies[:shift]:
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                graphs[name].replay()
            end.record()
            end.synchronize()
            results[name]['rounds_us'].append(start.elapsed_time(end) * 1000 / 6400)
            verify()
    for row in results.values():
        row['kernel_us'] = statistics.median(row['rounds_us'])
    assert torch.equal(x.cpu(), cpu)
    result = dict(status='pass', scene=args.scene, programs=programs, blocks=40, warmup=10, graph_steps=64,
                  iterations=100, rounds=5, rotated_order=True, exact_outputs_and_guards=True, package=triton.__file__,
                  policies=results)
    args.output.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print({name: row['kernel_us'] for name, row in results.items()}, flush=True)


if __name__ == '__main__':
    main()
