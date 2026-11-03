"""Exact stabilized mHC division: all denominator mantissas and integer RN32 oracle.

Run in the normal FlagTree environment on a free GPU, for example:
  python repro_mhc_normalized_division.py --device-uuid GPU-... --output .local/division-repro.json
No model weights or compiler binaries are inspected.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from local_optimizations.sinkhorn_normal_division import ASM as NORMAL_PATH_ASM

ASM = tl.constexpr(NORMAL_PATH_ASM)


@triton.jit
def division_probe(actual, reference, xbits, ybits, control):
    xe = tl.load(control)
    ye = tl.load(control + 1)
    seed = tl.load(control + 2)
    for base in tl.range(tl.program_id(0) * 1024, 8388608, 128 * 1024):
        i = base + tl.arange(0, 1024)
        mantissa = (i.to(tl.uint32) * 1664525 + seed.to(tl.uint32) * 1013904223) & 0x7fffff
        xb = mantissa | ((xe + 127).to(tl.uint32) << 23)
        yb = (i.to(tl.uint32) & 0x7fffff) | ((ye + 127).to(tl.uint32) << 23)
        x = tl.cast(xb, tl.float32, bitcast=True)
        y = tl.cast(yb, tl.float32, bitcast=True)
        q = tl.inline_asm_elementwise(ASM, constraints='=f,f,f', args=[x, y],
                                      dtype=tl.float32, is_pure=True, pack=1)
        tl.store(actual + i, q)
        tl.store(reference + i, libdevice.div_rn(x, y))
        tl.store(xbits + i, xb)
        tl.store(ybits + i, yb)


def integer_rn32(xb, yb):
    mx = (xb & 0x7fffff) | 0x800000
    my = (yb & 0x7fffff) | 0x800000
    power = ((xb >> 23) & 255) - ((yb >> 23) & 255)
    lower = 0 if mx >= my else -1
    exponent = lower + power
    q, r = divmod(mx << (23 - lower), my)
    if 2 * r > my or 2 * r == my and q & 1:
        q += 1
    if q == 1 << 24:
        q >>= 1
        exponent += 1
    assert 0 < exponent + 127 < 255
    return ((exponent + 127) << 23) | (q & 0x7fffff)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True).splitlines()
    assert not [r for r in rows if r.startswith(args.device_uuid)], 'Requested GPU is occupied'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_uuid
    import torch
    from triton.flagmega.runtime.prepared import ResourceContract, prepare_jit_kernel
    from triton.flagmega.runtime.nvidia_launch import compilation_options
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    count = 8388608
    arrays = [torch.full((count + 32,), -172, dtype=dtype, device='cuda')
              for dtype in (torch.float32, torch.float32, torch.int32, torch.int32)]
    actual, reference, xb, yb = [v[16:-16] for v in arrays]
    control = torch.tensor([-99, -20, 1, 0], dtype=torch.int32, device='cuda')
    contract = ResourceContract(compute_num_warps=8, resident_blocks_per_sm=1,
                                requires_grid_wide_sync=True)
    options = compilation_options(contract)
    options.update(launch_cooperative_grid=True, enable_fp_fusion=False)
    program = prepare_jit_kernel(division_probe, (actual, reference, xb, yb, control), (),
                                 grid=(128, 1, 1), contract=contract,
                                 allocation_device='cuda:0', **options)
    assert program.resource_report['spill_bytes'] == 0
    with torch.cuda.graph(graph := torch.cuda.CUDAGraph(), stream=stream):
        program.launch()
    indices = (torch.arange(1280, device='cuda', dtype=torch.int64) * 65537 + 17) % count
    cases = [(-99 + i * 99 // 31, -20 + i % 22, i + 1) for i in range(32)]
    cases += [(-99, 1, 97), (0, -20, 99), (-50, -20, 113), (0, 1, 127)]
    begin = time.monotonic()
    for case, (xe, ye, seed) in enumerate(cases):
        control.copy_(torch.tensor([xe, ye, seed, 0], dtype=torch.int32, device='cuda'))
        for replay in range(3):
            if replay < 2:
                graph.replay()
            else:
                program.launch()
            torch.cuda.synchronize()
            mismatch = actual.view(torch.int32).ne(reference.view(torch.int32))
            bad = int(mismatch.sum().item())
            assert bad == 0, (case, bad, int(mismatch.nonzero()[0].item()))
            assert bool(torch.isfinite(actual).all()) and bool(actual.gt(0).all())
            for value in arrays:
                assert bool(value[:16].eq(-172).all()) and bool(value[-16:].eq(-172).all())
        xs, ys = xb[indices].cpu().tolist(), yb[indices].cpu().tolist()
        got = actual.view(torch.int32)[indices].cpu().tolist()
        assert all(integer_rn32(x, y) == q for x, y, q in zip(xs, ys, got, strict=True))
    result = dict(passed=True, diagnostic_only=True, full_model_acceptance=False,
                  exhaustive_pairs=count * len(cases), integer_RN32_oracle_pairs=1280 * len(cases),
                  two_graph_and_eager=True, all_bytes_equal=True, canaries=True,
                  seconds=time.monotonic() - begin, resources=program.resource_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
