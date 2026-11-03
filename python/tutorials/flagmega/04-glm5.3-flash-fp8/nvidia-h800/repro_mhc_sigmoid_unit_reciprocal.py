"""Exact mHC sigmoid unit reciprocal across every normal y>=1 binary32 value.

Run in the normal FlagTree environment on a free GPU:
  python repro_mhc_sigmoid_unit_reciprocal.py --device-uuid GPU-... --output .local/sigmoid-repro.json
Uses the actual generated helper template, complete mantissas for exponents0..127,
two CUDA Graph replays, eager execution and an independent integer RN32/FTZ oracle.
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
# SourceFileLoader gives Triton the real template filename and source lines.
# This exercises the helper used by full generated programs.
import importlib.machinery
import importlib.util
import sys

helper_path = Path(__file__).parent / 'local_optimizations/sigmoid-reciprocal.py.jinja'
helper_loader = importlib.machinery.SourceFileLoader('_flagmega_sigmoid_repro_helper', str(helper_path))
helper_spec = importlib.util.spec_from_loader(helper_loader.name, helper_loader)
helper_module = importlib.util.module_from_spec(helper_spec)
helper_module.__dict__.update(triton=triton, tl=tl, libdevice=libdevice)
sys.modules[helper_spec.name] = helper_module
helper_loader.exec_module(helper_module)
unit_reciprocal = helper_module._flagmega_sigmoid_unit_reciprocal


@triton.jit
def division_probe(actual, reference, xbits, ybits, control):
    xe = tl.load(control)
    ye = tl.load(control + 1)
    seed = tl.load(control + 2)
    for base in tl.range(tl.program_id(0) * 1024, 8388608, 128 * 1024):
        i = base + tl.arange(0, 1024)
        mantissa = (i.to(tl.uint32) * 1664525 + seed.to(tl.uint32) * 1013904223) & 0x7fffff
        xb = tl.full((1024,), 0x3f800000, tl.uint32)
        yb = (i.to(tl.uint32) & 0x7fffff) | ((ye + 127).to(tl.uint32) << 23)
        x = tl.cast(xb, tl.float32, bitcast=True)
        y = tl.cast(yb, tl.float32, bitcast=True)
        q = unit_reciprocal(y)
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
    if exponent + 127 <= 0:
        return 0
    assert exponent + 127 < 255
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
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    import uuid
    assert uuid.UUID(actual_uuid.removeprefix("GPU-")) == uuid.UUID(args.device_uuid.removeprefix("GPU-"))
    import pynvml as nvml
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByUUID(args.device_uuid)
    torch.cuda.synchronize()
    registration = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
    observed_pids = [p.pid for p in registration]
    assert len(observed_pids) == 1, observed_pids
    owned_pid = observed_pids[0]
    hardware = []

    def inventory():
        pids = [p.pid for p in nvml.nvmlDeviceGetComputeRunningProcesses(handle)]
        sm = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
        temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
        thermal_mask = (nvml.nvmlClocksEventReasonSwThermalSlowdown
                        | nvml.nvmlClocksEventReasonHwThermalSlowdown)
        thermal = bool(nvml.nvmlDeviceGetCurrentClocksEventReasons(handle) & thermal_mask)
        row = dict(timestamp=time.time(), pids=pids, sm_mhz=sm,
                   temperature=temperature, thermal=thermal)
        hardware.append(row)
        assert pids == [owned_pid] and sm == 1980 and temperature < 78 and not thermal, row

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
    cases = [(0, ye, ye + 1) for ye in range(128)]
    begin = time.monotonic()
    for case, (xe, ye, seed) in enumerate(cases):
        control.copy_(torch.tensor([xe, ye, seed, 0], dtype=torch.int32, device='cuda'))
        for replay in range(3):
            if replay < 2:
                graph.replay()
            else:
                program.launch()
            torch.cuda.synchronize()
            inventory()
            mismatch = actual.view(torch.int32).ne(reference.view(torch.int32))
            bad = int(mismatch.sum().item())
            assert bad == 0, (case, bad, int(mismatch.nonzero()[0].item()))
            assert bool(torch.isfinite(actual).all()) and bool(actual.ge(0).all())
            for value in arrays:
                assert bool(value[:16].eq(-172).all()) and bool(value[-16:].eq(-172).all())
        xs, ys = xb[indices].cpu().tolist(), yb[indices].cpu().tolist()
        got = actual.view(torch.int32)[indices].cpu().tolist()
        assert all(integer_rn32(x, y) == q for x, y, q in zip(xs, ys, got, strict=True))
    result = dict(passed=True, diagnostic_only=True, full_model_acceptance=False,
                  denominator_domain="every normal binary32 y>=1; exponent0..127; FTZ and infinity edge",
                  exhaustive_pairs=count * len(cases), integer_RN32_oracle_pairs=1280 * len(cases),
                  two_graph_and_eager=True, all_bytes_equal=True, canaries=True,
                  actual_UUID=args.device_uuid, initial_GPU_owner_set_empty=True,
                  observed_registration_PIDs=observed_pids, hardware=hardware,
                  seconds=time.monotonic() - begin, resources=program.resource_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
