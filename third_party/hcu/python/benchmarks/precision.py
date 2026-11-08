"""HCU arithmetic and data movement measurements for the FlagTree HCU backend."""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=None)
parser.add_argument('--kernels', type=Path, default=None)
args = parser.parse_args()
visible = os.environ.get('HIP_VISIBLE_DEVICES', '')
if not visible.isdigit() or os.environ.get('CUDA_VISIBLE_DEVICES') != visible:
    raise ValueError('Set HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES to the same single physical device')
for sample in range(2):
    smi = subprocess.run(['rocm-smi', '-d', visible, '--showuse', '--showmemuse'], capture_output=True, text=True,
                         timeout=30)
    print('DEVICE_SAMPLE', sample, smi.stdout, flush=True)
    use = re.search(r'HCU use \(%\):\s*([\d.]+)', smi.stdout)
    memory = re.search(r'HCU memory use \(%\):\s*([\d.]+)', smi.stdout)
    if smi.returncode or not use or not memory or float(use[1]) or float(memory[1]):
        raise RuntimeError('The selected device is not idle')
    if sample == 0:
        time.sleep(2)

import torch
import triton
from triton.testing import do_bench

torch.set_num_threads(8)
torch.cuda.set_device(0)
target = triton.runtime.driver.active.get_current_target()
if target.backend != 'hip' or target.arch.split(':')[0] != 'gfx936':
    raise ValueError('This benchmark is validated for the FlagTree HCU gfx936 backend')
default_kernels = Path(__file__).resolve().parent.parent / 'test/unit/language/test_precision.py'
module_path = args.kernels or Path(os.environ.get('HCU_TEST_KERNELS', str(default_kernels)))
spec = importlib.util.spec_from_file_location('precision_measurement_kernels', module_path)
kernels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernels)
out = args.output or Path(os.environ.get('HCU_BENCHMARK_OUTPUT', 'hcu-precision-' + str(time.time_ns())))
out.mkdir(parents=True)
(out / 'kernels.py').write_bytes(module_path.read_bytes())
results = []


def save_ir(label, compiled):
    for stage in ('ttir', 'ttgir', 'llir', 'amdgcn'):
        (out / (label + '.' + stage)).write_text(compiled.asm[stage])
    asm = compiled.asm['amdgcn']
    body = asm.split('\n' + compiled.metadata.name + ':', 1)[1]
    body = re.split(r'^\s*\.Lfunc_end\d+:', body, maxsplit=1, flags=re.MULTILINE)[0]
    return dict(collections.Counter(re.findall(r'^\s+([a-z]+_[a-zA-Z0-9_]+)\b', body, re.MULTILINE)))


def measure(label, launch, check, reference=None, io_bytes=0, flops=0, integer_ops=0, **context):
    start = time.perf_counter()
    compiled = launch()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - start) * 1000
    baseline = reference() if reference else None
    check()
    measured = do_bench(launch, warmup=25, rep=100, quantiles=[.5, .2, .8])
    reference_ms = do_bench(reference, warmup=25, rep=100, quantiles=[.5, .2, .8])[0] if reference else None
    check()
    row = dict(case=label, cold_compile_and_first_launch_ms=cold_ms, warm_median_us=measured[0] * 1000,
               warm_p20_us=measured[1] * 1000, warm_p80_us=measured[2] * 1000, n_regs=compiled.n_regs,
               shared_bytes=compiled.metadata.shared, kernel_static_instruction_counts=save_ir(label,
                                                                                               compiled), **context)
    if reference:
        row.update(reference_median_us=reference_ms * 1000, relative_to_reference=measured[0] / reference_ms,
                   reference_n_regs=baseline.n_regs, reference_shared_bytes=baseline.metadata.shared,
                   reference_kernel_static_instruction_counts=save_ir(label + '.reference', baseline))
    if io_bytes:
        row.update(logical_io_bytes=io_bytes, effective_GB_s=io_bytes / (measured[0] * 1e6))
    if flops:
        row['logical_dot_TFLOP_s'] = flops / (measured[0] * 1e9)
    if integer_ops:
        row['logical_integer_dot_TOP_s'] = integer_ops / (measured[0] * 1e9)
    results.append(row)
    print('BENCH_CASE', json.dumps(row), flush=True)
    (out / 'partial_results.json').write_text(json.dumps(results, indent=2))


def check_buffer(buffer, expected, storage=None, sentinel=1234.5):
    actual = buffer[16:-16].cpu()
    if storage:
        actual = actual.view(storage)
        expected = expected.view(storage)
        guards = buffer.view(storage)
    else:
        guards = buffer
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=0, atol=0)
    assert torch.all(guards[:16] == sentinel) and torch.all(guards[-16:] == sentinel)


for m, n, k in [(512, 512, 512), (1024, 1024, 1024)]:
    for fmt in ('e2m1', 'e4m3'):
        for scaled in (False, True):
            generator = torch.Generator().manual_seed(1493)
            a, ra = kernels.precision_operand(fmt, (m, k), 1, generator, 'cuda')
            b, rb = kernels.precision_operand(fmt, (k, n), 0, generator, 'cuda')
            sa = (126 + torch.arange(m * (k // 32)).reshape(m, k // 32) % 3).to(torch.uint8) if scaled else None
            sb = (126 + (torch.arange(n * (k // 32)).reshape(n, k // 32) + 1) % 3).to(torch.uint8) if scaled else None
            if scaled:
                ra *= torch.exp2(sa.double() - 127).repeat_interleave(32, 1)
                rb *= torch.exp2(sb.double() - 127).repeat_interleave(32, 1).T
            expected = (ra @ rb).float()
            native_a, native_b = ra.bfloat16().cuda(), rb.bfloat16().cuda()
            sa, sb = (sa.cuda(), sb.cuda()) if scaled else (None, None)
            output = torch.full((m * n + 32, ), 1234.5, device='cuda')
            reference_output = torch.full_like(output, 1234.5)
            args = (a, *a.stride(), sa, b, *b.stride(), sb, None, output[16:-16], m, n, k, fmt, fmt)
            native_args = (native_a, *native_a.stride(), None, native_b, *native_b.stride(), None, None,
                           reference_output[16:-16], m, n, k, 'bf16', 'bf16')
            grid = (triton.cdiv(m, 64), triton.cdiv(n, 64))

            def launch():
                return kernels.precision_scaled_dot_loop[grid](*args, BM=64, BN=64, BK=32, num_warps=4, num_stages=2)

            def reference():
                return kernels.precision_scaled_dot_loop[grid](*native_args, BM=64, BN=64, BK=32, NATIVE=True,
                                                               num_warps=4, num_stages=2)

            def check():
                check_buffer(output, expected)
                check_buffer(reference_output, expected)

            measure(f'gemm-{fmt}-{m}x{n}x{k}-' + ('both' if scaled else 'none'), launch, check, reference,
                    flops=2 * m * n * k, num_warps=4, num_stages=2, tile=[64, 64, 32],
                    comparison='predecoded and prescaled BF16 operands; same logical dot and launch options')

generator = torch.Generator().manual_seed(1493)
ai = torch.randint(-8, 8, (64, 32), generator=generator)
bi = torch.randint(-8, 8, (32, 64), generator=generator)
expected = (ai @ bi).int()
native_a, native_b = ai.to(dtype=torch.int8, device='cuda'), bi.to(dtype=torch.int8, device='cuda')
for fmt in ('int8', 'int4'):
    if fmt == 'int4':
        a = ((ai[:, ::2] & 15) | ((ai[:, 1::2] & 15) << 4)).to(dtype=torch.uint8, device='cuda')
        b = ((bi[::2, :] & 15) | ((bi[1::2, :] & 15) << 4)).to(dtype=torch.uint8, device='cuda')
    else:
        a, b = ai.to(dtype=torch.int8, device='cuda'), bi.to(dtype=torch.int8, device='cuda')
    output = torch.full((4096 + 32, ), 42, dtype=torch.int32, device='cuda')
    reference_output = torch.full_like(output, 42)

    def launch():
        if fmt == 'int4':
            return kernels.precision_int4_dot[(1, )](a, b, None, output[16:-16], 64, 64, 32, 1, 0, num_warps=4,
                                                     num_stages=1)
        return kernels.precision_dot[(1, )](a, b, None, output[16:-16], 64, 64, 32, num_warps=4, num_stages=1)

    def reference():
        return kernels.precision_dot[(1, )](native_a, native_b, None, reference_output[16:-16], 64, 64, 32, num_warps=4,
                                            num_stages=1)

    def check():
        check_buffer(output, expected, sentinel=42)
        if fmt != 'int8':
            check_buffer(reference_output, expected, sentinel=42)

    measure('single-cta-dot-' + fmt, launch, check, reference if fmt != 'int8' else None, integer_ops=2 * 64 * 64 * 32,
            comparison='native INT8 dot with identical integer values', num_warps=4, num_stages=1)

for fmt in ('int4', 'int8'):
    bound = 8
    av = torch.randint(-bound, bound, (16, 64), generator=generator)
    bv = torch.randint(-bound, bound, (64, 16), generator=generator)
    if fmt == 'int4':
        a = ((av[:, ::2] & 15) | ((av[:, 1::2] & 15) << 4)).to(dtype=torch.uint8, device='cuda')
        b = ((bv[::2, :] & 15) | ((bv[1::2, :] & 15) << 4)).to(dtype=torch.uint8, device='cuda')
    else:
        a, b = av.to(dtype=torch.int8, device='cuda'), bv.to(dtype=torch.int8, device='cuda')
    sa = ((torch.arange(32).reshape(16, 2) % 5).float() - 2) / 2
    sb = ((torch.arange(32).reshape(16, 2) % 7).float() - 3) / 4
    expected = torch.full((16, 16), .25)
    for group in range(2):
        partial = (av[:, group * 32:(group + 1) * 32] @ bv[group * 32:(group + 1) * 32, :]).int().float()
        expected += partial * sa[:, group, None] * sb[None, :, group]
    sa, sb, acc = sa.cuda(), sb.cuda(), torch.full((16, 16), .25, device='cuda')
    output = torch.full((256 + 32, ), 1234.5, device='cuda')
    measure(
        'single-cta-integer-scaled-' + fmt, lambda: kernels.precision_integer_scaled[(1, )]
        (a, sa, b, sb, acc, output[16:-16], 16, 16, 64, fmt, fmt, num_warps=4, num_stages=1, enable_fp_fusion=False),
        lambda: check_buffer(output, expected), integer_ops=2 * 16 * 16 * 64, group_size=32, num_warps=4, num_stages=1,
        enable_fp_fusion=False)

# Cover launch overhead, a masked tail, and large-vector throughput.
for count in (256, 16390, 2**20):
    source_f32 = ((torch.arange(count) % 31).float() - 15) / 4
    source_i32 = ((torch.arange(count) % 262144) - 131072).int()
    source_i32[:4] = torch.tensor([16842753, -16842753, -(2**31), 2**31 - 1], dtype=torch.int32)
    source_i8 = (torch.arange(count) % 256).to(torch.uint8).view(torch.int8)
    for label, host, dtype in [('fp32-fp8e4m3fn', source_f32, torch.float8_e4m3fn),
                               ('int32-bf16', source_i32, torch.bfloat16), ('int8-fp32', source_i8, torch.float32)]:
        data = host.cuda()
        expected = host.to(dtype)
        if label == 'int32-bf16':
            expected.view(torch.int16)[:4] = torch.tensor([0x4b81, 0xcb81 - 65536, 0xcf00 - 65536, 0x4f00],
                                                          dtype=torch.int16)
        storage = torch.uint8 if dtype == torch.float8_e4m3fn else torch.int16 if dtype == torch.bfloat16 else None
        output = (torch.full(
            (count +
             32, ), 165, dtype=torch.uint8, device='cuda').view(dtype) if dtype == torch.float8_e4m3fn else torch.full(
                 (count + 32, ), 42, dtype=dtype, device='cuda'))
        measure(
            f'cast-{label}-n{count}', lambda: kernels.precision_cast[
                (triton.cdiv(count, 1024), )](data, output[16:-16], count, 1024),
            lambda: check_buffer(output, expected, storage, 165 if dtype == torch.float8_e4m3fn else 0x4228
                                 if dtype == torch.bfloat16 else 42),
            io_bytes=count * (host.element_size() + output.element_size()), elements=count, num_warps=4, num_stages=1)

    data = source_f32.cuda()
    expected_codes = torch.zeros(count, dtype=torch.uint8)
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    distance = (source_f32.abs()[:, None] - levels).abs()
    expected_codes = distance.argmin(1)
    even = torch.where((distance == distance.min(1, keepdim=True).values) & (torch.arange(8) % 2 == 0), torch.arange(8),
                       8).min(1).values
    expected_codes = torch.where(even < 8, even, expected_codes).to(
        torch.uint8) | (torch.signbit(source_f32).to(torch.uint8) << 3)
    expected_packed = expected_codes[::2] | (expected_codes[1::2] << 4)
    output = torch.full((count // 2 + 32, ), 165, dtype=torch.uint8, device='cuda')
    measure(
        f'cast-fp32-fp4-n{count}', lambda: kernels.precision_fp4_encode[(triton.cdiv(count, 1024), )]
        (data, output[16:-16], count, 'rtne', 1024), lambda: check_buffer(output, expected_packed, sentinel=165),
        io_bytes=count * 4 + count // 2, logical_values=count)

    packed_count = count // 2
    host = (torch.arange(packed_count) % 256).to(torch.uint8)
    data = host.cuda()
    codes = torch.stack((host & 15, host >> 4), -1).reshape(-1).long()
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    expected = table[codes].bfloat16()
    decoded = torch.full((count + 32, ), 42, dtype=torch.bfloat16, device='cuda')
    measure(
        f'cast-fp4-bf16-n{count}', lambda: kernels.precision_fp4_decode[(triton.cdiv(packed_count, 1024), )]
        (data, None, decoded[16:-16], packed_count, 1024), lambda: check_buffer(decoded, expected, torch.int16, 0x4228),
        io_bytes=packed_count + count * 2, logical_values=count)

    expected_int4 = torch.stack(((host.to(torch.int8) << 4) >> 4, host.to(torch.int8) >> 4), -1).reshape(-1)
    unpacked = torch.full((count + 32, ), 42, dtype=torch.int8, device='cuda')
    repacked = torch.full((packed_count + 32, ), 165, dtype=torch.uint8, device='cuda')

    def check_int4():
        check_buffer(unpacked, expected_int4, sentinel=42)
        check_buffer(repacked, host, sentinel=165)

    measure(
        f'cast-int4-roundtrip-n{count}', lambda: kernels.precision_int4_roundtrip[
            (triton.cdiv(packed_count, 1024), )](data, unpacked[16:-16], repacked[16:-16], packed_count, 1024),
        check_int4, io_bytes=packed_count * 4, logical_values=count)

    for dtype in (torch.int32, torch.uint8, torch.float8_e4m3fn):
        storage = torch.int32 if dtype == torch.int32 else torch.uint8
        host = (torch.arange(count) * 37).to(storage)
        data = host.cuda().view(dtype)
        for async_request in (False, True):
            output = torch.full((count + 32, ), 42, dtype=storage, device='cuda').view(dtype)
            measure(
                'copy-' + str(dtype).removeprefix('torch.') + ('-async' if async_request else '-sync') + f'-n{count}',
                lambda: kernels.precision_tle_copy[(triton.cdiv(count, 1024), )]
                (data, output[16:-16], count, 1, async_request), lambda: check_buffer(output, host, storage, 42),
                io_bytes=count * host.element_size() * 2, elements=count, num_warps=4, num_stages=1)
        output = torch.full((count + 32, ), 42, dtype=storage, device='cuda').view(dtype)
        measure('copy-' + str(dtype).removeprefix('torch.') + f'-shared-roundtrip-n{count}',
                lambda: kernels.precision_shared_copy[(triton.cdiv(count, 1024), )](data, output[16:-16], count, True),
                lambda: check_buffer(output, host, storage, 42), io_bytes=count * host.element_size() * 2,
                elements=count, num_warps=4, num_stages=1)

report = dict(
    target='hip/gfx936', physical_device=int(visible), triton_version=triton.__version__,
    runtime_package=str(Path(triton.__file__).parent), build_snapshot=os.environ.get('HCU_BUILD_SNAPSHOT'),
    kernel_source=str(module_path), kernel_source_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
    timing='GPU events; 25 ms warmup, 100 ms measurement; median/p20/p80; launch overhead included', instruction_scope=
    'Static kernel-body counts, including address/prologue instructions; loops counted once and callees not expanded.',
    io_scope=
    'Application input/output bytes; excludes internal LDS traffic. BF16 comparison changes operand storage widths.',
    artifacts=str(out), results=results)
(out / 'results.json').write_text(json.dumps(report, indent=2))
print('BENCHMARK_RESULT_JSON', json.dumps(report), flush=True)
