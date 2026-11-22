"""Audit the cubins loaded by default runtime prepare, without model timing."""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

from .build import plain, pin, save

SELECTED = {2: 6, 4: 6, 6: 6, 8: 4, 9: 6, 12: 6, 14: 6,
            16: 5, 18: 6, 20: 4, 21: 6, 24: 6}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    dest = args.out / 'native'
    dest.mkdir()
    work = Path(__file__).resolve().parents[2]
    repo = work.parents[4]
    sys.path.insert(0, str(work / 'benchmark'))
    from benchmark_model import DecodeContext
    from triton.backends.nvidia.compiler import get_ptxas
    import triton.flagmega.runtime.module as runtime_module
    import torch

    torch.set_num_threads(1)
    manifest = json.loads((args.artifact / 'artifact.json').read_text())
    calls = {}
    for index, spec in enumerate(manifest['codegen']['kernels']):
        calls.setdefault(spec['entry_symbol'], []).append({**spec, 'runtime_index': index})
    ptxas = Path(get_ptxas(120).path)
    report = dict(schema='flagmega.runtime-loaded-cubin-audit/v1',
                  artifact=pin(args.artifact / 'artifact.json'), harness=pin(__file__),
                  compiler=pin(repo / 'python/triton/_C/libtriton.so'), ptxas=pin(ptxas),
                  ptxas_version=subprocess.check_output([str(ptxas), '--version'], text=True),
                  default_bootstrap=True, explicit_native_registration=False,
                  ordinary_timing=False, model_forward_calls=0, performance_accepted=False,
                  status='running', resources=[], utc=datetime.now(timezone.utc).isoformat(),
                  program_execution=manifest['codegen']['program_execution'],
                  boundary='CUfunction loaded by normal default runtime prepare; 128 threads, actual dynamic shared')
    result_path = args.out / 'runtime-resources.json'
    save(result_path, report)
    original = runtime_module.prepare_jit_kernel
    observed = {}
    cuda = ctypes.CDLL('libcuda.so.1')
    context = None
    preparation_index = 0

    def observe(kernel, arguments, dynamic_argument_indices, **kw):
        nonlocal preparation_index
        index = preparation_index
        preparation_index += 1
        prepared = original(kernel, arguments, dynamic_argument_indices, **kw)
        compiled = prepared.compiled_kernel
        number = int(compiled.name.rsplit('_', 1)[-1])
        if number not in SELECTED:
            return prepared
        invocation = manifest['codegen']['kernels'][index]
        assert invocation['entry_symbol'] == compiled.name
        actual_invocation = dict(runtime_index=index, symbol=invocation['symbol'],
            call_path=invocation.get('call_path'), CUfunction=int(compiled.function),
            scalar_arguments={name: value for name, value in zip(kernel.arg_names, arguments)
                              if not callable(getattr(value, 'data_ptr', None))})
        if compiled.hash in observed:
            row = next(r for r in report['resources'] if r['compiled_hash'] == compiled.hash)
            row['prepared_invocations'].append(actual_invocation)
            return prepared
        assert compiled.function is not None
        assert compiled.kernel == compiled.asm['cubin']
        handle = ctypes.c_void_p(compiled.function)

        def attr(index):
            value = ctypes.c_int()
            rc = cuda.cuFuncGetAttribute(ctypes.byref(value), index, handle)
            if rc:
                raise RuntimeError(('cuFuncGetAttribute', index, rc))
            return value.value

        capacity = ctypes.c_int()
        dynamic_shared = int(compiled.metadata.shared)
        rc = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(capacity), handle, 128, ctypes.c_size_t(dynamic_shared))
        if rc:
            raise RuntimeError(('cuOccupancyMaxActiveBlocksPerMultiprocessor', rc))
        files = {}
        for kind in ('ttir', 'ttgir', 'llir', 'ptx', 'cubin'):
            path = dest / (compiled.name + '.' + kind)
            data = compiled.asm[kind]
            path.write_bytes(data if isinstance(data, bytes) else data.encode())
            files[kind] = pin(path)
        for binary, flags, extension in (
            ('cuobjdump', ['--dump-resource-usage'], 'resources.txt'),
            ('cuobjdump', ['--dump-sass'], 'sass'),
            ('nvdisasm', ['-c', '-gi'], 'sass-source'),
            ('nvdisasm', ['-c', '-gp'], 'sass-ptx')):
            path = dest / (compiled.name + '.' + extension)
            path.write_text(subprocess.check_output(
                ['/usr/local/cuda/bin/' + binary, *flags, files['cubin']['path']], text=True))
            files[extension] = pin(path)
        import re
        usage = Path(files['resources.txt']['path']).read_text()
        sass = Path(files['sass']['path']).read_text()
        resource_field = lambda key: int(re.search(r'\b' + key + r':(\d+)', usage)[1])
        row = dict(entry=compiled.name, compiled_hash=compiled.hash,
                   CUfunction=int(compiled.function), CUmodule=int(compiled.module),
                   loaded_cubin_sha256=hashlib.sha256(compiled.kernel).hexdigest(),
                   source=pin(args.artifact / 'program_sources' / (compiled.name + '.py')),
                   signature=compiled.src.signature, attrs=compiled.src.attrs,
                   constexprs={str(k): str(v) for k, v in compiled.src.constants.items()},
                   compile_options={k: v for k, v in kw.items()
                                    if k not in ('contract', 'allocation_device', 'grid')},
                   compiled_metadata={**compiled.metadata._asdict(),
                                      'target': asdict(compiled.metadata.target)},
                   invocation_mapping=calls[compiled.name],
                   prepared_invocations=[actual_invocation],
                   grid=list(prepared.grid), threads_per_CTA=128,
                   contract=asdict(prepared.contract), files=files,
                   registers_per_thread=attr(4), queried_ctas_per_sm=capacity.value,
                   required_ctas_per_sm=SELECTED[number],
                   static_shared_bytes=attr(1), dynamic_shared_bytes=dynamic_shared,
                   local_bytes_per_thread=attr(3), spill_words_reported=compiled.n_spills,
                   stack_bytes=resource_field('STACK'), cubin_shared_bytes=resource_field('SHARED'),
                   ptxas_spill_store_bytes=compiled.metadata.ptxas_spill_store_bytes,
                   ptxas_spill_load_bytes=compiled.metadata.ptxas_spill_load_bytes,
                   ptxas_stack_frame_bytes=compiled.metadata.ptxas_stack_frame_bytes,
                   global_scratch_bytes=prepared.resource_report['global_scratch_bytes'],
                   dynamic_argument_indices=list(dynamic_argument_indices),
                   argument_layouts=[dict(dtype=str(v.dtype), shape=list(v.shape),
                                          stride=list(v.stride()), alignment_mod_16=v.data_ptr() % 16)
                                     if callable(getattr(v, 'data_ptr', None))
                                     else dict(type=type(v).__name__, value=v) for v in arguments],
                   forbidden_SASS=[line for line in sass.splitlines()
                                   if re.search(r'\b(?:LDS|STS|LDL|STL|BAR|LDSM|LDGSTS)(?:\.|\b)', line)])
        row['driver_cuobjdump_registers_equal'] = row['registers_per_thread'] == compiled.n_regs == resource_field('REG')
        row['zero_spill_local_stack_shared'] = all(row[k] == 0 for k in (
            'static_shared_bytes', 'dynamic_shared_bytes', 'local_bytes_per_thread',
            'spill_words_reported', 'stack_bytes', 'cubin_shared_bytes',
            'ptxas_spill_store_bytes', 'ptxas_spill_load_bytes', 'ptxas_stack_frame_bytes',
            'global_scratch_bytes')) and not row['forbidden_SASS']
        row['admission_passed'] = capacity.value >= SELECTED[number]
        row['GLU_register_gate_passed'] = number not in (8, 20) or row['registers_per_thread'] <= 107
        row['same_loaded_object'] = row['loaded_cubin_sha256'] == files['cubin']['sha256']
        observed[compiled.hash] = prepared
        report['resources'].append(row)
        save(result_path, report)
        print(compiled.name, row['registers_per_thread'], capacity.value, flush=True)
        return prepared

    with open('/tmp/flagmega-gpu0.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            runtime_module.prepare_jit_kernel = observe
            context = DecodeContext(args.artifact)
            assert 'agent_optimizations.iq4_streaming.native' not in sys.modules
            assert len(report['resources']) == 12
            loaded = {p.compiled_kernel.hash: p.compiled_kernel.function for p in context.runtime._prepared}
            assert all(loaded[r['compiled_hash']] == r['CUfunction'] for r in report['resources'])
            report['default_prepare_passed'] = True
            report['passed'] = all(all(r[k] for k in (
                'same_loaded_object', 'driver_cuobjdump_registers_equal',
                'zero_spill_local_stack_shared', 'admission_passed', 'GLU_register_gate_passed'))
                                   for r in report['resources'])
            report['status'] = 'runtime-resource-gates-passed' if report['passed'] else 'runtime-resource-gates-failed'
        except BaseException as error:
            report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            runtime_module.prepare_jit_kernel = original
            if context is not None:
                context.close()
            save(result_path, report)


if __name__ == '__main__':
    main()
