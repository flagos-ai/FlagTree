"""Export exact emitted integrated H001 entries, without kernel launches.

Reproduction: PYTHONPATH=<repo>/python:<work> python -m
agent_optimizations.iq4_streaming.evidence_resources --artifact <artifact>
--baseline <baseline-resource01/resources.json> --out <fresh-attempt>.
This compiles GPU kernels with the installed compiler; it never rebuilds it.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from .audit_h001 import audit
from .qualify import resource


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    baseline = {r['entry']: r for r in json.loads(args.baseline.read_text())['resources']}
    report = dict(status='running', artifact=pin(args.artifact/'artifact.json'),
                  baseline=pin(args.baseline), executable=pin(__file__),
                  python=sys.executable, kernel_launches=0, ordinary_timing=False,
                  performance_accepted=False, resources=[])
    (args.out/'resources.json').write_text(json.dumps(report, indent=2)+'\n')
    # All 12 selected IQ4 entries. K/N and launch grids are the emitted ABI.
    with open('/tmp/flagmega-gpu0.lock', 'a') as lock:
        while True:
            fcntl.flock(lock, fcntl.LOCK_EX)
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True)
            if not apps.strip():
                break
            fcntl.flock(lock, fcntl.LOCK_UN)
            print('waiting for idle GPU: '+apps.strip(), flush=True)
            time.sleep(5)
        report['lease'] = dict(path=lock.name, pid=os.getpid(), apps_before=apps,
                              utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        report['device'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,pci.bus_id,driver_version,clocks.sm,clocks.mem,temperature.gpu', '--format=csv,noheader'], text=True).strip()
        try:
            for i in (2, 4, 6, 8, 9, 12, 14, 16, 18, 20, 21, 24):
                entry = f'flagmega_main_program_{i}'
                source = (args.artifact/'program_sources'/f'{entry}.py').resolve()
                before = pin(source)
                spec = importlib.util.spec_from_file_location('h001_evidence_'+entry, source)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                fn = getattr(mod, entry)
                # Read pointer types directly from the emitted launch bindings.
                inv = mod.FLAGMEGA_KERNEL_PROGRAMS[0]
                dtype = dict(bfloat16='*bf16', float32='*fp32', int32='*i32',
                             int64='*i64', uint8='*u8', bool='i1')
                signature = {}
                for name, binding in zip(fn.arg_names, inv['argument_bindings']):
                    if binding['kind'] == 'pointer':
                        signature[name] = dtype[binding['dtype']]
                    elif binding['kind'] == 'immediate':
                        signature[name] = {'int32':'i32', 'int64':'i64', 'bool':'i1'}[binding['dtype']]
                    else:
                        signature[name] = '*i64' if name == '_flagmega_function_offsets' else '*u8'
                assert len(signature) == len(fn.arg_names)
                options = dict(num_warps=inv['num_warps'], enable_fp_fusion=True)
                if 'maxnreg' in inv:
                    options['maxnreg'] = inv['maxnreg']
                kernel = triton.compile(ASTSource(fn, signature=signature),
                                        target=GPUTarget('cuda', 120, 32),
                                        options=options)
                static = audit(kernel, args.out, entry)
                r = resource(kernel, args.out, entry)
                # Preserve source and PTX line annotations in the actual cubin.
                for flags, ext in [(['-c','-gi'], 'sass-source'), (['-c','-gp'], 'sass-ptx')]:
                    p = args.out/f'{entry}.{ext}'
                    p.write_text(subprocess.check_output(['/usr/local/cuda/bin/nvdisasm', *flags, str(args.out/f'{entry}.cubin')], text=True))
                    r['files'][ext] = pin(p)
                assert before == pin(source), 'emitted source changed during compilation'
                tree = ast.parse(source.read_text())
                helper = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                              and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                                      and c.func.id == '_iq4_stream_dot' for c in ast.walk(n)))
                call = next(c for c in ast.walk(helper) if isinstance(c, ast.Call)
                            and isinstance(c.func, ast.Name) and c.func.id == '_iq4_stream_dot')
                K = ast.literal_eval(call.args[5])
                N = inv['grid'][0]*4
                family = 'glu' if i in (8,20) else 'interleaved' if i == 16 else 'projection'
                b = baseline.get(entry)
                required = b['occupancy_api_ctas_per_sm'] if b else 4 if family == 'glu' else 5
                zero_spill = (r['spill_words_reported'] == r['local_bytes_per_thread'] == r['stack_bytes'] == 0)
                row = dict(entry=entry, calls=inv['calls'], N=N, K=K, family=family,
                           compile_options=options,
                           source=before, signature=signature, grid=inv['grid'],
                           resources=r, baseline=b, static=static,
                           admission=dict(required_ctas_per_sm=required,
                                          actual_ctas_per_sm=r['queried_ctas_per_sm'],
                                          passed=r['queried_ctas_per_sm'] >= required),
                           zero_spill=zero_spill,
                           glu_register_gate=dict(applicable=family=='glu', maximum=107,
                                                  passed=family!='glu' or r['registers_per_thread'] <= 107),
                           payload_sector=dict(status='pending', value=None),
                           activation_sector=dict(status='pending', value=None),
                           eligible_warps=dict(status='pending', value=None),
                           placement=dict(status='pending', reason='capacity query without kernel launches'))
                report['resources'].append(row)
                (args.out/'resources.json').write_text(json.dumps(report, indent=2)+'\n')
                print(entry, N, K, r['registers_per_thread'], r['queried_ctas_per_sm'],
                      'admission='+str(row['admission']['passed']), flush=True)
            report['status'] = 'complete-resource-gates-failed' if any(
                not r['admission']['passed'] or not r['glu_register_gate']['passed'] or not r['zero_spill']
                for r in report['resources']) else 'complete-resource-self-checks-passed'
        except BaseException as e:
            report.update(status='failed', error=repr(e), traceback=traceback.format_exc())
            raise
        finally:
            (args.out/'resources.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
