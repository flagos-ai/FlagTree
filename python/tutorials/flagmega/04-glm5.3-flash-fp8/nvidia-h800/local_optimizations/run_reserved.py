"""Run full TP4 validation or measurement with the original memory boundary."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('numerics', 'measure'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--exclude-device', type=int, action='append', default=[])
    args = parser.parse_args()
    here = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    sdk = args.sdk.resolve()
    warmup = json.loads((here/'warmup.json').read_text())
    source = here/'artifact/generated_kernels.py'
    assert warmup['jit_ok'] and warmup['joint_resources_ok']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == warmup['source_sha256']
    revision = json.loads((here/'local_optimizations/source-revision.json').read_text())
    assert hashlib.sha256((sdk/'libflagcx_device.bc').read_bytes()).hexdigest() == \
        revision['measured_device_bc_sha256']
    # A fresh host build includes its own paths and ELF metadata. Record the
    # actual library identity and validate its protocol through the full run.
    host_sha256 = hashlib.sha256((sdk/'libflagcx.so').read_bytes()).hexdigest()
    runner = (here/'local_optimizations/numerics_digests.py' if args.mode == 'numerics'
              else here/'benchmark_gpu.py')
    command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1',
               '--nproc-per-node=4', '--rdzv-backend=c10d',
               '--rdzv-endpoint=127.0.0.1:0', '--local-addr=127.0.0.1',
               str(runner), '--trial', str(here), '--output', str(output)]
    if args.mode == 'numerics':
        reference = (args.reference or here/'local_optimizations/full-state-reference-digests.json').resolve()
        assert reference.is_file()
        command += ['--reference', str(reference), '--mode', 'check-owned']
    exclusion = sorted({0, 6, *args.exclude_device})
    reserve = [sys.executable, str(here/'local_optimizations/reserve_probe_memory_run.py'),
               '--output', str(output), '--num-devices', '4',
               '--budget-bytes', '80000017920']
    for device in exclusion:
        reserve += ['--exclude-device', str(device)]
    reserve += ['--', *command]
    environment = dict(os.environ, BASH_ENV='/dev/null', ENV='/dev/null',
                       PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', GLOO_SOCKET_IFNAME='lo',
                       FLAGCX_MODULE_PATH=str(sdk))
    environment.pop('TRITON_PTXAS_PATH', None)
    environment.pop('TRITON_CACHE_DIR', None)
    inputs = [runner, here/'warmup.json', source, sdk/'libflagcx.so', sdk/'libflagcx_device.bc']
    for directory in ('local_optimizations',):
        inputs.extend(p for p in (here/directory).rglob('*')
                      if p.is_file() and p.suffix in {'.py', '.jinja'})
    if args.mode == 'numerics':
        inputs.append(reference)
    identities = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    begin = time.monotonic()
    child = subprocess.Popen(reserve, env=environment)
    code = child.wait()
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
                    for p, digest in identities.items())
    receipt = dict(actual_exit_code=code, exit_code=code, real_child_wait=True,
                   supervisor_pid=child.pid, seconds=time.monotonic()-begin,
                   timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   sdk_host_sha256=host_sha256, source_receipts_unchanged=unchanged,
                   files=identities, performance_acceptance=False)
    if output.exists():
        (output/'supervisor-exit.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'files'}), flush=True)
    if code:
        raise SystemExit(code)
    assert unchanged, 'Reproduction inputs changed during execution'
    reports = [json.loads((output/f'rank-{rank}.json').read_text()) for rank in range(4)]
    assert all(r['passed'] and r['source_sha256'] == warmup['source_sha256'] for r in reports)
    expected = {e['symbol']: e['hash'] for e in warmup['entries']}
    assert all(r['compiled_entries'] == expected and len(r['resources']) == 228
               and all(k['spill_bytes'] == 0 for k in r['resources']) for r in reports)
    tokens = [3847, 284, 1373, 8282, 4218, 3755, 947, 906]
    count = 2 if args.mode == 'numerics' else 12
    assert all(len(r['requests']) == count
               and all(q['tokens'] == tokens for q in r['requests']) for r in reports)
    if args.mode == 'numerics':
        assert all(q['all_state_bytes_equal'] for r in reports for q in r['requests'])
    (output/'verified.json').write_text(json.dumps(dict(passed=True, mode=args.mode,
        source_sha256=warmup['source_sha256'], ranks=4, requests_per_rank=count,
        all_official8_passed=True, all_zero_spill=True, normal_runtime=True,
        state_byte_digests_verified=args.mode == 'numerics',
        sdk_host_sha256=host_sha256, performance_acceptance=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
