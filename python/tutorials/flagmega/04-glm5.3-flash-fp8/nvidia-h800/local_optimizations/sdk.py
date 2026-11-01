"""Build the measured FlagCX protocol from a pinned revision and local patches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--cuda', type=Path, default=Path('/usr/local/cuda-12.9'))
    parser.add_argument('--nccl', type=Path, default=Path('/usr'))
    parser.add_argument('--clang', default='/usr/bin/clang-18')
    parser.add_argument('--gcc-install-dir', type=Path,
                        default=Path('/usr/lib/gcc/x86_64-linux-gnu/13'))
    parser.add_argument('--opt', default='/usr/bin/opt-18')
    parser.add_argument('--llvm-dis', default='/usr/bin/llvm-dis-18')
    parser.add_argument('--llvm-as', default='/usr/bin/llvm-as-18')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    revision = json.loads((here/'source-revision.json').read_text())
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    env = dict(os.environ, BASH_ENV='/dev/null', ENV='/dev/null')
    receipts = []

    def run(label, command, cwd=None):
        begin = time.monotonic()
        with (output/f'{label}.log').open('w') as log:
            child = subprocess.Popen(command, cwd=cwd, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            code = child.wait()
        record = dict(label=label, command=command, cwd=str(cwd) if cwd else None,
                      actual_exit_code=code, real_child_wait=True,
                      seconds=time.monotonic()-begin)
        receipts.append(record)
        (output/'receipts.json').write_text(json.dumps(receipts, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        if code:
            raise SystemExit(code)

    if args.source:
        source = args.source.resolve()
        head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                       text=True).strip()
        assert head == revision['base_commit'], (head, revision['base_commit'])
        # A supplied source must already contain exactly these patches.
        for patch in revision['patch_order']:
            run('check-applied-'+Path(patch).stem,
                ['git', 'apply', '--reverse', '--check', str(here.parent/'patches'/patch)], source)
    else:
        source = output/'source'
        run('init', ['git', 'init', str(source)])
        run('remote', ['git', 'remote', 'add', 'origin', revision['repository']], source)
        run('fetch', ['git', 'fetch', '--depth=1', 'origin', revision['base_commit']], source)
        run('checkout', ['git', 'checkout', '--detach', 'FETCH_HEAD'], source)
        for patch in revision['patch_order']:
            run('apply-'+Path(patch).stem, ['git', 'apply', str(here.parent/'patches'/patch)], source)

    run('host-contract', [sys.executable, str(source/revision['host_test'])], source)
    host = output/'host-build'
    bitcode = output/'bitcode-build'
    common = ['SHELL=/bin/bash', 'USE_NVIDIA=1', f'DEVICE_HOME={args.cuda}',
              f'CCL_HOME={args.nccl}']
    run('host', ['make', f'-j{args.jobs}', *common, 'COMPILE_KERNEL=0',
                 f'BUILDDIR={host}', str(host/'lib/libflagcx.so')], source)
    run('bitcode', ['make', '-j2', *common, f'BUILDDIR={bitcode}',
                    f'CLANG={args.clang} --gcc-install-dir={args.gcc_install_dir}',
                    f'OPT={args.opt}', f'LLVM_DIS={args.llvm_dis}',
                    f'LLVM_AS={args.llvm_as}', 'llvm_ir'], source/'bindings/ir/nvidia')
    runtime = output/'runtime'
    runtime.mkdir()
    shutil.copytree(source/'flagcx/include', runtime/'include')
    shutil.copy2(host/'lib/libflagcx.so', runtime/'libflagcx.so')
    shutil.copy2(bitcode/'lib/libflagcx_device.bc', runtime/'libflagcx_device.bc')
    files = {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in runtime.rglob('*') if p.is_file()}
    report = dict(runtime=str(runtime), revision=revision['base_commit'], files=files,
                  measured_device_bc_identical=files['libflagcx_device.bc'] ==
                      revision['measured_device_bc_sha256'],
                  measured_host_so_identical=files['libflagcx.so'] ==
                      revision['measured_host_so_sha256'],
                  no_driver_calls=True, gpu_acceptance=False,
                  performance_acceptance=False)
    (output/'runtime.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
