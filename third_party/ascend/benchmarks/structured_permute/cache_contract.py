"""Cross-process cold/hot, dispatch, partial-stage and device-bitcode cache checks.

Each child owns its NPU context. The parent only mutates its dedicated cache.
No installed package or SDK file is modified to inject the alternate bitcode.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def worker(args):
    import torch
    import torch_npu
    import triton
    from triton.backends.ascend import compiler
    from bench import nested_kernel
    if args.bitcode:
        compiler.get_structured_permutation_bitcode = lambda: args.bitcode.resolve()
    calls = []
    real_run = subprocess.run

    def traced(cmd, *pos, **kw):
        if isinstance(cmd, (tuple, list)) and cmd:
            executable = Path(str(cmd[0])).name
            if executable in ('triton-mlir-opt', 'bishengir-opt', 'bishengir-compile', 'hivmc'):
                calls.append(executable)
        return real_run(cmd, *pos, **kw)

    subprocess.run = traced
    torch.npu.set_device(0)
    q, batch = 32, 41
    cpu = torch.arange(batch * 4 * q, dtype=torch.int32).remainder(97).float()
    x, z = cpu.npu(), torch.zeros(batch * q, device='npu')
    y, t, j = [torch.full_like(x, -111) for _ in range(3)]
    options = {} if args.explicit is None else {'enable_auto_blockify': bool(args.explicit)}
    kernel = nested_kernel[(batch, )](x, z, y, t, j, Q=q, **options)
    torch.npu.synchronize()
    ref = cpu.reshape(batch, 4, q)
    expected = ref[:, [0, 2, 1, 3], :].reshape(-1)
    transposed = ref.transpose(1, 2).contiguous().reshape(-1)
    verified = (torch.equal(y.cpu(), expected) and torch.equal(t.cpu(), transposed)
                and torch.equal(j.cpu(), transposed))
    bitcode = compiler.get_structured_permutation_bitcode()
    result = dict(status='pass' if verified else 'wrong_output', outputs_verified=verified, hash=kernel.hash,
                  compiler_calls=calls, enable_auto_blockify=kernel.metadata.enable_auto_blockify,
                  uses_structured_permutation=kernel.metadata.uses_structured_permutation,
                  metadata_group=kernel.metadata_group, bitcode_sha256=hashlib.sha256(bitcode.read_bytes()).hexdigest(),
                  package=triton.__file__, backend=compiler.__file__)
    args.output.write_text(json.dumps(result, indent=2, default=str) + '\n')
    assert verified, result


def suite(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = root / 'cache'
    assert not cache.exists(), 'Use a new result directory; cold-cache checks must really be cold'
    records = {}

    def run(name, auto, explicit=None, bitcode=None, binary_only=False):
        env = os.environ.copy()
        env.pop('TRITON_ALWAYS_COMPILE', None)
        env['TRITON_CACHE_DIR'] = str(root / 'cache-binary-only' if binary_only else cache)
        if binary_only: env['TRITON_STORE_BINARY_ONLY'] = '1'
        else: env.pop('TRITON_STORE_BINARY_ONLY', None)
        env['TRITON_ALL_BLOCKS_PARALLEL'] = str(auto)
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--output', str(root / (name + '.json'))]
        if explicit is not None: cmd += ['--explicit', str(explicit)]
        if bitcode: cmd += ['--bitcode', str(bitcode)]
        with (root / (name + '.log')).open('w') as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        path = root / (name + '.json')
        row = json.loads(path.read_text()) if path.exists() else {'status': 'child_failed'}
        assert result.returncode == 0 and row['status'] == 'pass', (name, row)
        records[name] = row
        return row

    cold0, hot0 = run('cold-default0', 0), run('hot-default0', 0)
    assert cold0['hash'] == hot0['hash'] and not hot0['compiler_calls']
    cold1, hot1 = run('cold-default1', 1), run('hot-default1', 1)
    assert cold0['hash'] != cold1['hash'], 'Dispatch contracts must not share a kernel cache key'
    assert cold1['compiler_calls'] and not hot1['compiler_calls']
    explicit0 = run('explicit0-env1', 1, explicit=0)
    explicit1 = run('explicit1-env0', 0, explicit=1)
    assert explicit0['enable_auto_blockify'] is False
    assert explicit1['enable_auto_blockify'] is True
    assert explicit0['hash'] == cold0['hash'] and explicit1['hash'] == cold1['hash']

    # Remove only the final binary: earlier MLIR/bytecode and metadata remain.
    binaries = [Path(p) for p in hot1['metadata_group'].values() if str(p).endswith('.npubin')]
    assert binaries
    for path in binaries:
        assert path.resolve().is_relative_to(cache)
        path.unlink()
    partial = run('partial-final-binary', 1)
    assert partial['hash'] == cold1['hash']
    assert 'bishengir-compile' in partial['compiler_calls']
    assert partial['outputs_verified'] and partial['uses_structured_permutation']

    # Link a valid, unused function into a private copy of the device library.
    from triton.backends.ascend.compiler import get_structured_permutation_bitcode
    original = get_structured_permutation_bitcode()
    probe, probe_bc, variant = root / 'probe.ll', root / 'probe.bc', root / 'variant.bc'
    probe.write_text('define internal void @flagtree_cache_contract_probe() { ret void }\n')
    subprocess.run([args.llvm_as, str(probe), '-o', str(probe_bc)], check=True)
    subprocess.run([args.llvm_link, str(original), str(probe_bc), '-o', str(variant)], check=True)
    changed = run('changed-device-bitcode', 1, bitcode=variant)
    changed_hot = run('hot-changed-device-bitcode', 1, bitcode=variant)
    assert changed['bitcode_sha256'] != cold1['bitcode_sha256']
    assert changed['hash'] != cold1['hash'] and changed['compiler_calls']
    assert changed_hot['hash'] == changed['hash'] and not changed_hot['compiler_calls']
    restored = run('restored-device-bitcode', 1)
    assert restored['hash'] == cold1['hash'] and not restored['compiler_calls']
    binary_cold = run('binary-only-cold', 0, binary_only=True)
    binary_hot = run('binary-only-hot', 0, binary_only=True)
    assert binary_cold['hash'] == binary_hot['hash'] and not binary_hot['compiler_calls']
    assert any(name.endswith('.npubin') for name in binary_cold['metadata_group'])
    assert not any(
        name.endswith(('.ttir', '.ttadapter', '.mlirbc', '.bcmlir')) for name in binary_cold['metadata_group'])
    (root /
     'summary.json').write_text(json.dumps(dict(status='pass', cases=len(records), records=records), indent=2) + '\n')
    print(f'PASS: {len(records)} cross-process cache contracts', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--root', type=Path)
    p.add_argument('--explicit', type=int, choices=[0, 1])
    p.add_argument('--bitcode', type=Path)
    p.add_argument('--llvm-as', default='llvm-as')
    p.add_argument('--llvm-link', default='llvm-link')
    args = p.parse_args()
    worker(args) if args.worker else suite(args)
