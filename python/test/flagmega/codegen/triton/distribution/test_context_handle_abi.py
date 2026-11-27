# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Communication handles are runtime data, independent of the compiled kernel."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _compile_context(package_path, memory, communication, mode_token=None):
    import torch
    from triton.flagmega.runtime.module import _torch_dtype
    from triton.flagmega.runtime.nvidia_launch import compilation_options
    from triton.flagmega.runtime.prepared import ResourceContract
    from triton.runtime import DistributedRtContext

    path = Path(package_path)
    package = json.loads((path / 'codegen.json').read_text())
    spec = importlib.util.spec_from_file_location('_flagmega_context_cache_probe', path / package['source'])
    generated = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = generated
    spec.loader.exec_module(generated)
    values = {}
    for arg in package['runtime_binding']['arguments']:
        values[arg['name']] = torch.empty(256, dtype=_torch_dtype(
            torch, package['test_argument_dtypes'][arg['name']]), device='cuda')
    for pool in package['runtime_binding']['pools']:
        values[pool['name']] = torch.empty(256, dtype=torch.uint8, device='cuda')
    DistributedRtContext._reset()
    values['device_dptr'] = DistributedRtContext(int(communication), int(memory))
    values['device_mem_ptr'] = int(memory)
    values['device_comm_ptr'] = int(communication)
    values['device_context_mode'] = (DistributedRtContext().is_lite_mode
                                     if mode_token is None else bool(int(mode_token)))
    arguments = tuple(values[name] for name in package['signature_arguments'])
    contract = ResourceContract(compute_num_warps=package['num_warps'], resident_blocks_per_sm=1)
    kernel = getattr(generated, package['symbol']).run(
        *arguments, grid=tuple(package['grid']), warmup=True, **compilation_options(contract))
    address_literals = any(str(value) in kernel.asm['ptx'] for value in (int(memory), int(communication)))
    # The fake handles are compilation inputs only: never launch this kernel.
    print(json.dumps({'kernel_hash': kernel.hash, 'constants': [int(memory), int(communication)],
                      'address_literals': address_literals,
                      'lite_mode': DistributedRtContext().is_lite_mode}), flush=True)


def _package(tmp_path):
    from triton.flagmega.codegen.triton import render_triton_package
    from triton.flagmega.ir import verify_buffer_plan

    from python.test.flagmega.codegen.triton.kernels.norm_apply.test_cross_chip_stats_runtime import (
        cross_chip_norm_module,
    )

    module = cross_chip_norm_module()
    package = render_triton_package(module, tmp_path)
    plan = verify_buffer_plan(module)
    package['test_argument_dtypes'] = {arg['name']: plan.buffer_map[arg['buffer']].dtype.value
                                      for arg in package['runtime_binding']['arguments']}
    (tmp_path / 'codegen.json').write_text(json.dumps(package) + '\n')
    return package


def test_generated_handle_arguments_have_a_runtime_abi(tmp_path):
    package = _package(tmp_path)
    assert package.get('device_context_abi') == 'runtime-handles/v2'
    assert 'device_dptr' not in package['signature_arguments']
    assert 'device_mem_ptr' in package['signature_arguments']
    assert 'device_comm_ptr' in package['signature_arguments']
    assert 'device_context_mode' in package['signature_arguments']
    assert 'device_context_mode: tl.constexpr' in (tmp_path / package['source']).read_text()
    assert 'device_dptr: tl.constexpr' not in (tmp_path / package['source']).read_text()


def test_context_addresses_do_not_recompile_the_same_generated_kernel(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('SM90 CUDA is required')
    _package(tmp_path)
    hashes = []
    for base in (0x7F1000000000, 0x7F2000000000):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path),
                                 str(base + 0x1000), str(base + 0xE00)],
                                capture_output=True, text=True, timeout=120, check=False,
                                env=dict(os.environ, FLAGTREE_LITE_DIST='0'))
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(result.stdout.splitlines()[-1])
        assert not report['address_literals'], 'Process addresses leaked into PTX constants'
        hashes.append(report['kernel_hash'])
    assert hashes[0] == hashes[1], 'The same kernel was specialized on process-local communication addresses'


def test_injected_context_mode_has_a_distinct_compilation_identity(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('SM90 CUDA is required')
    _package(tmp_path)
    reports = []
    for mode in ('0', '1'):
        # A compile-only control for the ABI discriminator. Keep the backend
        # fixed so unrelated injected-parameter passes cannot hide a missing
        # discriminator in the generated signature. Never execute fake handles.
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path),
                                 str(0x7F1000001000), str(0x7F1000000E00), mode],
                                capture_output=True, text=True, timeout=120, check=False,
                                env=dict(os.environ, FLAGTREE_LITE_DIST='0',
                                         TRITON_CACHE_DIR=str(tmp_path / f'cache-{mode}')))
        assert result.returncode == 0, result.stdout + result.stderr
        reports.append(json.loads(result.stdout.splitlines()[-1]))
    assert reports[0]['kernel_hash'] != reports[1]['kernel_hash'], reports


@pytest.mark.parametrize('names', [('device_mem_ptr', 'device_comm_ptr'),
                                  ('device_context_mode', 'device_dptr')])
def test_user_tensor_names_cannot_shadow_runtime_handles(tmp_path, names):
    from triton.flagmega import ir as fm
    from triton.flagmega.codegen.triton import render_triton_package
    from triton.flagmega.compiler import Compiler

    tensor = fm.tensor_type('bfloat16', (1, 512))
    mesh = fm.Placement((4, 1), 'cx', 'cb')
    shard = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 128)), mesh)

    class Graph(fm.Module):
        def forward(self):
            x = self.input(names[0], tensor, id=names[0])
            y = self.input(names[1], tensor, id=names[1])
            a, b = (fm.F.distributed.boxing(value, shard) for value in (x, y))
            result = fm.F.distributed.boxing(fm.F.math.add(a, b), tensor)
            self.function('main', (x, y), (result,))

    module = Compiler().compile(Graph(dialect='distributed', stage='frozen_constants', entry='main',
                               metadata={'auto_distribution': {'placement': mesh.to_data()}}).build()).module
    package = render_triton_package(module, tmp_path)
    signature = package['signature_arguments']
    assert len(signature) == len(set(signature))
    assert {'device_mem_ptr', 'device_comm_ptr'} <= set(signature)
    assert all(arg['name'] not in ('device_mem_ptr', 'device_comm_ptr', 'device_context_mode', 'device_dptr')
               for arg in package['runtime_binding']['arguments'])


def test_host_descriptor_cannot_shadow_a_runtime_handle(tmp_path):
    from dataclasses import replace

    from triton.flagmega.codegen.triton import render_triton_package
    from triton.flagmega.codegen.triton.package_plan import plain_package_value
    from triton.flagmega.errors import CodegenError

    from python.test.flagmega.codegen.triton.kernels.norm_apply.test_cross_chip_stats_runtime import (
        cross_chip_norm_module,
    )
    from python.test.flagmega.runtime.descriptors.helpers import single_spec

    module = cross_chip_norm_module()
    package_plan = plain_package_value(module.metadata['codegen_package_plan'])
    package_plan['host_tensor_descriptor_specs'] = [single_spec(name='device_mem_ptr', source='workspace')]
    module = replace(module, metadata={**dict(module.metadata), 'codegen_package_plan': package_plan})
    with pytest.raises(CodegenError, match='unique argument names'):
        render_triton_package(module, tmp_path)


if __name__ == '__main__':
    _compile_context(*sys.argv[1:])
