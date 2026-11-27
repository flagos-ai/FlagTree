# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Declared device libraries must participate in the ordinary JIT lookup."""
import sys
from types import SimpleNamespace

from triton.flagmega.runtime.nvidia_launch import compilation_options
from triton.flagmega.runtime.prepared import ResourceContract
from triton.runtime.jit import compute_cache_key


def _provider(monkeypatch, libraries):
    class Distributed:
        def get_extern_libs(self):
            return libraries

    monkeypatch.setitem(sys.modules, 'triton.backends.nvidia.distributed',
                        SimpleNamespace(Distributed=Distributed))


def test_library_selection_changes_real_jit_lookup_and_returns_to_prior_key(monkeypatch):
    libraries = {'libflagcx': '/test/runtime-A/libflagcx_device.bc'}
    _provider(monkeypatch, libraries)
    contract = ResourceContract(compute_num_warps=8, resident_blocks_per_sm=1)
    first = compilation_options(contract)
    libraries['libflagcx'] = '/test/runtime-B/libflagcx_device.bc'
    second = compilation_options(contract)
    libraries['libflagcx'] = '/test/runtime-A/libflagcx_device.bc'
    third = compilation_options(contract)
    # This is the same lookup performed before JIT's backend option parsing.
    specialization = [('*u8', 'D'), ('u64', 'D'), ('u64', 'D')]
    key_cache = {}
    keys = [compute_cache_key(key_cache, specialization, options)
            for options in (first, second, third)]
    assert keys[0] != keys[1]
    assert keys[0] == keys[2]
    assert first['extern_libs']['libflagcx'] == '/test/runtime-A/libflagcx_device.bc'
    assert second['extern_libs']['libflagcx'] == '/test/runtime-B/libflagcx_device.bc'
    kernel_cache = {keys[0]: 'compiled-A', keys[1]: 'compiled-B'}
    assert [kernel_cache[key] for key in keys] == ['compiled-A', 'compiled-B', 'compiled-A']


def test_disabled_distributed_provider_preserves_launch_bounds(monkeypatch):
    _provider(monkeypatch, {})
    contract = ResourceContract(compute_num_warps=4, resident_blocks_per_sm=2)
    assert compilation_options(contract) == {'num_warps': 4, 'ptx_options': '--minnctapersm=2'}


def test_other_backend_does_not_consult_nvidia_library_provider(monkeypatch):
    class Distributed:
        def get_extern_libs(self):
            raise AssertionError('NVIDIA library provider queried for BI-V150')

    monkeypatch.setitem(sys.modules, 'triton.backends.nvidia.distributed',
                        SimpleNamespace(Distributed=Distributed))
    contract = ResourceContract(compute_num_warps=4, resident_blocks_per_sm=1)
    assert compilation_options(contract, 'iluvatar-bi-v150') == {'num_warps': 4}
