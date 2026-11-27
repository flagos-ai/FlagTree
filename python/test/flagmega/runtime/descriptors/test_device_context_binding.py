# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Runtime handles precede descriptors and remain static prepared arguments."""

from types import SimpleNamespace

import pytest
import triton.flagmega.runtime.module as runtime_module
from triton.flagmega.errors import ArtifactError
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

from .helpers import single_spec
from .test_call_graph_runtime_binding import _Prepared


def _runtime(monkeypatch, abi, *, lite=False, descriptor_name='weight_descriptor'):
    monkeypatch.setattr(runtime_module, 'verify_buffer_plan', lambda module: object())
    names = ('device_mem_ptr', 'device_comm_ptr') if abi else ('device_dptr',)
    if abi == 'runtime-handles/v2':
        names = (*names, 'device_context_mode')
    codegen = {
        'runtime_binding': {'arguments': ({'name': 'source', 'buffer': 'input'},),
                            'pools': (), 'signature': ('source',)},
        'distributed_device_dptr': True,
        'host_tensor_descriptor_specs': (single_spec(source='source', name=descriptor_name),),
        'signature_arguments': ('source', *names, descriptor_name),
        'dynamic_argument_indices': (0, 1 + len(names)),
        'symbol': 'entry', 'num_warps': 1, 'grid': (1, 1, 1),
    }
    if abi:
        codegen['device_context_abi'] = abi
    result = GeneratedTirCallGraphModule(None, {'codegen': codegen}, SimpleNamespace(metadata={}), object())
    result._device_dptr = SimpleNamespace(get_packed_data=lambda: (11, 22), is_lite_mode=lite)
    result._mark_loaded('cpu')
    prepared, captured = _Prepared(), {}

    def prepare(kernel, arguments, dynamic_indices, **options):
        captured.update(arguments=arguments, dynamic_indices=dynamic_indices)
        return prepared

    monkeypatch.setattr(runtime_module, 'prepare_jit_kernel', prepare)
    return result, prepared, captured


@pytest.mark.parametrize('abi', (None, 'runtime-handles/v1', 'runtime-handles/v2'))
@pytest.mark.parametrize('lite', (False, True))
def test_context_and_external_descriptors_keep_their_abi_positions(monkeypatch, abi, lite):
    torch = pytest.importorskip('torch')
    runtime, prepared, captured = _runtime(monkeypatch, abi, lite=lite)
    source = torch.empty((3, 4))
    updated = torch.empty((5, 4))
    runtime.prepare(source)
    runtime.run_into(updated)
    if abi == 'runtime-handles/v2':
        assert captured['arguments'][1:4] == (11, 22, lite)
        assert captured['dynamic_indices'] == (0, 4)
    elif abi:
        assert captured['arguments'][1:3] == (11, 22)
        assert captured['dynamic_indices'] == (0, 3)
    else:
        assert captured['arguments'][1] is runtime._device_dptr
        assert captured['dynamic_indices'] == (0, 2)
    assert captured['arguments'][-1].base.data_ptr() == source.data_ptr()
    assert prepared.launches[0][0][0] is updated
    assert prepared.launches[0][0][1].base.data_ptr() == updated.data_ptr()


def test_unknown_context_abi_is_rejected(monkeypatch):
    with pytest.raises(ArtifactError, match='Unsupported distributed device context ABI'):
        _runtime(monkeypatch, 'unknown/v99')


@pytest.mark.parametrize('name', ('device_mem_ptr', 'device_comm_ptr', 'device_context_mode'))
def test_descriptor_cannot_shadow_context_arguments(monkeypatch, name):
    with pytest.raises(ArtifactError, match='context argument'):
        _runtime(monkeypatch, 'runtime-handles/v2', descriptor_name=name)
