# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Template counter slots: runtime-owned int64 counters, zeroed at prepare,
passed after the device context and before descriptors."""

from types import SimpleNamespace

import pytest
import triton.flagmega.runtime.module as runtime_module
from triton.flagmega.errors import ArtifactError
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

from .helpers import single_spec
from .test_call_graph_runtime_binding import _Prepared


def _runtime(monkeypatch, slots, argument):
    monkeypatch.setattr(runtime_module, 'verify_buffer_plan', lambda module: object())
    names = ('device_dptr', *((argument,) if argument else ()))
    codegen = {
        'runtime_binding': {'arguments': ({'name': 'source', 'buffer': 'input'},),
                            'pools': (), 'signature': ('source',)},
        'distributed_device_dptr': True,
        'host_tensor_descriptor_specs': (single_spec(source='source', name='weight_descriptor'),),
        'signature_arguments': ('source', *names, 'weight_descriptor'),
        'dynamic_argument_indices': (0, 1 + len(names)),
        'symbol': 'entry', 'num_warps': 1, 'grid': (1, 1, 1),
    }
    if slots is not None:
        codegen.update(counter_slots=slots, counter_slot_argument=argument)
    result = GeneratedTirCallGraphModule(None, {'codegen': codegen}, SimpleNamespace(metadata={}), object())
    result._device_dptr = SimpleNamespace(get_packed_data=lambda: (11, 22), is_lite_mode=False)
    result._mark_loaded('cpu')
    captured = {}

    def prepare(kernel, arguments, dynamic_indices, **options):
        captured.update(arguments=arguments, dynamic_indices=dynamic_indices)
        return _Prepared()

    monkeypatch.setattr(runtime_module, 'prepare_jit_kernel', prepare)
    return result, captured


def test_prepare_zeroes_and_passes_the_counter_slots(monkeypatch):
    torch = pytest.importorskip('torch')
    runtime, captured = _runtime(monkeypatch, 2, '_flagmega_counters')
    # Left non-zero by a previous launch sequence; prepare restores the contract.
    runtime._counter_values = torch.full((2,), 41, dtype=torch.int64)
    runtime.prepare(torch.empty((3, 4)))
    assert captured['arguments'][1] is runtime._device_dptr
    assert captured['arguments'][2] is runtime._counter_values
    assert runtime._counter_values.tolist() == [0, 0]
    assert captured['dynamic_indices'] == (0, 3)


def test_artifacts_without_counter_slots_pass_no_counter(monkeypatch):
    torch = pytest.importorskip('torch')
    runtime, captured = _runtime(monkeypatch, None, None)
    runtime.prepare(torch.empty((3, 4)))
    assert runtime._counter_values is None
    assert len(captured['arguments']) == 3


@pytest.mark.parametrize(('slots', 'argument'), ((1, None), (0, '_flagmega_counters'), (-1, '_flagmega_counters')))
def test_inconsistent_counter_slot_abi_is_rejected(monkeypatch, slots, argument):
    with pytest.raises(ArtifactError, match='counter-slot ABI'):
        _runtime(monkeypatch, slots, argument)
