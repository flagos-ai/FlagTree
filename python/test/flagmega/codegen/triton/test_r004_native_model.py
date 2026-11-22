# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The model contract cannot redefine or impersonate standalone v4."""
from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.rules.ntt.r004_model_policy import prepare_model_column_distribution
from triton.flagmega.rules.ntt.r004_native_model import (
    SCHEMA, REQUEST_KEY, native_model_contract, native_model_request, prepare_native_model,
)
from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import standalone_request
from triton.flagmega.passes.tir.materialize_kernel_definitions import r004_phase_definitions
from triton.flagmega.rules.ntt.r004_column_phase_bindings import column_phase_plan
from .test_r004_model_policy import repeated_effect_fixture
from .test_r004_phase_provider import coarse_definition


def model():
    return prepare_native_model(prepare_model_column_distribution(repeated_effect_fixture()))


def test_model_contract_reload_and_default(tmp_path):
    unrequested = repeated_effect_fixture()
    assert native_model_request(unrequested) is None
    m = model()
    assert standalone_request(m) is None
    assert m.metadata[REQUEST_KEY]['schema'] == SCHEMA != 'r004.gdn.standalone/v4'
    assert 'root' not in m.metadata[REQUEST_KEY]['arena']
    fm.emit_module(m, tmp_path/'model.py')
    loaded = fm.load_module(tmp_path/'model.py')
    assert loaded.semantic_hash == m.semantic_hash
    assert prepare_native_model(loaded) is loaded


@pytest.mark.parametrize('key,value', [('schema','r004.gdn.standalone/v4'),
    ('readonly_rows',1), ('recurrent_calls',47), ('native_layer',3),
    ('state_binding','parent-plus-layer-twice'), ('arena_binding','shared-pointer'),
    ('constructor','standalone-12-roots')])
def test_malformed_model_contract_rejected_by_verifier(key, value):
    m = model()
    request = {**m.metadata[REQUEST_KEY], key:value}
    with pytest.raises(IRVerificationError, match='native model'):
        fm.verify_module(replace(m, metadata={**m.metadata, REQUEST_KEY:request}))


def test_model_standalone_exclusive_and_late_install():
    m = model()
    with pytest.raises(IRVerificationError):
        prepare_native_model(replace(m, stage='bufferized_tir'))
    with pytest.raises(IRVerificationError):
        native_model_request(replace(m, metadata={**m.metadata,
            'r004_standalone_request':dict(schema='r004.gdn.standalone/v4',
                                         module='r004_T_full_p5_v4',readonly_layer=0)}))


def test_model_phase_definition_scope_and_selected_state():
    from triton.flagmega.ir.tir.verify import verify_prim_function
    from triton.flagmega.ir.tir import KernelDispatch
    coarse = coarse_definition()
    plan = column_phase_plan()
    coarse = replace(coarse, dispatch=KernelDispatch('nn.gdn_recurrent_core',
        arguments=('input',), outputs=('output',), writes=('output',), candidate=plan['implementation'],
        parameters=dict(family='gdn_recurrent', variant='r004_column_v4', phase_plan=plan,
                        native_model_contract=native_model_contract())))
    definitions = r004_phase_definitions(coarse)
    assert [d.name for d in definitions] == ['r004_model_T_prep','r004_model_T_core','r004_model_T_y']
    for d in definitions:
        verify_prim_function(d)
        assert d.attrs['kernel_num_warps'] == 4
        assert d.attrs['kernel_maxnreg'] == 80
        assert d.dispatch.facts['internal_grid_barriers'] == 0
    core = definitions[1]
    state = next(p for p in core.parameters if p.name == 'StateParent')
    assert [d.fixed_value for d in state.type.shape] == [48,128,128]
    assert list(core.attrs['depends_on']) == ['r004_model_T_prep']
