# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Corrected model subject must reject stale v4 contracts and mixed plans."""
from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError, IRSchemaError
from triton.flagmega.rules.ntt.r004_model_policy_v6 import prepare_model_column_distribution
from triton.flagmega.rules.ntt.r004_native_model import (
    native_model_contract, native_model_request, prepare_native_model, model_phase_symbol,
)
from triton.flagmega.rules.ntt.r004_column_phase_bindings import column_phase_plan as v4_plan
from triton.flagmega.rules.ntt.r004_column_phase_bindings_v6 import column_phase_plan
from triton.flagmega.rules.ntt.r004_model_policy import model_distribution_target
from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import _same_contract
from triton.flagmega.targets import NvidiaSm120Target
from .test_r004_model_policy import repeated_effect_fixture


def model():
    return prepare_native_model(prepare_model_column_distribution(repeated_effect_fixture()))


def test_corrected_subject_reload_and_provider(tmp_path):
    m = model()
    assert _same_contract(native_model_request(m), native_model_contract(6))
    assert native_model_contract(6)['schema'] == 'r004.gdn.native-model/v2'
    assert _same_contract(m.node_map['recurrent'].attrs['phase_plan'], column_phase_plan())
    fm.emit_module(m, tmp_path / 'model.py')
    fresh = fm.load_module(tmp_path / 'model.py')
    assert fresh.semantic_hash == m.semantic_hash
    assert model_distribution_target(NvidiaSm120Target(), fresh).distribution_policy.identity.endswith(
        column_phase_plan()['plan_body_sha256'])
    assert [model_phase_symbol(p, native_model_request(fresh)) for p in ('prep', 'core', 'y')] == [
        'r004_model_T_prep_v6', 'r004_model_T_core_v6', 'r004_model_T_y_v6']


@pytest.mark.parametrize('field,value', [
    ('schema', 'r004.gdn.native-model/v1'),
    ('implementation', 'tir.gdn_recurrent.r004_column_v4'),
    ('numerical_subject_sha256', native_model_contract()['numerical_subject_sha256']),
    ('norm_primitive', 'r004.norm128/unknown'),
    ('native_layer', 1), ('arena_binding', 'unplanned-global'),
])
def test_mixed_model_subject_rejected(field, value):
    m = model()
    request = {**native_model_request(m), field: value}
    with pytest.raises(IRVerificationError):
        native_model_request(replace(m, metadata={**m.metadata, 'r004_native_model_request': request}))
    with pytest.raises(IRVerificationError):
        model_phase_symbol('core', request)


def test_v4_plan_and_policy_cannot_impersonate_v6():
    m = model()
    bad = replace(m, nodes=tuple(replace(n, attrs={**n.attrs, 'phase_plan': v4_plan()})
                               if n.op == 'nn.gdn_recurrent_core' else n for n in m.nodes))
    with pytest.raises((IRVerificationError, IRSchemaError)):
        prepare_model_column_distribution(bad)
    policy = {**m.metadata['r004_model_policy'], 'candidate': 'r004_T_full_p5_v4'}
    with pytest.raises(IRVerificationError):
        model_distribution_target(NvidiaSm120Target(), replace(m, metadata={**m.metadata, 'r004_model_policy': policy}))


@pytest.mark.parametrize('schema', [None, 'unknown-package'])
def test_missing_package_identity_cannot_escape_native_verification(monkeypatch, schema):
    from types import SimpleNamespace
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
    import triton.flagmega.runtime.r004_native_model as native
    from triton.flagmega.errors import ArtifactError

    def reject(codegen, kernels, module):
        raise ArtifactError('native verification reached')

    monkeypatch.setattr(native, 'verified_model_programs', reject)
    owner = SimpleNamespace(ir_module=model(), kernels=(), codegen={})
    if schema is not None:
        owner.codegen['kernel_program_schema'] = schema
    with pytest.raises(ArtifactError, match='native verification reached'):
        GeneratedTirCallGraphModule._kernel_launch_specs(owner)
