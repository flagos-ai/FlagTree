"""Issued v6 identity admission preserves strict standalone structural contracts."""
from copy import deepcopy
from dataclasses import replace
import pytest
from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.rules.ntt.r004_column_phase_bindings_v6 import column_phase_plan
from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import (
    configure_standalone_phases, standalone_request, validate_phase_plan,
)
from triton.flagmega.codegen.triton.kernels.gdn_recurrent.r004_contract_v4 import native_arguments
from python.test.flagmega.codegen.triton.test_r004_phase_bindings import boundary

REQUEST = dict(schema='r004.gdn.standalone/v4', module='r004_T_full_p5_v6', readonly_layer=0)


def test_issued_v6_routes_through_exact_twelve_root_contract():
    module = replace(boundary(requested=False), metadata={'r004_standalone_request': REQUEST})
    assert standalone_request(module) == REQUEST
    configured = configure_standalone_phases(module)
    plan = next(n.attrs['phase_plan'] for n in configured.nodes if n.op == 'nn.gdn_recurrent_core')
    plan = validate_phase_plan(plan)
    assert plan == column_phase_plan()
    assert [e['dynamic_argument_indices'] for e in plan['entries']] == [[0, 1], [2, 5, 6], [1, 3]]
    assert native_arguments(plan)['core'] == ('QN', 'KN', 'QKV', 'Beta', 'G', 'StateParent', 'Layer', 'Core')
    assert plan['pool']['nbytes'] == 41472 and len(plan['roots']) == 12


@pytest.mark.parametrize('payload', [
    [], dict(REQUEST, module='r004_T_full_p5_v7'), dict(REQUEST, module='r004_T_full_p5_v6_extra'),
    dict(REQUEST, module='r004_native_model'), dict(REQUEST, fallback=True),
    dict(REQUEST, schema='r004.gdn.standalone/v6'), dict(REQUEST, readonly_layer=True),
    dict(REQUEST, readonly_layer=-1), dict(REQUEST, readonly_layer=48),
    {'schema': REQUEST['schema'], 'module': REQUEST['module']},
])
def test_unissued_or_malformed_request_is_rejected(payload):
    module = replace(boundary(requested=False), metadata={'r004_standalone_request': payload})
    with pytest.raises(IRVerificationError):
        standalone_request(module)


@pytest.mark.parametrize('field,value', [('norm_primitive', 'r004.norm128/old'),
    ('schema', 'r004.gdn.column-register/v4'), ('extra', True)])
def test_v6_plan_never_accepts_stale_or_extra_fields(field, value):
    plan = deepcopy(column_phase_plan())
    plan[field] = value
    with pytest.raises(IRSchemaError):
        validate_phase_plan(plan)


def test_v6_request_rejects_mixed_c_phase_plan():
    from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import reference_phase_plan
    b = boundary(requested=False)
    nodes = tuple(replace(n, attrs={**n.attrs, 'phase_plan': reference_phase_plan()})
                  if n.op == 'nn.gdn_recurrent_core' else n for n in b.nodes)
    module = replace(b, nodes=nodes, metadata={'r004_standalone_request': REQUEST})
    with pytest.raises(IRVerificationError, match='Mixed standalone'):
        configure_standalone_phases(module)
