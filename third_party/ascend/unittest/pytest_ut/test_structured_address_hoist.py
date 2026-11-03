"""Host checks for physical UB address-table reuse and its rejection cases."""
import importlib
from pathlib import Path
import subprocess

import pytest

compiler = importlib.import_module('triton.backends.ascend.compiler')


@pytest.fixture(scope='module', autouse=True)
def assign_npu():
    yield


def test_physical_address_hoist_admission():
    source = Path(__file__).resolve().parents[1] / 'Conversion/structured_address_hoist.mlir'
    result = subprocess.check_output([
        compiler._get_triton_mlir_opt_path(),
        str(source), '--hoist-structured-permutation-addresses', '--allow-unregistered-dialect', '--verify-each'
    ], text=True)
    admitted = {'packed_double', 'packed_single', 'unaligned_source', 'before_loop_vector_init'}
    rejected = {
        'index_extra_read', 'physical_alias', 'unknown_alias', 'index_write', 'after_loop_read', 'strided_indices',
        'unaligned_indices', 'unaligned_scratch', 'unaligned_destination', 'overlapping_tables',
        'source_destination_alias', 'conditional_gather', 'unknown_source', 'zero_iterations', 'one_iteration',
        'two_iterations', 'unmodeled_scalar_effect', 'outside_loop_read_before', 'before_loop_dma_write',
        'before_loop_unknown_effect'
    }
    for name in admitted | rejected:
        body = result.split('func.func @' + name + '(')[1].split('\n  }')[0]
        if name in admitted:
            assert body.count('call @custom_flagtree_prepare_addresses') == 1, name
            assert body.count('call @custom_flagtree_addressed_gather_float') == 1, name
            assert body.index('prepare_addresses') < body.index('scf.for'), name
            assert 'call @custom_flagtree_stride_gather_float' not in body, name
        else:
            assert 'call @custom_flagtree_prepare_addresses' not in body, name
            assert body.count('call @custom_flagtree_stride_gather_float') == 1, name

    dynamic = result.split('func.func @dynamic_iterations(')[1].split('\n  }')[0]
    assert dynamic.count('call @custom_flagtree_prepare_addresses') == 1
    assert dynamic.count('call @custom_flagtree_addressed_gather_float') == 1
    assert dynamic.count('call @custom_flagtree_stride_gather_float') == 1
    assert dynamic.count('scf.if') == 2


@pytest.mark.parametrize('option', ['structured_address_hoisting', 'structured_store_sync_optimization'])
def test_address_hoist_option_is_hashed(option):
    from triton.backends.compiler import GPUTarget
    backend = compiler.AscendBackend(GPUTarget('npu', 'Ascend910B4', 0))
    default = backend.parse_options({})
    enabled = backend.parse_options({option: True})
    disabled = backend.parse_options({option: False})
    assert getattr(default, option)
    assert default.hash() == enabled.hash() and default.hash() != disabled.hash()


@pytest.mark.parametrize('option', ['structured_address_hoisting', 'structured_store_sync_optimization'])
@pytest.mark.parametrize('value', [0, 1, 'true', None])
def test_address_hoist_rejects_non_boolean(value, option):
    with pytest.raises(ValueError, match='must be a bool'):
        compiler.NPUOptions(**{option: value})


def test_store_sync_requires_disjoint_direct_dma():
    source = Path(__file__).resolve().parents[1] / 'Conversion/structured_store_sync.mlir'
    result = subprocess.check_output(
        [compiler._get_triton_mlir_opt_path(),
         str(source), '--coalesce-structured-store-sync', '--verify-each'], text=True)
    expected = {
        'disjoint': 0, 'overlap': 1, 'revisit': 1, 'unknown_base': 2, 'atomic': 3, 'dynamic_mode': 3, 'unaligned_ub': 3,
        'strided_gm': 3, 'intervening_write': 1, 'intervening_unknown': 1, 'other_pipeline': 0, 'cross_pipeline': 1,
        'final_wait': 1, 'intervening_control': 1
    }
    for name, count in expected.items():
        body = result.split('func.func @' + name + '(')[1].split('\n  }')[0]
        assert body.count('pipe_barrier[<PIPE_MTE3>]') == count, name
        if name == 'other_pipeline':
            assert body.count('pipe_barrier[<PIPE_V>]') == 1
        if name == 'cross_pipeline':
            assert body.count('hivm.hir.set_flag') == 1
