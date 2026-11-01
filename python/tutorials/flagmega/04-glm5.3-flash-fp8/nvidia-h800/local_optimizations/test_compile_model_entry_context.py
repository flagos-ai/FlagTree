"""Real implementation selection must work from file and package entry contexts."""
import importlib
import os
from pathlib import Path
import runpy
import sys
import pytest
import torch

tutorial = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(tutorial))

def _entry_symbols(entry_context):
    if entry_context == 'standalone_file':
        path = Path(os.environ.get('FLAGMEGA_ENTRY_SCRIPT', str(tutorial / 'local_optimizations/compile_model.py')))
        return runpy.run_path(str(path), run_name='compile_entry_context_probe')
    return vars(importlib.import_module('local_optimizations.compile_model'))

@pytest.mark.parametrize('entry_context', ['standalone_file', 'package_module'])
def test_real_catalog_preserves_complete_adaptive_selection(entry_context):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    from local_optimizations.native_tp4 import native_implementation_model
    catalog = _entry_symbols(entry_context)['catalog']
    selected = catalog(native_implementation_model(offline_projection_weights=True))
    indexers = [v for v in selected.implementations if v.family == 'dsa_indexer' and v.id.endswith('.joint_pool_query')]
    assert indexers and selected.preferences['dsa_indexer'][0] == indexers[0].id
    for impl in indexers:
        assert impl.parameters['pool_score_schedule'] == 'joint_adaptive_complete_pools_absorbed_v1'
        assert impl.parameters['pool_score_ctas'] == impl.parameters['absorbed_query_ctas'] == 64
        assert (impl.parameters['long_pool_score_ctas'], impl.parameters['long_absorbed_query_ctas'], impl.parameters['long_min_base']) == (96, 32, 1024)
        assert impl.facts['internal_grid_barriers'] == 3
    assert not torch.cuda.is_initialized()

@pytest.mark.parametrize('entry_context', ['standalone_file', 'package_module'])
def test_partition_entry_reaches_normal_module_validation(entry_context):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    class ReachedModuleValidation(RuntimeError):
        pass
    class ModuleBoundaryProbe:
        def __getattr__(self, name):
            raise ReachedModuleValidation(name)
    # Invalid module access deliberately stops at the normal validation boundary.
    # A missing package context fails earlier with ImportError instead.
    with pytest.raises(ReachedModuleValidation):
        _entry_symbols(entry_context)['partition'](ModuleBoundaryProbe())
    assert not torch.cuda.is_initialized()
