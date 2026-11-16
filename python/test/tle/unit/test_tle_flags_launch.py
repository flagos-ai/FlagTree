"""CPU-only checks of the selected barrier's actual launcher boundary."""
import json
from types import SimpleNamespace

import pytest

from triton.experimental.tle.grid_barrier_runtime import bind_launch_contract, validate_launch_contract


def metadata(**updates):
    result = dict(tle_flags_launch_contract=json.dumps(dict(schema='tle.flags32/v1',
                  domain=[8, 16], grid=[16, 8, 1])), num_ctas=1,
                  cluster_dims=(1, 1, 1), launch_cooperative_grid=True)
    result.update(updates)
    return SimpleNamespace(**result)


def test_default_exact_callable():
    fn = lambda *args: args
    assert bind_launch_contract(fn, SimpleNamespace()) is fn


def test_selected_normal_direct_boundary():
    calls = []
    fn = bind_launch_contract(lambda *args: calls.append(args), metadata())
    fn(16, 8, 1, 'stream', 'function')
    assert calls == [(16, 8, 1, 'stream', 'function')]
    for grid in [(128, 1, 1), (8, 16, 1), (16, 1, 8), (16, 8, 2), (0, 8, 1)]:
        with pytest.raises(ValueError):
            fn(*grid, 'stream', 'function')
    assert len(calls) == 1


@pytest.mark.parametrize('updates', [dict(num_ctas=2), dict(cluster_dims=(2, 1, 1)),
                         dict(launch_cooperative_grid=False),
                         dict(tle_flags_launch_contract=''),
                         dict(tle_flags_launch_contract='{"grid":[16,8,1]}')])
def test_invalid_contract_cannot_reach_launch(updates):
    calls = []
    with pytest.raises(ValueError):
        bind_launch_contract(lambda *args: calls.append(args), metadata(**updates))
    assert not calls


def test_prepared_rejects_before_allocation(monkeypatch):
    import triton.flagmega.runtime.prepared as prepared
    calls = []
    monkeypatch.setattr(prepared, '_prepare_global_scratch', lambda *a, **kw: calls.append('allocate'))
    compiled = SimpleNamespace(metadata=metadata(), src=SimpleNamespace(attrs={}))
    with pytest.raises(ValueError, match='launch grid'):
        prepared.PreparedKernel(compiled, (), (), grid=(128, 1, 1), contract=object())
    assert not calls


def test_missing_selected_domain_is_rejected():
    bad = json.dumps(dict(schema='tle.flags32/v1', domain=[], grid=[16, 8, 1]))
    with pytest.raises(ValueError):
        validate_launch_contract(metadata(tle_flags_launch_contract=bad), (16, 8, 1))


def test_normal_cuda_launcher_checks_before_c_call(monkeypatch):
    import triton.backends.nvidia.driver as driver
    calls = []
    monkeypatch.setattr(driver, 'compile_module_from_src',
                        lambda **kwargs: SimpleNamespace(launch=lambda *args: calls.append(args)))
    monkeypatch.setattr(driver, 'library_dirs', lambda: [])
    m = metadata(global_scratch_size=0, global_scratch_align=128,
                 profile_scratch_size=0, profile_scratch_align=1, launch_pdl=False)
    launcher = driver.CudaLauncher(SimpleNamespace(constants={}, signature={0: '*i32'}), m)
    with pytest.raises(ValueError, match='launch grid'):
        launcher(128, 1, 1, 0, 0)
    assert calls == []
