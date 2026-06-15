import os
import pytest
import tempfile
from typing import Optional, Set


def _is_gcu500():
    """Check if the current target architecture is GCU500."""
    arch = os.environ.get("COMPILE_ARCH", "")
    if arch == "gcu500":
        return True
    if os.environ.get("INTERNAL_GCU_SIM", "").upper() == "DRACO":
        return True
    return False


def _patch_triton_for_gcu500():
    """Apply runtime patches to upstream triton for GCU500 compatibility.

    Only called when _is_gcu500() is True (COMPILE_ARCH=gcu500 or
    INTERNAL_GCU_SIM=DRACO).
    """
    import importlib

    # Tolerate missing amd/nvidia backends during entry-point discovery
    # so that the enflame backend can be found on GCU500 test hosts where
    # the upstream triton package still registers amd/nvidia entry points.
    import triton.backends as _backends

    def _safe_discover():
        backends = {}
        if os.environ.get("TRITON_BACKENDS_IN_TREE") == "1":
            root = os.path.dirname(_backends.__file__)
            for name in os.listdir(root):
                if not os.path.isdir(os.path.join(root, name)) or name.startswith('__'):
                    continue
                try:
                    compiler = importlib.import_module(f"triton.backends.{name}.compiler")
                    driver = importlib.import_module(f"triton.backends.{name}.driver")
                    backends[name] = _backends.Backend(
                        _backends._find_concrete_subclasses(compiler, _backends.BaseBackend),
                        _backends._find_concrete_subclasses(driver, _backends.DriverBase))
                except Exception:
                    pass
            return backends

        import sys
        if sys.version_info >= (3, 10):
            from importlib.metadata import entry_points
        else:
            from importlib_metadata import entry_points
        for ep in entry_points().select(group="triton.backends"):
            try:
                compiler = importlib.import_module(f"{ep.value}.compiler")
                driver = importlib.import_module(f"{ep.value}.driver")
                backends[ep.name] = _backends.Backend(
                    _backends._find_concrete_subclasses(compiler, _backends.BaseBackend),
                    _backends._find_concrete_subclasses(driver, _backends.DriverBase))
            except Exception:
                pass
        return backends

    _backends._discover_backends = _safe_discover
    _backends.backends.clear()
    _backends.backends.update(_safe_discover())

    # Fix bfloat16 cast: perform on CPU first to avoid missing
    # device-side topsop kernels on GCUSIM.
    import triton_gcu.triton
    import triton._internal_testing as _testing
    _orig_to_triton = _testing.to_triton

    def _patched_to_triton(x, device, dst_type=None):
        import torch
        t = x.dtype.name
        if t == 'float32' and dst_type == 'bfloat16':
            return torch.tensor(x).bfloat16().to(device)
        return _orig_to_triton(x, device=device, dst_type=dst_type)

    _testing.to_triton = _patched_to_triton


def pytest_configure(config):
    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")
    if _is_gcu500():
        _patch_triton_for_gcu500()


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default='gcu')


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        from triton import knobs
        with knobs.cache.scope():
            knobs.cache.dir = tmpdir
            yield tmpdir


def _fresh_knobs_impl(monkeypatch, skipped_attr: Optional[Set[str]] = None):
    from triton import knobs

    if skipped_attr is None:
        skipped_attr = set()

    knobs_map = {
        name: knobset
        for name, knobset in knobs.__dict__.items()
        if isinstance(knobset, knobs.base_knobs) and knobset != knobs.base_knobs and name not in skipped_attr
    }

    # We store which variables we need to unset below in finally because
    # monkeypatch doesn't appear to reset variables that were never set
    # before the monkeypatch.delenv call below.
    env_to_unset = []
    prev_propagate_env = knobs.propagate_env

    def fresh_function():
        nonlocal env_to_unset
        for name, knobset in knobs_map.items():
            setattr(knobs, name, knobset.copy().reset())
            for knob in knobset.knob_descriptors.values():
                if knob.key in os.environ:
                    monkeypatch.delenv(knob.key, raising=False)
                else:
                    env_to_unset.append(knob.key)
        knobs.propagate_env = True
        return knobs

    def reset_function():
        for name, knobset in knobs_map.items():
            setattr(knobs, name, knobset)
        for k in env_to_unset:
            if k in os.environ:
                del os.environ[k]
        knobs.propagate_env = prev_propagate_env

    return fresh_function, reset_function


@pytest.fixture
def fresh_knobs(monkeypatch):
    fresh_function, reset_function = _fresh_knobs_impl(monkeypatch)
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def fresh_knobs_except_libraries(monkeypatch):
    """
    A variant of `fresh_knobs` that keeps library path
    information from the environment as these may be
    needed to successfully compile kernels.
    """
    fresh_function, reset_function = _fresh_knobs_impl(monkeypatch, skipped_attr={"build", "nvidia", "amd"})
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def with_allocator():
    import triton
    from triton.runtime._allocation import NullAllocator
    from triton._internal_testing import default_alloc_fn

    triton.set_allocator(default_alloc_fn)
    try:
        yield
    finally:
        triton.set_allocator(NullAllocator())


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    skipped = terminalreporter.stats.get("skipped", [])
    if skipped:
        terminalreporter.write_sep("=", "detailed skipped tests")
        for report in skipped:
            node_id = report.nodeid
            skip_info = ""

            if isinstance(report.longrepr, tuple):
                skip_info = report.longrepr[2] if len(report.longrepr) > 2 else str(report.longrepr)
            else:
                skip_info = str(report.longrepr).split("\n")[-1]

            terminalreporter.write_line(f"{node_id} - {skip_info}")
