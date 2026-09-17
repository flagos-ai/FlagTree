import pytest
import tempfile


def pytest_configure(config):
    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")
    config.addinivalue_line("markers", "performance: benchmark-oriented tests")


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        from triton import knobs

        with knobs.cache.scope(), knobs.runtime.scope():
            knobs.cache.dir = tmpdir
            yield tmpdir


@pytest.fixture
def fresh_knobs():
    """
    Resets all knobs except ``build``, ``nvidia``, and ``amd`` (preserves
    library paths needed to compile kernels).
    """
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl(skipped_attr={"build", "nvidia", "amd"})
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def fresh_knobs_including_libraries():
    """
    Resets ALL knobs including ``build``, ``nvidia``, and ``amd``.
    Use for tests that verify initial values of these knobs.
    """
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def with_allocator(device):
    import triton
    import torch
    from triton.runtime._allocation import NullAllocator

    allocator_device = device
    if allocator_device not in ("cuda", "musa"):
        from triton.runtime import driver

        target = driver.active.get_current_target()
        allocator_device = "musa" if target.backend == "musa" else "cuda"

    def alloc_fn(size: int, align: int, stream):
        return torch.empty(size, dtype=torch.int8, device=allocator_device)

    triton.set_allocator(alloc_fn)
    try:
        yield
    finally:
        triton.set_allocator(NullAllocator())
