import pytest
import tempfile


def pytest_configure(config):
    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")
    config.addinivalue_line(
        "markers",
        "remote_integration: exercise an external service and require explicit opt-in",
    )


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")
    # flagtree
    parser.addoption(
        "--first-parameter-only",
        action="store_true",
        help="Only run the first parameter combination for each test function",
    )


# flagtree: only run the first parameter combination for each test function
def pytest_collection_modifyitems(config, items):
    if config.getoption("--first-parameter-only"):
        selected = []
        deselected = []
        seen_parameterized_tests = set()

        for item in items:
            # Non-parameterized tests have no callspec and should always run.
            if not hasattr(item, "callspec"):
                selected.append(item)
                continue

            test_key = (item.parent.nodeid, item.originalname)
            if test_key in seen_parameterized_tests:
                deselected.append(item)
            else:
                seen_parameterized_tests.add(test_key)
                selected.append(item)

        items[:] = selected
        if deselected:
            config.hook.pytest_deselected(items=deselected)


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
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def fresh_knobs_except_libraries():
    """
    A variant of `fresh_knobs` that keeps library path
    information from the environment as these may be
    needed to successfully compile kernels.
    """
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl(skipped_attr={"build", "nvidia", "amd"})
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
