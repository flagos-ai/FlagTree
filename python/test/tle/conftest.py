import pytest

from triton.experimental.tle import (
    get_supported_primitives,
    primitive_name,
)

_MARKER = "require_tle"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "require_tle(*primitives): run only when the active backend supports every named TLE primitive",
    )


def _required_primitives(item):
    required = []
    for marker in item.iter_markers(name=_MARKER):
        if marker.kwargs:
            raise pytest.UsageError("require_tle accepts primitive names as positional arguments only")
        if not marker.args:
            raise pytest.UsageError("require_tle requires at least one TLE primitive name")
        required.extend(primitive_name(primitive) for primitive in marker.args)
    return frozenset(required)


def pytest_runtest_setup(item):
    from triton._flagtree_backend import get_active_backend_name

    required = _required_primitives(item)
    if not required:
        return
    try:
        backend_name = get_active_backend_name()
    except Exception as exc:
        pytest.skip(f"cannot determine TLE primitive support for the active backend: {exc}")
    supported = get_supported_primitives(backend_name)
    missing = sorted(required - supported)
    if missing:
        pytest.skip(f"backend {backend_name!r} does not support required TLE primitive(s): {', '.join(missing)}")
