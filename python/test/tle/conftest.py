import pytest

from triton.experimental.tle import (
    is_tle_supported,
    primitive_name,
)

_MARKER = "skip_if_tle_unsupported"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_if_tle_unsupported(*primitives): skip when the active backend does not support every named TLE primitive",
    )


def _get_required_primitives_list(item):
    required_list = []
    for marker in item.iter_markers(name=_MARKER):
        if marker.kwargs:
            raise pytest.UsageError("skip_if_tle_unsupported accepts primitive names as positional arguments only")
        if not marker.args:
            raise pytest.UsageError("skip_if_tle_unsupported requires at least one TLE primitive name")
        required_list.extend(primitive_name(primitive) for primitive in marker.args)
    return frozenset(required_list)


def pytest_runtest_setup(item):
    required_list = _get_required_primitives_list(item)
    if not required_list:  # no required_list primitives declared
        return
    if not is_tle_supported(required_list):
        pytest.skip(f"current backend does not support tle required_list: {', '.join(sorted(required_list))}")
