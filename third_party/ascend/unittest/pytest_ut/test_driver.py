# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import types
from unittest.mock import MagicMock

import pytest

_MOCK_AICORE_NUM = 24
_MOCK_AIVECTOR_NUM = _MOCK_AICORE_NUM * 2


def _make_mock_npu_utils():
    """Create a mock with real _get_npu_device_limit_form_env and get_device_properties
    bound, while get_device_aicore returns a fixed int.

    get_device_aicore cannot be bound via types.MethodType because it is decorated
    with @functools.lru_cache(), whose wrapper object is not a descriptor and does
    not produce a correct bound method. Using return_value avoids this issue."""
    try:
        from triton.backends.ascend.driver import NPUUtils
    except ImportError:
        pytest.skip("Ascend backend not available")
    mock = MagicMock()
    # _get_npu_device_limit_form_env calls self.get_device_aicore(); configure it
    # to return a fixed int instead of binding the real lru_cached method.
    mock.get_device_core.return_value = _MOCK_AICORE_NUM, _MOCK_AIVECTOR_NUM
    # Bind real methods (not lru_cached) so the actual parsing/property logic runs.
    mock._get_npu_device_limit_form_env = types.MethodType(NPUUtils._get_npu_device_limit_form_env, mock)
    mock.get_device_properties = types.MethodType(NPUUtils.get_device_properties, mock)
    return mock, NPUUtils


@pytest.mark.parametrize(
    "env_value,expected_aic,expected_aiv",
    [
        # env not set -> return device defaults
        (None, _MOCK_AICORE_NUM, _MOCK_AIVECTOR_NUM),
        # valid values within device limit
        ("12,24", 12, 24),
        (f"{_MOCK_AICORE_NUM},{_MOCK_AIVECTOR_NUM}", _MOCK_AICORE_NUM, _MOCK_AIVECTOR_NUM),
        ("1,2", 1, 2),
        # leading/trailing spaces are stripped by .strip()
        ("  12,24  ", 12, 24),
        ("12 , 24  ", 12, 24),
        ("12, 24", 12, 24),
    ],
)
def test_npu_device_limit_env_var_valid(monkeypatch, env_value, expected_aic, expected_aiv):
    """Test _get_npu_device_limit_form_env returns user-specified values for valid input."""
    mock_utils, NPUUtils = _make_mock_npu_utils()

    if env_value is None:
        monkeypatch.delenv("NPU_DEVICE_LIMIT", raising=False)
    else:
        monkeypatch.setenv("NPU_DEVICE_LIMIT", env_value)

    aic, aiv = NPUUtils._get_npu_device_limit_form_env(mock_utils)
    assert aic == expected_aic
    assert aiv == expected_aiv


@pytest.mark.parametrize(
    "env_value",
    [
        # non-positive values
        "0,48",
        "24,0",
        "0,0",
        # exceeds device limit
        f"{_MOCK_AICORE_NUM + 1},"
        f"{_MOCK_AIVECTOR_NUM}",
        f"{_MOCK_AICORE_NUM},"
        f"{_MOCK_AIVECTOR_NUM + 1}",
        "100,200",
        # invalid format
        "abc",
        "12",  # missing second value
        "12,24,36",  # too many values
        "12.5,24",  # float not matched
        "",  # empty string
        "-1,48",  # negative sign not matched
        "2,6",
        "2,5",
    ],
)
def test_npu_device_limit_env_var_invalid(monkeypatch, env_value):
    """Test _get_npu_device_limit_form_env raises ValueError for invalid input."""
    mock_utils, NPUUtils = _make_mock_npu_utils()
    monkeypatch.setenv("NPU_DEVICE_LIMIT", env_value)

    with pytest.raises(ValueError, match="NPU_DEVICE_LIMIT"):
        NPUUtils._get_npu_device_limit_form_env(mock_utils)


def test_npu_device_limit_get_device_properties(monkeypatch):
    """Test get_device_properties respects NPU_DEVICE_LIMIT."""
    mock_utils, NPUUtils = _make_mock_npu_utils()

    monkeypatch.setenv("NPU_DEVICE_LIMIT", "8,16")
    props = NPUUtils.get_device_properties(mock_utils, "npu")
    assert props["num_aicore"] == 8
    assert props["num_vectorcore"] == 16
    assert props["max_shared_mem"] == 1


def test_npu_device_limit_default_when_unset(monkeypatch):
    """Test get_device_properties returns hardware defaults when env var is unset."""
    mock_utils, NPUUtils = _make_mock_npu_utils()

    monkeypatch.delenv("NPU_DEVICE_LIMIT", raising=False)
    props = NPUUtils.get_device_properties(mock_utils, "npu")
    assert props["num_aicore"] == _MOCK_AICORE_NUM
    assert props["num_vectorcore"] == _MOCK_AIVECTOR_NUM


def test_npu_device_limit_propagates_error(monkeypatch):
    """Test get_device_properties propagates ValueError from _get_npu_device_limit_form_env."""
    mock_utils, NPUUtils = _make_mock_npu_utils()

    monkeypatch.setenv("NPU_DEVICE_LIMIT", "0,48")
    with pytest.raises(ValueError, match="non-positive"):
        NPUUtils.get_device_properties(mock_utils, "npu")


def test_npu_device_limit_no_caching_of_env_var(monkeypatch):
    """Test that get_device_properties re-reads the env var on every call.

    get_device_properties is not lru_cached, so changing the env var between
    calls takes effect immediately."""
    mock_utils, NPUUtils = _make_mock_npu_utils()

    monkeypatch.setenv("NPU_DEVICE_LIMIT", "8,16")
    props1 = NPUUtils.get_device_properties(mock_utils, "npu")
    assert props1["num_aicore"] == 8

    # Change env var - get_device_properties is not cached, so the new value takes effect
    monkeypatch.setenv("NPU_DEVICE_LIMIT", "16,32")
    props2 = NPUUtils.get_device_properties(mock_utils, "npu")
    assert props2["num_aicore"] == 16


def test_npu_device_limit_get_aicore_num(monkeypatch):
    """Test get_aicore_num / get_aivector_core_num read from get_device_properties."""
    mock_utils, NPUUtils = _make_mock_npu_utils()

    monkeypatch.setenv("NPU_DEVICE_LIMIT", "8,16")

    assert NPUUtils.get_aicore_num(mock_utils) == 8
    assert NPUUtils.get_aivector_core_num(mock_utils) == 16
