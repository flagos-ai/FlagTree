import os
import platform
import hashlib
from unittest.mock import patch, mock_open
import unittest
import pytest
from triton.backends.ascend.utils import (
    get_cann_version_file_hash,
    get_machine_arch,
)


@patch.object(platform, "machine")
def test_get_machine_arch_normal_case(mock_machine):
    mock_machine.return_value = "x86_64"
    result = get_machine_arch()
    assert result == "x86_64"


@patch.object(platform, "machine")
def test_get_machine_arch_arm_case(mock_machine):
    mock_machine.return_value = "arm64"
    result = get_machine_arch()
    assert result == "aarch64"


@patch.object(platform, "machine")
def test_get_machine_arch_unknown_case(mock_machine):
    mock_machine.return_value = "unknown"
    with pytest.raises(KeyError):
        get_machine_arch()


def test_get_cann_version_file_hash_normal_case():
    test_version = "9.0.0"
    test_inner_version = "v100rc00"
    test_file_context = f"version={test_version}\ninnerversion={test_inner_version}".encode("utf-8")
    with patch("builtins.open", mock_open(read_data=test_file_context)):
        result = get_cann_version_file_hash()
        excepted_hash = hashlib.sha256(test_file_context).hexdigest()
        assert result == excepted_hash


def test_get_cann_version_file_hash_file_not_find_case():
    with patch("builtins.open") as mock_open_file:
        mock_open_file.side_effect = FileNotFoundError
        with pytest.raises(FileNotFoundError):
            get_cann_version_file_hash()
