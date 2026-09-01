from __future__ import annotations

import pytest

from triton.flagtune.contract.identity import (
    ModelIdentity,
    gpu_metadata,
    make_platform_key,
)
from triton.flagtune.contract.operator_schema import parse_model_config


def _model_config(**updates):
    config = {
        "format_version": 5,
        "model_version": "1.0.0",
        "platform_key": "nvidia-h20",
        "op_id": "flaggems/mm",
        "variant": "gemv",
        "dtype_key": "bf16-bf16-bf16",
        "dtypes": ["bfloat16", "bfloat16", "bfloat16"],
        "gpu": {
            "backend": "cuda",
            "vendor": "nvidia",
            "device_name": "NVIDIA H20-3e",
            "architecture": "sm90",
            "platform_key": "nvidia-h20",
        },
        "inputs": {"M": {}},
        "params": {"BLOCK": {"values": [32]}},
        "features": ["M"],
    }
    config.update(updates)
    return config


def test_h20_product_aliases_use_one_platform_key():
    assert make_platform_key("NVIDIA", "NVIDIA H20") == "nvidia-h20"
    assert make_platform_key("NVIDIA", "NVIDIA H20-3e") == "nvidia-h20"


def test_metax_product_name_does_not_duplicate_vendor_in_platform_key():
    assert make_platform_key("MetaX", "MetaX C550") == "metax-c550"


def test_ppu_product_name_uses_thead_platform_key():
    assert make_platform_key("T-Head", "PPU-ZW810E") == "thead-zw810e"


def test_mthreads_product_name_does_not_duplicate_vendor_in_platform_key():
    assert make_platform_key("MThreads", "MTT S5000") == "mthreads-s5000"


def test_hygon_product_name_does_not_duplicate_vendor_in_platform_key():
    assert make_platform_key("Hygon", "Hygon BW") == "hygon-bw"


def test_model_config_accepts_metax_maca_backend():
    config = _model_config(
        platform_key="metax-c550",
        gpu={
            "backend": "maca",
            "vendor": "metax",
            "device_name": "MetaX C550",
            "architecture": "sm80",
            "platform_key": "metax-c550",
        },
    )

    assert parse_model_config(config).op_id == "flaggems/mm"


def test_model_config_accepts_mthreads_musa_backend():
    config = _model_config(
        platform_key="mthreads-s5000",
        gpu={
            "backend": "musa",
            "vendor": "mthreads",
            "device_name": "MTT S5000",
            "architecture": "musa31",
            "platform_key": "mthreads-s5000",
        },
    )

    assert parse_model_config(config).op_id == "flaggems/mm"


@pytest.mark.parametrize("format_version", [5.0, True])
def test_model_config_requires_exact_integer_format_version(format_version):
    with pytest.raises(ValueError, match="format_version"):
        parse_model_config({**_model_config(), "format_version": format_version})


def test_gpu_metadata_exposes_platform_key_and_architecture():
    metadata = gpu_metadata(
        backend="cuda",
        vendor="NVIDIA",
        device_name="NVIDIA H20-3e",
        architecture="sm90",
    )

    assert metadata["platform_key"] == "nvidia-h20"
    assert metadata["architecture"] == "sm90"


def test_model_identity_uses_platform_key():
    identity = ModelIdentity("nvidia-h20", "flaggems/mm", "gemv", "bf16-bf16-bf16")
    assert identity.platform_key == "nvidia-h20"
    assert identity.artifact_key == "nvidia-h20/flaggems/mm/gemv/bf16-bf16-bf16"
