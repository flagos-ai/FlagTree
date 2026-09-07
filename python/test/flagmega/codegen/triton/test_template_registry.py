# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
)
from triton.flagmega.errors import CodegenError


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_registry_prefers_architecture_then_platform_then_generic(tmp_path):
    _write(tmp_path / "kernels/matmul/gemv.py.jinja", "generic {{ value }}")
    _write(tmp_path / "kernels/matmul/platforms/nvidia/gemv.py.jinja", "nvidia {{ value }}")
    _write(
        tmp_path / "kernels/matmul/platforms/nvidia/sm90/gemv.py.jinja",
        "sm90 {{ value }}",
    )
    registry = TritonTemplateRegistry(tmp_path)

    assert registry.render_kernel(
        KernelTemplateSpec("matmul", "gemv", "nvidia", "sm90"), {"value": 7}
    ).source == "sm90 7"
    assert registry.render_kernel(
        KernelTemplateSpec("matmul", "gemv", "nvidia", "sm80"), {"value": 7}
    ).source == "nvidia 7"
    assert registry.render_kernel(
        KernelTemplateSpec("matmul", "gemv", "amd", "gfx942"), {"value": 7}
    ).source == "generic 7"


def test_registry_uses_strict_undefined_and_reports_the_selected_template(tmp_path):
    _write(tmp_path / "kernels/matmul/gemv.py.jinja", "{{ required_value }}")
    registry = TritonTemplateRegistry(tmp_path)

    with pytest.raises(CodegenError, match="kernels/matmul/gemv.py.jinja"):
        registry.render_kernel(KernelTemplateSpec("matmul", "gemv"), {})


@pytest.mark.parametrize("name", ["MatMul", "matmul/glob", "../matmul", ""])
def test_registry_rejects_non_snake_or_path_like_names(tmp_path, name):
    with pytest.raises(CodegenError):
        TritonTemplateRegistry(tmp_path).resolve(KernelTemplateSpec(name, "gemv"))
