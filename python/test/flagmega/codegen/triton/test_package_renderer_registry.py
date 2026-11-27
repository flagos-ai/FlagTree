# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton import PackageRendererRegistry, PackageRendererSpec
from triton.flagmega.errors import CodegenError


def _module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value = builder.var("value", fm.tensor_type("float32", [1]), id="value")
    builder.function("main", [value], [value])
    return builder.build(entry="main")


def _spec(name: str, kind: str, matches=lambda _module: True):
    return PackageRendererSpec(
        name,
        kind,
        matches,
        lambda _module: {"symbol": "entry", "grid": [1, 1, 1]},
        lambda _descriptor, _version: "source",
    )


def test_package_renderer_registry_requires_one_exact_match():
    registry = PackageRendererRegistry()
    first = registry.register(_spec("first", "first/v1"))
    assert registry.resolve(_module()) is first

    registry.register(_spec("second", "second/v1"))
    with pytest.raises(CodegenError, match="ambiguous.*first.*second"):
        registry.resolve(_module())


def test_package_renderer_registry_rejects_duplicate_name_and_kind():
    registry = PackageRendererRegistry()
    registry.register(_spec("first", "first/v1"))
    with pytest.raises(CodegenError, match="already registered"):
        registry.register(_spec("first", "other/v1"))
    with pytest.raises(CodegenError, match="kind.*already registered"):
        registry.register(_spec("other", "first/v1"))


def test_package_renderer_registry_reports_no_match():
    registry = PackageRendererRegistry()
    registry.register(_spec("never", "never/v1", lambda _module: False))
    with pytest.raises(CodegenError, match="No Triton entry-package renderer"):
        registry.resolve(_module())
