# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""split_k_gemv / split_k_packed_k_major_gemv are portable wrappers (no vendor
intrinsics), so they must resolve for every target that inherits the portable
implementation catalog -- not only nvidia/sm90. When they lived under
platforms/nvidia/sm90 only, sm89 and RDNA retained the split-K candidates (no
feature requirements) but reached bufferized_tir and then failed package
rendering with "No Triton template implements dense_matmul/split_k_...".
"""

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry

_VARIANTS = ("split_k_gemv", "split_k_packed_k_major_gemv")
_PORTABLE_TARGETS = [
    ("nvidia", "sm90"),
    ("nvidia", "sm89"),
    ("amd", "gfx1100"),
    ("amd", "gfx1201"),
]


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("platform,architecture", _PORTABLE_TARGETS)
def test_split_k_templates_resolve_generically(variant, platform, architecture):
    registry = TritonTemplateRegistry()
    resolved = registry.resolve(KernelTemplateSpec("dense_matmul", variant, platform, architecture))
    # Every portable target resolves to the single generic template; the
    # fallback must not require a platform-specific copy.
    assert resolved == f"kernels/dense_matmul/{variant}.py.jinja"
