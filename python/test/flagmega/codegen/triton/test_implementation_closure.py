# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.kernel_call_renderers import _FAMILY_ENCODERS
from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
)
from triton.flagmega.targets.portable_triton_implementations import (
    portable_triton_implementation_model,
)


def test_every_catalog_implementation_has_a_renderer_and_resolvable_template():
    registry = TritonTemplateRegistry()
    for implementation in portable_triton_implementation_model().implementations:
        assert implementation.family in _FAMILY_ENCODERS, implementation.id
        assert registry.resolve(KernelTemplateSpec(
            implementation.family,
            implementation.variant,
            "nvidia",
            "sm90",
        ))
