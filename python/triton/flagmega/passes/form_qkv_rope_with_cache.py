# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compatibility entry point; formation is a local dataflow rule."""

from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.neutral.form_qkv_rope_with_cache import form_qkv_rope_with_cache_rule


def form_qkv_rope_with_cache(module):
    return DataflowPass("FormQKVRoPEWithCache", (form_qkv_rope_with_cache_rule(),)).run(module)
