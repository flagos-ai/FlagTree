# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Instantiate the production projection template without model state."""

import importlib.util

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry


def build_projection_kernel(directory, value_tile):
    source = TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec("gdn_recurrent", "persistent", "nvidia", "sm90"),
        {"recurrent_value_tile": value_tile, "head_block": 128, "query_scale_repr": "0.125"},
    ).source
    wrapper = '''
@triton.jit
def projection_kernel(source, weight, rows, active, output,
                      K: tl.constexpr, TILE_K: tl.constexpr, VALUE_TILE: tl.constexpr,
                      UNIFORM: tl.constexpr):
    offsets = tl.program_id(0) * VALUE_TILE + tl.arange(0, VALUE_TILE)
    row_ids = tl.load(rows + offsets)
    mask = tl.load(active + offsets)
    result = _flagmega_dense_rows(source, weight, row_ids, mask, K, TILE_K, UNIFORM)
    tl.store(output + offsets, result)
'''
    path = directory / "projection_kernel.py"
    path.write_text("import triton\nimport triton.language as tl\n" + source + wrapper)
    spec = importlib.util.spec_from_file_location("flagmega_projection_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.projection_kernel, path
