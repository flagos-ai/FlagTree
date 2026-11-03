# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Exact complete-expert top8 with an unsigned total-order bitonic network."""

import hashlib
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader, Environment
from triton.flagmega.errors import CodegenError

KEY = "kernels/staged_routing/local.py.jinja"
SCHEDULE = "complete_E288_uint64_warp_partitioned_top8_v2"
_active = False


def describe(module, previous):
    descriptor = previous(module)
    calls = [call for call in descriptor["render_calls"]
             if call["family"] == "staged_routing" and call.get("router_quant_overlap")]
    if len(calls) != 2:
        raise CodegenError("Ordered router top8 requires both complete routed regions")
    for call in calls:
        expected = dict(capacity=288, tile=512, rows=1, k=8, selected_tile=8,
                        scoring="sigmoid", arithmetic="tanh_scaled_reciprocal",
                        has_bias=True, normalize=True, scaling="2.5")
        if any(call.get(key) != value for key, value in expected.items()):
            raise CodegenError("Ordered router top8 requires its complete original score and normalization contract")
        call["ordered_packed_top8"] = SCHEDULE
    descriptor["implementation_override"]["inputs"][Path(__file__).name] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    descriptor["ordered_complete_router_top8"] = dict(
        schedule=SCHEDULE, experts=288, routes=8,
        original_nan_first_score_descending_index_ascending=True,
        original_singleton_maxnum_semantics_and_sum8_div_rn=True,
        original_all_quantization_and_dispatch_owners_retained=True)
    return descriptor


def transform(source):
    offsets = "        _fm_offsets = tl.arange(0, {{ call.tile }})\n"
    start = "        _fm_available = _fm_mask\n"
    end = "{% if call.normalize %}\n"
    if source.count(offsets) != 1 or source.count(start) != 1 or source.count(end) != 1:
        raise CodegenError("Original complete staged routing selection boundary changed")
    first = source.index(start)
    last = source.index(end, first)
    original = source[first:last]
    if original.count("_flagmega_staged_routing_pair_max") != 1:
        raise CodegenError("Original staged routing total-order comparator changed")
    packed = '''{% if call.ordered_packed_top8 | default(false) %}
        # Canonical zero matches floating equality; all NaNs share one score key.
        _fm_bits = tl.where(_fm_candidates == 0., 0., _fm_candidates).to(tl.uint32, bitcast=True)
        _fm_order = tl.where((_fm_bits & 2147483648) != 0, ~_fm_bits, _fm_bits ^ 2147483648)
        _fm_order = tl.where(_fm_candidates != _fm_candidates, 4294967295, _fm_order).to(tl.uint32)
        _fm_key = (_fm_order.to(tl.uint64) << 32) | (~_fm_offsets.to(tl.uint32)).to(tl.uint64)
        # Every valid float key, including -inf, is greater than padding zero.
        _fm_key = tl.where(_fm_mask, _fm_key, 0)
        _fm_selected_offsets = tl.arange(0, {{ call.selected_tile }})
        # A global top8 winner belongs to its own group's top8.
        _fm_group_keys = tle.gpu.set_layout(tl.reshape(_fm_key, (4, 128)), tle.gpu.BlockEncoding([1, 4], [1, 32], [{{ num_warps }}, 1], [1, 0]))
        _fm_group_best = tl.topk(_fm_group_keys, {{ call.k }}, dim=1)
        _fm_candidates32 = tle.gpu.set_layout(tl.reshape(_fm_group_best, (32,)), tle.gpu.BlockEncoding([1], [32], [{{ num_warps }}], [0]))
        _fm_selected_keys = tl.topk(_fm_candidates32, {{ call.k }})
        _fm_selected_indices = (~(_fm_selected_keys & 4294967295).to(tl.uint32)).to(tl.int64)
        _fm_selected_values = tl.gather(_fm_probabilities, _fm_selected_indices.to(tl.int32), 0)
        # The original singleton maxnum ignores a NaN probability against -inf.
        _fm_selected_values = tl.where(_fm_selected_values == _fm_selected_values, _fm_selected_values, -float("inf"))
        # Preserve the original sum8 tree instead of the topk register layout.
        _fm_sum_layout: tl.constexpr = tle.gpu.BlockEncoding([1], [32], [{{ num_warps }}], [0])
        _fm_selected_values = tle.gpu.set_layout(_fm_selected_values, _fm_sum_layout)
{% else %}
'''
    result = source[:first] + packed + original + "{% endif %}\n" + source[last:]
    return result.replace(offsets, offsets + '''{% if call.ordered_packed_top8 | default(false) %}
        _fm_offsets = tle.gpu.set_layout(_fm_offsets, tle.gpu.BlockEncoding([4], [32], [{{ num_warps }}], [0]))
{% endif %}
''')


def activate():
    global _active
    if _active:
        return
    from triton.flagmega.codegen.triton import renderer, tir_package
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec

    previous = tir_package.describe_tir_package

    def wrapped(module):
        return describe(module, previous)

    tir_package.describe_tir_package = wrapped
    spec = renderer.renderer_registry._specs["bufferized-tir"]
    renderer.renderer_registry._specs[spec.name] = PackageRendererSpec(
        spec.name, spec.kind, spec.matches, wrapped, spec.render)
    original = Environment.get_template

    def selected(self, name, *args, **kwargs):
        if name in ("module.py.jinja", KEY):
            source = self.loader.get_source(self, KEY)[0]
            if "call.ordered_packed_top8" not in source:
                self.loader = ChoiceLoader([DictLoader({KEY: transform(source)}), self.loader])
        return original(self, name, *args, **kwargs)

    Environment.get_template = selected
    _active = True
