"""The exact top8 overlay retains unrelated routing and normalization contracts."""
import copy

import pytest

from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.errors import CodegenError
from local_optimizations.router_ordered_top8 import KEY, SCHEDULE, describe, transform


def _descriptor():
    call = dict(family="staged_routing", capacity=288, tile=512, rows=1, k=8,
                selected_tile=8, scoring="sigmoid", arithmetic="tanh_scaled_reciprocal",
                has_bias=True, normalize=True, scaling="2.5",
                router_quant_overlap={"quant_symbol": "original_quant"})
    return dict(render_calls=[copy.deepcopy(call), copy.deepcopy(call),
                             dict(family="staged_routing", arithmetic="exp_grouped_reciprocal")],
                implementation_override=dict(inputs={}))


def test_native_template_keeps_original_selection_and_all_normalization():
    source = (TritonTemplateRegistry().root / KEY).read_text()
    result = transform(source)
    start = source.index("        _fm_available = _fm_mask\n")
    end = source.index("{% if call.normalize %}\n", start)
    assert source[start:end] in result
    assert result.endswith(source[end:])
    assert result.count("tl.topk(_fm_group_keys, {{ call.k }}, dim=1)") == 1
    assert result.count("tl.topk(_fm_candidates32, {{ call.k }})") == 1
    assert "BlockEncoding([4], [32], [{{ num_warps }}], [0])" in result
    assert "_fm_candidates == 0." in result and "_fm_candidates != _fm_candidates" in result
    assert "~_fm_offsets.to(tl.uint32)" in result
    assert "_fm_selected_values == _fm_selected_values" in result
    assert "BlockEncoding([1], [32], [{{ num_warps }}], [0])" in result


def test_complete_descriptor_only_selects_reviewed_router_regions():
    original = _descriptor()
    result = describe(None, lambda _: copy.deepcopy(original))
    assert result["render_calls"][2] == original["render_calls"][2]
    for before, after in zip(original["render_calls"][:2], result["render_calls"][:2], strict=True):
        assert after.pop("ordered_packed_top8") == SCHEDULE
        assert after == before


@pytest.mark.parametrize("field,value", [("capacity", 256), ("k", 4), ("selected_tile", 16),
                                         ("arithmetic", "standard"), ("has_bias", False),
                                         ("normalize", False), ("scaling", 1.0)])
def test_changed_original_contract_is_rejected(field, value):
    descriptor = _descriptor()
    descriptor["render_calls"][0][field] = value
    with pytest.raises(CodegenError, match="original score and normalization contract"):
        describe(None, lambda _: descriptor)


def test_missing_complete_routed_region_is_rejected():
    descriptor = _descriptor()
    descriptor["render_calls"].pop(0)
    with pytest.raises(CodegenError, match="both complete routed regions"):
        describe(None, lambda _: descriptor)


def test_changed_template_boundary_is_rejected():
    source = (TritonTemplateRegistry().root / KEY).read_text()
    with pytest.raises(CodegenError, match="selection boundary"):
        transform(source.replace("        _fm_available = _fm_mask\n", ""))


def test_changed_load_layout_boundary_is_rejected():
    source = (TritonTemplateRegistry().root / KEY).read_text()
    with pytest.raises(CodegenError, match="selection boundary"):
        transform(source.replace("        _fm_offsets = tl.arange(0, {{ call.tile }})\n", ""))
