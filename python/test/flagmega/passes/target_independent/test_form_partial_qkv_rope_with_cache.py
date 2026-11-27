# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.form_qkv_rope_with_cache import form_qkv_rope_with_cache
from test_form_qkv_rope_with_cache import QKVRoPERegion


@pytest.mark.parametrize("rotary_dim", [16, 32, 48, 64])
@pytest.mark.parametrize("wide", [False, True])
def test_partial_rope_forms_cache_region(rotary_dim, wide, tmp_path):
    source = QKVRoPERegion(rotary_dim=rotary_dim, wide=wide).build()
    result = form_qkv_rope_with_cache(source)
    fused = [node for node in result.nodes if node.op == "nn.qkv_rope_with_cache"]
    assert len(fused) == 1
    assert fused[0].attrs["rotary_dim"] == rotary_dim
    assert not any(node.op in {"nn.rope", "nn.update_paged_attention_kv_cache"} for node in result.nodes)
    assert result.node_map["updated"].type == source.node_map["updated"].type
    assert fm.load_module(fm.emit_module(result, tmp_path / "partial.py")) == result
