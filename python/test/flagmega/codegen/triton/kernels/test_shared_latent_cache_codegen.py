# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from pathlib import Path
import ast, json
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig
from python.test.flagmega.ir.ops.nn._shared_latent_cache_fixtures import SharedModule

root = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "lanes,layout,split_axis,cyclic,heads,reusable",
    [
        ((), ("seq", "head", "dim"), None, False, 4, False),
        ((2, 4), ("head", "dim", "seq"), "head", True, 7, False),
        ((16,), ("dim", "seq", "head"), "dim", False, 4, False),
        ((8,), ("seq", "head", "dim"), "head", False, 4, True),
    ],
)
def test_normal_lowering_consumes_shared_scalar_lanes_without_concat(
    tmp_path, lanes, layout, split_axis, cyclic, heads, reusable
):
    mesh = fm.Placement((2, 2), "yx", "bb")
    g = SharedModule(
        PagedAttentionStateConfig(3, heads, 32),
        lanes=lanes,
        layout=layout,
        placement=mesh,
        split_axis=split_axis,
        cyclic=cyclic,
        reusable=reusable,
    ).build()
    c = Compiler(CompileOptions(placement=mesh))
    m = c.compile(g).module
    restored = fm.load_module(fm.emit_module(m, tmp_path / "shared.py"))
    assert restored.semantic_hash == m.semantic_hash
    path = write_artifact(restored, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    source = (path / "generated_kernels.py").read_text()
    assert "flagmega-kernel: concat/" not in source and "flagmega-kernel: pack/" not in source
    (dispatch,) = (
        d.dispatch
        for d in m.kernel_definitions
        if d.dispatch.microkernel.family == "update_paged_attention_kv_cache_pair"
    )
    assert dispatch.semantic_attrs["shared_kv"] is True
    tree = ast.parse(source)
    loads = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cache_value" for t in n.targets)
    ]
    assert len(loads) == 1
    stores = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "tl.store"
        and len(n.args) > 1
        and isinstance(n.args[1], ast.Name)
        and n.args[1].id == "cache_value"
    ]
    assert len(stores) == 2


def test_omitted_and_explicit_false_keep_identical_legacy_source(tmp_path):
    sources = []
    for shared in (None, False):
        graph = SharedModule(PagedAttentionStateConfig(3, 4, 32), shared_kv=shared).build()
        compiled = Compiler().compile(graph).module
        artifact = write_artifact(compiled, tmp_path / str(shared), target="nvidia-sm90", emit_executable=True)
        sources.append((artifact / "generated_kernels.py").read_bytes())
    assert sources[0] == sources[1]


def test_shared_input_removes_the_combined_activation_buffer(tmp_path):
    plans = {}
    for shared in (None, True):
        g = SharedModule(PagedAttentionStateConfig(3, 4, 32), shared_kv=shared, lanes=(2, 4)).build()
        m = Compiler().compile(g).module
        fm.verify_buffer_plan(m)
        p = write_artifact(m, tmp_path / str(shared), target="nvidia-sm90", emit_executable=True)
        manifest = json.loads((p / "artifact.json").read_text())
        plans[shared] = sum(
            pool["nbytes"] for pool in manifest["codegen"]["runtime_binding"]["pools"] if pool["storage"] != "rdata"
        )
        assert ("flagmega-kernel: concat/" in (p / "generated_kernels.py").read_text()) is (shared is None)
    assert plans[True] < plans[None]
