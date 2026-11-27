# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
import ast
import json
from pathlib import Path
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.artifacts import write_artifact
from python.test.flagmega.ir.ops.nn._mla_head_fixtures import graph, policies


def compile_projection(path, **kwargs):
    mesh = kwargs.get("mesh")
    source = graph(**kwargs)
    module = Compiler(CompileOptions(placement=mesh) if mesh else CompileOptions()).compile(source).module
    fm.verify_buffer_plan(module)
    restored = fm.load_module(fm.emit_module(module, path / "compiled.py"))
    assert restored.semantic_hash == module.semantic_hash
    artifact = write_artifact(restored, path / "artifact", target="nvidia-sm90", emit_executable=True)
    return module, artifact


@pytest.mark.parametrize(
    "direction,fp8,cyclic,staged,scale_sharded,reusable",
    [
        ("absorb_query", False, False, False, False, False),
        ("expand_value", False, True, False, False, False),
        ("absorb_query", True, True, False, False, True),
        ("expand_value", True, False, True, True, True),
    ],
)
def test_normal_pipeline_emits_only_owned_complete_heads(
    tmp_path, direction, fp8, cyclic, staged, scale_sharded, reusable
):
    mesh = fm.Placement((2, 2), "yx", "bb")
    heads = 7 if cyclic else 8
    hp, rp = policies(mesh, heads=heads, cyclic=cyclic, staged=staged)
    m, p = compile_projection(
        tmp_path,
        direction=direction,
        fp8=fp8,
        mesh=mesh,
        heads=heads,
        head_policy=hp,
        row_policy=rp,
        scale_sharded=scale_sharded,
        reusable=reusable,
    )
    text = (p / "generated_kernels.py").read_text()
    ast.parse(text)
    assert "_fm_mla_global_head" in text
    assert "flagmega-kernel: pack/" not in text and "flagmega-kernel: unpack/" not in text
    (dispatch,) = (d.dispatch for d in m.kernel_definitions if d.dispatch.microkernel.family == "mla_projection")
    assert dispatch.semantic_attrs["direction"] == direction
