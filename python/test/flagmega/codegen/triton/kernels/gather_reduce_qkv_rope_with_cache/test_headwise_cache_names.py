# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The headwise-partial QKV write stores K/V through the renderer's per-head
cache_offset names (qkv_k_/qkv_v_cache_physical_block, _block_offset); its
per-token cache position must bind them, or the kernel fails to compile."""

import ast
import builtins

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from python.test.flagmega.passes.test_fuse_qkv_statistics import qkv_statistics_graph


def _undefined_names(source):
    tree = ast.parse(source)
    module_names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    module_names |= {target.id for node in tree.body if isinstance(node, ast.Assign)
                     for target in node.targets if isinstance(target, ast.Name)}
    module_names |= {(alias.asname or alias.name).split(".")[0] for node in tree.body
                     if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    undefined = set()
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        bound = {argument.arg for argument in function.args.args}
        bound |= {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        undefined |= {(function.name, node.id) for node in ast.walk(function)
                      if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                      and node.id not in bound | module_names and not hasattr(builtins, node.id)}
    return undefined


@pytest.mark.parametrize("target", ("amd-gfx1100", "amd-gfx1201", "nvidia-sm90"))
def test_headwise_partial_cache_write_reads_only_defined_names(tmp_path, target):
    source, _ = qkv_statistics_graph(head_dim=64, rotary_dim=None, use_mean=False, heads_tuple=(4, 2, 2))
    compiled = Compiler(CompileOptions(target=target)).compile(source).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target=target, emit_executable=True)
    generated = (artifact / "generated_kernels.py").read_text()
    assert "qkv_k_cache_physical_block" in generated  # the headwise K/V cache store
    assert _undefined_names(generated) == set()
