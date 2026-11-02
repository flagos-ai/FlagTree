# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import import_qwen3_model
from triton.flagmega.options import CompileOptions, parse_placement
from triton.flagmega.targets.amd_gfx1100 import AmdGfx1100Target
from triton.flagmega.targets.amd_gfx1201 import AmdGfx1201Target

pytest.importorskip("torch")

from python.test.flagmega.qwen3.helpers import full_checkpoint_wmma  # noqa: E402

DEFAULT = fm.Placement((8, 16), "yx", "bb")
WIDE = fm.Placement((8, 24), "yx", "bb")


@pytest.mark.parametrize("target_type", [AmdGfx1100Target, AmdGfx1201Target])
def test_rdna_targets_keep_the_default_mesh(target_type):
    assert target_type().options.placements == (DEFAULT, )


@pytest.mark.parametrize("target_type", [AmdGfx1100Target, AmdGfx1201Target])
def test_rdna_with_placement_only_changes_the_mesh(target_type):
    base = target_type()
    wide = base.with_placement(WIDE)
    assert type(wide) is target_type
    assert wide.options.placements == (WIDE, )
    assert base.options.placements == (DEFAULT, )
    assert (wide.name, wide.policy_version, wide.codegen_architecture) == (base.name, base.policy_version,
                                                                            base.codegen_architecture)
    assert wide.triton_implementation_model == base.triton_implementation_model
    assert {field: getattr(wide.options, field)
            for field in vars(wide.options) if field != "placements"} == {
                field: getattr(base.options, field)
                for field in vars(base.options) if field != "placements"
            }
    with pytest.raises(TypeError):
        base.with_placement("yx=8x24/bb")


def test_cli_placement_reaches_the_rdna_target():
    compiler = Compiler(CompileOptions(target="amd-gfx1100", placement=parse_placement("yx=8x24/bb")))
    assert compiler.target.options.placements == (WIDE, )
    assert Compiler(CompileOptions(target="amd-gfx1100")).target.options.placements == (DEFAULT, )


def test_rdna_split_plan_on_the_wide_mesh_keeps_heads_sharded():
    checkpoint = full_checkpoint_wmma(
        num_hidden_layers=1,
        hidden=24 * 8,
        intermediate=128,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=8,
        vocab=64,
    )
    options = CompileOptions(target="amd-gfx1100", placement=parse_placement("yx=8x24/bb"))
    module = Compiler(options).compile(import_qwen3_model(checkpoint), stop_after="decompose-paged-attention").module
    partials = [node for node in module.nodes if node.op == "ntt.paged_attention_partial"]
    assert partials
    # 24 heads divide the x extent 24, so the default y context split already keeps them sharded.
    assert {(node.attrs["split_hierarchy_axis"], node.attrs["split_count"]) for node in partials} == {(0, 8)}
