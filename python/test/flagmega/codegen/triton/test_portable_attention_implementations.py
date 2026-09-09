# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path
import ast
import inspect
import re
from types import SimpleNamespace

from triton.flagmega.codegen.triton.portable_implementations import (
    portable_attention_implementations,
    portable_attention_preferences,
)
from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
)
from triton.flagmega.targets.nvidia.implementations import (
    sm90_triton_implementation_model, )


def test_attention_catalog_is_target_neutral_and_has_real_generic_templates():
    implementations = portable_attention_implementations()
    preferences = portable_attention_preferences()

    assert {value.family for value in implementations} == set(preferences)
    assert "qkv_rope_with_cache" in preferences
    assert all(value.facts["portable_triton"] is True for value in implementations)
    registry = TritonTemplateRegistry()
    for implementation in implementations:
        spec = KernelTemplateSpec(
            implementation.family,
            implementation.variant,
            "nvidia",
            "sm90",
        )
        assert registry.resolve(spec) == (f"kernels/{implementation.family}/{implementation.variant}.py.jinja")
        rendered = registry.render_kernel(spec, {}).source
        compile(rendered, f"{implementation.family}.py", "exec")

    source = Path(__file__).parents[4].joinpath("triton", "flagmega", "codegen", "triton",
                                                "portable_implementations.py").read_text(encoding="utf-8").lower()
    for target_spelling in ("nvidia", "sm90", "cuda", "mma", "tma"):
        # Match identifier/name components, not the "tma" inside "softmax".
        assert not re.search(rf"(?<![a-z0-9]){target_spelling}(?![a-z0-9])", source)


def test_sm90_model_composes_portable_attention_instead_of_redeclaring_it():
    portable = {value.id: value for value in portable_attention_implementations()}
    model = sm90_triton_implementation_model()

    for implementation_id, implementation in portable.items():
        assert model.implementation(implementation_id) == implementation
        assert implementation_id in model.preferences[implementation.family]

    # A physical target may prefer a legal specialized implementation, but
    # composing it must preserve every portable fallback and its relative
    # order.  Requiring a portable candidate to remain first would disable
    # target-owned performance selection.
    portable_ids = frozenset(portable)
    for family, expected in portable_attention_preferences().items():
        actual = tuple(implementation_id for implementation_id in model.preferences[family]
                       if implementation_id in portable_ids)
        assert actual == expected

    source = inspect.getsource(
        __import__(
            "triton.flagmega.targets.nvidia.implementations",
            fromlist=("sm90_triton_implementation_model", ),
        ))
    assert "portable_attention_implementations()" in source
    for implementation_id in portable:
        assert f'"{implementation_id}"' not in source


def test_partial_attention_exposes_independent_tile_variants_and_prefers_no_spill_tile():
    implementations = tuple(value for value in portable_attention_implementations()
                            if value.family == "paged_attention_partial")

    assert {value.id: (value.variant, value.parameters["token_tile"])
            for value in implementations} == {
                "tir.paged_attention_partial.decode_t16": ("decode_t16", 16),
                "tir.paged_attention_partial.decode_t32": ("decode_t32", 32),
            }
    assert portable_attention_preferences()["paged_attention_partial"] == (
        "tir.paged_attention_partial.decode_t16",
        "tir.paged_attention_partial.decode_t32",
    )


def test_attention_scalar_controls_are_values_not_pointer_operands():
    registry = TritonTemplateRegistry()
    rendered = {
        implementation.family:
        registry.render_kernel(
            KernelTemplateSpec(
                implementation.family,
                implementation.variant,
                "nvidia",
                "sm90",
            ),
            {},
        ).source
        for implementation in portable_attention_implementations()
    }

    # Rank-zero TIR controls are lowered through the scalar ABI.  Loading them
    # as pointers makes literal/folded controls fail during Triton AST lowering.
    assert "tl.load(layer_id)" not in rendered["paged_attention_partial"]
    update = rendered["update_paged_attention_kv_cache"]
    assert "tl.load(layer_id)" not in update
    assert "tl.load(advance_sequence" not in update


def test_attention_length_includes_the_cache_slot_just_published_by_update():
    registry = TritonTemplateRegistry()
    rendered = {
        implementation.family:
        registry.render_kernel(
            KernelTemplateSpec(
                implementation.family,
                implementation.variant,
                "nvidia",
                "sm90",
            ),
            {},
        ).source
        for implementation in portable_attention_implementations()
    }

    source = rendered["paged_attention_partial"]
    length = next(node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "context_length" for target in node.targets))
    expression = compile(ast.Expression(length), "attention_length", "eval")
    tl = SimpleNamespace(load=lambda value: SimpleNamespace(to=lambda dtype: value), int32=int)
    for slot in (0, 1, 31, 255, 256):
        for query_token in (0, 1, 31):
            assert eval(expression,
                        {"tl": tl, "slot_mapping": slot, "query_token": query_token}) == slot + query_token + 1
    assert "tl.load(sequence_lengths)" not in source


def test_partial_attention_does_not_apply_compact_owner_offset_twice():
    implementation = next(value for value in portable_attention_implementations()
                          if value.family == "paged_attention_partial")
    source = TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec(
            implementation.family,
            implementation.variant,
            "nvidia",
            "sm90",
        ),
        {},
    ).source

    assert "partial_index = 0" in source
    assert "partial_index = head" not in source
    assert "context_shard * num_query_heads" not in source
