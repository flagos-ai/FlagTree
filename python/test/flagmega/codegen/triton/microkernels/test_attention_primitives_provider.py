# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
import inspect
from pathlib import Path

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.implementation import (
    TritonImplementation,
    TritonImplementationModel,
)
from triton.flagmega.codegen.triton.microkernels import (
    AttentionPrimitiveMicroKernelProvider,
    TIRMicroKernelContext,
)
from triton.flagmega.errors import CodegenError

from .helpers import semantic_packed_qkv_module


def _implementation_model():
    implementations = tuple(
        TritonImplementation(
            f"test.{family}.{variant}",
            family,
            variant,
            {"schedule_token": ordinal},
            {"mode": "decode"},
            facts={"portable_reference": variant == "reference"},
        )
        for family in (
            "rotary_embedding",
            "rope",
            "update_paged_attention_kv_cache",
        )
        for ordinal, variant in enumerate(("reference", "optimized"), start=1)
    )
    return TritonImplementationModel(
        implementations,
        {
            family: (f"test.{family}.optimized", f"test.{family}.reference")
            for family in (
                "rotary_embedding",
                "rope",
                "update_paged_attention_kv_cache",
            )
        },
        "test-attention-machine/v1",
    )


def _context(op, attrs, *, arguments, outputs):
    dispatch = fm.T.kernel_dispatch(
        semantic_op=op,
        semantic_candidate=f"semantic.{op}",
        arguments=arguments,
        outputs=outputs,
        semantic_attrs=attrs,
        reads=arguments,
        writes=outputs,
    )
    tensor = fm.tensor_type("bfloat16", (1, 16, 128))
    names = tuple(dict.fromkeys((*arguments, *outputs)))
    function = fm.T.prim_function(
        "attention_primitive",
        "triton",
        tuple(
            fm.T.prim_parameter(
                name,
                tensor,
                fm.T.PrimParameterRole.OUTPUT
                if name in outputs else fm.T.PrimParameterRole.INPUT,
            )
            for name in names
        ),
        fm.T.sequential((dispatch,)),
        fm.T.return_((
            fm.T.return_binding(fm.T.value_ref(outputs[0], tensor), outputs[0]),
        )),
    )
    module = replace(semantic_packed_qkv_module(), prim_functions=(function,))
    return TIRMicroKernelContext(
        module, function, dispatch, _implementation_model()
    )


@pytest.mark.parametrize(
    ("op", "attrs", "arguments", "outputs", "family"),
    (
        (
            "nn.rotary_embedding",
            {"head_dim": 128, "theta": 1_000_000.0, "attention_scaling": 1.0},
            ("reference", "state"),
            ("cos", "sin"),
            "rotary_embedding",
        ),
        (
            "nn.rope",
            {},
            ("input", "cos", "sin"),
            ("result",),
            "rope",
        ),
        (
            "ntt.vectorized_rope",
            {},
            ("input", "cos", "sin"),
            ("result",),
            "rope",
        ),
        (
            "nn.update_paged_attention_kv_cache",
            {"cache_kind": "key", "layout": ("seq", "head", "dim")},
            ("slots", "state", "layer_id", "advance_sequence"),
            ("result",),
            "update_paged_attention_kv_cache",
        ),
    ),
)
def test_provider_queries_injected_family_and_uses_its_preference(
    op, attrs, arguments, outputs, family
):
    proposal = AttentionPrimitiveMicroKernelProvider().propose(
        _context(op, attrs, arguments=arguments, outputs=outputs)
    )

    assert proposal is not None
    assert tuple(value.id for value in proposal.candidates) == (
        f"test.{family}.reference",
        f"test.{family}.optimized",
    )
    assert proposal.default_candidate == f"test.{family}.optimized"


def test_provider_rejects_malformed_semantic_layout_before_catalog_lookup():
    context = _context(
        "nn.update_paged_attention_kv_cache",
        {"cache_kind": "key", "layout": ("seq", "seq", "dim")},
        arguments=("slots", "state", "layer_id", "advance_sequence"),
        outputs=("result",),
    )

    with pytest.raises(CodegenError, match="layout"):
        AttentionPrimitiveMicroKernelProvider().propose(context)


def test_attention_microkernel_provider_contains_no_machine_or_tile_policy():
    source = Path(
        inspect.getsourcefile(AttentionPrimitiveMicroKernelProvider) or ""
    ).read_text(encoding="utf-8").lower()

    for spelling in (
        "nvidia",
        "sm90",
        "block_n =",
        "block_k =",
        "num_warps =",
        "num_stages =",
        "mesh_size",
    ):
        assert spelling not in source
