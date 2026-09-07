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
    QKVRoPEWithCacheMicroKernelProvider,
    TIRMicroKernelContext,
    default_triton_microkernel_registry,
)
from triton.flagmega.errors import CodegenError

from .helpers import semantic_packed_qkv_module


def _context(*, attrs=None, arguments=None, outputs=("result_0", "result_1")):
    arguments = arguments or (
        "qkv",
        "q_scale",
        "k_scale",
        "q_bias",
        "k_bias",
        "cos",
        "sin",
        "state",
        "layer_id",
        "advance_sequence",
    )
    attrs = attrs or {
        "q_axis": 2,
        "q_epsilon": 1e-6,
        "q_use_mean": False,
        "k_axis": 2,
        "k_epsilon": 1e-6,
        "k_use_mean": False,
        "qkv_layout": ("seq", "head", "dim"),
        "attention_layout": ("seq", "head", "dim"),
    }
    dispatch = fm.T.kernel_dispatch(
        semantic_op="nn.qkv_rope_with_cache",
        semantic_candidate="semantic.nn.qkv_rope_with_cache",
        arguments=arguments,
        outputs=outputs,
        semantic_attrs=attrs,
        reads=arguments,
        writes=tuple(
            value for value in ("state", *outputs) if value in (*arguments, *outputs)
        ),
        effect_kind="read_write",
        effect_resource="paged_attention_kv_cache",
    )
    function = fm.T.prim_function(
        "qkv_rope_cache",
        "triton",
        (),
        fm.T.sequential((dispatch,)),
    )
    module = replace(semantic_packed_qkv_module(), prim_functions=(function,))
    implementations = tuple(
        TritonImplementation(
            f"test.qkv_rope_with_cache.{variant}",
            "qkv_rope_with_cache",
            variant,
            {"elements_per_program": elements},
            {"mode": "decode"},
            facts={"portable_reference": variant == "reference"},
        )
        for variant, elements in (("reference", 64), ("optimized", 128))
    )
    model = TritonImplementationModel(
        implementations,
        {
            "qkv_rope_with_cache": (
                "test.qkv_rope_with_cache.optimized",
                "test.qkv_rope_with_cache.reference",
            )
        },
        "test-qkv-rope-machine/v1",
    )
    return TIRMicroKernelContext(module, function, dispatch, model)


def test_provider_selects_injected_generic_decode_implementation():
    provider = QKVRoPEWithCacheMicroKernelProvider()

    proposal = provider.propose(_context())

    assert proposal is not None
    assert tuple(value.id for value in proposal.candidates) == (
        "test.qkv_rope_with_cache.reference",
        "test.qkv_rope_with_cache.optimized",
    )
    assert proposal.default_candidate == "test.qkv_rope_with_cache.optimized"
    assert isinstance(
        default_triton_microkernel_registry().provider_for(
            "nn.qkv_rope_with_cache"
        ),
        QKVRoPEWithCacheMicroKernelProvider,
    )


@pytest.mark.parametrize(
    ("attrs", "arguments", "message"),
    (
        (
            {**_context().dispatch.semantic_attrs, "q_axis": 3},
            None,
            "axis",
        ),
        (
            {**_context().dispatch.semantic_attrs, "qkv_layout": ("seq", "seq", "dim")},
            None,
            "layout",
        ),
        (
            None,
            ("qkv",),
            "requires 10 arguments",
        ),
    ),
)
def test_provider_rejects_malformed_semantic_contract(attrs, arguments, message):
    with pytest.raises(CodegenError, match=message):
        QKVRoPEWithCacheMicroKernelProvider().propose(
            _context(attrs=attrs, arguments=arguments)
        )


def test_provider_has_no_model_machine_or_tile_policy():
    source = Path(
        inspect.getsourcefile(QKVRoPEWithCacheMicroKernelProvider) or ""
    ).read_text(encoding="utf-8").lower()

    for spelling in (
        "qwen",
        "nvidia",
        "sm90",
        "block_n =",
        "block_k =",
        "num_warps =",
        "num_stages =",
    ):
        assert spelling not in source
