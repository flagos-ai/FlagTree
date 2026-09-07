# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm


def test_explicit_empty_tir_alias_contract_never_queries_semantic_registry():
    tensor = fm.tensor_type("bfloat16", (1, 16))
    dispatch = fm.KernelDispatch(
        semantic_op="tir.synthetic.fused",
        semantic_candidate="tir.synthetic.fused",
        arguments=("input",),
        outputs=("result",),
        inplace_alias_candidates=(),
        reads=("input",),
        writes=("result",),
    )
    prim_function = fm.PrimFunction(
        name="synthetic",
        module_kind="triton",
        parameters=(
            fm.PrimParameter("input", tensor, fm.PrimParameterRole.INPUT),
            fm.PrimParameter("result", tensor, fm.PrimParameterRole.OUTPUT),
        ),
        body=fm.Sequential((dispatch,)),
        results=fm.Return((fm.ReturnBinding(fm.ValueRef("result", tensor), "result"),)),
    )
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    input_value = builder.var("input", tensor, id="input")
    result = builder.call(
        "tir.call",
        (input_value,),
        tensor,
        id="result",
        attrs={"callee": "synthetic"},
    )
    builder.function("main", (input_value,), (result,))
    module = replace(
        builder.build(entry="main"), prim_functions=(prim_function,)
    )

    plan = fm.make_buffer_plan(module)

    assert plan.buffer_map["result"].alias is None


def test_missing_legacy_alias_contract_remains_distinct_from_explicit_empty():
    common = {
        "semantic_op": "tir.synthetic.fused",
        "semantic_candidate": "tir.synthetic.fused",
        "arguments": ("input",),
        "outputs": ("result",),
        "reads": ("input",),
        "writes": ("result",),
    }
    legacy = fm.KernelDispatch(**common)
    explicit = fm.KernelDispatch(**common, inplace_alias_candidates=())

    assert "inplace_alias_candidates" not in legacy.to_data()
    assert explicit.to_data()["inplace_alias_candidates"] == {"$tuple": []}
    assert fm.tir_from_data(legacy.to_data()).inplace_alias_candidates is None
    assert fm.tir_from_data(explicit.to_data()).inplace_alias_candidates == ()
