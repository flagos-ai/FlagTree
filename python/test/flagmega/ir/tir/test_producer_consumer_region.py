# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError


def _dispatch() -> fm.KernelDispatch:
    return fm.T.kernel_dispatch(
        semantic_op="test.copy",
        arguments=("input",),
        outputs=("output",),
        semantic_facts={
            "flops": 7,
            "bytes_read": 16,
            "bytes_written": 16,
            "synchronizations": 0,
        },
        reads=("input",),
        writes=("output",),
    )


def _region(dispatch=None):
    dispatch = dispatch or _dispatch()
    stage = fm.T.pipeline_stage("copy_stage", dispatch)
    return fm.T.producer_consumer_region(
        fm.T.sequential((stage,)),
        fm.T.sequential((stage,)),
    )


def _module(region) -> fm.IRModule:
    tensor = fm.tensor_type("float32", (4,))
    function = fm.T.prim_function(
        "copy",
        "triton",
        (
            fm.T.prim_parameter("input", tensor, fm.T.PrimParameterRole.INPUT),
            fm.T.prim_parameter("output", tensor, fm.T.PrimParameterRole.OUTPUT),
        ),
        fm.T.sequential((region,)),
        fm.T.return_((
            fm.T.return_binding(fm.T.value_ref("output", tensor), "output"),
        )),
    )
    builder = fm.IRBuilder(dialect="bufferized_tir", stage="bufferized_tir")
    source = builder.var("source", tensor, id="source")
    builder.prim_function(function)
    result = builder.call(
        "tir.call", (source,), tensor, id="result", attrs={"callee": "copy"}
    )
    builder.function("main", (source,), (result,))
    return fm.verify_module(builder.build(entry="main"))


def test_region_requires_matching_stage_drain_and_handoff_structure():
    dispatch = _dispatch()
    first = fm.T.pipeline_stage("first", dispatch)
    second = fm.T.pipeline_stage("second", dispatch)

    with pytest.raises(IRSchemaError, match="identical IDs and order"):
        fm.T.producer_consumer_region(
            fm.T.sequential((first,)), fm.T.sequential((second,))
        )
    with pytest.raises(IRSchemaError, match="before executing"):
        fm.T.producer_consumer_region(
            fm.T.sequential((fm.T.pipeline_drain("first"), first)),
            fm.T.sequential((first,)),
        )
    with pytest.raises(IRSchemaError, match="duplicate pipeline handoff"):
        fm.T.producer_consumer_region(
            fm.T.sequential((
                first,
                fm.T.pipeline_handoff("ready"),
                fm.T.pipeline_handoff("ready"),
            )),
            fm.T.sequential((first,)),
        )


def test_region_cost_counts_one_semantic_kernel_and_one_logical_marker():
    dispatch = _dispatch()
    stage = fm.T.pipeline_stage("copy_stage", dispatch)
    handoff = fm.T.pipeline_handoff("ready")
    region = fm.T.producer_consumer_region(
        fm.T.sequential((stage, handoff)),
        fm.T.sequential((stage, handoff)),
    )

    cost = fm.estimate_tir_cost(region)

    assert cost.flops == 7
    assert cost.bytes_read == 16
    assert cost.bytes_written == 16
    assert cost.synchronizations == 1


def test_region_round_trips_as_editable_python_and_readable_script(tmp_path: Path):
    module = _module(_region())

    checkpoint = fm.emit_module(module, tmp_path / "pipeline.py")
    loaded = fm.load_module(checkpoint)
    python_source = checkpoint.read_text(encoding="utf-8")
    script = fm.script_source(module)

    assert loaded.semantic_hash == module.semantic_hash
    assert "T.producer_consumer_region(" in python_source
    assert "T.pipeline_stage(" in python_source
    assert "T.ProducerConsumerRegion" in script
    assert "produce:" in script and "consume:" in script
    assert "T.PipelineStage('copy_stage')" in script


def test_kernel_dispatch_identity_survives_single_stage_region_lowering():
    dispatch = _dispatch()
    function = _module(_region(dispatch)).prim_function_map["copy"]

    assert fm.kernel_dispatch_of(function) == dispatch
