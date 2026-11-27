# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.tir_package import (
    describe_tir_package,
    render_tir_package,
)

_AUX_PIPELINE = (
    "tir.dense_matmul.tensor_descriptor_smem_pipeline_aux_gemv"
)


def test_auxiliary_pipeline_emits_three_roles_and_direct_output_publication(
    compile_pipeline_module,
):
    package = describe_tir_package(compile_pipeline_module(
        reusable=False,
        implementation=_AUX_PIPELINE,
    ))
    schedule = package["pipeline_schedule"]
    source = render_tir_package(package, "unit")

    compile(source, "pipeline.py", "exec")
    assert len(schedule["auxiliary_events"]) == 1
    stage = schedule["stages"][0]
    auxiliary = stage["auxiliary_consumer"]
    assert auxiliary["channel_indices"] == (0,)
    assert not auxiliary["consumer_shared_workspace_indices"]
    assert auxiliary["warps"] == 4
    assert auxiliary["registers"] == 160
    assert len(stage["workspaces"]) == 1
    assert stage["workspaces"][0]["pipe_owned"] is True
    assert source.count("def flagmega_main__auxiliary_consumer(") == 1
    assert "[4, 1]" in source
    assert "[160, 32]" in source
    assert "pipeline_auxiliary_start_writer.commit(0)" in source
    assert "pipeline_auxiliary_start_reader.wait(0)" in source
    assert "pipeline_auxiliary_done_writer.commit(0)" in source
    assert "dense_accumulator.to(tl.bfloat16)" in source


def test_auxiliary_reader_owns_drain_across_reused_pipeline_calls(
    compile_pipeline_module,
):
    package = describe_tir_package(compile_pipeline_module(
        reusable=True,
        implementation=_AUX_PIPELINE,
    ))
    schedule = package["pipeline_schedule"]
    source = render_tir_package(package, "unit")

    auxiliary_drains = [
        value for value in schedule["auxiliary_events"]
        if value["kind"] == "drain"
    ]
    primary_drains = [
        value for value in schedule["consumer_events"]
        if value["kind"] == "drain"
    ]
    assert len(auxiliary_drains) == 1
    assert not primary_drains
    assert source.count(".pipe.wait_drained()") == 2
