# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir import lower_transfer_pipeline_regions

from .helpers import module, pipeline_dispatch, prim_function


def test_retired_private_frames_require_complete_writes_before_the_first_read():
    import pytest
    from types import SimpleNamespace
    from triton.flagmega.errors import IRVerificationError
    from triton.flagmega.ir.bufferization import MemorySharingScope
    from triton.flagmega.ir.memory_effect import MemoryAccessMode, MemoryAccessScope, MemoryEffect
    from triton.flagmega.passes.tir.bufferize.synchronization import _Access
    from triton.flagmega.passes.tir.lower_transfer_pipeline_regions import _verify_publication_accesses

    def access(mode, start=0, size=512, *, physical='private-frame', scope=MemorySharingScope.BLOCK):
        return _Access('read' if mode == MemoryAccessMode.READ else 'write', 'value', 'block_local_data', physical,
                       start, size, 'read' if mode == MemoryAccessMode.READ else 'write', None, None,
                       MemoryEffect(mode, MemoryAccessScope.BLOCK), ('whole_resource', None), False, False, scope)

    old = access(MemoryAccessMode.WRITE)
    read = access(MemoryAccessMode.READ)
    plan = SimpleNamespace(buffer_map={'value': SimpleNamespace(role='temporary')})
    check = lambda values: _verify_publication_accesses(values, [(0, old)], 'main', 2, plan)
    check([access(MemoryAccessMode.WRITE, 0, 256), access(MemoryAccessMode.WRITE, 256, 256), read])
    for values in ([read, access(MemoryAccessMode.WRITE)], [access(MemoryAccessMode.WRITE, 0, 256), read],
                   [access(MemoryAccessMode.WRITE, 0, 256),
                    access(MemoryAccessMode.WRITE, 257, 255),
                    read], [access(MemoryAccessMode.WRITE, physical='different-frame'), read]):
        with pytest.raises(IRVerificationError, match='Shared/BLOCK/private'):
            check(values)
    global_write = access(MemoryAccessMode.WRITE, scope=MemorySharingScope.CHIP)
    global_read = access(MemoryAccessMode.READ, scope=MemorySharingScope.CHIP)
    _verify_publication_accesses([global_read], [(0, global_write)], 'main', 2, plan)


def _region(result):
    function = result.prim_functions[0]
    return function.body.fields[0]


def test_single_pipeline_dispatch_gets_explicit_producer_and_consumer_roles():
    dispatch = pipeline_dispatch("kernel", 0, shared_start=0)

    result = lower_transfer_pipeline_regions(module(prim_function("kernel", (dispatch, ))))
    region = _region(result)

    assert isinstance(region, fm.ProducerConsumerRegion)
    assert [type(value) for value in region.produce_body.fields] == [fm.PipelineStage]
    assert [type(value) for value in region.consume_body.fields] == [fm.PipelineStage]
    assert region.produce_body.fields[0].operation is dispatch
    assert region.consume_body.fields[0].operation is dispatch


def test_non_pipeline_function_is_structurally_unchanged():
    dispatch = pipeline_dispatch("kernel", 0, shared_start=0, pipelined=False)
    original = module(prim_function("kernel", (dispatch, )))

    assert lower_transfer_pipeline_regions(original) is original


def test_overlapping_pipeline_stages_drain_previous_shared_owner():
    first = pipeline_dispatch("kernel", 0, shared_start=0)
    second = pipeline_dispatch("kernel", 1, shared_start=32)

    region = _region(lower_transfer_pipeline_regions(module(prim_function("kernel", (first, second)))))

    producer_drains = [value.stage_id for value in region.produce_body.fields if isinstance(value, fm.PipelineDrain)]
    consumer_drains = [value.stage_id for value in region.consume_body.fields if isinstance(value, fm.PipelineDrain)]
    assert producer_drains == ["kernel_transfer_stage_0"]
    assert consumer_drains == producer_drains


def test_disjoint_pipeline_stages_do_not_add_synchronization():
    first = pipeline_dispatch("kernel", 0, shared_start=0)
    second = pipeline_dispatch("kernel", 1, shared_start=64)

    region = _region(lower_transfer_pipeline_regions(module(prim_function("kernel", (first, second)))))

    assert not any(isinstance(value, fm.PipelineDrain) for value in region.consume_body.fields)
    assert not any(isinstance(value, fm.PipelineHandoff) for value in region.consume_body.fields)


def test_ordinary_shared_owner_hands_reused_range_to_pipeline_producer():
    ordinary = pipeline_dispatch("kernel", 0, shared_start=0, pipelined=False)
    pipeline = pipeline_dispatch("kernel", 1, shared_start=32)

    region = _region(lower_transfer_pipeline_regions(module(prim_function("kernel", (ordinary, pipeline)))))

    consumer_handoff = next(value for value in region.consume_body.fields if isinstance(value, fm.PipelineHandoff))
    producer_handoff = next(value for value in region.produce_body.fields if isinstance(value, fm.PipelineHandoff))
    assert consumer_handoff.handoff_id == producer_handoff.handoff_id
    assert region.consume_body.fields.index(consumer_handoff) == 1
    assert region.produce_body.fields.index(producer_handoff) == 0


def test_structured_ordinary_shared_owner_is_an_atomic_consumer_boundary():
    ordinary = pipeline_dispatch("kernel", 0, shared_start=0, pipelined=False)
    conditional = fm.T.if_then_else(
        fm.T.immediate(True, fm.tensor_type("bool", ())),
        fm.T.sequential((ordinary, )),
    )
    pipeline = pipeline_dispatch("kernel", 1, shared_start=32)
    function = prim_function("kernel", ())
    function = replace(
        function,
        body=fm.T.sequential((conditional, pipeline)),
    )

    region = _region(lower_transfer_pipeline_regions(module(function)))
    consumer_handoff = next(value for value in region.consume_body.fields if isinstance(value, fm.PipelineHandoff))
    producer_handoff = next(value for value in region.produce_body.fields if isinstance(value, fm.PipelineHandoff))

    assert region.consume_body.fields[0] is conditional
    assert region.consume_body.fields.index(consumer_handoff) == 1
    assert producer_handoff.handoff_id == consumer_handoff.handoff_id
