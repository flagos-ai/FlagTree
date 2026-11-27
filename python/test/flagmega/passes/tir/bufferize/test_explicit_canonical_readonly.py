# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit readonly coordinates constrain both allocation and descriptors."""
from dataclasses import replace
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.distributed_abi import distributed_buffer_abi
from triton.flagmega.passes.tir.bufferize import BufferizationOptions

@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "bfloat16", "float32"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("views", [0, 1, 2])
def test_explicit_canonical_readonly_layout_survives_descriptor_and_view_planning(dtype, grouped, views):
    kind=fm.DistributedType(fm.tensor_type(dtype,(16,512)),
        (fm.SBP.broadcast(),fm.SBP.split_contiguous((0,),128)),
        fm.Placement((4,8,16),"cyx","cbb"))
    local_kind=replace(kind,placement=fm.Placement((4,1,1),"cyx","cbb"))
    metadata={"canonical_global":"true"}
    if grouped:
        metadata["rdata_group"]={"name":"weights","index":0,"count":1}
    b=fm.IRBuilder(dialect="tir",stage="selected_tir")
    w=b.weight("weight",kind,source="fixture.safetensors",key="weight",id="weight",metadata=metadata)
    current=w
    for index in range(views):
        current=b.call("distributed.sharded_view",(current,),local_kind,id=f"view{index}",attrs={"new_type":local_kind})
    output=b.call("tir.kernel",(current,),local_kind if views else kind,id="output")
    b.function("main",(),(output,))
    plan=fm.make_buffer_plan(b.build(entry="main"),
        options=replace(BufferizationOptions.generic(),rdata_chip_sharding=True))
    source=plan.buffer_map["weight"]
    expected_bytes=16*512*fm.DType(dtype).itemsize
    for name in ("weight",*(f"view{i}" for i in range(views))):
        descriptor=plan.buffer_map[name]
        assert descriptor.distributed_storage_kind is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
        assert descriptor.nbytes==descriptor.mem_span.nbytes==expected_bytes
        assert descriptor.strides==(512,1) and descriptor.component_stride_bytes==0
        assert descriptor.distributed_backing_type is None
        assert descriptor.mem_span.must_alias(source.mem_span)
        for chip in range(4):
            mesh_width=128 if name=="weight" else 1
            abi=distributed_buffer_abi(descriptor,chip*mesh_width)
            for column in range(128):
                assert abi.storage_axis_coordinate(1,column)==abi.logical_axis_coordinate(1,column)==chip*128+column
