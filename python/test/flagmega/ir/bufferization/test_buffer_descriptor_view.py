from triton.flagmega.ir.bufferization import (
    AliasKind,
    BufferDescriptor,
    MemSpan,
    PhysicalBuffer,
)
from triton.flagmega.ir.types import data_type


def test_buffer_subview_shares_storage_and_records_typed_provenance():
    physical = PhysicalBuffer("workspace:0", "workspace", 256, 16, 128)
    source = BufferDescriptor(
        id="source",
        dtype=data_type("bool"),
        shape=(256,),
        strides=(1,),
        storage="workspace",
        alignment=16,
        mem_span=MemSpan(physical),
        function="main",
    )

    view = source.subview(
        "middle",
        dtype=data_type("bool"),
        shape=(32,),
        strides=(1,),
        byte_offset=48,
        byte_size=32,
        alignment=16,
    )

    assert view.mem_span.buffer is physical
    assert view.mem_span.start.fixed_value == 48
    assert view.offset == 176
    assert view.alias.source == "source"
    assert view.alias.kind is AliasKind.VIEW
    assert view.mem_span.is_within(source.mem_span)


def test_buffer_descriptor_serializes_only_mem_span_not_legacy_alias_fields():
    physical = PhysicalBuffer("rdata:0", "rdata", 64, 16)
    descriptor = BufferDescriptor(
        "weight",
        data_type("bool"),
        (64,),
        (1,),
        "rdata",
        16,
        MemSpan(physical),
    )

    data = descriptor.to_data()

    assert data["mem_span"]["buffer"] == "rdata:0"
    assert "physical_id" not in data
    assert "byte_offset" not in data
    assert "alias_of" not in data
