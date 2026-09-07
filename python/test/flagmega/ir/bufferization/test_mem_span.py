from triton.flagmega.ir.bufferization import MemSpan, PhysicalBuffer
from triton.flagmega.ir.dim_expr import dim


def test_mem_span_defines_overlap_and_exact_alias_on_one_physical_buffer():
    physical = PhysicalBuffer("workspace:0", "workspace", 128, 16, 64)
    whole = MemSpan(physical)
    prefix = whole.subspan(0, 32)
    overlap = whole.subspan(16, 32)
    suffix = whole.subspan(64, 32)

    assert whole.must_alias(MemSpan(physical, 0, 128))
    assert prefix.may_alias(overlap)
    assert not prefix.must_alias(overlap)
    assert not prefix.may_alias(suffix)
    assert prefix.offset == 64
    assert overlap.offset == 80


def test_equal_arena_ranges_of_distinct_physical_buffers_are_not_aliases():
    first = PhysicalBuffer("workspace:0", "workspace", 64, 16, 0)
    reused = PhysicalBuffer("workspace:1", "workspace", 64, 16, 0)

    assert not MemSpan(first).may_alias(MemSpan(reused))
    assert not MemSpan(first).must_alias(MemSpan(reused))


def test_symbolic_subspans_use_dim_expr_proofs_and_conservative_unknowns():
    n = dim("n", minimum=0, maximum=1024)
    physical = PhysicalBuffer("external:0", "external", n + 64, 16)
    left = MemSpan(physical, n, 16)
    adjacent = MemSpan(physical, n + 16, 8)
    m = dim("m", minimum=0, maximum=1024)
    unknown = MemSpan(physical, m, 8)

    assert left.must_alias(MemSpan(physical, n, 16))
    assert not left.may_alias(adjacent)
    assert unknown.may_alias(left)


def test_mem_span_serialization_reuses_canonical_physical_buffer_object():
    physical = PhysicalBuffer("rdata:0", "rdata", 256, 64, 512)
    restored = MemSpan.from_data(
        MemSpan(physical, 32, 48).to_data(),
        {physical.id: physical},
    )

    assert restored.buffer is physical
    assert restored.start.fixed_value == 32
    assert restored.size.fixed_value == 48
    assert restored.offset == 544
