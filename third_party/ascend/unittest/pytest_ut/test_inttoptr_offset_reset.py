import re

import pytest
import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import ASTSource
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, ttir_to_linalg


@triton.jit
def _inttoptr_static_offset_kernel(
    block_table_ptrs,
    out,
    BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)

    # Load a raw i64 address and cast to pointer — creates tt.int_to_ptr.
    block_table_ptr = tl.load(block_table_ptrs).to(tl.pointer_type(tl.int32))

    # Scalar addptr with a constant offset — creates a ReinterpretCastOp
    # whose result type has a non-zero static offset in the strided layout.
    block_table_ptr = block_table_ptr + 1

    block_numbers = tl.load(block_table_ptr + offsets)
    slot_ids = block_numbers.to(tl.int64)
    tl.store(out + offsets, slot_ids)


@triton.jit
def _inttoptr_dyn_offset_kernel(
    block_table_ptrs,
    dyn_offset,
    out,
    BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)

    block_table_ptr = tl.load(block_table_ptrs).to(tl.pointer_type(tl.int32))

    # Scalar addptr with a runtime offset — creates a ReinterpretCastOp
    # whose result type has a dynamic offset (?) in the strided layout.
    block_table_ptr = block_table_ptr + tl.load(dyn_offset)

    block_numbers = tl.load(block_table_ptr + offsets)
    slot_ids = block_numbers.to(tl.int64)
    tl.store(out + offsets, slot_ids)


@triton.jit
def _inttoptr_indirect_load_kernel(
    block_table_ptrs,
    offsets_ptr,
    out,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    offsets = tl.load(offsets_ptr + offsets)
    raw_ptr = tl.load(block_table_ptrs).to(tl.pointer_type(tl.int32))
    raw_ptrs = tl.broadcast_to(raw_ptr, (BLOCK, ))
    values = tl.load(raw_ptrs + offsets)
    tl.store(out + offsets, values)


@triton.jit
def _inttoptr_branch_local_kernel(base_ptrs, flag_ptr, out):
    """Use an int_to_ptr only inside each branch, without a loop backedge."""
    raw_addr = tl.load(base_ptrs)
    flag = tl.load(flag_ptr)
    if flag != 0:
        ptr = raw_addr.to(tl.pointer_type(tl.int32))
        value = tl.load(ptr)
    else:
        ptr = raw_addr.to(tl.pointer_type(tl.int32)) + 4
        value = tl.load(ptr)
    tl.store(out, value)


@triton.jit
def _inttoptr_for_carrier_kernel(base_ptrs, steps_ptr, out):
    """Carry a complete scalar pointer through a for backedge."""
    raw_addr = tl.load(base_ptrs)
    steps = tl.load(steps_ptr)
    ptr = raw_addr.to(tl.pointer_type(tl.int32))
    for _ in tl.range(0, steps):
        ptr = ptr + 1
    tl.store(out, tl.load(ptr))


@triton.jit
def _inttoptr_while_carrier_kernel(base_ptrs, steps_ptr, out):
    """Carry the complete i64 address through a while backedge."""
    raw_addr = tl.load(base_ptrs)
    steps = tl.load(steps_ptr)
    address = raw_addr
    iteration = 0
    while iteration < steps:
        # The loaded address is byte-addressed; one int32 element is four bytes.
        address = address + 4
        iteration += 1
    ptr = address.to(tl.pointer_type(tl.int32))
    tl.store(out, tl.load(ptr))


@triton.jit
def _inttoptr_nested_if_kernel(base_ptrs, steps_ptr, flag_ptr, out):
    """Update a complete pointer carrier on both sides of a nested if."""
    raw_addr = tl.load(base_ptrs)
    steps = tl.load(steps_ptr)
    flag = tl.load(flag_ptr)
    ptr = raw_addr.to(tl.pointer_type(tl.int32))
    for iteration in tl.range(0, steps):
        if flag != 0:
            ptr = ptr + 1
        else:
            ptr = ptr + 2
    tl.store(out, tl.load(ptr))


def _compile_to_adapter(kernel, signature, constants):
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    options = NPUOptions()
    ttir = ast_to_ttir(kernel, src, context, options, {}, {})
    return str(ttir_to_linalg(ttir, {**options.__dict__}, options, named_ops=True))


def test_inttoptr_static_offset_reset():
    """Static offset (constant 1) baked into base address."""
    block = 8

    block_table_cpu = torch.arange(16, dtype=torch.int32)
    block_table = block_table_cpu.npu()
    out = torch.empty((block, ), dtype=torch.int64, device="npu")

    block_table_ptrs = torch.tensor([block_table.data_ptr()], dtype=torch.int64).npu()

    _inttoptr_static_offset_kernel[(1, )](
        block_table_ptrs,
        out,
        BLOCK_SIZE=4,
        BLOCK=block,
    )
    torch.npu.synchronize()

    # offset 1 => skip block_table[0]
    expected = block_table_cpu[1:1 + block].to(torch.int64)
    torch.testing.assert_close(out.cpu(), expected.cpu())


def test_inttoptr_indirect_load_uses_scalar_view():
    """An opaque int_to_ptr tensor load keeps the one-element carrier view."""
    adapter = _compile_to_adapter(
        _inttoptr_indirect_load_kernel,
        {"block_table_ptrs": "*i64", "offsets_ptr": "*i32", "out": "*i32"},
        {"BLOCK": 8},
    )
    assert "hivm.hir.pointer_cast" in adapter
    assert "annotation.mark" in adapter
    assert "memref.reinterpret_cast" in adapter
    assert re.search(r"memref<1xi32, strided<\[1\]>>", adapter)


def test_inttoptr_indirect_load_scalar_view_npu():
    """A runtime int_to_ptr base keeps its one-element view in the helper ABI."""
    source_cpu = torch.arange(16, dtype=torch.int32)
    source = source_cpu.npu()
    base_ptrs = torch.tensor([source.data_ptr()], dtype=torch.int64).npu()
    offsets = torch.arange(8, dtype=torch.int32).npu()
    out = torch.empty(8, dtype=torch.int32, device="npu")

    _inttoptr_indirect_load_kernel[(1, )](base_ptrs, offsets, out, BLOCK=8)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), source_cpu[:8])


@pytest.mark.parametrize("flag", [0, 1])
def test_inttoptr_branch_local(flag):
    source_cpu = torch.arange(16, dtype=torch.int32)
    source = source_cpu.npu()
    base_ptrs = torch.tensor([source.data_ptr()], dtype=torch.int64).npu()
    flag_ptr = torch.tensor([flag], dtype=torch.int32).npu()
    out = torch.empty(1, dtype=torch.int32, device="npu")

    _inttoptr_branch_local_kernel[(1, )](base_ptrs, flag_ptr, out)
    torch.npu.synchronize()
    expected = source_cpu[0 if flag else 4:1 if flag else 5]
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.parametrize("steps", [0, 1, 4])
def test_inttoptr_for_carrier(steps):
    source_cpu = torch.arange(32, dtype=torch.int32)
    source = source_cpu.npu()
    base_ptrs = torch.tensor([source.data_ptr()], dtype=torch.int64).npu()
    steps_ptr = torch.tensor([steps], dtype=torch.int32).npu()
    out = torch.empty(1, dtype=torch.int32, device="npu")

    _inttoptr_for_carrier_kernel[(1, )](base_ptrs, steps_ptr, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), source_cpu[steps:steps + 1])


@pytest.mark.parametrize("steps", [0, 1, 4])
def test_inttoptr_while_carrier(steps):
    source_cpu = torch.arange(32, dtype=torch.int32)
    source = source_cpu.npu()
    base_ptrs = torch.tensor([source.data_ptr()], dtype=torch.int64).npu()
    steps_ptr = torch.tensor([steps], dtype=torch.int32).npu()
    out = torch.empty(1, dtype=torch.int32, device="npu")

    _inttoptr_while_carrier_kernel[(1, )](base_ptrs, steps_ptr, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), source_cpu[steps:steps + 1])


@pytest.mark.parametrize("flag", [0, 1])
@pytest.mark.parametrize("steps", [0, 1, 4])
def test_inttoptr_nested_if(flag, steps):
    source_cpu = torch.arange(32, dtype=torch.int32)
    source = source_cpu.npu()
    base_ptrs = torch.tensor([source.data_ptr()], dtype=torch.int64).npu()
    steps_ptr = torch.tensor([steps], dtype=torch.int32).npu()
    flag_ptr = torch.tensor([flag], dtype=torch.int32).npu()
    out = torch.empty(1, dtype=torch.int32, device="npu")

    _inttoptr_nested_if_kernel[(1, )](base_ptrs, steps_ptr, flag_ptr, out)
    torch.npu.synchronize()
    offset = steps if flag else 2 * steps
    torch.testing.assert_close(out.cpu(), source_cpu[offset:offset + 1])


def test_inttoptr_dyn_offset_reset():
    """Dynamic offset (runtime value) baked into base address."""
    block = 8

    block_table_cpu = torch.arange(16, dtype=torch.int32)
    block_table = block_table_cpu.npu()
    out = torch.empty((block, ), dtype=torch.int64, device="npu")

    block_table_ptrs = torch.tensor([block_table.data_ptr()], dtype=torch.int64).npu()
    dyn_offset = torch.tensor([2], dtype=torch.int32).npu()

    _inttoptr_dyn_offset_kernel[(1, )](
        block_table_ptrs,
        dyn_offset,
        out,
        BLOCK_SIZE=4,
        BLOCK=block,
    )
    torch.npu.synchronize()

    # dynamic offset 2 => skip block_table[0:2]
    expected = block_table_cpu[2:2 + block].to(torch.int64)
    torch.testing.assert_close(out.cpu(), expected.cpu())
