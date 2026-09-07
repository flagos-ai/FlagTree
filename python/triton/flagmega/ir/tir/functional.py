# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Statically declared, snake-case TIR construction namespace."""

from triton.flagmega.ir.bufferization import MemSpan, PhysicalBuffer
from triton.flagmega.ir.tir.auxiliary_consumer_contract import TIRAuxiliaryConsumerContract
from triton.flagmega.ir.tir.barrier import Barrier, BarrierScope
from triton.flagmega.ir.tir.binary import Binary
from triton.flagmega.ir.tir.block import Block
from triton.flagmega.ir.tir.buffer import Buffer, DistributedBufferStorageKind
from triton.flagmega.ir.tir.buffer_tuple import BufferTuple
from triton.flagmega.ir.tir.buffer_load import BufferLoad
from triton.flagmega.ir.tir.buffer_region import BufferRegion
from triton.flagmega.ir.tir.buffer_store import BufferStore
from triton.flagmega.ir.tir.evaluate import Evaluate
from triton.flagmega.ir.tir.execution_function import ExecutionFunction
from triton.flagmega.ir.tir.for_loop import For, LoopMode, LoopPartition
from triton.flagmega.ir.tir.if_then_else import IfThenElse
from triton.flagmega.ir.tir.immediate import Immediate
from triton.flagmega.ir.tir.inplace_alias_candidate import InplaceAliasCandidate
from triton.flagmega.ir.tir.let import Let
from triton.flagmega.ir.tir.kernel_dispatch import KernelDispatch
from triton.flagmega.ir.tir.kernel_definition import KernelDefinition
from triton.flagmega.ir.tir.kernel_invoke import KernelInvoke
from triton.flagmega.ir.tir.microkernel_selection import TIRMicroKernelSelection
from triton.flagmega.ir.tir.memory_pool_frame import MemoryPoolFrame
from triton.flagmega.ir.tir.prim_function import PrimFunction, PrimParameter, PrimParameterRole
from triton.flagmega.ir.tir.prim_call_binding import PrimCallBinding
from triton.flagmega.ir.tir.prim_function_call import PrimFunctionCall
from triton.flagmega.ir.tir.pipeline_drain import PipelineDrain
from triton.flagmega.ir.tir.pipeline_handoff import PipelineHandoff
from triton.flagmega.ir.tir.pipeline_stage import PipelineStage
from triton.flagmega.ir.tir.producer_consumer_region import ProducerConsumerRegion
from triton.flagmega.ir.tir.range import Range
from triton.flagmega.ir.tir.return_stmt import Return, ReturnBinding
from triton.flagmega.ir.tir.scalar_var import ScalarVar
from triton.flagmega.ir.tir.sequential import Sequential
from triton.flagmega.ir.tir.shared_workspace_descriptor import TIRSharedWorkspaceDescriptor
from triton.flagmega.ir.tir.synchronization_range import SynchronizationRange
from triton.flagmega.ir.tir.transfer_pipeline_channel import TIRTransferPipelineChannel
from triton.flagmega.ir.tir.transfer_pipeline_contract import TIRTransferPipelineContract
from triton.flagmega.ir.tir.value_ref import ValueRef
from triton.flagmega.ir.tir.workspace_requirement import WorkspaceLifetime, WorkspaceRequirement


class T:
    """Typed aliases retain constructor signatures and IDE completion."""

    physical_buffer = PhysicalBuffer
    kernel_definition = KernelDefinition
    kernel_invoke = KernelInvoke
    barrier = Barrier
    mem_span = MemSpan
    scalar_var = ScalarVar
    immediate = Immediate
    binary = Binary
    range = Range
    buffer = Buffer
    buffer_tuple = BufferTuple
    buffer_region = BufferRegion
    buffer_load = BufferLoad
    buffer_store = BufferStore
    value_ref = ValueRef
    inplace_alias_candidate = InplaceAliasCandidate
    kernel_dispatch = KernelDispatch
    microkernel_selection = TIRMicroKernelSelection
    memory_pool_frame = MemoryPoolFrame
    shared_workspace_descriptor = TIRSharedWorkspaceDescriptor
    synchronization_range = SynchronizationRange
    transfer_pipeline_channel = TIRTransferPipelineChannel
    auxiliary_consumer_contract = TIRAuxiliaryConsumerContract
    transfer_pipeline_contract = TIRTransferPipelineContract
    workspace_requirement = WorkspaceRequirement
    evaluate = Evaluate
    execution_function = ExecutionFunction
    sequential = Sequential
    for_loop = For
    let = Let
    if_then_else = IfThenElse
    block = Block
    return_binding = ReturnBinding
    return_ = Return
    prim_parameter = PrimParameter
    prim_function = PrimFunction
    prim_call_binding = PrimCallBinding
    prim_function_call = PrimFunctionCall
    pipeline_stage = PipelineStage
    pipeline_drain = PipelineDrain
    pipeline_handoff = PipelineHandoff
    producer_consumer_region = ProducerConsumerRegion

    LoopMode = LoopMode
    BarrierScope = BarrierScope
    LoopPartition = LoopPartition
    PrimParameterRole = PrimParameterRole
    DistributedBufferStorageKind = DistributedBufferStorageKind
    WorkspaceLifetime = WorkspaceLifetime


__all__ = ["T"]
