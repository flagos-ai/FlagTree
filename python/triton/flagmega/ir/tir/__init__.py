# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""First-class tensor IR plus physical bufferization compatibility APIs."""

from triton.flagmega.ir.bufferization import (
    AliasInfo,
    AliasKind,
    BufferDescriptor,
    BufferPlan,
    CallMemoryPoolBinding,
    FunctionMemoryPool,
    MemSpan,
    PhysicalAllocation,
    PhysicalBuffer,
)
from triton.flagmega.ir.bufferization.verify import verify_buffer_plan
from triton.flagmega.ir.tir.base import TIRCost, TIRNode, TIRStmt, TIRValue
from triton.flagmega.ir.tir.barrier import Barrier, BarrierScope
from triton.flagmega.ir.tir.auxiliary_consumer_contract import TIRAuxiliaryConsumerContract
from triton.flagmega.ir.tir.binary import Binary
from triton.flagmega.ir.tir.block import Block
from triton.flagmega.ir.tir.buffer import Buffer, DistributedBufferStorageKind
from triton.flagmega.ir.tir.buffer_tuple import BufferTuple
from triton.flagmega.ir.tir.buffer_load import BufferLoad
from triton.flagmega.ir.tir.buffer_region import BufferRegion
from triton.flagmega.ir.tir.buffer_store import BufferStore
from triton.flagmega.ir.tir.evaluate import Evaluate
from triton.flagmega.ir.tir.execution_function import ExecutionFunction
from triton.flagmega.ir.tir.execution import execution_calls_of, execution_items_of, verify_execution_functions
from triton.flagmega.ir.tir.execution_kind import (
    KernelExecutionKind,
    kernel_execution_kind,
    kernel_execution_kind_for_call,
)
from triton.flagmega.ir.tir.for_loop import For, LoopMode, LoopPartition
from triton.flagmega.ir.tir.functional import T
from triton.flagmega.ir.tir.if_then_else import IfThenElse
from triton.flagmega.ir.tir.inplace_alias_candidate import InplaceAliasCandidate
from triton.flagmega.ir.tir.immediate import Immediate
from triton.flagmega.ir.tir.let import Let
from triton.flagmega.ir.tir.kernel_dispatch import (
    KernelDispatch,
    kernel_dispatch_for_call,
    kernel_dispatch_of,
)
from triton.flagmega.ir.tir.microkernel_selection import TIRMicroKernelSelection
from triton.flagmega.ir.tir.memory_pool_frame import MemoryPoolFrame
from triton.flagmega.ir.tir.prim_function import PrimFunction, PrimParameter, PrimParameterRole
from triton.flagmega.ir.tir.prim_call_binding import PrimCallBinding
from triton.flagmega.ir.tir.prim_function_call import PrimFunctionCall
from triton.flagmega.ir.tir.kernel_definition import KernelDefinition
from triton.flagmega.ir.tir.kernel_invoke import KernelInvoke
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
from triton.flagmega.ir.tir.serialization import tir_from_data, tir_to_data
from triton.flagmega.ir.tir.visitor import TIRRewriter, TIRVisitor, estimate_tir_cost, iter_tir_children
from triton.flagmega.ir.tir.verify import verify_prim_function


def make_buffer_plan(module, *, alignment: int = 256, options=None, allocation_session=None) -> BufferPlan:
    from triton.flagmega.passes.tir.bufferize.planner import plan_buffers

    return plan_buffers(module, alignment=alignment, options=options, allocation_session=allocation_session)


__all__ = [
    "KernelDefinition", "KernelInvoke",
    "AliasInfo", "AliasKind", "Barrier", "BarrierScope", "Binary", "Block", "Buffer", "BufferDescriptor", "BufferTuple",
    "BufferLoad", "BufferPlan", "BufferRegion", "BufferStore", "CallMemoryPoolBinding", "Evaluate", "ExecutionFunction", "For", "FunctionMemoryPool",
    "DistributedBufferStorageKind", "IfThenElse", "Immediate", "InplaceAliasCandidate", "KernelDispatch", "KernelExecutionKind", "Let", "LoopMode", "LoopPartition", "MemSpan",
    "MemoryPoolFrame", "PhysicalAllocation", "PhysicalBuffer", "PipelineDrain", "PipelineHandoff",
    "PipelineStage", "PrimCallBinding", "PrimFunction", "PrimFunctionCall", "PrimParameter", "ProducerConsumerRegion",
    "PrimParameterRole", "Range", "Return", "ReturnBinding", "ScalarVar", "Sequential",
    "T", "TIRAuxiliaryConsumerContract", "TIRCost", "TIRMicroKernelSelection", "TIRNode", "TIRRewriter", "TIRSharedWorkspaceDescriptor", "TIRStmt", "TIRTransferPipelineChannel", "TIRTransferPipelineContract", "TIRValue", "TIRVisitor",
    "estimate_tir_cost", "execution_calls_of", "execution_items_of", "iter_tir_children", "make_buffer_plan", "tir_from_data", "tir_to_data",
    "SynchronizationRange", "ValueRef", "WorkspaceLifetime", "WorkspaceRequirement", "kernel_dispatch_for_call", "kernel_dispatch_of", "kernel_execution_kind", "kernel_execution_kind_for_call", "verify_buffer_plan",
    "verify_execution_functions", "verify_prim_function",
]
