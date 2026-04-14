/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GCU_OPS
#define GCU_OPS

include "Dialect/GCU/IR/GCUDialect.td"
include "Dialect/GCU/IR/GCUAttrDefs.td"
include "Dialect/GCU/IR/GCUInterfaces.td"
include "Dialect/GCU/IR/GCUTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td" // SymbolUserOpInterface
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface
include "mlir/Interfaces/CastInterfaces.td" // CastOpInterface
include "mlir/Interfaces/FunctionInterfaces.td" // FunctionOpInterface
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // BranchOpInterface
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType

//
// Op Base
//
class GCU_Op<string mnemonic, list<Trait> traits = []> :
    Op<GCU_Dialect, mnemonic,
       !listconcat(traits, [])> {
}




//
// AllocESLEngine op
//
def GCU_AllocESLEngineOp : GCU_Op<"alloc_esl_engine", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a ESL engine";
  let description = [{
    `gcu.alloc_esl_engine` allocate a ESL engine.
  }];
  let arguments = (ins
      I64:$ibgda_info_ptr, I64:$qp_shared_states_ptr,
      I32:$rank, I32:$num_ranks
  );
  let results = (outs GCU_ESLEngineType:$engine);
  let assemblyFormat = "$ibgda_info_ptr $qp_shared_states_ptr $rank $num_ranks attr-dict `:` type($engine)";
}

//
// AllocESLEndpoint op
//
def GCU_AllocESLEndpointOp : GCU_Op<"alloc_esl_endpoint", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a ESL endpoint";
  let description = [{
    `gcu.alloc_esl_endpoint` allocate a ESL endpoint.
  }];
  let results = (outs GCU_ESLEndpointType:$endpoint);
  let assemblyFormat = "attr-dict `:` type($endpoint)";
}

//
// ESLGetQPOp op
//
def GCU_ESLGetQPOp : GCU_Op<"esl_get_qp", [MemoryEffectsOpInterface]> {
  let summary = "Get queue pair from esl engine";
  let description = [{
    `gcu.esl_get_qp` get queue pair from esl engine.
  }];
  let arguments = (ins
      GCU_ESLEngineType:$engine,
      I32:$dst_rank,
      I32:$qp_idx,
      GCU_ESLEndpointType:$endpoint
  );

  let assemblyFormat = "$engine `,` $dst_rank `,` $qp_idx `,` $endpoint attr-dict `:` type($endpoint)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(3),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// ESLGetNumRCPerPEOp op
//
def GCU_ESLGetNumRCPerPEOp : GCU_Op<"esl_get_num_rc_per_pe", [MemoryEffectsOpInterface]> {
  let summary = "Get number of RC connections per PE";
  let description = [{
    `gcu.esl_get_num_rc_per_pe` number of RC connections per PE.
  }];
  let arguments = (ins GCU_ESLEngineType:$engine);
  let results = (outs I32:$num_rc_per_pe);
  let assemblyFormat = "$engine attr-dict `:` type($num_rc_per_pe)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// ESLGetWqeSlotsOp op
//
def GCU_ESLGetWqeSlotsOp : GCU_Op<"esl_get_wqe_slots", [MemoryEffectsOpInterface]> {
  let summary = "Atomically allocate WQE slots for a QP";
  let description = [{
    `gcu.esl_get_wqe_slots` Atomically allocate WQE slots for a QP.
    Ensures thread-safe allocation of WQE indices when multiple threads compete for the same QP.
  }];
  let arguments = (ins
      GCU_ESLEngineType:$engine,
      GCU_ESLEndpointType:$endpoint,
      I32:$num_wqes
  );
  let results = (outs I64:$wqe_index);
  let assemblyFormat = "$engine `,` $endpoint `,` $num_wqes attr-dict `:` type($wqe_index)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// ESLRingDbOp op
//
def GCU_ESLRingDbOp : GCU_Op<"esl_ring_db", [MemoryEffectsOpInterface]> {
  let summary = "Trigger doorbell for specified QP";
  let description = [{
    `gcu.esl_ring_db` Trigger doorbell for specified QP.
    Ensures ordered doorbell triggering when multiple threads share the same QP.
  }];
  let arguments = (ins
      GCU_ESLEngineType:$engine,
      GCU_ESLEndpointType:$endpoint,
      I64:$wqe_idx,
      I32:$num_wqes
  );
  let results = (outs I64:$wait_id);
  let assemblyFormat = "$engine `,` $endpoint `,` $wqe_idx `,` $num_wqes attr-dict `:` type($wait_id)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// ESLWaitDoneOp op
//
def GCU_ESLWaitDoneOp : GCU_Op<"esl_wait_done", [MemoryEffectsOpInterface]> {
  let summary = " Wait for transmission to complete";
  let description = [{
    `gcu.esl_wait_done`  Wait for transmission to complete.
  }];
  let arguments = (ins
      GCU_ESLEngineType:$engine,
      GCU_ESLEndpointType:$endpoint,
      I64:$wait_id
  );
  let assemblyFormat = "$engine `,` $endpoint `,` $wait_id attr-dict `:` type($engine)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
    }
  }];
}

def GCU_ESLSendOp : GCU_Op<"esl_send", [MemoryEffectsOpInterface]> {
  let summary = "Send data using specified QP and WQE";
  let description = [{
    `gcu.esl_send` send data using specified QP and WQE'.
  }];
  let arguments = (ins
      GCU_ESLEngineType:$engine,
      GCU_PtrType:$dst,
      GCU_PtrType:$src,
      GCU_ESLEndpointType:$endpoint,
      I64:$wqe_idx,
      I64:$nelem,
      I1:$fence,
      GCU_ESLAtomicOpTypeAttr:$atomic_op_t
  );
  let assemblyFormat = "$engine `,` $dst `,` $src `,` $endpoint `,` $wqe_idx `,` $nelem `,` $fence attr-dict `:` type($dst) `,` type($src)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(3),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(3),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// ESLGetRdmaPeerBaseOp op
//
def GCU_ESLGetRdmaPeerBaseOp : GCU_Op<"esl_get_rdma_peer_base", [MemoryEffectsOpInterface]> {
  let summary = "Get rdma peer base from 'prims'";
  let description = [{
    `gcu.esl_get_rdma_peer_base` get rdma peer base from 'prims'.
  }];
  let arguments = (ins
      GCU_PtrType:$prims,
      I32:$rank
  );
  let results = (outs I64:$rdma_peer_base);
  let assemblyFormat = "$prims `,` $rank attr-dict `:` type($prims) `,` type($rdma_peer_base)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// ESLGetIbgdaInfoPtrOp op
//
def GCU_ESLGetIbgdaInfoPtrOp : GCU_Op<"esl_get_ibgda_info_ptr", [MemoryEffectsOpInterface]> {
  let summary = "get ibgda information pointer in 'prims'";
  let description = [{
    `gcu.esl_get_ibgda_info_ptr` get ibgda information pointer in 'prims'.
  }];
  let arguments = (ins GCU_PtrType:$prims);
  let results = (outs I64:$ibgda_info_ptr);
  let assemblyFormat = "$prims attr-dict `:` type($prims) `,` type($ibgda_info_ptr)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// ESLGetQpSharedStatesPtrOp op
//
def GCU_ESLGetQpSharedStatesPtrOp : GCU_Op<"esl_get_qp_shared_states_ptr", [MemoryEffectsOpInterface]> {
  let summary = "Get queue pair shared states pointer in 'prims'";
  let description = [{
    `gcu.esl_get_qp_shared_states_ptr` get queue pair shared states pointer in 'prims'.
  }];
  let arguments = (ins GCU_PtrType:$prims);
  let results = (outs I64:$qp_shared_states_ptr);
  let assemblyFormat = "$prims attr-dict `:` type($prims) `,` type($qp_shared_states_ptr)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// AllocDTE Op
//
def GCU_AllocDTEOp : GCU_Op<"alloc_dte", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a DTE context";
  let description = [{
    `gcu.alloc_dte` allocate a DTE context. If addrespace != private, the dte context needs
    to be called by `init` explicitly.
  }];
  let results = (outs GCU_DTEType:$dte);
  let assemblyFormat = "attr-dict `:` type($dte)";
}

//
// InitDTE Op
//
def GCU_InitDTEOp : GCU_Op<"init_dte", [MemoryEffectsOpInterface]> {
  let summary = "Initialize a DTE context";
  let description = [{
    `gcu.init_dte` initialize a DTE context and only for non-private dte contexts.
  }];
  let arguments = (ins GCU_DTEType:$dte);
  let assemblyFormat = "$dte attr-dict `:` type($dte)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// ConnectDTE Op
//
def GCU_ConnectDTEOp : GCU_Op<"connect_dte", [MemoryEffectsOpInterface]> {
  let summary = "Connect two DTE contexts";
  let description = [{
    `gcu.connect_dte` connect two DTE contexts.
  }];
  let arguments = (ins GCU_DTEType:$from, GCU_DTEType:$to);
  let assemblyFormat = "$from `,` $to attr-dict `:` type($from) `to` type($to)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// DestroyDTE Op
//
def GCU_DestroyDTEOp : GCU_Op<"destroy_dte", [MemoryEffectsOpInterface]> {
  let summary = "Destroy a DTE context";
  let description = [{
    `gcu.destroy_dte` destroy a DTE context and only for non-private dte contexts.
  }];
  let arguments = (ins GCU_DTEType:$dte);
  let assemblyFormat = "$dte attr-dict `:` type($dte)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// DeallocDTE Op
//
def GCU_DeallocDTEOp : GCU_Op<"dealloc_dte", [MemoryEffects<[MemFree]>]> {
  let summary = "Deallocate a DTE context";
  let description = [{
    `gcu.dealloc_dte` deallocate a DTE context. If addrespace != private, the dte context needs
    to be called by `destroy` explicitly.
  }];
  let arguments = (ins GCU_DTEType:$dte);
  let assemblyFormat = "$dte attr-dict `:` type($dte)";
}

//
// TriggerDTE Op
//
def GCU_TriggerDTEOp : GCU_Op<"trigger_dte", [MemoryEffectsOpInterface]> {
  let summary = "trigger a dte operation";
  let description = [{
    `gcu.trigger_dte` trigger a dte operation.
  }];
  let arguments = (ins GCU_DTEType:$dte);
  let assemblyFormat = "$dte attr-dict `:` type($dte)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// WaitDTE Op
//
def GCU_WaitDTEOp : GCU_Op<"wait_dte", [MemoryEffectsOpInterface]> {
  let summary = "wait a dte operation";
  let description = [{
    `gcu.wait_dte` wait a dte operation.
  }];
  let arguments = (ins GCU_DTEType:$dte);
  let assemblyFormat = "$dte attr-dict `:` type($dte)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// SetDstAddrOp Op
//
def GCU_SetDstAddrOp : GCU_Op<"set_dst_addr", [MemoryEffectsOpInterface]> {
  let summary = "Set destination address for the operation bound to the DTE context.";
  let description = [{
    `gcu.set_dst_addr` set destination address for the operation bound to the DTE context..
  }];
  let arguments = (ins GCU_DTEType:$dte, GCU_PtrType:$addr);
  let assemblyFormat = "$dte `,` $addr attr-dict `:` type($dte) `,` type($addr)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// SetSrcOffsetOp Op
//
def GCU_SetSrcOffsetOp : GCU_Op<"set_src_offset", [MemoryEffectsOpInterface]> {
  let summary = "Set source address offset for the operation bound to the DTE";
  let description = [{
    `gcu.set_src_offset` set source address offset for the operation bound to the DTE.
    `dim` is an int number representing the dimension.
    `offset ` is an int number representing the offset.
  }];
  let arguments = (ins GCU_DTEType:$dte, I32:$dim, I32:$offset);
  let assemblyFormat = "$dte `,` $dim `,` $offset attr-dict `:` type($dte)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// SetDstOffsetOp Op
//
def GCU_SetDstOffsetOp : GCU_Op<"set_dst_offset", [MemoryEffectsOpInterface]> {
  let summary = " Set destination address offset for the operation bound to the DTE.";
  let description = [{
    `gcu.set_dst_offset` set destination address offset for the operation bound to the DTE.
    `dim` is an int number representing the dimension.
    `offset `is an int number representing the offset.
  }];
  let arguments = (ins GCU_DTEType:$dte, I32:$dim, I32:$offset);
  let assemblyFormat = "$dte `,` $dim `,` $offset attr-dict `:` type($dte)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Memset Op
//
def GCU_MemsetAsyncOp : GCU_Op<"memset_async", [MemoryEffectsOpInterface]> {
  let summary = "Memset";
  let description = [{
    `gcu.memset_async` set a value to a buffer.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, GCU_FpInt:$value);
  let assemblyFormat = "$dte `,` $dst `,` $value attr-dict `:` type($dte) `,` type($dst) `,` type($value)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Memcpy Op
//
def GCU_MemcpyAsyncOp : GCU_Op<"memcpy_async", [MemoryEffectsOpInterface]> {
  let summary = "Memcpy";
  let description = [{
    `gcu.memcpy_async` copy a data from source to dst.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src);
  let assemblyFormat = "$dte `,` $dst `,` $src attr-dict `:` type($dte) `,` type($dst) `,` type($src)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Slice Op
//
def GCU_SliceAsyncOp : GCU_Op<"slice_async", [MemoryEffectsOpInterface]> {
  let summary = "Slice";
  let description = [{
    `gcu.slice_async` slice a data from source to dst.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src, Variadic<I32>:$offsets, AnyType:$default_value);
  let assemblyFormat = "$dte `,` $dst `,` $src `[` $offsets `]` `,` $default_value attr-dict `:` type($dte) `,` type($dst) `,` type($src) `,` type($default_value)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// SlicePad Op
//
def GCU_SlicePadAsyncOp : GCU_Op<"slice_pad_async", [AttrSizedOperandSegments, MemoryEffectsOpInterface]> {
  let summary = "Slice and pad";
  let description = [{
    `gcu.slice_pad_async` slice a data from source to dst with pad value.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src, Variadic<I32>:$offsets, Variadic<I32>:$slice_shape, AnyType:$pad_value);
  let assemblyFormat = "$dte `,` $dst `,` $src `[` $offsets `]` `[` $slice_shape `]` `,` $pad_value attr-dict `:` type($dte) `,` type($dst) `,` type($src) `,` type($pad_value)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Deslice Op
//
def GCU_DesliceAsyncOp : GCU_Op<"deslice_async", [MemoryEffectsOpInterface]> {
  let summary = "Deslice";
  let description = [{
    `gcu.deslice_async` deslice a data from source to dst.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src, Variadic<I32>:$offsets);
  let assemblyFormat = "$dte `,` $dst `[` $offsets `]` `,` $src attr-dict `:` type($dte) `,` type($dst) `,` type($src)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Slice_Deslice Op
//
def GCU_SliceDesliceAsyncOp : GCU_Op<"slice_deslice_async", [AttrSizedOperandSegments, MemoryEffectsOpInterface]> {
  let summary = "Slice_Deslice";
  let description = [{
    `gcu.slice_deslice_async` slice_deslice a data from source to dst.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src, Variadic<I32>:$offsets, Variadic<I32>:$slice_shape, Variadic<I32>:$dst_offsets);
  let assemblyFormat = "$dte `,` $dst `[` $offsets `]` `[` $slice_shape `]` `[` $dst_offsets `]` `,` $src attr-dict `:` type($dte) `,` type($dst) `,` type($src)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Transpose Op
//
def GCU_TransposeAsyncOp : GCU_Op<"transpose_async", [MemoryEffectsOpInterface]> {
  let summary = "Transpose";
  let description = [{
    `gcu.transpose_async` transpose a data from source to dst.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src, Variadic<I32>:$layout);
  let assemblyFormat = "$dte `,` $dst `,` $src `[` $layout `]` attr-dict `:` type($dte) `,` type($dst) `,` type($src)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Broadcast Op
//
def GCU_BroadcastAsyncOp : GCU_Op<"broadcast_async", [MemoryEffectsOpInterface]> {
  let summary = "Broadcast";
  let description = [{
    `gcu.broadcast_async` broadcast a data from source to dst.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src);
  let assemblyFormat = "$dte `,` $dst `,` $src attr-dict `:` type($dte) `,` type($dst) `,` type($src)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Slice_Broadcast Op
//
def GCU_SliceBroadcastAsyncOp : GCU_Op<"slice_broadcast_async", [AttrSizedOperandSegments, MemoryEffectsOpInterface]> {
  let summary = "Slice and broadcast";
  let description = [{
    `gcu.slice_broadcast_async` slice_broadcast a data from source to dst.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src, Variadic<I32>:$offsets, Variadic<I32>:$slice_shape);
  let assemblyFormat = "$dte `,` $dst `,` $src `[` $offsets `]` `[` $slice_shape `]` attr-dict `:` type($dte) `,` type($dst) `,` type($src)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Slice_Transpose Op
//
def GCU_SliceTransposeAsyncOp : GCU_Op<"slice_transpose_async", [AttrSizedOperandSegments, MemoryEffectsOpInterface]> {
  let summary = "Slice and transpose";
  let description = [{
    `gcu.slice_transpose_async` slice_transpose a data from source to dst.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src, Variadic<I32>:$offsets, Variadic<I32>:$layout, GCU_FpInt:$value);
  let assemblyFormat = "$dte `,` $dst `,` $src `[` $offsets `]` `[` $layout `]` `,` $value attr-dict `:` type($dte) `,` type($dst) `,` type($src) `,` type($value)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Transpose_Deslice Op
//
def GCU_TransposeDesliceAsyncOp : GCU_Op<"transpose_deslice_async", [AttrSizedOperandSegments, MemoryEffectsOpInterface]> {
  let summary = "Transpose and delice";
  let description = [{
    `gcu.transpose_deslice_async` transpose_deslice a data from source to dst.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src, Variadic<I32>:$layout, Variadic<I32>:$offsets);
  let assemblyFormat = "$dte `,` $dst`[` $offsets `]`  `,` $src  `[` $layout `]` attr-dict `:` type($dte) `,` type($dst) `,` type($src)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Memset_Deslice Op
//
def GCU_MemsetDesliceAsyncOp : GCU_Op<"memset_deslice_async", [MemoryEffectsOpInterface]> {
  let summary = "Memset and deslice";
  let description = [{
    `gcu.memset_deslice_async` memset_deslice a data from source to dst.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src, Variadic<I32>:$offsets, GCU_FpInt:$value);
  let assemblyFormat = "$dte `,` $dst `[` $offsets `]` `,` $src `,` $value attr-dict `:` type($dte) `,` type($dst) `,` type($src) `,` type($value)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// mirror_tb Op
//
def GCU_MirrortbAsyncOp : GCU_Op<"mirror_tb_async", [MemoryEffectsOpInterface]> {
  let summary = "mirror_tb";
  let description = [{
    `gcu.mirror_tb_async` top-blow flip.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src);
  let assemblyFormat = "$dte `,` $dst `,` $src attr-dict `:` type($dte) `,` type($dst) `,` type($src)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// mirror_lr Op
//
def GCU_MirrorlrAsyncOp : GCU_Op<"mirror_lr_async", [MemoryEffectsOpInterface]> {
  let summary = "mirror_lr";
  let description = [{
    `gcu.mirror_lr_async` left-right flip.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src);
  let assemblyFormat = "$dte `,` $dst `,` $src attr-dict `:` type($dte) `,` type($dst) `,` type($src)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// mirror_tb_pad Op
//
def GCU_MirrortbPadAsyncOp : GCU_Op<"mirror_tb_pad_async", [AttrSizedOperandSegments, MemoryEffectsOpInterface]> {
  let summary = "mirror_tb_pad";
  let description = [{
    `gcu.mirror_tb_pad_async` top-blow flip src store to dst then pad on high low mid.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src, Variadic<I32>:$pad_low, Variadic<I32>:$pad_high, Variadic<I32>:$pad_mid, AnyType:$pad_value);
  let assemblyFormat = "$dte `,` $dst `,` $src `,` `[` $pad_low `]` `[` $pad_high `]` `[` $pad_mid `]` `,`$pad_value attr-dict `:` type($dte) `,` type($dst) `,` type($src) `,` type($pad_value)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// mirror_lr_pad Op
//
def GCU_MirrorlrPadAsyncOp : GCU_Op<"mirror_lr_pad_async", [AttrSizedOperandSegments, MemoryEffectsOpInterface]> {
  let summary = "mirror_lr_pad";
  let description = [{
    `gcu.mirror_lr_pad_async` left-right flip src store to dst then pad on high low mid.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src, Variadic<I32>:$pad_low, Variadic<I32>:$pad_high, Variadic<I32>:$pad_mid, AnyType:$pad_value);
  let assemblyFormat = "$dte `,` $dst `,` $src `,` `[` $pad_low `]` `[` $pad_high `]` `[` $pad_mid `]` `,`$pad_value attr-dict `:` type($dte) `,` type($dst) `,` type($src) `,` type($pad_value)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// mirror_tb_deslice Op
//
def GCU_MirrortbDesliceAsyncOp : GCU_Op<"mirror_tb_deslice_async", [MemoryEffectsOpInterface]> {
  let summary = "mirror_tb_deslice";
  let description = [{
    `gcu.mirror_tb_deslice_async` top-blow flip then deslice to dst.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src, Variadic<I32>:$dst_offsets);
  let assemblyFormat = "$dte `,` $dst `[` $dst_offsets `]` `,` $src attr-dict `:` type($dte) `,` type($dst) `,` type($src)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// mirror_lr_deslice Op
//
def GCU_MirrorlrDesliceAsyncOp : GCU_Op<"mirror_lr_deslice_async", [MemoryEffectsOpInterface]> {
  let summary = "mirror_lr_deslice";
  let description = [{
    `gcu.mirror_lr_deslice_async` left-right flip then deslice to dst.
  }];
  let arguments = (ins GCU_DTEType:$dte, AnyMemRef:$dst, AnyMemRef:$src, Variadic<I32>:$dst_offsets);
  let assemblyFormat = "$dte `,` $dst `[` $dst_offsets `]` `,` $src attr-dict `:` type($dte) `,` type($dst) `,` type($src)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// DynamicSharedMemory Op
//
def GCU_DynamicSharedMemoryOp : GCU_Op<"dynamic_shared_memory", [Pure]>
{
  let summary = "Get the memref for dynamic shared memory";

  let description = [{
  }];
  let arguments = (ins);
  let results = (outs Arg<MemRefRankOf<[I8], [1]>>:$resultMemref);
  let assemblyFormat = [{ attr-dict `:` type($resultMemref) }];
  let hasVerifier = 1;
}

//
// AllocBarrier Op
//
def GCU_AllocBarrierOp : GCU_Op<"alloc_barrier", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a barrier";
  let description = [{
    `gcu.alloc_barrier` allocate a barrier and the barrier needs to be called by `init` explicitly.
  }];
  let results = (outs GCU_BarrierType:$barrier);
  let assemblyFormat = "attr-dict `:` type($barrier)";
  let hasVerifier = 1;
}

//
// InitBarrier Op
//
def GCU_InitBarrierOp : GCU_Op<"init_barrier", [MemoryEffectsOpInterface]> {
  let summary = "Initialize a barrier";
  let description = [{
    `gcu.init_barrier` initialize a barrier.
  }];
  let arguments = (ins GCU_BarrierType:$barrier, I32:$count);
  let assemblyFormat = "$barrier `,` $count attr-dict `:` type($barrier) `,` type($count)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// ArriveAndWaitBarrier Op
//
def GCU_ArriveAndWaitBarrierOp : GCU_Op<"arrive_and_wait_barrier", [MemoryEffectsOpInterface]> {
  let summary = "Arrive and wait a barrier";
  let description = [{
    `gcu.arrive_and_wait_barrier` arrive and wait a barrier.
  }];
  let arguments = (ins GCU_BarrierType:$barrier);
  let assemblyFormat = "$barrier attr-dict `:` type($barrier)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// ArriveBarrier Op
//
def GCU_ArriveBarrierOp : GCU_Op<"arrive_barrier", [MemoryEffectsOpInterface]> {
  let summary = "Arrive a barrier";
  let description = [{
    `gcu.arrive_barrier` arrive a barrier.
  }];
  let arguments = (ins GCU_BarrierType:$barrier);
  let assemblyFormat = "$barrier attr-dict `:` type($barrier)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// WaitBarrier Op
//
def GCU_WaitBarrierOp : GCU_Op<"wait_barrier", [MemoryEffectsOpInterface]> {
  let summary = "Wait a barrier";
  let description = [{
    `gcu.wait_barrier` wait a barrier.
  }];
  let arguments = (ins GCU_BarrierType:$barrier);
  let assemblyFormat = "$barrier attr-dict `:` type($barrier)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// DestroyBarrier Op
//
def GCU_DestroyBarrierOp : GCU_Op<"destroy_barrier", [MemoryEffectsOpInterface]> {
  let summary = "Destroy a barrier";
  let description = [{
    `gcu.destroy_barrier` destroy a barrier.
  }];
  let arguments = (ins GCU_BarrierType:$barrier);
  let assemblyFormat = "$barrier attr-dict `:` type($barrier)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// DeallocBarrier Op
//
def GCU_DeallocBarrierOp : GCU_Op<"dealloc_barrier", [MemoryEffects<[MemFree]>]> {
  let summary = "Deallocate a barrier";
  let description = [{
    `gcu.dealloc_barrier` deallocate a barrier and the barrier needs to be called by `destroy` explicitly.
  }];
  let arguments = (ins GCU_BarrierType:$barrier);
  let assemblyFormat = "$barrier attr-dict `:` type($barrier)";
}

//
// AllocPipeline Op
//
def GCU_AllocPipelineOp : GCU_Op<"alloc_pipeline", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a pipeline";
  let description = [{
    `gcu.alloc_pipeline` allocate a pipeline and the pipeline needs to be called by `init` explicitly.
  }];
  let results = (outs GCU_PipelineType:$pipeline);
  let assemblyFormat = "attr-dict `:` type($pipeline)";
  let hasVerifier = 1;
}

//
// InitPipeline Op
//
def GCU_InitPipelineOp : GCU_Op<"init_pipeline", [MemoryEffectsOpInterface]> {
  let summary = "Initialize a pipeline";
  let description = [{
    `gcu.init_pipeline` initialize a pipeline.
  }];
  let arguments = (ins GCU_PipelineType:$pipeline);
  let assemblyFormat = "$pipeline attr-dict `:` type($pipeline)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// ProducerAcquire Op
//
def GCU_ProducerAcquireOp : GCU_Op<"producer_acquire"> {
  let summary = "Producer acquires a token to fill buffers";
  let description = [{
    `gcu.producer_acquire` producer acquires a token to fill buffers.
  }];
  let arguments = (ins GCU_PipelineType:$pipeline);
  let assemblyFormat = "$pipeline attr-dict `:` type($pipeline)";
}

//
// ProducerCommit Op
//
def GCU_ProducerCommitOp : GCU_Op<"producer_commit"> {
  let summary = "Producer commits the buffer changes";
  let description = [{
    `gcu.producer_commit` producer commits the buffer changes.
  }];
  let arguments = (ins GCU_PipelineType:$pipeline);
  let assemblyFormat = "$pipeline attr-dict `:` type($pipeline)";
}

//
// ConsumerWait Op
//
def GCU_ConsumerWaitOp : GCU_Op<"consumer_wait"> {
  let summary = "Consumer awaits buffer readiness";
  let description = [{
    `gcu.consumer_wait` consumer awaits buffer readiness.
  }];
  let arguments = (ins GCU_PipelineType:$pipeline);
  let assemblyFormat = "$pipeline attr-dict `:` type($pipeline)";
}

//
// ConsumerRelease Op
//
def GCU_ConsumerReleaseOp : GCU_Op<"consumer_release"> {
  let summary = "Consumer releases the token";
  let description = [{
    `gcu.consumer_release` consumer releases the token.
  }];
  let arguments = (ins GCU_PipelineType:$pipeline);
  let assemblyFormat = "$pipeline attr-dict `:` type($pipeline)";
}

//
// WarpSpecialize Op
//
def GCU_WarpSpecializeOp : GCU_Op<"warp_specialize", [
  RecursiveMemoryEffects, RecursivelySpeculatable,
  DeclareOpInterfaceMethods<RegionBranchOpInterface>
]> {
  let summary = "asynchronously execute code on multiple warpgroups";
  let description = [{
    The `gcu.warp_specialize` op represents executing different code
    simultaneously on different warp groups. A warp group is a group of
    power-of-2 warps, which can be a different number of warps than in the
    enclosing region.

    The "default" region of the op represents the code executed by the currently
    executing warp group. This region is allowed to implicitly capture. The op
    contains a number of "partition" regions that are isolated from above. They
    must be isolated because these regions represent different layout domains,
    as the number of warps is different.

    Semantically, execution of each region starts simultaneously for each warp
    group, and all warp groups are joined at the end of the op.

    Example:

    ```mlir
    %0 = gcu.warp_specialize(%a, %b)
    default {
      %out = some_operation(%a) // implicit capture of `%a`
      gcu.warp_yield %out : i32
    }
    partition0(%arg0: i32, %arg1: i32) num_warps(8) {
      some_async_dispatch(%arg0, %arg1)
      gcu.warp_return
    }
    partition1(%arg0: i32, %arg1: i32) num_warps(1) {
      some_async_dispatch(%arg0, %arg1)
      gcu.warp_return
    } : (i32, i32) -> i32
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$explicitCaptures,
    DenseI32ArrayAttr:$partitionNumWarps,
    OptionalAttr<DenseI32ArrayAttr>:$warpGroupStartIds,
    OptionalAttr<DenseI32ArrayAttr>:$requestedRegisters,
    OptionalAttr<DenseI32ArrayAttr>:$actualRegisters
  );
  let results = (outs Variadic<AnyType>:$defaultPassthrough);

  let regions = (region
    MinSizedRegion<1>:$defaultRegion,
    SizedRegion<1>:$partitionOpHolder
  );

  let extraClassDeclaration = [{
    RegionRange getPartitionRegions();

    // Get the size and alignment of the capture list.
    std::pair<uint64_t, uint64_t> getCaptureSizeAlign();
    // Get the total number of extra warps required.
    unsigned getTotalPartitionWarps();
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes,
                   "ArrayRef<int32_t>":$partitionNumWarps,
                   "unsigned":$numPartitionRegions)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "ValueRange":$explicitCaptures,
                   "ArrayRef<int32_t>":$partitionNumWarps)>,
  ];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizeMethod = 1;
}

//
// WarpSpecializePartitions Op
//
def GCU_WarpSpecializePartitionsOp : GCU_Op<"warp_specialize.partitions", [
  IsolatedFromAbove, RecursiveMemoryEffects, RecursivelySpeculatable,
  Terminator, HasParent<"WarpSpecializeOp">
]> {
  let summary = "container op for `gcu.warp_specialize`";
  let description = [{
    Because MLIR requires entire operations be isolated from above, this op
    contains the actual isolated from above regions of `gcu.warp_specialize`.
  }];

  let regions = (region VariadicRegion<MinSizedRegion<1>>:$partitionRegions);
}

//
// WarpYield Op
//
def GCU_WarpYieldOp : GCU_Op<"warp_yield", [
  Pure, Terminator, ReturnLike, HasParent<"WarpSpecializeOp">,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>
]> {
  let summary = "yield from the default region of `gcu.warp_specialize`";
  let description = [{
    The `gcu.warp_yield` operation is the terminator for the "default" region of
    a `gcu.warp_specialize` operation. The operands are passed transparently as
    the SSA results of the `gcu.warp_specialize` operation.

    Example:

    ```mlir
    gcu.warp_yield %a, %b : i32, tensor<32xbf16, #blocked>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$values);

  let assemblyFormat = "($values^)? attr-dict (`:` type($values)^)?";
  let hasVerifier = 1;
}

//
// WarpReturn Op
//
def GCU_WarpReturnOp : GCU_Op<"warp_return", [
  Pure, Terminator, ReturnLike, HasParent<"WarpSpecializePartitionsOp">
]> {
  let summary = "implicit terminator from partition regions";
  let description = [{
    The `gcu.warp_return` operation is the implicit terminator that ends the
    partition regions of a `gcu.warp_specialize` op. It has no operands as these
    regions cannot return anything.

    TODO: Support returning uniform values from partition regions.
  }];

  let assemblyFormat = "attr-dict";
}

//
// VectorConverter Op
//
def GCU_VectorConvertOp : GCU_Op<"vector_convert", [Pure]> {
  let summary = "Convert a vector type";
  let description = [{
    `gcu.vector_convert` convert a vector type.
  }];
  let arguments = (ins Variadic<GCU_Vector1D>:$inputs);
  let results = (outs Variadic<GCU_Vector1D>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `to` type($outputs) ";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//
// PtrToMemRef Op
//
def GCU_PtrToMemRefOp : GCU_Op<"ptr2memref", [Pure]>
{
  let summary = "Convert a pointer to memref";

  let description = [{
  }];
  let arguments = (ins GCU_PtrType:$ptr);
  let results = (outs Arg<MemRefRankOf<[AnyType], [1]>>:$resultMemref);
  let assemblyFormat = [{ $ptr attr-dict `:` type($ptr) `to` type($resultMemref) }];
  let hasVerifier = 1;
}

//
// MemRefToPtr Op
//
def GCU_MemRefToPtrOp : GCU_Op<"memref2ptr", [Pure]>
{
  let summary = "Convert a memref to pointer";

  let description = [{
  }];
  let arguments = (ins AnyMemRef:$memref);
  let results = (outs GCU_PtrType:$ptr);
  let assemblyFormat = [{ $memref attr-dict `:` type($memref) `to` type($ptr) }];
  let hasVerifier = 1;
}

//
// PtrToInt Op
//
def GCU_PtrToIntOp : GCU_Op<"ptr2int", [Pure]>
{
  let summary = "Convert a pointer to integer";

  let description = [{
  }];
  let arguments = (ins GCU_PtrType:$ptr);
  let results = (outs I64:$result);
  let assemblyFormat = [{ $ptr attr-dict `:` type($ptr) }];
}


//
// IntToPtr Op
//
def GCU_IntToPtrOp : GCU_Op<"int2ptr", [Pure]>
{
  let summary = "Convert an integer to pointer";

  let description = [{
  }];
  let arguments = (ins I64:$value);
  let results = (outs GCU_PtrType:$ptr);
  let assemblyFormat = [{ $value attr-dict `:` type($ptr) }];
}

//
// GetMemRefOffset Op
//
def GCU_GetMemRefOffsetOp : GCU_Op<"get_memref_offset", [Pure]>
{
  let summary = "Get a memref offset";

  let description = [{
  }];
  let arguments = (ins AnyMemRef:$memref, Variadic<Index>:$offsets);
  let results = (outs Index:$storage_offset);
  let assemblyFormat = [{ $memref $offsets attr-dict `:` type($memref) }];
}

//
// MaterializeInDestination Op
//
def GCU_MaterializeInDestinationOp : GCU_Op<"materialize_in_destination", [MemoryEffectsOpInterface]>
{
  let summary = "Materialize a tensor to memref and reuse dest memref";

  let description = [{
  }];
  let arguments = (ins AnyTensor:$source, AnyMemRef:$dest);
  let assemblyFormat = [{ $source `in` $dest attr-dict `:` type($source) `in` type($dest)}];
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
    }
  }];
}

def GCU_ExternElementwiseOp : GCU_Op<"extern_elementwise_op", [SameVariadicOperandSize]> {
  let summary = "gcu libdevice api for gcu400";
  let arguments = (ins Variadic<VectorOfRank<[1]>>:$srcs, StrAttr:$symbol);
  let results = (outs VectorOfRank<[1]>:$result);

  let assemblyFormat = "$srcs attr-dict `:` type($srcs) `to` type($result) ";
}

//
// BuiltinElementwiseOp
//
def GCU_BuiltinElementwiseOp : GCU_Op<"builtin_elementwise_op", [MemoryEffectsOpInterface, AttrSizedOperandSegments]> {
  let summary = "Builtin elementwise operation";
  let description = [{
      "Call an elementwise function $symbol implemented in builtin"
  }];
  let arguments = (ins AnyMemRef:$output, Variadic<AnyTypeOf<[AnyMemRef, AnyI8, AnyI16, AnyI32, AnyI64, F16, F32, BF16]>>:$inputs, StrAttr:$symbol, Variadic<F32>:$params);
  let assemblyFormat = [{$output `,` $inputs ( `,` $params^ )? attr-dict `:` type($output) `,` type($inputs) ( `,` type($params)^ )?}];
  let builders = [
    OpBuilder<(ins
      "mlir::Value":$output,
      "mlir::ValueRange":$inputs,
      "mlir::StringAttr":$symbol)
      , [{
    return build($_builder, $_state, output, inputs, symbol, {});}]>,
    OpBuilder<(ins
      "mlir::Value":$output,
      "mlir::ValueRange":$inputs,
      "mlir::ValueRange":$params,
      "mlir::StringAttr":$symbol)
      , [{
    return build($_builder, $_state, output, inputs, symbol, params);}]>
  ];
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                          SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}


//
// MFence Op
//
def GCU_MFenceOp : GCU_Op<"mfence", []>
{
  let summary = "memory fence";
  let arguments = (ins DefaultValuedAttr<GCU_MemoryFenceAttr,
                        "::mlir::gcu::MFenceType::Memory">:$mfence_type);
  let description = [{
  }];
  let assemblyFormat = [{ attr-dict }];
}

//
// GatherLoad Op
//
def GCU_GatherLoadOp : GCU_Op<"gather_load", [MemoryEffectsOpInterface]> {
  let summary = "Gather load";
  let description = [{
    `gcu.gather_load` gather a tensor from source to dst.
  }];
  let arguments = (ins GCU_PtrType:$dst, GCU_PtrType:$src, GCU_PtrType:$offsets, GCU_PtrType:$masks, GCU_PtrType:$others, I32:$size);
  let assemblyFormat = "$dst `,` $src `,` $offsets `,` $masks `,` $others `,` $size attr-dict `:` type($dst) `,` type($src) `,` type($offsets) `,` type($masks) `,` type($others) `,` type($size)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(3),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(4),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// ScatterStore Op
//
def GCU_ScatterStoreOp : GCU_Op<"scatter_store", [MemoryEffectsOpInterface]> {
  let summary = "Scatter store";
  let description = [{
    `gcu.scatter_store` scatter a tensor from source to dst.
  }];
  let arguments = (ins GCU_PtrType:$dst, GCU_PtrType:$src, GCU_PtrType:$offsets, GCU_PtrType:$masks, I32:$size);
  let assemblyFormat = "$dst `,` $src `,` $offsets `,` $masks `,` $size attr-dict `:` type($dst) `,` type($src) `,` type($offsets) `,` type($masks) `,` type($size)";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(3),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// MatMul Op
//
def GCU_MatMulOp : GCU_Op<"matmul", [MemoryEffectsOpInterface]> {
  let summary = "Matrix multiplication with bias";
  let description = [{
    Performs a two dimensional matrix multiplication.
  }];
  let arguments = (ins AnyMemRef:$out, AnyMemRef:$lhs, AnyMemRef:$rhs, Optional<AnyMemRef>:$bias,
      DefaultValuedAttr<GCU_InputPrecisionAttr, "::mlir::gcu::InputPrecision::IEEE">:$inputPrecision);
  let assemblyFormat = "$out `,` $lhs `,` $rhs (`,` $bias^)?  attr-dict `:` type($out) `,` type($lhs) `,` type($rhs) (`,` type($bias)^)?";
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
    }
  }];
  let hasVerifier = 1;
}

//
// Assert Op
//
def GCU_AssertOp : GCU_Op<"assert", []>
{
  let summary = "device assert";
  let description = [{
    gcu device assert.
  }];
  let arguments = (ins I1:$condition, StrAttr:$message, StrAttr:$file, StrAttr:$func, I32Attr:$line);
  let assemblyFormat = [{
    $condition attr-dict `:` type($condition)
  }];
}

//
// triton debug Op
//
def GCU_TritonBorder : GCU_Op<"border", []>
{
  let summary = "triton border";
  let description = [{
  }];
  let assemblyFormat = [{ attr-dict }];
}

//
// Reduce Op
//

def GCU_ReduceMemRef : MemRefRankOf<[AnyIntOfWidths<[8, 16, 32, 64]>, BF16, F16, F32], [3]>;

def GCU_ReduceOp : GCU_Op<"reduce", [MemoryEffectsOpInterface]> {
  let summary = "Reduction using specified combination algorithm";
  let arguments = (ins GCU_ReduceOperationAttr:$op, GCU_ReduceMemRef:$out, GCU_ReduceMemRef:$in, Optional<GCU_ReduceMemRef>:$workspace, I32Attr:$axis);
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
  let hasVerifier = 1;
}

//
// Reduce argxxxOp
//

def GCU_ReduceArgOp : GCU_Op<"reduce_arg", [MemoryEffectsOpInterface]> {
  let summary = "Reduction using specified combination algorithm";
  let arguments = (ins GCU_ReduceOperationAttr:$op, GCU_ReduceMemRef:$out, GCU_ReduceMemRef:$outIndex, GCU_ReduceMemRef:$in, GCU_ReduceMemRef:$inIndex,  Optional<GCU_ReduceMemRef>:$workspace, I32Attr:$axis);
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(3),
                           SideEffects::DefaultResource::get());
    }
  }];
  let hasVerifier = 1;
}

def GCU_VectorMovSftOp : GCU_Op<"vector_movsft", [AllTypesMatch<["vin", "vout"]>]> {
  let summary = "vector move shift";
  let arguments = (ins GCU_VectorMovSftModeAttr:$mode, VectorOfLengthAndType<[16, 32, 64], [I32, I64, F32]>:$vin, I32Attr:$unit);
  let results = (outs VectorOfLengthAndType<[16, 32, 64], [I32, I64, F32]>:$vout);
}

// for performance tool

//
// BeginClock
//

def GCU_BeginClockOp : GCU_Op<"begin_clock", []>
{
  let summary = "get clock start counter";

  let description = [{
  }];
  let results = (outs I64:$clock);
  let assemblyFormat = [{ attr-dict `:` type($clock) }];
}

//
// EndClock
//

def GCU_EndClockOp : GCU_Op<"end_clock", []>
{
  let summary = "get clock end counter";

  let description = [{
  }];
  let results = (outs I64:$clock);
  let assemblyFormat = [{ attr-dict `:` type($clock) }];
}

//
// GetGlobalThreadId Op
//
def GCU_GetGlobalThreadIdOp : GCU_Op<"get_global_thread_id", [Pure]>
{
  let summary = "Get the global thread id";

  let description = [{
    `gcu.get_global_thread_id` returns the global thread id of the current thread.
  }];
  let results = (outs Index:$result);
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

def GCU_TarInitOp : GCU_Op<"tar_init", []> {
  let summary = "Thread Address Register only for gcu400";
  let arguments = (ins I64:$in);
  let results = (outs GCU_TarType:$out);
}

def GCU_TarLoadOp : GCU_Op<"tar_load", []> {
  let summary = "vector load from TAR only for gcu400";
  let arguments = (ins GCU_TarType:$src_addr, GCU_TarType:$stride);
  let results = (outs GCU_VectorType:$v, GCU_TarType:$dst_addr);
}

def GCU_TarStoreOp : GCU_Op<"tar_store", []> {
  let summary = "vector store to TAR only for gcu400";
  let arguments = (ins GCU_VectorType:$v, GCU_TarType:$src_addr, GCU_TarType:$stride);
  let results = (outs GCU_TarType:$dst_addr);
}

def GCU_TarGatherOp : GCU_Op<"tar_gather", []> {
  let summary = "vector gather load to TAR only for gcu400";
  let arguments = (ins GCU_TarType:$src_addr, I32:$num, GCU_VectorType:$other, Optional<GCU_PtrType>:$mask);
  let results = (outs GCU_VectorType:$v, GCU_TarType:$dst_addr);
}

def GCU_TarScatterOp : GCU_Op<"tar_scatter", []> {
  let summary = "vector scatter store to TAR only for gcu400";
  let arguments = (ins GCU_TarType:$src_addr, GCU_VectorType:$v, I32:$num, Optional<GCU_PtrType>:$mask);
  let results = (outs GCU_TarType:$dst_addr);
}

def GCU_VectorStepOp : GCU_Op<"vector_step", [Pure,
     PredOpTrait<"source operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]> {
  let summary = "a linear sequence of values";
  let arguments = (ins AnyIntOfWidths<[8, 16, 32, 64]>:$start);
  let results = (outs GCU_VectorType:$result);
}

def GCU_MemMapOp : GCU_Op<"mem_map", []> {
  let summary = "mmu map memory, only for gcu300";
  let arguments = (ins GCU_PtrType:$ptr, I32:$num);
  let results = (outs I32:$addr);
}

def GCU_MemUnmapOp : GCU_Op<"mem_unmap", []> {
  let summary = "mmu unmap memory, only for gcu300";
  let arguments = (ins I32:$addr, I32:$num);
}

def GCU_AtomicRMWOp : GCU_Op<"atomic_rmw", []> {
    let summary = "atomic rmw";
    let description = [{
        load data at $ptr, do $rmw_op with $val, and store result to $ptr.
        return old value at $ptr
    }];
    let hasVerifier = 1;

    let arguments = (ins
      GCU_AtomicRMWAttr:$atomic_rmw_op,
      GCU_PtrType:$ptr,
      AnyTypeOf<[AnyMemRef, AnyI8, AnyI16, AnyI32, AnyI64, F16, F32, BF16]>:$val,
      GCU_MemSemanticAttr:$sem,
      GCU_MemSyncScopeAttr:$scope
    );

    let results = (outs AnyTypeOf<[AnyI8, AnyI16, AnyI32, AnyI64, F16, F32, BF16, RankedTensorOf<[AnyI8, AnyI16, AnyI32, AnyI64, F16, F32, BF16]>]>:$result);

    let assemblyFormat = [{
      $atomic_rmw_op `,` $sem `,` $scope `,` $ptr `,` $val attr-dict `:`
      functional-type(operands, $result)
    }];
}

def GCU_AtomicCASOp : GCU_Op<"atomic_cas", []> {
    let summary = "atomic cas";
    let description = [{
        compare $cmp with data $old at location $ptr,
        if $old == $cmp, store $val to $ptr,
        else store $old to $ptr,
        return $old
    }];
    let hasVerifier = 1;

    let arguments = (ins
      GCU_PtrType:$ptr,
      AnyTypeOf<[AnyMemRef, AnyI8, AnyI16, AnyI32, AnyI64, F16, F32, BF16]>:$cmp,
      AnyTypeOf<[AnyMemRef, AnyI8, AnyI16, AnyI32, AnyI64, F16, F32, BF16]>:$val,
      GCU_MemSemanticAttr:$sem,
      GCU_MemSyncScopeAttr:$scope
    );

    let results = (outs AnyTypeOf<[AnyI8, AnyI16, AnyI32, AnyI64, F16, F32, BF16, RankedTensorOf<[AnyI8, AnyI16, AnyI32, AnyI64, F16, F32, BF16]>]>:$result);

    let assemblyFormat = [{
      $sem `,` $scope `,` $ptr `,` $cmp  `,` $val attr-dict `:`
      functional-type(operands, $result)
    }];
}

#endif // GCU_OPS
