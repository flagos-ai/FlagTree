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
  let arguments = (ins AnyMemRef:$output, Variadic<AnyTypeOf<[AnyMemRef, AnyI8, AnyI16, AnyI32, F16, F32, BF16]>>:$inputs, StrAttr:$symbol, Variadic<F32>:$params);
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
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(3),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
    }
  }];
  let hasVerifier = 1;
}

def GCU_VectorMovSftOp : GCU_Op<"vector_movsft", [AllTypesMatch<["vin", "vout"]>]> {
  let summary = "vector move shift";
  let arguments = (ins GCU_VectorMovSftModeAttr:$mode, VectorOfLengthAndType<[32], [I32, F32]>:$vin, I32Attr:$unit);
  let results = (outs VectorOfLengthAndType<[32], [I32, F32]>:$vout);
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
      AnyTypeOf<[AnyMemRef, AnyI8, AnyI16, AnyI32, F16, F32, BF16]>:$val,
      Optional<AnyType>:$mask,
      GCU_MemSemanticAttr:$sem,
      GCU_MemSyncScopeAttr:$scope
    );

    let results = (outs AnyTypeOf<[AnyI8, AnyI16, AnyI32, F16, F32, BF16, RankedTensorOf<[AnyI8, AnyI16, AnyI32, F16, F32, BF16]>]>:$result);

    let assemblyFormat = [{
      $atomic_rmw_op `,` $sem `,` $scope `,` $ptr `,` $val (`,` $mask^)?  attr-dict `:`
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
      AnyTypeOf<[AnyMemRef, AnyI8, AnyI16, AnyI32, F16, F32, BF16]>:$cmp,
      AnyTypeOf<[AnyMemRef, AnyI8, AnyI16, AnyI32, F16, F32, BF16]>:$val,
      GCU_MemSemanticAttr:$sem,
      GCU_MemSyncScopeAttr:$scope
    );

    let results = (outs AnyTypeOf<[AnyI8, AnyI16, AnyI32, F16, F32, BF16, RankedTensorOf<[AnyI8, AnyI16, AnyI32, F16, F32, BF16]>]>:$result);

    let assemblyFormat = [{
      $sem `,` $scope `,` $ptr `,` $cmp  `,` $val attr-dict `:`
      functional-type(operands, $result)
    }];
}

#endif // GCU_OPS
