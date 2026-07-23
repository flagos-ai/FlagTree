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
#ifndef MemrefExt_OPS
#define MemrefExt_OPS

include "Dialect/MemrefExt/IR/MemrefExtDialect.td"
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
class MemrefExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<MemrefExt_Dialect, mnemonic,
       !listconcat(traits, [])> {
}
//
// Memset Op (tag version)
//
def GCU_MemsetStartOp : MemrefExt_Op<"memset_start", [MemoryEffectsOpInterface]> {
  let summary = "Memset";
  let description = [{
    `memref_ext.memset_start` msmet a value to buffer.
  }];
  let arguments = (ins AnyMemRef:$dst,
                       AnyTypeOf<[AnySignlessInteger, AnyFloat]>:$value,
                       AnyMemRef:$tag_memref,
                       Variadic<Index>:$tag_indices);

  let assemblyFormat = [{$dst `,` $value `,` $tag_memref `[` $tag_indices `]` attr-dict `:`
    type($dst) `,` type($value) `,` type($tag_memref)}];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Slice Op (tag version)
//
def GCU_SliceStartOp : MemrefExt_Op<"slice_start",
                                    [AttrSizedOperandSegments,
                                     MemoryEffectsOpInterface]> {
  let summary = "Slice";
  let description = [{
    `memref_ext.slice_start` slice a data from source to dst.
  }];
  let arguments = (ins AnyMemRef:$dst,
                       AnyMemRef:$src,
                       Variadic<I32>:$offsets,
                       AnyType:$default_value,
                       AnyMemRef:$tag_memref,
                       Variadic<Index>:$tag_indices);

  let assemblyFormat = [{$dst `,` $src `[` $offsets `]` `,` $default_value `,` $tag_memref `[` $tag_indices `]` attr-dict `:`
    type($dst) `,` type($src) `,` type($default_value) `,` type($tag_memref)}];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(4),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// SlicePad Op (tag version)
//
def GCU_SlicePadStartOp : MemrefExt_Op<"slice_pad_start",
                                      [AttrSizedOperandSegments, MemoryEffectsOpInterface]> {
  let summary = "Slice";
  let description = [{
    `memref_ext.slice_pad_start` slice a data from source to dst with pad value.
  }];
  let arguments = (ins AnyMemRef:$dst,
                       AnyMemRef:$src,
                       Variadic<I32>:$offsets,
                       Variadic<I32>:$slice_shape,
                       AnyType:$pad_value,
                       AnyMemRef:$tag_memref,
                       Variadic<Index>:$tag_indices);

  let assemblyFormat = [{$dst `,` $src `[` $offsets `]` `[` $slice_shape `]` `,` $pad_value `,` $tag_memref `[` $tag_indices `]` attr-dict `:`
    type($dst) `,` type($src) `,` type($pad_value) `,` type($tag_memref)}];

  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(5),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Deslice Op (tag version)
//
def GCU_DesliceStartOp : MemrefExt_Op<"deslice_start", [AttrSizedOperandSegments, MemoryEffectsOpInterface]> {
  let summary = "Deslice";
  let description = [{
    `memref_ext.deslice_start` deslice a data from source to dst.
  }];
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, Variadic<I32>:$offsets, AnyMemRef:$tag_memref, Variadic<Index>:$tag_indices);
  let assemblyFormat = [{$dst `,` $src `[` $offsets `]` $tag_memref `[` $tag_indices `]` attr-dict `:`
    type($dst) `,` type($src) `,` type($tag_memref)}];
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(3),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Slice_Deslice Op (tag version)
//
def GCU_SliceDesliceStartOp : MemrefExt_Op<"slice_deslice_start", [AttrSizedOperandSegments, MemoryEffectsOpInterface]> {
  let summary = "Slice_Deslice";
  let description = [{
    `memref_ext.slice_deslice_start` deslice a data from source to dst.
  }];
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, Variadic<I32>:$offsets, Variadic<I32>:$slice_shape, Variadic<I32>:$dst_offsets, AnyMemRef:$tag_memref, Variadic<Index>:$tag_indices);
  let assemblyFormat = [{$dst `,` $src `[` $offsets `]` `[` $slice_shape `]` `[` $dst_offsets `]` $tag_memref `[` $tag_indices `]` attr-dict `:`
    type($dst) `,` type($src) `,` type($tag_memref)}];
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(5),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Transpose Op (tag version)
//
def GCU_TransposeStartOp : MemrefExt_Op<"transpose_start", [AttrSizedOperandSegments, MemoryEffectsOpInterface]> {
  let summary = "Transpose";
  let description = [{
    `memref_ext.transpose_start` transpose a data from source to dst.
  }];
  let arguments = (ins AnyMemRef:$dst,
                       AnyMemRef:$src,
                       Variadic<I32>:$layout,
                       AnyMemRef:$tag_memref,
                       Variadic<Index>:$tag_indices);

  let assemblyFormat = [{$dst `,` $src `[` $layout `]` $tag_memref `[` $tag_indices `]` attr-dict `:`
    type($dst) `,` type($src) `,` type($tag_memref)}];
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(3),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Broadcast Op (tag version)
//
def GCU_BroadcastStartOp : MemrefExt_Op<"broadcast_start", [MemoryEffectsOpInterface]> {
  let summary = "Broadcast";
  let description = [{
    `memref_ext.broadcast_start` broadcast a data from source to dst.
  }];
  let arguments = (ins AnyMemRef:$dst,
                       AnyMemRef:$src,
                       AnyMemRef:$tag_memref,
                       Variadic<Index>:$tag_indices);
  let assemblyFormat = [{$dst `,` $src `,` $tag_memref `[` $tag_indices `]` attr-dict `:`
    type($dst) `,` type($src) `,` type($tag_memref)}];
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(2),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Slice_Broadcast Op (tag version)
//
def GCU_SliceBroadcastStartOp : MemrefExt_Op<"slice_broadcast_start", [AttrSizedOperandSegments, MemoryEffectsOpInterface]> {
  let summary = "Slice and broadcast";
  let description = [{
    `memref_ext.slice_broadcast_start` slice_broadcast a data from source to dst.
  }];
  let arguments = (ins AnyMemRef:$dst,
                       AnyMemRef:$src,
                       Variadic<I32>:$offsets,
                       Variadic<I32>:$slice_shape,
                       AnyMemRef:$tag_memref,
                       Variadic<Index>:$tag_indices);
  let assemblyFormat = [{$dst `,` $src `[` $offsets `]` `[` $slice_shape `]` `,` $tag_memref `[` $tag_indices `]` attr-dict `:`
    type($dst) `,` type($src) `,` type($tag_memref)}];
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(4),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Slice_Transpose Op (tag version)
//
def GCU_SliceTransposeStartOp : MemrefExt_Op<"slice_transpose_start", [AttrSizedOperandSegments, MemoryEffectsOpInterface]> {
  let summary = "Slice and transpose";
  let description = [{
    `memref_ext.slice_transpose_start` slice_transpose a data from source to dst.
  }];
  let arguments = (ins AnyMemRef:$dst,
                       AnyMemRef:$src,
                       Variadic<I32>:$offsets,
                       Variadic<I32>:$layout,
                       AnyTypeOf<[AnySignlessInteger, AnyFloat]>:$value,
                       AnyMemRef:$tag_memref,
                       Variadic<Index>:$tag_indices);
  let assemblyFormat = [{$dst `,` $src `[` $offsets `]` `[` $layout `]` `,` $value `,` $tag_memref `[` $tag_indices `]` attr-dict `:`
    type($dst) `,` type($src) `,` type($value) `,` type($tag_memref)}];
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(5),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Transpose_Deslice Op (tag version)
//
def GCU_TransposeDesliceStartOp : MemrefExt_Op<"transpose_deslice_start", [AttrSizedOperandSegments, MemoryEffectsOpInterface]> {
  let summary = "Transpose and deslice";
  let description = [{
    `memref_ext.transpose_deslice_start` transpose_deslice a data from source to dst.
  }];
  let arguments = (ins AnyMemRef:$dst,
                       AnyMemRef:$src,
                       Variadic<I32>:$layout,
                       Variadic<I32>:$offsets,
                       AnyMemRef:$tag_memref,
                       Variadic<Index>:$tag_indices);
  let assemblyFormat = [{$dst `[` $offsets `]`  `,` $src `[` $layout `]` `,` $tag_memref `[` $tag_indices `]` attr-dict `:`
    type($dst) `,` type($src) `,` type($tag_memref)}];
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(4),
                           SideEffects::DefaultResource::get());
    }
  }];
}

//
// Memset_Deslice Op (tag version)
//
def GCU_MemsetDesliceStartOp : MemrefExt_Op<"memset_deslice_start", [AttrSizedOperandSegments, MemoryEffectsOpInterface]> {
  let summary = "Memset and deslice";
  let description = [{
    `memref_ext.memset_deslice_start` memset_deslice a data from source to dst.
  }];
  let arguments = (ins AnyMemRef:$dst,
                       AnyMemRef:$src,
                       Variadic<I32>:$offsets,
                       AnyTypeOf<[AnySignlessInteger, AnyFloat]>:$value,
                       AnyMemRef:$tag_memref,
                       Variadic<Index>:$tag_indices);
  let assemblyFormat = [{$dst `[` $offsets `]` `,` $src `,` $value `,` $tag_memref `[` $tag_indices `]` attr-dict `:`
    type($dst) `,` type($src) `,` type($value) `,` type($tag_memref)}];
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(4),
                           SideEffects::DefaultResource::get());
    }
  }];
}

#endif // MemrefExt_OPS
