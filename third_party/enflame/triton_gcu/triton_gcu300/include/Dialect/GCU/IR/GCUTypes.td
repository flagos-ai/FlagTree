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
#ifndef GCU_TYPES
#define GCU_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Dialect/GCU/IR/GCUDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//
// Types
//
class GCUTypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<GCU_Dialect, name, traits> {
    // Used by dte ctx
    let mnemonic = _mnemonic;
}

// Floating-point Type
def GCU_Float : AnyTypeOf<[F8E4M3FNUZ, F8E4M3FN, F8E4M3B11FNUZ, F8E5M2, F16, BF16, F32, F64], "floating-point">;
def GCU_FloatTensor : TensorOf<[GCU_Float]>;
def GCU_FloatLike : AnyTypeOf<[GCU_Float, GCU_FloatTensor]>;

// Boolean Type
// GCU_Bool -> I1
def GCU_BoolTensor : TensorOf<[I1]>;
def GCU_BoolLike : AnyTypeOf<[I1, GCU_BoolTensor]>;

// Integer Type
def GCU_Int : AnyTypeOf<[I1, I8, I16, I32, I64], "integer">;
def GCU_IntTensor : TensorOf<[GCU_Int]>;
def GCU_IntLike : AnyTypeOf<[GCU_Int, GCU_IntTensor]>;

// I32 Type
// GCU_I32 -> I32
// GCU_I32Tensor -> I32Tensor
def GCU_I32Like : AnyTypeOf<[I32, I32Tensor]>;

// I64 Type
// GCU_I64 -> I64
// GCU_I64Tensor -> I64Tensor
def GCU_I64Like : AnyTypeOf<[I64, I64Tensor]>;

// Tensor Type
def GCU_FpIntTensor : AnyTypeOf<[GCU_FloatTensor, GCU_IntTensor]>;
def GCU_Tensor : AnyTypeOf<[GCU_FpIntTensor]>;

// DTE Ctx Type
def GCU_DTEType : GCUTypeDef<"DTE", "dte", [MemRefElementTypeInterface]> {
    let summary = "DTE ctx type (`::mlir::gcu::DTEType`) in GCU IR type system";
    let description = [{
        DTE ctx type in GCU IR type system.
    }];
    let parameters = (ins "AddressSpaceAttr":$addressSpace);
    let assemblyFormat = [{ `<` $addressSpace `>` }];
}

// Barrier Type
def GCU_BarrierType : GCUTypeDef<"Barrier", "barrier", [MemRefElementTypeInterface]> {
    let summary = "DTE ctx type (`::mlir::gcu::BarrierType`) in GCU IR type system";
    let description = [{
        Barrier type in GCU IR type system.
    }];
    let parameters = (ins "AddressSpaceAttr":$addressSpace);
    let assemblyFormat = [{ `<` $addressSpace `>` }];
}

// Pointer Type
def GCU_PtrType : GCUTypeDef<"Ptr", "ptr", [MemRefElementTypeInterface]> {
    let summary = "Pointer type (`::mlir::gcu::PtrType`) in GCU IR type system";
    let description = [{
        Pointer type in GCU IR type system.
    }];
    let parameters = (ins "Type":$elementType);
    let assemblyFormat = [{ `<` $elementType `>` }];
}

// Any Type in IR
def GCU_FpInt : AnyTypeOf<[GCU_Float, GCU_Int]>;
def GCU_Type : AnyTypeOf<[GCU_FloatLike, GCU_IntLike, GCU_FpIntTensor, GCU_DTEType]>;

// Vector 1d
def GCU_Vector1D : VectorOfRank<[1]>;


// todo_AT
// TileDesc Type
def GCU_TileDescType : GCUTypeDef<"TileDesc", "tile_desc", []> {
    let summary = "DTE ctx type (`::mlir::gcu::TileDescType`) in GCU IR type system";
    let description = [{
        Tile descriptor type in GCU IR type system.
    }];
    let parameters = (ins "RankedTensorType":$desc);
    let assemblyFormat = [{ `<` $desc `>` }];
}

class VectorOfRankAndLengthAndType<list<int> allowedRanks,
                                   list<int> allowedLengths,
                                   list<Type> allowedTypes> : AllOfType<
  [VectorOfRank<allowedRanks>, VectorOfLength<allowedLengths>,
   VectorOfAnyRankOf<allowedTypes>],
  VectorOfRank<allowedRanks>.summary #
  VectorOfLength<allowedLengths>.summary #
  VectorOfAnyRankOf<allowedTypes>.summary,
  "::mlir::VectorType">;

def GCU_TarType : VectorOfLengthAndType<[1],[I64]>;

def GCU_FP8 : AnyTypeOf<[F8E4M3FN, F8E5M2]>;

def GCU_VectorType : AnyTypeOf<[VectorOfRankAndLengthAndType<[1], [512, 1024, 2048], [I1, I8, GCU_FP8]>,
                                VectorOfRankAndLengthAndType<[1], [256, 512, 1024], [I16, F16, BF16]>,
                                VectorOfRankAndLengthAndType<[1], [128, 256, 512], [I32, F32]>,
                                VectorOfRankAndLengthAndType<[1], [64, 128, 256], [I64]>]>;

#endif //GCU_TYPES
