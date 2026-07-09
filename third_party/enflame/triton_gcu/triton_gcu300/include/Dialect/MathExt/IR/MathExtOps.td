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
#ifndef MATH_EXT_OPS
#define MATH_EXT_OPS

include "Dialect/MathExt/IR/MathExtDialect.td"
include "Dialect/MathExt/IR/MathExtTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//
// Op Base
//
class MathExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<MathExt_Dialect, mnemonic,
       !listconcat(traits, [])> {
}

//
// Umulhi Op
//
def MathExt_UmulhiOp : MathExt_Op<"umulhi", [SameOperandsAndResultType, Pure] #
                                            ElementwiseMappable.traits> {
  let summary = "Umulhi";
  let description = [{
    Performs unsigned multiplication of two I32/I64 inputs and returns the high 32/64 bits of the result as an I32/I64 output.
  }];
  let results = (outs MathExt_IntLike:$result);
  let arguments = (ins MathExt_IntLike:$lhs, MathExt_IntLike:$rhs);
}

//
// Histogram Op
//
def MathExt_HistogramOp : MathExt_Op<"histogram", [MemoryEffectsOpInterface]> {
  let summary = "Histogram";
  let description = [{
    Return the histogram of the input tensor. The number of bins is equal to
    the dimension of the output tensor. Each bins has a width of 1 and bins
    start at 0.

    Optional attributes:
    - `use_mradix`: Use mradix hardware acceleration for radix sort.
      Requires operand element type to be i32 with shape in (0, 32].
  }];
  let arguments = (ins AnyMemRef:$result, AnyMemRef:$operand, OptionalAttr<UnitAttr>:$use_mradix);
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
  let hasVerifier = 1;
}

//
// InclusiveScan Op
//
def MathExt_InclusiveScanOp : MathExt_Op<"inclusive_scan", [MemoryEffectsOpInterface]> {
  let summary = "Inclusive prefix sum (1D only)";
  let description = [{
    Compute inclusive prefix sum (cumulative sum) of a 1D input memref and
    store the result in the output memref. Both result and operand must be
    rank-1 memrefs with the same shape.

    Optional attributes:
    - `use_miota`: Use miota hardware acceleration for bool-valued add-scan.
      Requires operand element type to be i32 with values in {0, 1}.
    - `reverse`: Compute reverse (right-to-left) inclusive prefix sum.
  }];
  let arguments = (ins
    AnyMemRef:$result,
    AnyMemRef:$operand,
    OptionalAttr<UnitAttr>:$use_miota,
    OptionalAttr<UnitAttr>:$reverse
  );
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
  let hasVerifier = 1;
}

#endif // MATH_EXT_OPS
