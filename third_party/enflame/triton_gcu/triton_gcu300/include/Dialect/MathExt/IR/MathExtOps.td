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
// One op for every histogram form the GCU backend supports.  The counted value
// ("digit") for one key element is always produced by the scalar `body` region;
// the region is mandatory so there is a single op definition:
//
//   math_ext.histogram %result, %key, %num_bins, %size_in, %mask {
//   ^bb(%e: <key elem>):
//     ...                     // arith on %e, may capture enclosing scalars
//     math_ext.yield %digit   // i32 digit for every non-identity region
//   }
//
//   - Counting raw bin indices is the identity region `math_ext.yield %e`.
//     The backend lowers it either to the generic scalar builtin (common
//     scene) or, when `use_mradix` is set, to the raw-value MRADIX builtin.
//   - A radix digit is `(e >> shift) & (num_bins - 1)`; the backend recognises
//     that exact region and extracts the shift from the region and lowers it
//     to the tuned MRADIX digit-shift builtin (MRADIX + shift scene).
//   - Any other region is vectorised onto one MRADIX vector and inlined into
//     the count loop (MRADIX + region scene), so no digit tile is ever
//     materialised.  A 32-bit floating-point key is accepted as long as the
//     region bitcasts it to i32 before the digit arithmetic.
//
// `result` is the count *buffer*, not the bin vector: it may be padded up to
// one OACC line (128 i32) so the tuned builtin can drain the whole MRADIX
// count vector with a single 512B store; only the first `num_bins` entries are
// meaningful.  `num_bins` is the number of logical bins; the digit shift is
// part of the region.
//
// `size_in` is the number of leading keys that are counted (a prefix
// `lane < N` mask folded by the fusion pass); the whole key buffer is counted
// when it equals the buffer length.  `mask`, when present, is a byte-per-lane
// i8 (or i1) buffer that masks individual lanes.
//
// Optional attributes:
// - `use_mradix`: Use the MRADIX hardware unit for the identity region.  It is
//   only meaningful there: a non-identity region is always counted by MRADIX.
//   Requires an i32 key, an i32 result and num_bins in (0, 32].
def MathExt_HistogramOp : MathExt_Op<"histogram", [
    MemoryEffectsOpInterface, SingleBlock]> {
  let summary = "Histogram";
  let description = [{
    Counts the value produced by `body` for every valid key element and writes
    the per-bin counts to `result`.  See the comment above for the exact
    semantics of `num_bins`, `size_in`, `mask` and `use_mradix`.
  }];
  let arguments = (ins
    AnyMemRef:$result,
    AnyMemRef:$key,
    I32:$num_bins,
    I32:$size_in,
    Optional<AnyMemRef>:$mask,
    OptionalAttr<UnitAttr>:$use_mradix
  );
  let regions = (region SizedRegion<1>:$body);
  let extraClassDeclaration = [{
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
        effects) {
      effects.emplace_back(MemoryEffects::Write::get(),
                           &getOperation()->getOpOperand(0),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(),
                           &getOperation()->getOpOperand(1),
                           SideEffects::DefaultResource::get());
      if (getMask())
        effects.emplace_back(MemoryEffects::Read::get(),
                             &getOperation()->getOpOperand(4),
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

//
// Yield Op
//
def MathExt_YieldOp : MathExt_Op<"yield", [
    ParentOneOf<["HistogramOp"]>, Pure, Terminator]> {
  let summary = "terminator for a MathExt region";
  let arguments = (ins Variadic<AnyType>:$operands);
}
#endif // MATH_EXT_OPS
