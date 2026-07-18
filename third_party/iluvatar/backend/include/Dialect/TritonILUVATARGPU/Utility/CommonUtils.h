// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITON_THIRD_PARTY_ILUVATAR_INCLUDE_DIALECT_TRITONILUVATARGPU_UTILITY_COMMONUTILS_H_
#define TRITON_THIRD_PARTY_ILUVATAR_INCLUDE_DIALECT_TRITONILUVATARGPU_UTILITY_COMMONUTILS_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir::triton::ILUVATAR {
using ElemLocationKey = SmallVector<std::pair<StringAttr, int32_t>>;

SmallVector<scf::ForOp> getLeafForOps(triton::FuncOp funcOp);

// [FIXME LL] Kill this function
SmallVector<unsigned> getShapePerCTATile(RankedTensorType tensorTy);

// Build element coordinates for a given register ID.
// All other hardware dimensions (lane, warp, block) are set to 0.
ElemLocationKey getElemCoordinatesFromRegisters(LinearLayout ll, unsigned regId,
                                                MLIRContext *ctx);

// Extract register ID from element coordinates.
// Returns std::nullopt if non-register dimensions are non-zero.
std::optional<int> getRegFromCoordinates(LinearLayout ll,
                                         ElemLocationKey coordinates,
                                         MLIRContext *ctx);

} // namespace mlir::triton::ILUVATAR

namespace mlir::LLVM::ILUVATAR {

struct DotChainInfo {
  bool isHeadDot = false;
  bool useAsA = false;
  bool useAsB = false;
  bool isTailDot = false;
  bool defAsA = false;
  bool defAsB = false;
};

// Analyze chain-dot relationships, crossing scf.for loop boundaries (split-K
// FA, etc.).
void analyzeDotChain(mlir::triton::DotOpInterface dotOp, DotChainInfo &info);

// Check if the result of this tl.dot is used as opA or opB of another tl.dot.
bool isChainDotHead(mlir::triton::DotOpInterface dotOp, unsigned opIdx = 0);

// Check if an operand of this tl.dot comes from another tl.dot.
bool isChainDotTail(mlir::triton::DotOpInterface dotOp);

} // namespace mlir::LLVM::ILUVATAR

#endif // TRITON_THIRD_PARTY_ILUVATAR_INCLUDE_DIALECT_TRITONILUVATARGPU_UTILITY_COMMONUTILS_H_
