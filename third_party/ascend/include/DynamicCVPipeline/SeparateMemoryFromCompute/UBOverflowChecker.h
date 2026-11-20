/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ADAPTER_UB_OVERFLOW_CHECKER_H
#define TRITON_ADAPTER_UB_OVERFLOW_CHECKER_H

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>

namespace mlir {
namespace triton {

struct BufferInfo {
  enum class Kind {
    Annot,
    Unannot,
  };

  Kind kind = Kind::Unannot;
  memref::AllocOp allocOp = nullptr;
  annotation::MarkOp markOp = nullptr;
  int64_t originalSize = 0;
  int64_t reducedSize = 0;
  int64_t alignedSize = 0;
  int64_t multiBufferCount = 1;
  scf::ForOp forOp = nullptr;
  bool fromHint = false;
};

struct TensorInfo {
  tensor::EmptyOp emptyOp = nullptr;
  int64_t originalSize = 0;
  int64_t reducedSize = 0;
  int64_t alignedSize = 0;
};

struct UBEstimateResult {
  int64_t totalBits = 0;
};

namespace UBConstants {

/// UB total space in bits (248 KB).
inline constexpr int64_t UB_SPACE_SIZE_BITS = 2031616;

/// TileAndBindSubBlock halves dim0.
inline constexpr int64_t K_SUB_BLOCK_DIM = 2;

/// EnableStrideAlign align-to-32-bytes.
inline constexpr int64_t STRIDE_ALIGN_BYTES = 32;

/// 256-bit align-up unit.
inline constexpr int64_t ALIGN_UNIT_BITS = 256;

} // namespace UBConstants

int getAlignUnit(Type elementType);

SmallVector<BufferInfo> collectBuffers(ModuleOp module);

SmallVector<TensorInfo> collectTensorEmpties(ModuleOp module);

void computeBufferSize(BufferInfo &buf);

void computeTensorSize(TensorInfo &tensor);

UBEstimateResult checkUBOverflow(ArrayRef<BufferInfo> buffers,
                                 ArrayRef<TensorInfo> tensors);

LogicalResult pruneMultiBufferMarks(ModuleOp module);

class UBOverflowCheckerPass
    : public PassWrapper<UBOverflowCheckerPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UBOverflowCheckerPass)

  UBOverflowCheckerPass() = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const final { return "ub-overflow-check"; }

  llvm::StringRef getDescription() const final {
    return "Estimate UB usage and prune multi_buffer marks if overflow "
           "detected";
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createUBOverflowCheckerPass();
void registerUBOverflowCheckerPasses();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_UB_OVERFLOW_CHECKER_H
