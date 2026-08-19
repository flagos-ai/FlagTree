/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_WARPSPECIALIZEUTILITY_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_WARPSPECIALIZEUTILITY_H

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"
#include <functional>

namespace mlir {
namespace triton {

// Forward declaration
class TritonLLVMIRRewriter;

//===----------------------------------------------------------------------===//
// convertOpTypes
//===----------------------------------------------------------------------===//

/// Convert operand types, region argument types, and result types of a
/// an operation using the provided type converter. This is used for
/// WarpSpecializeOp and related operations during lowering to LLVM.
void convertOpTypes(Operation *op, const TypeConverter &typeConverter);

//===----------------------------------------------------------------------===//
// elideTrivialCaptures
//===----------------------------------------------------------------------===//

/// Attempt to eliminate captures by rematerializing trivial computations into
/// each partition region.
void elideTrivialCaptures(LLVM::LLVMFuncOp func,
                          ArrayRef<gpu::WarpSpecializeOp> wsOps);

//===----------------------------------------------------------------------===//
// lowerWarpSpecializeCommon
//===----------------------------------------------------------------------===//

/// Phase indicator for register reallocation during warp specialization.
enum class RegisterReallocPhase {
  SwitchLoopStart,       // Reallocate at the beginning of switch loop
  WorkerPartitionStart,  // Reallocate at worker partition region start
  WorkerPartitionEnd,    // Reallocate at worker partition region end
  DefaultPartitionStart, // Reallocate at default partition region start
  DefaultPartitionEnd    // Reallocate at default partition region end
};

/// Callbacks for backend-specific operations during warp specialization
/// lowering.
struct WarpSpecializeCallbacks {
  /// Create a barrier to synchronize threads across the whole CTA
  std::function<void(TritonLLVMIRRewriter &, unsigned barIdx)> createAllBarrier;

  /// Reallocate registers.
  /// regionNumber is only used for WorkerPartitionStart and WorkerPartitionEnd
  /// phases.
  std::function<void(TritonLLVMIRRewriter &, gpu::WarpSpecializeOp,
                     RegisterReallocPhase, unsigned regionNumber)>
      reallocRegisters;
};

/// Common implementation of warp specialize lowering.
/// Uses callbacks for backend-specific barrier and register reallocation
/// operations.
LogicalResult lowerWarpSpecializeCommon(
    LLVM::LLVMFuncOp func, ArrayRef<gpu::WarpSpecializeOp> wsOps, Block *entry,
    Block *header, Block *switchLoop, Value wid, MLIRContext *ctx,
    unsigned defaultNumWarps, unsigned totalNumWarps,
    const TargetInfoBase &targetInfo, const WarpSpecializeCallbacks &callbacks,
    unsigned switchLoopBarrierIdx);

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_WARPSPECIALIZEUTILITY_H
