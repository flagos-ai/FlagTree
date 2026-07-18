#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include <cctype>
#include <limits>

#include "tle/dialect/include/IR/VerfiyUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include <iostream>

namespace mlir::triton::tle {
namespace RemotePointers {
llvm::LogicalResult verifyDeviceSpace(mlir::Value src, mlir::Value result) {
  // flagcxGetIntraPointerC accept raw device pointers represented as signless
  // i64 values.
  if (!src.getType().isSignlessInteger(64))
    return failure();

  if (auto tensorTy = dyn_cast<RankedTensorType>(result.getType())) {
    auto ptr = dyn_cast<triton::PointerType>(tensorTy.getElementType());
    if (!ptr)
      return failure();
    return success();
  }
  return success();
}
} // namespace RemotePointers

} // namespace mlir::triton::tle
