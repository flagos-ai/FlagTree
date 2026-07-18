//===- VerfiyUtils.h - Verfiy utils for TLE dialect -----------------------===//

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

namespace mlir::triton::tle {

namespace RemotePointers {
llvm::LogicalResult verifyDeviceSpace(mlir::Value src, mlir::Value result);
}

} // namespace mlir::triton::tle
