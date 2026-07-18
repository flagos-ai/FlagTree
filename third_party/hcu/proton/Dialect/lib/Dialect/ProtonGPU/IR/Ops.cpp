// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "Dialect/ProtonGPU/IR/Ops.cpp.inc"

#include "Dialect/ProtonGPU/IR/OpsEnums.cpp.inc"

namespace mlir {
namespace triton {
namespace proton {
namespace gpu {

// -- CircularRecordOp --
LogicalResult CircularStoreOp::verify() {
  auto scopeId = getScopeId();
  auto segmentType = getSegment().getType();
  auto granularity = segmentType.getGranularity();
  auto selectedIds = segmentType.getSelectIds();
  auto bufferSizeInBytes = segmentType.getNBytes();
  auto mod = getOperation()->getParentOfType<ModuleOp>();

  int numWarps = getTotalNumWarps(mod);

  int segmentNum = selectedIds.empty() ? numWarps : selectedIds.size();
  if (!llvm::isPowerOf2_32(bufferSizeInBytes / segmentNum))
    return emitOpError("profiling buffer segment size must be power of 2");

  if (scopeId < 0 || scopeId > 255)
    return emitOpError("scope id must be in [0, 255]");

  return success();
}

// -- SegmentAllocOp --
LogicalResult SegmentAllocOp::verify() {
  auto segmentType = getSegment().getType();
  auto granularity = segmentType.getGranularity();
  auto selectIds = segmentType.getSelectIds();
  if (granularity != Granularity::WARP && selectIds.size()) {
    return emitOpError(
        "only warp granularity supports non-empty selectIds for now");
  }
  return success();
}

// -- InitCtxOp --
LogicalResult InitCtxOp::verify() {
  if (getOperation()->getParentOfType<triton::gpu::WarpSpecializeOp>())
    return emitOpError(
        "can't initialize proton context in a warp specialized op");
  return success();
}

} // namespace gpu
} // namespace proton
} // namespace triton
} // namespace mlir
