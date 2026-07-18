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

#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {
SmallVector<scf::ForOp> getLeafForOps(triton::FuncOp funcOp) {
  SmallVector<scf::ForOp> allOps;
  funcOp->walk([&](scf::ForOp forOp) { allOps.push_back(forOp); });

  SmallVector<scf::ForOp> leafOps;
  for (scf::ForOp forOp : allOps) {
    auto searchResult = forOp.getBody()->walk(
        [](scf::ForOp) { return WalkResult::interrupt(); });
    if (!searchResult.wasInterrupted())
      leafOps.push_back(forOp);
  }
  return leafOps;
}

SmallVector<unsigned> getShapePerCTATile(RankedTensorType tensorTy) {
  auto llEnc = triton::gpu::toLinearEncoding(tensorTy);
  auto sizePerThread = llEnc.getSizePerThread();
  auto threadsPerWarp = llEnc.getThreadsPerWarp();
  auto warpsPerCTA = llEnc.getWarpsPerCTA();
  SmallVector<unsigned> shape;
  for (auto [size, thread, warp] :
       llvm::zip(sizePerThread, threadsPerWarp, warpsPerCTA)) {
    shape.push_back(size * thread * warp);
  }
  return shape;
}

ElemLocationKey getElemCoordinatesFromRegisters(triton::LinearLayout ll,
                                                unsigned regId,
                                                MLIRContext *ctx) {
  StringAttr kReg = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  StringAttr kBlock = StringAttr::get(ctx, "block");

  SmallVector<std::pair<StringAttr, int32_t>> hardwareLocation = {
      {kReg, static_cast<int32_t>(regId)},
      {kLane, 0},
      {kWarp, 0},
      {kBlock, 0},
  };

  return ll.apply(hardwareLocation);
}

std::optional<int> getRegFromCoordinates(triton::LinearLayout ll,
                                         ElemLocationKey coordinates,
                                         MLIRContext *ctx) {
  auto dims = ll.pseudoinvert().apply(coordinates);
  StringAttr kReg = StringAttr::get(ctx, "register");
  assert(dims[0].first == kReg && "First dimension must be 'register'");

  int regId = dims[0].second; // "register"
  if (dims[1].second != 0 || dims[2].second != 0 || dims[3].second != 0)
    return std::nullopt;
  return regId;
}
} // namespace mlir::triton::AMD
