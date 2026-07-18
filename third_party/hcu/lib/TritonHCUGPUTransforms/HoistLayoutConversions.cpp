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

#include "TritonHCUGPUTransforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONHCUGPUHOISTLAYOUTCONVERSIONS
#include "TritonHCUGPUTransforms/Passes.h.inc"

namespace {

// Hoist convert_layout out of the loop if the src is defined out of the loop.
// This is a heuristic driven by optimizing fused attention kernels, in which
// we want to load Q tensor and keep it in register, instead of loading it
// (neither from global or shared memory) at every iteration of the loop.
static void hoistCvtDotOpOutOfLoop(ttg::ConvertLayoutOp cvtOp) {
  // Check the dst of cvt has dotOperand layout
  RankedTensorType rtType = dyn_cast<RankedTensorType>(cvtOp.getType());
  if (!rtType)
    return;
  Attribute encoding = rtType.getEncoding();
  if (!encoding)
    return;
  if (!isa<ttg::DotOperandEncodingAttr>(encoding))
    return;
  // Check the src of cvt is defined out of the loop
  auto srcDefOp = cvtOp.getSrc().getDefiningOp();
  if (srcDefOp) {
    scf::ForOp parentForOp = cvtOp->getParentOfType<scf::ForOp>();
    if (parentForOp && !parentForOp->isAncestor(srcDefOp)) {
      cvtOp->moveAfter(srcDefOp);
    }
  }
}

} // anonymous namespace

struct TritonHCUGPUHoistLayoutConversionsPass
    : public impl::TritonHCUGPUHoistLayoutConversionsBase<
          TritonHCUGPUHoistLayoutConversionsPass> {

  void runOnOperation() override {
    tt::FuncOp funcOp = getOperation();

    SmallVector<ttg::ConvertLayoutOp> cvtOps;
    funcOp.walk([&](ttg::ConvertLayoutOp cvtOp) { cvtOps.push_back(cvtOp); });

    for (auto cvtOp : cvtOps)
      hoistCvtDotOpOutOfLoop(cvtOp);
  }
};

} // namespace mlir
