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

#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_IR_TARGETFEATURES_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_IR_TARGETFEATURES_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <cassert>

namespace mlir::triton::nvidia_gpu {

class TargetFeatures {
public:
  explicit TargetFeatures(int computeCapability)
      : computeCapability(computeCapability) {}

  static TargetFeatures fromModuleOp(ModuleOp moduleOp) {
    auto targetAttr =
        moduleOp->getAttrOfType<StringAttr>(triton::gpu::AttrTargetName);
    assert(targetAttr && "Expected a target attribute on the module operation");

    StringRef targetName = targetAttr.getValue();
    assert(targetName.starts_with(kTargetPrefix) &&
           "expected target attribute to be prefixed with \"cuda:\"");

    int computeCapability;
    bool parseError = targetName.drop_front(sizeof(kTargetPrefix) - 1)
                          .getAsInteger(10, computeCapability);
    assert(!parseError &&
           "invalid compute capability string in target attribute");

    return TargetFeatures(computeCapability);
  }

  int getComputeCapability() const { return computeCapability; }

  bool supportClusterOps() const {
    return computeCapability >= 90 && computeCapability / 10 != 12;
  }

  bool supportMaximumMinimum() const { return computeCapability >= 80; }

  bool supportLdMatrix() const { return computeCapability >= 75; }
  bool supportStMatrix() const { return computeCapability >= 90; }
  bool supportLdStMatrixB8() const { return computeCapability >= 100; }

  bool supportBitwidth16Elementwise() const {
    // Hopper (sm90) and newer.
    return computeCapability >= 90;
  }

  bool supportBitwidth32Elementwise() const {
    // Blackwell (sm100) and newer.
    return computeCapability >= 100;
  }

  bool supportLdRed() const {
    // Blackwell (sm103) and newer, but exclude sm120 and sm121.
    return computeCapability >= 103 && computeCapability / 10 != 12;
  }

  bool supportsI8Tcgen05MMA() const { return computeCapability == 100; }
  bool supportsExclusiveTMEMAlloc() const { return computeCapability == 107; }
  int getMaxTMEMColumns() const {
    return supportsExclusiveTMEMAlloc() ? 576 : 512;
  }
  bool requiresFp4Padding() const {
    return computeCapability == 100 || computeCapability == 103 ||
           computeCapability == 110;
  }
  bool supports4xFp4Tcgen05MMA() const { return computeCapability == 107; }
  bool supports2xFp8Tcgen05MMA() const { return computeCapability == 107; }
  bool supportsReuseB() const { return computeCapability == 107; }
  bool supportsMbarMulticast() const { return computeCapability == 107; }

private:
  static constexpr char kTargetPrefix[] = "cuda:";

  int computeCapability;
};

} // namespace mlir::triton::nvidia_gpu

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_IR_TARGETFEATURES_H_
