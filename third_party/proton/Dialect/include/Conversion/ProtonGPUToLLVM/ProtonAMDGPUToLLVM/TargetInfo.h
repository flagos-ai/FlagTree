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

#ifndef PROTONGPU_TO_LLVM_TARGETINFO_AMD_H
#define PROTONGPU_TO_LLVM_TARGETINFO_AMD_H

#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h" // TODO(fywkevin): move amd TargetInfo.h to include/
#include <string>

namespace mlir::triton::proton::gpu::AMD {
class TargetInfo : public mlir::triton::proton::gpu::TargetInfoBase {
public:
  explicit TargetInfo(const mlir::triton::AMD::TargetInfo &helper,
                      std::string arch)
      : mlir::triton::proton::gpu::TargetInfoBase(helper),
        arch(std::move(arch)) {}

  const mlir::triton::AMD::TargetInfo &getTritonTargetInfo() const override {
    return static_cast<const mlir::triton::AMD::TargetInfo &>(helper);
  }

  Value clock(ConversionPatternRewriter &rewriter, Location loc,
              bool isClock64) const override;

  Value globalTime(ConversionPatternRewriter &rewriter,
                   Location loc) const override;

  Value processorId(ConversionPatternRewriter &rewriter,
                    Location loc) const override;

  int getAddressSpace(Attribute addressSpace) const override;

  int getIndexPtrAddrSpace() const override;

  ~TargetInfo() = default;

private:
  std::string arch;
};
} // namespace mlir::triton::proton::gpu::AMD

#endif // PROTONGPU_TO_LLVM_TARGETINFO_AMD_H
