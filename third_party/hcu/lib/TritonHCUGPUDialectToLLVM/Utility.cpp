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

#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;

namespace mlir::LLVM::HCU {

ElemLocationKey getElemCoordinatesFromRegisters(tt::LinearLayout ll,
                                                unsigned regId,
                                                MLIRContext *ctx) {
  StringAttr kReg = StringAttr::get(ctx, "register");
  SmallVector<std::pair<StringAttr, int32_t>> hardwareLocation;
  for (auto dimName : ll.getInDimNames()) {
    if (dimName == kReg)
      hardwareLocation.push_back({dimName, regId});
    else
      hardwareLocation.push_back({dimName, 0});
  }
  return ll.apply(hardwareLocation);
}

std::optional<int> getRegFromCoordinates(tt::LinearLayout ll,
                                         ElemLocationKey coordinates,
                                         MLIRContext *ctx) {
  auto hardwareLocation = ll.pseudoinvert().apply(coordinates);
  llvm::MapVector<ElemLocationKey, unsigned> elemToReg;
  StringAttr kReg = StringAttr::get(ctx, "register");
  for (auto location : hardwareLocation) {
    if (location.first == kReg)
      return location.second;
  }

  return {};
} // namespace mlir::triton

} // namespace mlir::LLVM::HCU
