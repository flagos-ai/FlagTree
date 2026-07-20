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

#include "triton/Dialect/TritonGPU/Transforms/LayoutPropagationUtility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <optional>
#include <utility>

namespace mlir::triton::gpu {

std::optional<std::pair<triton::LoadOp, LinearLayout>>
inferSourceLoadLayout(const LinearLayout &dstLayout, Operation *defOp) {
  if (!defOp)
    return std::nullopt;
  return inferSourceLoadLayout(
      LinearEncodingAttr::get(defOp->getContext(), dstLayout), defOp);
}

std::optional<std::pair<triton::LoadOp, LinearLayout>>
inferSourceLoadLayout(LinearEncodingAttr dstLayout, Operation *defOp) {
  Attribute curLayout = dstLayout;
  Operation *curOp = defOp;
  while (curOp) {
    if (isa<triton::LoadOp>(curOp))
      break; // Found the load op; we are done here.

    if (auto cvtOp = dyn_cast<ConvertLayoutOp>(curOp)) {
      // For convert op we keep the current layout to push through further.
      curOp = cvtOp.getSrc().getDefiningOp();
    } else {
      if (curOp->getNumOperands() != 1)
        break;
      curLayout = inferSrcEncoding(curOp, curLayout);
      curOp = curOp->getOperand(0).getDefiningOp();
    }
  }
  auto loadOp = dyn_cast_or_null<triton::LoadOp>(curOp);
  if (!loadOp)
    return std::nullopt;
  auto loadType = dyn_cast<RankedTensorType>(loadOp.getType());
  if (!loadType)
    return std::nullopt;

  return std::make_pair(
      loadOp,
      toLinearLayout(loadType.getShape(), cast<LinearEncodingAttr>(curLayout)));
}

} // namespace mlir::triton::gpu
