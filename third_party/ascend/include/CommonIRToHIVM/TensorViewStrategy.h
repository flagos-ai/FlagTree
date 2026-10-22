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

#ifndef COMMONIR_TO_HIVM_TENSORVIEW_STRATEGY_H
#define COMMONIR_TO_HIVM_TENSORVIEW_STRATEGY_H

#include "TensorViewAccessAnalysis.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace triton {

enum class TensorViewStrategy {
  Unknown,
  ScalarTransfer,
  PerIndexPayloadDMA,
  SingleAffineDMA,
};

struct TensorViewPlan {
  TensorViewStrategy strategy = TensorViewStrategy::ScalarTransfer;

  // Reserved for the unimplemented SingleAffineDMA strategy.
  int64_t indexStep = 0;
  bool isScalarTransfer() {
    return strategy == TensorViewStrategy::ScalarTransfer;
  }
  bool isPerIndexPayloadDMA() {
    return strategy == TensorViewStrategy::PerIndexPayloadDMA;
  }
  bool isSingleAffineDMA() {
    return strategy == TensorViewStrategy::SingleAffineDMA;
  }
};

FailureOr<TensorViewPlan>
chooseTensorViewPlan(const TensorViewAccessInfo &view,
                     const GatherScatterAccessInfo &access);

} // namespace triton
} // namespace mlir

#endif // COMMONIR_TO_HIVM_TENSORVIEW_STRATEGY_H
