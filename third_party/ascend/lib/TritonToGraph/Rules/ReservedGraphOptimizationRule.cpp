/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToGraph/GraphOptimizationRule.h"

#include "llvm/Support/Debug.h"

#include <memory>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

class ReservedGraphOptimizationRule final : public GraphOptimizationRule {
public:
  explicit ReservedGraphOptimizationRule(
      ReservedGraphOptimizationRuleOptions options)
      : options(options) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] registered graph optimization rule "
               << static_cast<GraphOptimizationRuleMask>(options.id) << " ("
               << getGraphOptimizationRuleName(options.id)
               << ") namespace=" << options.optionNamespace << "\n");
  }

  GraphOptimizationRuleId getId() const override { return options.id; }

  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::None;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    static_cast<void>(context);
    static_cast<void>(plans);
    ++rejectedUnimplemented;
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] graph optimization rule "
                            << getGraphOptimizationRuleName(options.id)
                            << " namespace=" << options.optionNamespace
                            << " candidates=0 rejected.unimplemented="
                            << rejectedUnimplemented << "\n");
    return success();
  }

private:
  ReservedGraphOptimizationRuleOptions options;
  unsigned rejectedUnimplemented = 0;
};

} // namespace

std::unique_ptr<GraphOptimizationRule> cfg::createReservedGraphOptimizationRule(
    ReservedGraphOptimizationRuleOptions options) {
  return std::make_unique<ReservedGraphOptimizationRule>(options);
}
