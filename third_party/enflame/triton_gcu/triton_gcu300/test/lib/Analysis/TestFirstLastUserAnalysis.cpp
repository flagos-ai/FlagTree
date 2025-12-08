/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <memory>

#include "Analysis/FirstLastUserAnalysis.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

namespace {

using namespace mlir;

struct TestFirstLastUserAnalysisPass
    : public PassWrapper<TestFirstLastUserAnalysisPass,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFirstLastUserAnalysisPass)

  StringRef getArgument() const final {
    return "test-first-last-user-analysis";
  }
  StringRef getDescription() const final {
    return "Test first last user analysis results.";
  }
  void runOnOperation() override {
    triton::gcu::FirstLastUserAnalysis &userAnalysis =
        getAnalysis<triton::gcu::FirstLastUserAnalysis>();

    Operation *moduleOp = getOperation();
    llvm::raw_ostream &os = llvm::errs();

    auto moduleTag = moduleOp->getAttrOfType<StringAttr>("test_tag");
    if (!moduleTag) {
      os << "No test_tag attribute found in module op.\n";
      return;
    }
    os << "test_tag: " << moduleTag.getValue() << "\n";

    moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag || op->getResults().empty())
        return;
      auto result = userAnalysis.getLastUserOp(op);
      os << tag.getValue() << " -> ";
      if (result) {
        auto resultTag = result->getAttrOfType<StringAttr>("tag");
        if (resultTag) {
          os << result->getAttrOfType<StringAttr>("tag").getValue() << "\n";
        } else {
          os << "<unknown>";
        }
      } else {
        os << "<unknown>";
      }
    });
  }
};

} // namespace

namespace mlir {
namespace test {

std::unique_ptr<::mlir::Pass> createTestFirstLastUserAnalysisPass() {
  return std::make_unique<TestFirstLastUserAnalysisPass>();
}

void registerTestFirstLastUserAnalysisPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createTestFirstLastUserAnalysisPass();
  });
}
} // namespace test
} // namespace mlir
