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

#include "mlir/Pass/Pass.h"
#include "third_party/proton/Dialect/include/Analysis/ScopeIdAllocation.h"

using namespace mlir;
using namespace triton::proton;

namespace {

struct TestScopeIdAllocationPass
    : public PassWrapper<TestScopeIdAllocationPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestScopeIdAllocationPass);

  TestScopeIdAllocationPass() = default;
  TestScopeIdAllocationPass(const TestScopeIdAllocationPass &other)
      : PassWrapper<TestScopeIdAllocationPass, OperationPass<ModuleOp>>(other) {
  }

  StringRef getArgument() const final {
    return "test-print-scope-id-allocation";
  }
  StringRef getDescription() const final {
    return "print the result of the scope id allocation pass";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    // Convert to std::string can remove quotes from opName
    ModuleScopeIdAllocation moduleScopeIdAllocation(moduleOp);
    moduleOp.walk([&](triton::FuncOp funcOp) {
      auto opName = SymbolTable::getSymbolName(funcOp).getValue().str();
      mlir::emitRemark(funcOp.getLoc(), opName);
      llvm::DenseMap<ScopeIdAllocation::ScopeId, ScopeIdAllocation::ScopeId>
          parentScopeIdMap;
      for (auto [childId, parentId] :
           moduleScopeIdAllocation.getScopeIdParents(funcOp)) {
        parentScopeIdMap.insert({childId, parentId});
      }
      funcOp.walk([&](RecordOp recordOp) {
        auto scopeId = moduleScopeIdAllocation.getOpScopeId(recordOp);
        mlir::emitRemark(recordOp.getLoc()) << "scope id = " << scopeId;
        int64_t parentId = -1;
        if (auto parentIt = parentScopeIdMap.find(scopeId);
            parentIt != parentScopeIdMap.end())
          parentId = parentIt->second;
        mlir::emitRemark(recordOp.getLoc()) << "scope parent id = " << parentId;
      });
    });
  }
};

} // namespace

namespace mlir {
namespace test {
namespace proton {
void registerTestScopeIdAllocationPass() {
  PassRegistration<TestScopeIdAllocationPass>();
}
} // namespace proton
} // namespace test
} // namespace mlir
