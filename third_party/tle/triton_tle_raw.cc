#include "IR/Dialect.h"
#include "ir.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/utils/include/Protocol.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace mlir;
namespace tle = triton::tle;

namespace {
SmallVector<Value> flatten(TritonOpBuilder &builder,
                           const TypedValue<LLVM::LLVMStructType> &val) {
  LLVM::LLVMStructType llvmStructTy = val.getType();
  const size_t rank = llvmStructTy.getBody().size();
  return llvm::map_to_vector(
      llvm::seq(rank), [&builder, &val](int64_t idx) -> Value {
        return builder.create<LLVM::ExtractValueOp>(val, SmallVector{idx});
      });
}
} // namespace

// Create an LLVM::CallOp that wraps an LLVM function, performing type
// conversion from Triton IR types to LLVM types and back using SignaturePattern
// and ReturnPattern.
//
// Overview:
// 1. Parse the LLVM IR text and extract the target function using Triton's MLIR
// context
// 2. Perform argument type conversion: TT IR types -> LLVM types via
// SignaturePattern::apply
//    - Inputs are TT IR types (tensor, pointer, scalar)
//    - SignaturePattern::apply converts each TT type to corresponding LLVM
//    types
//    - Collected operands are passed to LLVM::CallOp
// 3. Create an LLVM::CallOp with converted operands
// 4. Perform return type conversion: LLVM types -> TT IR types via
// ReturnPattern::apply
//    - LLVM::CallOp returns LLVM types
//    - ReturnPattern::apply converts each LLVM return to TT IR type
//
// Example type conversion for tensor:
//   - TT IR: tensor<128xi32> (RankedTensorType)
//   - LLVM func args: allocated_ptr, aligned_ptr, offset, size[0], stride[0]
//   - Conversion: SignaturePattern::apply extracts tensor into LLVM values
//
// Example type conversion for scalar:
//   - TT IR: i32 (IntegerType)
//   - LLVM func: 1 arg = i32
//   - Conversion: SignaturePattern::apply directly passes the scalar value
SmallVector<Value> createTLERawRegionByLLVMFunc(
    TritonOpBuilder &self, std::string_view text, std::string_view fnname,
    const std::vector<Value> &outputs, const std::vector<Value> &inputs) {
  ParserConfig config(self.getContext());
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(text, config);
  LLVM::LLVMFuncOp func = module->lookupSymbol<LLVM::LLVMFuncOp>(fnname);
  OpBuilder &builder = self.getBuilder();
  Operation *curOp = builder.getInsertionBlock()->getParentOp();
  while (curOp && curOp->getParentOp() && !isa<ModuleOp>(curOp)) {
    curOp = curOp->getParentOp();
  }
  ModuleOp curModule = cast<ModuleOp>(curOp);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(curModule.getBody());
    for (Operation &op : module->getOps()) {
      if ((!isa<SymbolOpInterface>(op) ||
           (isa<SymbolOpInterface>(op) &&
            !curModule.lookupSymbol(cast<SymbolOpInterface>(op).getName())))) {
        builder.clone(op);
      }
    }
  };
  LLVM::LLVMFuncOp funcOp = curModule.lookupSymbol<LLVM::LLVMFuncOp>(fnname);
  SmallVector<Value> operands = {};
  TypeRange tgts = func.getArgumentTypes();
  SmallVector<Value> outs = SmallVector<Value>(outputs.begin(), outputs.end()),
                     ins = SmallVector<Value>(inputs.begin(), inputs.end());
  for (Value src : llvm::concat<Value>(outs, ins)) {
    operands.append(tle::protocol::SignaturePattern::apply(self, tgts, src));
  }

  LLVM::CallOp callOp = self.create<LLVM::CallOp>(funcOp, operands);
  callOp.setAlwaysInline(true);
  SmallVector<Value> finalResults;
  tgts = ValueRange(outs).getTypes();
  for (Value result : callOp.getResults()) {
    SmallVector<Value> rets =
        tle::protocol::ReturnPattern::apply(self, tgts, result);
    finalResults.append(std::move(rets));
  }
  return finalResults;
}
