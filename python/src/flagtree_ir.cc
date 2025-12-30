#include "ir.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/FlagTree/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "llvm/ADT/SmallVectorExtras.h"

namespace py = pybind11;

class FlagTreeOpBuilder : public TritonOpBuilder {
public:
  flagtree::DSLRegionOp
  createEdslRegionByLLVMFunc(std::string_view text, std::string_view fnname,
                             const std::vector<Value> &outputs,
                             const std::vector<Value> &inputs);
};

flagtree::DSLRegionOp FlagTreeOpBuilder::createEdslRegionByLLVMFunc(
    std::string_view text, std::string_view fnname,
    const std::vector<Value> &outputs, const std::vector<Value> &inputs) {
  ParserConfig config(getContext());
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(text, config);
  LLVM::LLVMFuncOp func = module->lookupSymbol<LLVM::LLVMFuncOp>(fnname);
  OpBuilder &builder = getBuilder();
  SmallVector<Type> outputTys = llvm::map_to_vector(
      outputs, [](Value value) -> Type { return value.getType(); });
  SmallVector<Value> operands = llvm::to_vector(
      llvm::concat<Value>(SmallVector<Value>(outputs.begin(), outputs.end()),
                          SmallVector<Value>(inputs.begin(), inputs.end())));
  flagtree::DSLRegionOp dslRegionOp =
      create<flagtree::DSLRegionOp>(outputTys, operands);
  OpBuilder::InsertionGuard guard(builder);
  Region &body = dslRegionOp.getBody();
  SmallVector<Type> operandTys = llvm::map_to_vector(
      operands, [](Value value) -> Type { return value.getType(); });
  IRMapping mapper;
  auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(func.getArgument(0).getType());
  uint32_t as = ptrTy.getAddressSpace();
  for (auto [idx, oldBlock] : llvm::enumerate(func.getBlocks())) {
    if (idx == 0) {
      Block *newBlock = builder.createBlock(
          &body, {}, operandTys,
          SmallVector<Location>(operandTys.size(), getLastLoc()));
      SmallVector<Value> extractOps;
      for (const auto &input : body.getArguments()) {
        if (RankedTensorType tensorTy =
                dyn_cast<RankedTensorType>(input.getType())) {
          Type ty = LLVM::LLVMPointerType::get(getContext(), as);
          extractOps.push_back(
              create<flagtree::ExtractAllocatedPtrOp>(ty, input));
          extractOps.push_back(
              create<flagtree::ExtractAlignedPtrOp>(ty, input));
          extractOps.push_back(create<flagtree::ExtractOffsetOp>(input));
          const size_t rank = tensorTy.getRank();
          auto sizesOp = create<flagtree::ExtractSizesOp>(rank, input);
          auto stridesOp = create<flagtree::ExtractStridesOp>(rank, input);
          for (const auto &result : sizesOp.getResults()) {
            extractOps.push_back(result);
          }
          for (const auto &result : stridesOp.getResults()) {
            extractOps.push_back(result);
          }
        } else {
          extractOps.push_back(input);
        }
        for (auto [funcArg, extractOp] :
             llvm::zip(func.getArguments(), extractOps)) {
          mapper.map(funcArg, extractOp);
        }
      }
      mapper.map(&oldBlock, newBlock);
    } else {
      Block *newBlock = builder.createBlock(
          &body, {}, oldBlock.getArgumentTypes(),
          SmallVector<Location>(oldBlock.getNumArguments(), getLastLoc()));
      for (auto [oldArg, newArg] :
           llvm::zip(oldBlock.getArguments(), newBlock->getArguments())) {
        mapper.map(oldArg, newArg);
      }
      mapper.map(&oldBlock, newBlock);
    }
  }
  for (auto [oldBlock, newBlock] :
       llvm::zip(func.getBlocks(), body.getBlocks())) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&newBlock);
    for (Operation &operation : oldBlock.getOperations()) {
      if (LLVM::ReturnOp returnOp = dyn_cast<LLVM::ReturnOp>(operation)) {
        SmallVector<Value> yields;
        if (dslRegionOp.getNumResults() == 1) {
          flagtree::PackOp packOp = builder.create<flagtree::PackOp>(
              operation.getLoc(), dslRegionOp.getResult(0).getType(),
              mapper.lookup(returnOp.getArg()));
          yields.push_back(packOp.getOutput());
        } else {
          for (auto [idx, result] : llvm::enumerate(dslRegionOp.getResults())) {
            LLVM::ExtractValueOp operand = builder.create<LLVM::ExtractValueOp>(
                operation.getLoc(), mapper.lookup(returnOp.getArg()),
                SmallVector<int64_t>{static_cast<int64_t>(idx)});
            flagtree::PackOp packOp = builder.create<flagtree::PackOp>(
                operation.getLoc(), result.getType(), operand);
            yields.push_back(packOp.getOutput());
          }
        }
        builder.create<flagtree::YieldOp>(operation.getLoc(), yields);
      } else {
        builder.clone(operation, mapper);
      }
    }
  }
  return dslRegionOp;
}

void init_flagtree_ir(py::module &&m) {
  using ret = py::return_value_policy;

  py::class_<flagtree::DSLRegionOp>(m, "DSLRegionOp", py::module_local(),
                                    py::dynamic_attr())
      .def(
          "get_results",
          [](flagtree::DSLRegionOp &op) -> std::vector<OpResult> {
            auto results_range = op->getResults();
            return std::vector<OpResult>(results_range.begin(),
                                         results_range.end());
          },
          ret::reference)
      .def("dump", &flagtree::DSLRegionOp::dump);

  py::class_<flagtree::YieldOp>(m, "YieldOp", py::module_local(),
                                py::dynamic_attr())
      .def("dump", &flagtree::YieldOp::dump);

  py::class_<FlagTreeOpBuilder, TritonOpBuilder>(
      m, "FlagTreeOpBuilder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *>())
      .def("get_op_builder", &FlagTreeOpBuilder::getBuilder, ret::reference)
      .def("create_edsl_region_by_llvm_func",
           &FlagTreeOpBuilder::createEdslRegionByLLVMFunc);
}
