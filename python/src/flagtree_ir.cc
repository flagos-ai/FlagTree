#include "ir.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/FlagTree/IR/Dialect.h"

namespace py = pybind11;

class FlagTreeOpBuilder : public TritonOpBuilder {
public:
  flagtree::DSLRegionOp
  createEdslRegionByLLVMFunc(std::string_view text, std::string_view fnname,
                             const std::vector<Value> &inputs);
};

flagtree::DSLRegionOp FlagTreeOpBuilder::createEdslRegionByLLVMFunc(
    std::string_view text, std::string_view fnname,
    const std::vector<Value> &inputs) {
  ParserConfig config(getContext());
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(text, config);
  LLVM::LLVMFuncOp func = module->lookupSymbol<LLVM::LLVMFuncOp>(fnname);
  OpBuilder &builder = getBuilder();
  flagtree::DSLRegionOp dslRegionOp = create<flagtree::DSLRegionOp>(inputs);
  OpBuilder::InsertionGuard guard(builder);
  Region &body = dslRegionOp.getBody();
  SmallVector<Type> inputTys;
  for (const Value &input : inputs) {
    inputTys.push_back(input.getType());
  }
  Block *block =
      builder.createBlock(&body, {}, inputTys,
                          SmallVector<Location>(inputTys.size(), getLastLoc()));
  SmallVector<Value> extractOps;
  for (const auto &input : body.getArguments()) {
    if (RankedTensorType tensorTy =
            dyn_cast<RankedTensorType>(input.getType())) {
      extractOps.push_back(create<flagtree::ExtractAllocatedPtrOp>(input));
      extractOps.push_back(create<flagtree::ExtractAlignedPtrOp>(input));
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
  }
  IRMapping mapper;
  for (auto [input, funcArg] : llvm::zip(extractOps, func.getArguments())) {
    mapper.map(funcArg, input);
  }
  for (auto &op : func.getBody().getOps()) {
    if (isa<LLVM::ReturnOp>(op)) {
      builder.create<flagtree::YieldOp>(op.getLoc());
    } else {
      builder.clone(op, mapper);
    }
  }
  return dslRegionOp;
}

void init_flagtree_ir(py::module &&m) {
  using ret = py::return_value_policy;

  py::class_<flagtree::DSLRegionOp>(m, "DSLRegionOp", py::module_local(),
                                    py::dynamic_attr())
      .def("get_operation", &flagtree::DSLRegionOp::getOperation)
      .def("get_body", &flagtree::DSLRegionOp::getBody, ret::reference)
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
