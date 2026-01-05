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
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace py = pybind11;

class FlagTreeOpBuilder : public TritonOpBuilder {
public:
  flagtree::DSLRegionOp
  createEdslRegionByLLVMFunc(std::string_view text, std::string_view fnname,
                             const std::vector<Value> &outputs,
                             const std::vector<Value> &inputs,
                             const std::vector<std::string> &arg_type_hints);
};

flagtree::DSLRegionOp FlagTreeOpBuilder::createEdslRegionByLLVMFunc(
    std::string_view text, std::string_view fnname,
    const std::vector<Value> &outputs, const std::vector<Value> &inputs,
    const std::vector<std::string> &arg_type_hints) {
  ParserConfig config(getContext());
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(text, config);
  LLVM::LLVMFuncOp func = module->lookupSymbol<LLVM::LLVMFuncOp>(fnname);
  OpBuilder &builder = getBuilder();
  SmallVector<Type> outputTys = llvm::map_to_vector(
      outputs, [](Value value) -> Type { return value.getType(); });
  SmallVector<Value> operands = llvm::to_vector(
      llvm::concat<Value>(SmallVector<Value>(outputs.begin(), outputs.end()),
                          SmallVector<Value>(inputs.begin(), inputs.end())));
  
  // Step 1: Convert arg_type_hints to ArrayAttr
  ArrayAttr typeHintsAttr = nullptr;
  if (!arg_type_hints.empty()) {
    SmallVector<Attribute> typeHintAttrs;
    for (const auto &typeHint : arg_type_hints) {
      typeHintAttrs.push_back(StringAttr::get(getContext(), typeHint));
    }
    typeHintsAttr = ArrayAttr::get(getContext(), typeHintAttrs);
  }
  
  // Step 2: Create DSLRegionOp with type hints attribute
  SmallVector<NamedAttribute> attrs;
  if (typeHintsAttr) {
    attrs.push_back(NamedAttribute(
        StringAttr::get(getContext(), "arg_type_hints"), typeHintsAttr));
  }
  
  flagtree::DSLRegionOp dslRegionOp =
      create<flagtree::DSLRegionOp>(outputTys, operands, attrs);
  
  // Debug: Print arg_type_hints
  llvm::errs() << "[DEBUG] arg_type_hints size: " << arg_type_hints.size() << "\n";
  for (const auto &typeHint : arg_type_hints) {
    llvm::errs() << "[DEBUG] typeHint: " << typeHint << "\n";
  }
  if (typeHintsAttr) {
    llvm::errs() << "[DEBUG] typeHintsAttr created, size: " << typeHintsAttr.size() << "\n";
  }
  
  OpBuilder::InsertionGuard guard(builder);
  Region &body = dslRegionOp.getBody();
  SmallVector<Type> operandTys = llvm::map_to_vector(
      operands, [](Value value) -> Type { return value.getType(); });
  IRMapping mapper;
  
  // Step 3: Get arg_type_hints from attribute (rename to arg_types for clarity)
  SmallVector<std::string> arg_types;
  if (auto attr = dslRegionOp->getAttrOfType<ArrayAttr>("arg_type_hints")) {
    for (auto typeHintAttr : attr) {
      if (auto strAttr = dyn_cast<StringAttr>(typeHintAttr)) {
        arg_types.push_back(strAttr.str());
      }
    }
  }
  
  llvm::errs() << "[DEBUG] Extracted arg_types from attribute, size: " << arg_types.size() << "\n";
  
  // Step 4: Create extract operations in block 0, mapping DSLRegionOp operands to LLVM func args
  // We iterate through DSLRegionOp operands (which are outputs + inputs) and match them with
  // LLVM function arguments based on type hints
  uint32_t llvm_arg_idx = 0;  // Track position in LLVM function arguments
  SmallVector<Value> extractOps;  // Collect all extract operations for mapping
  
  for (auto [idx, oldBlock] : llvm::enumerate(func.getBlocks())) {
    if (idx == 0) {
      // Create the first block with operands as block arguments
      Block *newBlock = builder.createBlock(
          &body, {}, operandTys,
          SmallVector<Location>(operandTys.size(), getLastLoc()));
      
      builder.setInsertionPointToStart(newBlock);
      
      // Step 4: Iterate through DSLRegionOp operands and create extract operations
      // Note: operands = [outputs..., inputs...], arg_types corresponds to all operands
      for (size_t operand_idx = 0; operand_idx < operands.size(); ++operand_idx) {
        Value operand = operands[operand_idx];
        Value blockArg = newBlock->getArgument(operand_idx);
        std::string arg_type = operand_idx < arg_types.size() ? arg_types[operand_idx] : "";
        
        llvm::errs() << "[DEBUG] Processing operand " << operand_idx 
                     << ", TT type: " << blockArg.getType() 
                     << ", arg_type hint: " << arg_type << "\n";
        
        // Case 1: TT Tensor -> EDSL expects memref -> LLVM should have 5 args (3 + 2*rank)
        if (RankedTensorType tensorTy = dyn_cast<RankedTensorType>(blockArg.getType())) {
          const size_t rank = tensorTy.getRank();
          size_t expected_llvm_args = 3 + 2 * rank;  // allocated_ptr, aligned_ptr, offset, sizes..., strides...
          
          // Verify we have enough LLVM arguments
          if (llvm_arg_idx + expected_llvm_args > func.getNumArguments()) {
            llvm::errs() << "[ERROR] Not enough LLVM arguments for tensor. "
                         << "Expected " << expected_llvm_args << " args starting at index " << llvm_arg_idx
                         << ", but only " << (func.getNumArguments() - llvm_arg_idx) << " remaining\n";
            assert(false && "Not enough LLVM arguments for tensor");
          }
          
          // Get address space from the first LLVM argument (allocated_ptr)
          uint32_t as = cast<LLVM::LLVMPointerType>(func.getArgument(llvm_arg_idx).getType()).getAddressSpace();
          Type ptrTy = LLVM::LLVMPointerType::get(getContext(), as);
          
          llvm::errs() << "[DEBUG] Tensor case: rank=" << rank 
                       << ", expected_llvm_args=" << expected_llvm_args
                       << ", address_space=" << as << "\n";
          
          // Create extract operations for tensor components
          size_t extract_start_idx = extractOps.size();
          extractOps.push_back(create<flagtree::ExtractAllocatedPtrOp>(ptrTy, blockArg));
          extractOps.push_back(create<flagtree::ExtractAlignedPtrOp>(ptrTy, blockArg));
          extractOps.push_back(create<flagtree::ExtractOffsetOp>(blockArg));
          
          auto sizesOp = create<flagtree::ExtractSizesOp>(rank, blockArg);
          auto stridesOp = create<flagtree::ExtractStridesOp>(rank, blockArg);
          for (const auto &result : sizesOp.getResults()) {
            extractOps.push_back(result);
          }
          for (const auto &result : stridesOp.getResults()) {
            extractOps.push_back(result);
          }
          
          // Map LLVM function arguments to extract operation results
          for (size_t i = 0; i < expected_llvm_args; ++i) {
            mapper.map(func.getArgument(llvm_arg_idx + i), extractOps[extract_start_idx + i]);
          }
          
          llvm_arg_idx += expected_llvm_args;
          
        // Case 2: TT Pointer -> EDSL expects llvm.ptr -> LLVM should have 1 arg
        } else if (auto ptrTy = dyn_cast<triton::PointerType>(blockArg.getType())) {
          size_t expected_llvm_args = 1;
          
          // Verify we have enough LLVM arguments
          if (llvm_arg_idx + expected_llvm_args > func.getNumArguments()) {
            llvm::errs() << "[ERROR] Not enough LLVM arguments for pointer. "
                         << "Expected " << expected_llvm_args << " args starting at index " << llvm_arg_idx
                         << ", but only " << (func.getNumArguments() - llvm_arg_idx) << " remaining\n";
            assert(false && "Not enough LLVM arguments for pointer");
          }
          
          // Get address space from LLVM argument
          uint32_t as = cast<LLVM::LLVMPointerType>(func.getArgument(llvm_arg_idx).getType()).getAddressSpace();
          Type llvmPtrTy = LLVM::LLVMPointerType::get(getContext(), as);
          
          llvm::errs() << "[DEBUG] Pointer case: expected_llvm_args=" << expected_llvm_args
                       << ", address_space=" << as << "\n";
          
          // Create extract operation for pointer
          extractOps.push_back(create<flagtree::ExtractPtrOp>(llvmPtrTy, blockArg));
          
          // Map LLVM function argument to extract operation result
          mapper.map(func.getArgument(llvm_arg_idx), extractOps.back());
          
          llvm_arg_idx += expected_llvm_args;
          
        // Case 3: Scalar (i32, i64, etc.) -> EDSL expects same type -> LLVM should have 1 arg
        } else {
          size_t expected_llvm_args = 1;
          
          // Verify we have enough LLVM arguments
          if (llvm_arg_idx + expected_llvm_args > func.getNumArguments()) {
            llvm::errs() << "[ERROR] Not enough LLVM arguments for scalar. "
                         << "Expected " << expected_llvm_args << " args starting at index " << llvm_arg_idx
                         << ", but only " << (func.getNumArguments() - llvm_arg_idx) << " remaining\n";
            assert(false && "Not enough LLVM arguments for scalar");
          }
          
          llvm::errs() << "[DEBUG] Scalar case: type=" << blockArg.getType()
                       << ", expected_llvm_args=" << expected_llvm_args << "\n";
          
          // For scalars, use the block argument directly (no extract needed)
          extractOps.push_back(blockArg);
          
          // Map LLVM function argument to block argument
          mapper.map(func.getArgument(llvm_arg_idx), blockArg);
          
          llvm_arg_idx += expected_llvm_args;
        }
      }
      
      // Verify we consumed all LLVM function arguments
      if (llvm_arg_idx != func.getNumArguments()) {
        llvm::errs() << "[WARNING] Mismatch in LLVM argument count. "
                     << "Consumed " << llvm_arg_idx << " args, but function has " << func.getNumArguments() << " args\n";
      }
      
      mapper.map(&oldBlock, newBlock);
    } else {
      // For other blocks, just map block arguments
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
