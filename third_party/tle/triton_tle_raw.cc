#include "ir.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "tle/dialect/include/IR/Dialect.h"
#include <optional>
#include <regex>
#include <string>

using namespace mlir;
namespace tle = triton::tle;

// Helper function to parse address space from EDSL parameter type
// Returns the address space if found, or std::nullopt if not found/parse failed
static std::optional<uint32_t>
parseAddressSpaceFromTypeHint(const std::string &typeHint) {
  if (typeHint.empty()) {
    return std::nullopt;
  }

  // Parse memref format: "memref<shape, address_space>" or "!memref<shape,
  // address_space>" Match: optional !, memref<...anything..., address_space>
  std::regex memref_regex(R"(!?memref<[^>]*,\s*(\d+)\s*>)");
  std::smatch memref_match;
  if (std::regex_match(typeHint, memref_match, memref_regex)) {
    try {
      return static_cast<uint32_t>(std::stoul(memref_match[1].str()));
    } catch (...) {
      llvm::errs()
          << "[ERROR] Failed to parse address space from memref param type: "
          << typeHint << "\n";
      return std::nullopt;
    }
  }

  // Parse llvm.ptr format: "llvm.ptr<address_space>" or
  // "!llvm.ptr<address_space>" Match: optional !, llvm.ptr<address_space>
  std::regex ptr_regex(R"(!?llvm\.ptr<(\d+)>)");
  std::smatch ptr_match;
  if (std::regex_match(typeHint, ptr_match, ptr_regex)) {
    try {
      return static_cast<uint32_t>(std::stoul(ptr_match[1].str()));
    } catch (...) {
      llvm::errs()
          << "[ERROR] Failed to parse address space from llvm.ptr param type: "
          << typeHint << "\n";
      return std::nullopt;
    }
  }

  // If type hint is not empty but doesn't match expected formats, report error
  llvm::errs() << "[ERROR] Unsupported EDSL parameter type format: " << typeHint
               << "\n";
  llvm::errs() << "[ERROR] Expected format: \"memref<shape, address_space>\" "
                  "or \"llvm.ptr<address_space>\"\n";
  return std::nullopt;
}

// Create a DSLRegionOp that wraps an LLVM function, performing type conversion
// from Triton IR types to LLVM types based on EDSL function declarations.
//
// Overview:
// 1. Parse the LLVM IR text and extract the target function using Triton's MLIR
// context
// 2. Create a DSLRegionOp with EDSL function parameter types stored in
// attributes
// 3. Perform argument type conversion: TT IR types -> LLVM types (via extract
// operations)
//    - DSLRegionOp's operands are TT IR types (tensor, pointer, scalar)
//    - EDSL function declarations (stored in edsl_param_types attribute)
//    specify expected types
//    - LLVM function arguments are already in LLVM types
//    - We need to verify consistency: TT type -> EDSL param type -> LLVM func
//    arg type
//
// Example type conversion for tensor:
//   - TT IR: tensor<128xi32> (RankedTensorType)
//   - EDSL param type: "memref<?xi32, 3>" (stored in edsl_param_types
//   attribute)
//   - LLVM func: 5 args = allocated_ptr<3>, aligned_ptr<3>, offset, size[0],
//   stride[0]
//   - Conversion: Extract tensor into 5 LLVM values using
//   ExtractAllocatedPtrOp, etc.
//
// Example type conversion for scalar:
//   - TT IR: i32 (IntegerType)
//   - EDSL param type: "i32"
//   - LLVM func: 1 arg = i32
//   - Conversion: Use block argument directly
tle::DSLRegionOp createEdslRegionByLLVMFunc(
    TritonOpBuilder &self, std::string_view text, std::string_view fnname,
    const std::vector<Value> &outputs, const std::vector<Value> &inputs,
    const std::vector<std::string> &arg_type_hints,
    const std::vector<std::string> &arg_names) {
  // Stage 1: Parse LLVM IR and extract function using Triton's MLIR context
  ParserConfig config(self.getContext());
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(text, config);
  LLVM::LLVMFuncOp func = module->lookupSymbol<LLVM::LLVMFuncOp>(fnname);
  OpBuilder &builder = self.getBuilder();

  SmallVector<Type> outputTys = llvm::map_to_vector(
      outputs, [](Value value) -> Type { return value.getType(); });
  SmallVector<Value> operands = llvm::to_vector(
      llvm::concat<Value>(SmallVector<Value>(outputs.begin(), outputs.end()),
                          SmallVector<Value>(inputs.begin(), inputs.end())));

  // Stage 2: Create DSLRegionOp, create new body
  // The edsl_param_types contain the EDSL function parameter type declarations
  // (e.g., "memref<?xi32, 3>", "i32") These are stored as an ArrayAttr of
  // StringAttrs in the DSLRegionOp's "edsl_param_types" attribute The
  // edsl_param_names contain the EDSL function parameter names (e.g.,
  // "sum_buf", "indices") These are stored as an ArrayAttr of StringAttrs in
  // the DSLRegionOp's "edsl_param_names" attribute
  ArrayAttr edslParamTypesAttr = nullptr;
  if (!arg_type_hints.empty()) {
    SmallVector<Attribute> typeAttrs;
    for (const auto &typeHint : arg_type_hints) {
      typeAttrs.push_back(StringAttr::get(self.getContext(), typeHint));
    }
    edslParamTypesAttr = ArrayAttr::get(self.getContext(), typeAttrs);
  }

  ArrayAttr edslParamNamesAttr = nullptr;
  if (!arg_names.empty()) {
    SmallVector<Attribute> nameAttrs;
    for (const auto &name : arg_names) {
      nameAttrs.push_back(StringAttr::get(self.getContext(), name));
    }
    edslParamNamesAttr = ArrayAttr::get(self.getContext(), nameAttrs);
  }

  SmallVector<NamedAttribute> attrs;
  if (edslParamTypesAttr) {
    attrs.push_back(
        NamedAttribute(StringAttr::get(self.getContext(), "edsl_param_types"),
                       edslParamTypesAttr));
  }
  if (edslParamNamesAttr) {
    attrs.push_back(
        NamedAttribute(StringAttr::get(self.getContext(), "edsl_param_names"),
                       edslParamNamesAttr));
  }

  tle::DSLRegionOp dslRegionOp =
      self.create<tle::DSLRegionOp>(outputTys, operands, attrs);
  OpBuilder::InsertionGuard guard(builder);
  Region &body = dslRegionOp.getBody();
  SmallVector<Type> operandTys = llvm::map_to_vector(
      operands, [](Value value) -> Type { return value.getType(); });
  IRMapping mapper;

  // Stage 3: Argument type conversion and and establish mapping relationships
  // Convert TT IR types (DSLRegionOp operands) to LLVM types (LLVM function
  // arguments) For each DSLRegionOp operand:
  //   1. Check its TT IR type (tensor, pointer, scalar)
  //   2. Check the corresponding EDSL type hint from attribute (e.g.,
  //   "memref<?xi32, 3>", "i32")
  //   3. Verify the LLVM function has the expected number and types of
  //   arguments
  //   4. Create extract operations to convert TT IR values to LLVM values
  //   5. Map LLVM function arguments to extract operation results in IRMapping
  //
  // Type conversion examples:
  //   - TT tensor<128xi32> + EDSL "memref<?xi32, 3>" -> LLVM 5 args (ptr<3>,
  //   ptr<3>, offset, size, stride)
  //   - TT ptr<f32> + EDSL param type "llvm.ptr<1>" -> LLVM 1 arg (ptr<1>)
  //   - TT i32 + EDSL "i32" -> LLVM 1 arg (i32, passed through directly)
  uint32_t llvm_arg_idx = 0;     // Track position in LLVM function arguments
  SmallVector<Value> extractOps; // Collect all extract operations for mapping

  for (auto [idx, oldBlock] : llvm::enumerate(func.getBlocks())) {
    if (idx == 0) {
      // Create the first block with operands as block arguments
      Block *newBlock = builder.createBlock(
          &body, {}, operandTys,
          SmallVector<Location>(operandTys.size(), self.getLastLoc()));

      // Set insertion point to start: extract operations will be inserted at
      // the beginning
      builder.setInsertionPointToStart(newBlock);

      // Iterate through DSLRegionOp operands and create extract operations
      // Note: operands = [outputs..., inputs...], edsl_param_types corresponds
      // to all operands Extract edsl_param_names from attribute
      SmallVector<std::string> edsl_param_names_from_attr;
      if (auto edslParamNamesAttr =
              dslRegionOp->getAttrOfType<ArrayAttr>("edsl_param_names")) {
        for (auto nameAttr : edslParamNamesAttr) {
          if (auto strAttr = dyn_cast<StringAttr>(nameAttr)) {
            edsl_param_names_from_attr.push_back(strAttr.str());
          }
        }
      }

      for (size_t operand_idx = 0; operand_idx < operands.size();
           ++operand_idx) {
        Value operand = operands[operand_idx];
        Value blockArg = newBlock->getArgument(operand_idx);
        std::string arg_type = operand_idx < arg_type_hints.size()
                                   ? arg_type_hints[operand_idx]
                                   : "";
        std::string arg_name = operand_idx < edsl_param_names_from_attr.size()
                                   ? edsl_param_names_from_attr[operand_idx]
                                   : "";

        // Case 1: TT Tensor type conversion
        // TT IR: RankedTensorType (e.g., tensor<128xi32>)
        // EDSL param type: "memref<?xi32, 3>" (stored in edsl_param_types
        // attribute) LLVM func: 3 + 2*rank args = allocated_ptr<address_space>,
        // aligned_ptr<address_space>, offset, sizes[rank], strides[rank]
        // Conversion: Create ExtractAllocatedPtrOp, ExtractAlignedPtrOp,
        // ExtractOffsetOp, ExtractSizesOp, ExtractStridesOp
        if (RankedTensorType tensorTy =
                dyn_cast<RankedTensorType>(blockArg.getType())) {
          // Type consistency check: verify EDSL parameter type matches actual
          // Triton type If EDSL param type is "llvm.ptr<...>", it requires
          // triton::PointerType, not RankedTensorType
          if (!arg_type.empty()) {
            std::regex ptr_regex(R"(!?llvm\.ptr<\d+>)");
            if (std::regex_match(arg_type, ptr_regex)) {
              llvm::errs() << "[ERROR] Type mismatch for operand "
                           << operand_idx;
              if (!arg_name.empty()) {
                llvm::errs() << " (parameter: " << arg_name << ")";
              }
              llvm::errs() << "\n";
              llvm::errs() << "[ERROR] EDSL param type: " << arg_type
                           << " (expects triton::PointerType)\n";
              llvm::errs() << "[ERROR] Actual Triton type: "
                           << blockArg.getType() << " (RankedTensorType)\n";
              assert(false && "EDSL type hint mismatch: llvm.ptr requires "
                              "triton::PointerType, but got RankedTensorType");
            }
          }
          const size_t rank = tensorTy.getRank();
          size_t expected_llvm_args =
              3 + 2 * rank; // allocated_ptr, aligned_ptr, offset, sizes...,
                            // strides...

          // Verify we have enough LLVM arguments
          if (llvm_arg_idx + expected_llvm_args > func.getNumArguments()) {
            llvm::errs() << "[ERROR] Not enough LLVM arguments for tensor. "
                         << "Expected " << expected_llvm_args
                         << " args starting at index " << llvm_arg_idx
                         << ", but only "
                         << (func.getNumArguments() - llvm_arg_idx)
                         << " remaining\n";
            assert(false && "Not enough LLVM arguments for tensor");
          }

          // Get address space from the first LLVM argument (allocated_ptr)
          uint32_t llvm_as = cast<LLVM::LLVMPointerType>(
                                 func.getArgument(llvm_arg_idx).getType())
                                 .getAddressSpace();

          // Parse expected address space from EDSL parameter type (e.g.,
          // "memref<?xi32, 3>" -> 3)
          uint32_t expected_as = llvm_as; // Default to LLVM's address space
          if (auto parsed_as = parseAddressSpaceFromTypeHint(arg_type)) {
            expected_as = *parsed_as;

            // Verify address space consistency
            if (expected_as != llvm_as) {
              llvm::errs()
                  << "[ERROR] Address space mismatch for tensor operand "
                  << operand_idx << "\n";
              llvm::errs() << "[ERROR] EDSL hint: " << arg_type
                           << " (address space: " << expected_as << ")\n";
              llvm::errs() << "[ERROR] LLVM func arg address space: " << llvm_as
                           << "\n";
              assert(false && "Address space mismatch");
            }
          }

          Type ptrTy =
              LLVM::LLVMPointerType::get(self.getContext(), expected_as);

          // Create extract operations for tensor components
          size_t extract_start_idx = extractOps.size();
          extractOps.push_back(
              self.create<tle::ExtractAllocatedPtrOp>(ptrTy, blockArg));
          extractOps.push_back(
              self.create<tle::ExtractAlignedPtrOp>(ptrTy, blockArg));
          extractOps.push_back(self.create<tle::ExtractOffsetOp>(blockArg));

          auto sizesOp = self.create<tle::ExtractSizesOp>(rank, blockArg);
          auto stridesOp = self.create<tle::ExtractStridesOp>(rank, blockArg);
          for (const auto &result : sizesOp.getResults()) {
            extractOps.push_back(result);
          }
          for (const auto &result : stridesOp.getResults()) {
            extractOps.push_back(result);
          }

          // Map LLVM function arguments to extract operation results
          for (size_t i = 0; i < expected_llvm_args; ++i) {
            mapper.map(func.getArgument(llvm_arg_idx + i),
                       extractOps[extract_start_idx + i]);
          }

          llvm_arg_idx += expected_llvm_args;

          // Case 2: TT Pointer type conversion
          // TT IR: triton::PointerType (e.g., ptr<f32>)
          // EDSL param type: "llvm.ptr<1>" (stored in edsl_param_types
          // attribute) LLVM func: 1 arg = ptr<address_space> Conversion: Create
          // ExtractPtrOp to convert TT pointer to LLVM pointer
        } else if (auto ptrTy =
                       dyn_cast<triton::PointerType>(blockArg.getType())) {
          // Type consistency check: verify EDSL parameter type matches actual
          // Triton type If EDSL param type is "memref<...>", it requires
          // RankedTensorType, not triton::PointerType
          if (!arg_type.empty()) {
            std::regex memref_regex(R"(!?memref<[^>]*,\s*\d+\s*>)");
            if (std::regex_match(arg_type, memref_regex)) {
              llvm::errs() << "[ERROR] Type mismatch for operand "
                           << operand_idx;
              if (!arg_name.empty()) {
                llvm::errs() << " (parameter: " << arg_name << ")";
              }
              llvm::errs() << "\n";
              llvm::errs() << "[ERROR] EDSL param type: " << arg_type
                           << " (expects RankedTensorType)\n";
              llvm::errs() << "[ERROR] Actual Triton type: "
                           << blockArg.getType() << " (triton::PointerType)\n";
              assert(false && "EDSL type hint mismatch: memref requires "
                              "RankedTensorType, but got triton::PointerType");
            }
          }
          size_t expected_llvm_args = 1;

          // Verify we have enough LLVM arguments
          if (llvm_arg_idx + expected_llvm_args > func.getNumArguments()) {
            llvm::errs() << "[ERROR] Not enough LLVM arguments for pointer. "
                         << "Expected " << expected_llvm_args
                         << " args starting at index " << llvm_arg_idx
                         << ", but only "
                         << (func.getNumArguments() - llvm_arg_idx)
                         << " remaining\n";
            assert(false && "Not enough LLVM arguments for pointer");
          }

          // Get address space from LLVM argument
          uint32_t llvm_as = cast<LLVM::LLVMPointerType>(
                                 func.getArgument(llvm_arg_idx).getType())
                                 .getAddressSpace();

          // Parse expected address space from EDSL parameter type (e.g.,
          // "llvm.ptr<1>" -> 1)
          uint32_t expected_as = llvm_as; // Default to LLVM's address space
          if (auto parsed_as = parseAddressSpaceFromTypeHint(arg_type)) {
            expected_as = *parsed_as;

            // Verify address space consistency
            if (expected_as != llvm_as) {
              llvm::errs()
                  << "[ERROR] Address space mismatch for pointer operand "
                  << operand_idx << "\n";
              llvm::errs() << "[ERROR] EDSL hint: " << arg_type
                           << " (address space: " << expected_as << ")\n";
              llvm::errs() << "[ERROR] LLVM func arg address space: " << llvm_as
                           << "\n";
              assert(false && "Address space mismatch");
            }
          }

          Type llvmPtrTy =
              LLVM::LLVMPointerType::get(self.getContext(), expected_as);

          // Create extract operation for pointer
          extractOps.push_back(
              self.create<tle::ExtractPtrOp>(llvmPtrTy, blockArg));

          // Map LLVM function argument to extract operation result
          mapper.map(func.getArgument(llvm_arg_idx), extractOps.back());

          llvm_arg_idx += expected_llvm_args;

          // Case 3: Scalar type conversion
          // TT IR: IntegerType or FloatType (e.g., i32, i64, f32)
          // EDSL hint: Same type string (e.g., "i32", "i64")
          // LLVM func: 1 arg = same type (i32, i64, f32, etc.)
          // Conversion: Use block argument directly (no extract operation
          // needed)
        } else if (isa<IntegerType>(blockArg.getType()) ||
                   isa<FloatType>(blockArg.getType())) {
          // Type consistency check: verify EDSL parameter type matches actual
          // Triton type Scalars should not have memref or llvm.ptr param types
          if (!arg_type.empty()) {
            std::regex memref_regex(R"(!?memref<[^>]*,\s*\d+\s*>)");
            std::regex ptr_regex(R"(!?llvm\.ptr<\d+>)");
            bool is_memref_hint = std::regex_match(arg_type, memref_regex);
            bool is_ptr_hint = std::regex_match(arg_type, ptr_regex);
            if (is_memref_hint || is_ptr_hint) {
              llvm::errs() << "[ERROR] Type mismatch for operand "
                           << operand_idx;
              if (!arg_name.empty()) {
                llvm::errs() << " (parameter: " << arg_name << ")";
              }
              llvm::errs() << "\n";
              llvm::errs() << "[ERROR] EDSL param type: " << arg_type
                           << " (expects tensor or pointer type)\n";
              llvm::errs() << "[ERROR] Actual Triton type: "
                           << blockArg.getType() << " (scalar type)\n";
              assert(false && "EDSL type hint mismatch: memref/llvm.ptr "
                              "requires tensor/pointer type, but got scalar");
            }
          }
          size_t expected_llvm_args = 1;

          // Verify we have enough LLVM arguments
          if (llvm_arg_idx + expected_llvm_args > func.getNumArguments()) {
            llvm::errs() << "[ERROR] Not enough LLVM arguments for scalar. "
                         << "Expected " << expected_llvm_args
                         << " args starting at index " << llvm_arg_idx
                         << ", but only "
                         << (func.getNumArguments() - llvm_arg_idx)
                         << " remaining\n";
            assert(false && "Not enough LLVM arguments for scalar");
          }

          // For scalars, use the block argument directly (no extract needed)
          extractOps.push_back(blockArg);

          // Map LLVM function argument to block argument
          mapper.map(func.getArgument(llvm_arg_idx), blockArg);

          llvm_arg_idx += expected_llvm_args;
        } else {
          // Unsupported type: report error
          Type argType = blockArg.getType();
          llvm::errs() << "[ERROR] Unsupported operand type: " << argType
                       << " at operand index " << operand_idx << "\n";
          llvm::errs() << "[ERROR] Expected one of: RankedTensorType, "
                          "triton::PointerType, IntegerType, FloatType\n";
          llvm::errs() << "[ERROR] EDSL param type: " << arg_type << "\n";
          assert(false && "Unsupported operand type");
        }
      }

      // Verify we consumed all LLVM function arguments
      if (llvm_arg_idx != func.getNumArguments()) {
        llvm::errs() << "[WARNING] Mismatch in LLVM argument count. "
                     << "Consumed " << llvm_arg_idx
                     << " args, but function has " << func.getNumArguments()
                     << " args\n";
      }
      mapper.map(&oldBlock, newBlock);
    } else {
      // For other blocks, just map block arguments
      Block *newBlock = builder.createBlock(
          &body, {}, oldBlock.getArgumentTypes(),
          SmallVector<Location>(oldBlock.getNumArguments(), self.getLastLoc()));
      for (auto [oldArg, newArg] :
           llvm::zip(oldBlock.getArguments(), newBlock->getArguments())) {
        mapper.map(oldArg, newArg);
      }
      mapper.map(&oldBlock, newBlock);
    }
  }

  // Stage 4: Clone the LLVM function body to the DSLRegionOp body
  for (auto [oldBlock, newBlock] :
       llvm::zip(func.getBlocks(), body.getBlocks())) {
    OpBuilder::InsertionGuard guard(builder);
    // Use setInsertionPointToEnd because extract operations were inserted at
    // the start in Stage 3 Clone operations will be inserted after extract
    // operations
    builder.setInsertionPointToEnd(&newBlock);
    for (Operation &operation : oldBlock.getOperations()) {
      if (LLVM::ReturnOp returnOp = dyn_cast<LLVM::ReturnOp>(operation)) {
        SmallVector<Value> yields;
        if (dslRegionOp.getNumResults() == 1) {
          tle::PackOp packOp = builder.create<tle::PackOp>(
              operation.getLoc(), dslRegionOp.getResult(0).getType(),
              mapper.lookup(returnOp.getArg()));
          yields.push_back(packOp.getOutput());
        } else {
          for (auto [idx, result] : llvm::enumerate(dslRegionOp.getResults())) {
            LLVM::ExtractValueOp operand = builder.create<LLVM::ExtractValueOp>(
                operation.getLoc(), mapper.lookup(returnOp.getArg()),
                SmallVector<int64_t>{static_cast<int64_t>(idx)});
            tle::PackOp packOp = builder.create<tle::PackOp>(
                operation.getLoc(), result.getType(), operand);
            yields.push_back(packOp.getOutput());
          }
        }
        builder.create<tle::YieldOp>(operation.getLoc(), yields);
      } else {
        builder.clone(operation, mapper);
      }
    }
  }
  return dslRegionOp;
}
