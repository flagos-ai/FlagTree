#include "ir.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "tle/dialect/include/IR/Dialect.h"
#include <optional>
#include <regex>
#include <string>

// Create a DSLRegionOp that wraps an LLVM function, performing type
// conversion from
// Triton IR types to LLVM types based on EDSL function declarations.
//
// Overview:
// 1. Parse the LLVM IR text and extract the target function using
// Triton's MLIR context
// 2. Create a DSLRegionOp with EDSL function type hints stored in
// attributes
// 3. Perform argument type conversion: TT IR types -> LLVM types (via
// extract operations)
//    - DSLRegionOp's operands are TT IR types (tensor, pointer, scalar)
//    - EDSL function declarations (stored in arg_type_hints attribute)
//    specify expected types
//    - LLVM function arguments are already in LLVM types
//    - We need to verify consistency: TT type -> EDSL type hint -> LLVM
//    func arg type
//
// Example type conversion for tensor:
//   - TT IR: tensor<128xi32> (RankedTensorType)
//   - EDSL hint: "memref<?xi32, 3>" (stored in arg_type_hints attribute)
//   - LLVM func: 5 args = allocated_ptr<3>, aligned_ptr<3>, offset,
//   size[0], stride[0]
//   - Conversion: Extract tensor into 5 LLVM values using
//   ExtractAllocatedPtrOp, etc.
//
// Example type conversion for scalar:
//   - TT IR: i32 (IntegerType)
//   - EDSL hint: "i32"
//   - LLVM func: 1 arg = i32
//   - Conversion: Use block argument directly

using namespace mlir;
namespace tle = triton::tle;

// Helper function to parse address space from EDSL type hint
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
          << "[ERROR] Failed to parse address space from memref type hint: "
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
          << "[ERROR] Failed to parse address space from llvm.ptr type hint: "
          << typeHint << "\n";
      return std::nullopt;
    }
  }

  // If type hint is not empty but doesn't match expected formats, report error
  llvm::errs() << "[ERROR] Unsupported type hint format: " << typeHint << "\n";
  llvm::errs() << "[ERROR] Expected format: \"memref<shape, address_space>\" "
                  "or \"llvm.ptr<address_space>\"\n";
  return std::nullopt;
}

tle::DSLRegionOp createEdslRegionByLLVMFunc(
    TritonOpBuilder &self, std::string_view text, std::string_view fnname,
    const std::vector<Value> &outputs, const std::vector<Value> &inputs,
    const std::vector<std::string>
        &arg_type_hints) { // Stage 1: Parse LLVM IR and extract function using
                           // Triton's MLIR context
  ParserConfig config(self.getContext());
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(text, config);
  LLVM::LLVMFuncOp func = module->lookupSymbol<LLVM::LLVMFuncOp>(fnname);
  OpBuilder &builder = self.getBuilder();

  SmallVector<Type> outputTys = llvm::map_to_vector(
      outputs, [](Value value) -> Type { return value.getType(); });
  SmallVector<Value> operands = llvm::to_vector(
      llvm::concat<Value>(SmallVector<Value>(outputs.begin(), outputs.end()),
                          SmallVector<Value>(inputs.begin(), inputs.end())));

  // Stage 2: Create DSLRegionOp
  // The arg_type_hints contain the EDSL function parameter type declarations
  // (e.g., "memref<?xi32, 3>", "i32") These are stored as an ArrayAttr of
  // StringAttrs in the DSLRegionOp's "arg_type_hints" attribute
  ArrayAttr typeHintsAttr = nullptr;
  if (!arg_type_hints.empty()) {
    SmallVector<Attribute> typeHintAttrs;
    for (const auto &typeHint : arg_type_hints) {
      typeHintAttrs.push_back(StringAttr::get(self.getContext(), typeHint));
    }
    typeHintsAttr = ArrayAttr::get(self.getContext(), typeHintAttrs);
  }

  SmallVector<NamedAttribute> attrs;
  if (typeHintsAttr) {
    attrs.push_back(NamedAttribute(
        StringAttr::get(self.getContext(), "arg_type_hints"), typeHintsAttr));
  }

  tle::DSLRegionOp dslRegionOp =
      self.create<tle::DSLRegionOp>(outputTys, operands, attrs);
  OpBuilder::InsertionGuard guard(builder);
  Region &body = dslRegionOp.getBody();
  SmallVector<Type> operandTys = llvm::map_to_vector(
      operands, [](Value value) -> Type { return value.getType(); });
  IRMapping mapper;

  // Stage 3: Argument type conversion and validation
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
  //   - TT ptr<f32> + EDSL "llvm.ptr<1>" -> LLVM 1 arg (ptr<1>)
  //   - TT i32 + EDSL "i32" -> LLVM 1 arg (i32, passed through directly)
  uint32_t llvm_arg_idx = 0;     // Track position in LLVM function arguments
  SmallVector<Value> extractOps; // Collect all extract operations for mapping

  for (auto [idx, oldBlock] : llvm::enumerate(func.getBlocks())) {
    if (idx == 0) {
      // Create the first block with operands as block arguments
      Block *newBlock = builder.createBlock(
          &body, {}, operandTys,
          SmallVector<Location>(operandTys.size(), self.getLastLoc()));

      builder.setInsertionPointToStart(newBlock);

      // Iterate through DSLRegionOp operands and create extract operations
      // Note: operands = [outputs..., inputs...], arg_type_hints corresponds to
      // all operands
      for (size_t operand_idx = 0; operand_idx < operands.size();
           ++operand_idx) {
        Value operand = operands[operand_idx];
        Value blockArg = newBlock->getArgument(operand_idx);
        std::string arg_type = operand_idx < arg_type_hints.size()
                                   ? arg_type_hints[operand_idx]
                                   : "";

        // Case 1: TT Tensor type conversion
        // TT IR: RankedTensorType (e.g., tensor<128xi32>)
        // EDSL hint: "memref<?xi32, 3>" (stored in arg_type_hints attribute)
        // LLVM func: 3 + 2*rank args = allocated_ptr<address_space>,
        // aligned_ptr<address_space>, offset, sizes[rank], strides[rank]
        // Conversion: Create ExtractAllocatedPtrOp, ExtractAlignedPtrOp,
        // ExtractOffsetOp, ExtractSizesOp, ExtractStridesOp
        if (RankedTensorType tensorTy =
                dyn_cast<RankedTensorType>(blockArg.getType())) {
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

          // Parse expected address space from EDSL type hint (e.g.,
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
          // EDSL hint: "llvm.ptr<1>" (stored in arg_type_hints attribute)
          // LLVM func: 1 arg = ptr<address_space>
          // Conversion: Create ExtractPtrOp to convert TT pointer to LLVM
          // pointer
        } else if (auto ptrTy =
                       dyn_cast<triton::PointerType>(blockArg.getType())) {
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

          // Parse expected address space from EDSL type hint (e.g.,
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
          llvm::errs() << "[ERROR] EDSL type hint: " << arg_type << "\n";
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
