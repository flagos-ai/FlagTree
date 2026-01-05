// MIT License

// Copyright (c) 2025 The FlagOS Contributors

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// flagtree tle

#include "Python.h"
#include "Transforms/Passes.h"
#include "ir.h" // TritonOpBuilder
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "passes.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Casting.h"

namespace py = pybind11;
using namespace mlir;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;
namespace tle = triton::tle;

void init_triton_tle_ir(py::module &&m) {
  using ret = py::return_value_policy;

  // Get the existing builder class from the main ir module (TLX style)
  auto *builder_cls = ir::getBuilderClass();

  // Add TLE extensions to the existing TritonOpBuilder class
  builder_cls
      ->def("make_swizzled_shared_encoding_attr",
            [](TritonOpBuilder &self, unsigned vectorSize, unsigned perPhase,
               unsigned maxPhase, std::vector<unsigned> order,
               std::vector<unsigned> CTAsPerCGA,
               std::vector<unsigned> CTASplitNum,
               std::vector<unsigned> CTAOrder) {
              assert(order.size() == CTAsPerCGA.size() && "shape mismatch");
              assert(order.size() == CTASplitNum.size() && "shape mismatch");
              assert(order.size() == CTAOrder.size() && "shape mismatch");
              auto context = self.getBuilder().getContext();
              auto CTALayout = ttg::CTALayoutAttr::get(context, CTAsPerCGA,
                                                       CTASplitNum, CTAOrder);
              return mlir::cast<Attribute>(ttg::SwizzledSharedEncodingAttr::get(
                  context, vectorSize, perPhase, maxPhase, order, CTALayout));
            })
      .def("make_nv_mma_shared_encoding_attr",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              std::vector<unsigned> order, Type &elemType,
              std::vector<unsigned> CTAsPerCGA,
              std::vector<unsigned> CTASplitNum, std::vector<unsigned> CTAOrder,
              bool fp4Padded, bool swizzled) {
             /* Validation logic for user defined layout encoding begin */
             assert(shape.size() == order.size());
             assert(order.size() == CTAsPerCGA.size());
             assert(CTAsPerCGA.size() == CTASplitNum.size());
             assert(CTASplitNum.size() == CTAOrder.size());
             /* Validation logic for user defined layout encoding end */

             auto context = self.getBuilder().getContext();
             auto CTALayout = ttg::CTALayoutAttr::get(context, CTAsPerCGA,
                                                      CTASplitNum, CTAOrder);
             if (swizzled) {
               return mlir::cast<Attribute>(ttg::NVMMASharedEncodingAttr::get(
                   context, shape, order, CTALayout, elemType, fp4Padded));
             } else {
               return mlir::cast<Attribute>(ttg::NVMMASharedEncodingAttr::get(
                   context, /*swizzlingByteWidth=*/0,
                   /*transposed=*/order[0] == 0,
                   elemType.getIntOrFloatBitWidth(), fp4Padded, CTALayout));
             }
           })
      .def("make_tensor_memory_encoding_attr",
           [](TritonOpBuilder &self, unsigned blockM, unsigned blockN,
              bool unpacked, unsigned CTASplitM, unsigned CTASplitN) {
             auto context = self.getBuilder().getContext();
             return mlir::cast<Attribute>(ttng::TensorMemoryEncodingAttr::get(
                 context, blockM, blockN, unpacked, CTASplitM, CTASplitN));
           })
      .def("create_local_alloc",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              Type &elementType, Attribute &encoding) -> mlir::Value {
             auto context = self.getBuilder().getContext();
             auto memorySpace = ttg::SharedMemorySpaceAttr::get(context);
             auto memDesc =
                 ttg::MemDescType::get(shape, elementType, encoding,
                                       memorySpace, /*mutableMemory=*/true);
             return self.create<ttg::LocalAllocOp>(memDesc);
           })
      .def("create_local_alloc",
           [](TritonOpBuilder &self, Type resultTy, Value value) -> Value {
             return self.create<ttg::LocalAllocOp>(resultTy, value);
           })
      .def("create_tma_copy",
           [](TritonOpBuilder &self, Value src, Value dst,
              std::vector<Value> &indices) {
             self.create<ttg::TMACopyOp>(src, dst, indices);
             return;
           })
      .def("create_local_load",
           [](TritonOpBuilder &self, Type resultTy, Value memDesc) -> Value {
             return self.create<ttg::LocalLoadOp>(resultTy, memDesc);
           })
      .def("create_local_store",
           [](TritonOpBuilder &self, Value &dst, Value &regValues) -> void {
             self.create<ttg::LocalStoreOp>(regValues, dst);
           })
      .def("get_memdesc_type",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              Type &elementType, Attribute &encoding,
              std::string storage) -> Type {
             auto context = self.getBuilder().getContext();
             Attribute memorySpace;
             if (storage == "tmem")
               memorySpace = ttng::TensorMemorySpaceAttr::get(context);
             else if (storage == "smem") {
               memorySpace = ttg::SharedMemorySpaceAttr::get(context);
             } else {
               llvm_unreachable("Unknown storage type");
             }
             return ttg::MemDescType::get(shape, elementType, encoding,
                                          memorySpace, /*mutableMemory=*/true);
           });
}

void init_triton_tle_passes(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_early_assign_memory_space",
                     tle::createTritonTleEarlyAssignMemorySpace);
  ADD_PASS_WRAPPER_0("add_lower_async_load",
                     tle::createTritonTleLowerAsyncLoad);
  ADD_PASS_WRAPPER_0("add_lower_tma_copy", tle::createTritonTleLowerTmaCopy);
}

void init_tle_raw_ir(py::module &&m) {
  using ret = py::return_value_policy;

  py::class_<tle::DSLRegionOp>(m, "DSLRegionOp", py::module_local(),
                               py::dynamic_attr())
      .def(
          "get_results",
          [](tle::DSLRegionOp &op) -> std::vector<OpResult> {
            auto results_range = op->getResults();
            return std::vector<OpResult>(results_range.begin(),
                                         results_range.end());
          },
          ret::reference)
      .def("dump", &tle::DSLRegionOp::dump);

  py::class_<tle::YieldOp>(m, "YieldOp", py::module_local(), py::dynamic_attr())
      .def("dump", &tle::YieldOp::dump);

  auto *builder_cls = ir::getBuilderClass();
  builder_cls->def(
      "create_edsl_region_by_llvm_func",
      [](TritonOpBuilder &self, std::string_view text, std::string_view fnname,
         const std::vector<Value> &outputs, const std::vector<Value> &inputs,
         const std::vector<std::string> &arg_type_hints) {
        ParserConfig config(self.getContext());
        OwningOpRef<ModuleOp> module =
            parseSourceString<ModuleOp>(text, config);
        LLVM::LLVMFuncOp func = module->lookupSymbol<LLVM::LLVMFuncOp>(fnname);
        OpBuilder &builder = self.getBuilder();
        SmallVector<Type> outputTys = llvm::map_to_vector(
            outputs, [](Value value) -> Type { return value.getType(); });
        SmallVector<Value> operands = llvm::to_vector(llvm::concat<Value>(
            SmallVector<Value>(outputs.begin(), outputs.end()),
            SmallVector<Value>(inputs.begin(), inputs.end())));

        // Step 1: Convert arg_type_hints to ArrayAttr
        ArrayAttr typeHintsAttr = nullptr;
        if (!arg_type_hints.empty()) {
          SmallVector<Attribute> typeHintAttrs;
          for (const auto &typeHint : arg_type_hints) {
            typeHintAttrs.push_back(
                StringAttr::get(self.getContext(), typeHint));
          }
          typeHintsAttr = ArrayAttr::get(self.getContext(), typeHintAttrs);
        }

        // Step 2: Create DSLRegionOp with type hints attribute
        SmallVector<NamedAttribute> attrs;
        if (typeHintsAttr) {
          attrs.push_back(NamedAttribute(
              StringAttr::get(self.getContext(), "arg_type_hints"),
              typeHintsAttr));
        }

        tle::DSLRegionOp dslRegionOp =
            self.create<tle::DSLRegionOp>(outputTys, operands, attrs);

        // Debug: Print arg_type_hints
        llvm::errs() << "[DEBUG] arg_type_hints size: " << arg_type_hints.size()
                     << "\n";
        for (const auto &typeHint : arg_type_hints) {
          llvm::errs() << "[DEBUG] typeHint: " << typeHint << "\n";
        }
        if (typeHintsAttr) {
          llvm::errs() << "[DEBUG] typeHintsAttr created, size: "
                       << typeHintsAttr.size() << "\n";
        }

        OpBuilder::InsertionGuard guard(builder);
        Region &body = dslRegionOp.getBody();
        SmallVector<Type> operandTys = llvm::map_to_vector(
            operands, [](Value value) -> Type { return value.getType(); });
        IRMapping mapper;

        // Step 3: Get arg_type_hints from attribute (rename to arg_types for
        // clarity)
        SmallVector<std::string> arg_types;
        if (auto attr =
                dslRegionOp->getAttrOfType<ArrayAttr>("arg_type_hints")) {
          for (auto typeHintAttr : attr) {
            if (auto strAttr = dyn_cast<StringAttr>(typeHintAttr)) {
              arg_types.push_back(strAttr.str());
            }
          }
        }

        llvm::errs() << "[DEBUG] Extracted arg_types from attribute, size: "
                     << arg_types.size() << "\n";

        // Step 4: Create extract operations in block 0, mapping DSLRegionOp
        // operands to LLVM func args We iterate through DSLRegionOp operands
        // (which are outputs + inputs) and match them with LLVM function
        // arguments based on type hints
        uint32_t llvm_arg_idx = 0; // Track position in LLVM function arguments
        SmallVector<Value>
            extractOps; // Collect all extract operations for mapping

        for (auto [idx, oldBlock] : llvm::enumerate(func.getBlocks())) {
          if (idx == 0) {
            // Create the first block with operands as block arguments
            Block *newBlock = builder.createBlock(
                &body, {}, operandTys,
                SmallVector<Location>(operandTys.size(), self.getLastLoc()));

            builder.setInsertionPointToStart(newBlock);

            // Step 4: Iterate through DSLRegionOp operands and create extract
            // operations Note: operands = [outputs..., inputs...], arg_types
            // corresponds to all operands
            for (size_t operand_idx = 0; operand_idx < operands.size();
                 ++operand_idx) {
              Value operand = operands[operand_idx];
              Value blockArg = newBlock->getArgument(operand_idx);
              std::string arg_type =
                  operand_idx < arg_types.size() ? arg_types[operand_idx] : "";

              llvm::errs() << "[DEBUG] Processing operand " << operand_idx
                           << ", TT type: " << blockArg.getType()
                           << ", arg_type hint: " << arg_type << "\n";

              // Case 1: TT Tensor -> EDSL expects memref -> LLVM should have 5
              // args (3 + 2*rank)
              if (RankedTensorType tensorTy =
                      dyn_cast<RankedTensorType>(blockArg.getType())) {
                const size_t rank = tensorTy.getRank();
                size_t expected_llvm_args =
                    3 + 2 * rank; // allocated_ptr, aligned_ptr, offset,
                                  // sizes..., strides...

                // Verify we have enough LLVM arguments
                if (llvm_arg_idx + expected_llvm_args >
                    func.getNumArguments()) {
                  llvm::errs()
                      << "[ERROR] Not enough LLVM arguments for tensor. "
                      << "Expected " << expected_llvm_args
                      << " args starting at index " << llvm_arg_idx
                      << ", but only "
                      << (func.getNumArguments() - llvm_arg_idx)
                      << " remaining\n";
                  assert(false && "Not enough LLVM arguments for tensor");
                }

                // Get address space from the first LLVM argument
                // (allocated_ptr)
                uint32_t as = cast<LLVM::LLVMPointerType>(
                                  func.getArgument(llvm_arg_idx).getType())
                                  .getAddressSpace();
                Type ptrTy = LLVM::LLVMPointerType::get(self.getContext(), as);

                llvm::errs() << "[DEBUG] Tensor case: rank=" << rank
                             << ", expected_llvm_args=" << expected_llvm_args
                             << ", address_space=" << as << "\n";

                // Create extract operations for tensor components
                size_t extract_start_idx = extractOps.size();
                extractOps.push_back(
                    self.create<tle::ExtractAllocatedPtrOp>(ptrTy, blockArg));
                extractOps.push_back(
                    self.create<tle::ExtractAlignedPtrOp>(ptrTy, blockArg));
                extractOps.push_back(
                    self.create<tle::ExtractOffsetOp>(blockArg));

                auto sizesOp = self.create<tle::ExtractSizesOp>(rank, blockArg);
                auto stridesOp =
                    self.create<tle::ExtractStridesOp>(rank, blockArg);
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

                // Case 2: TT Pointer -> EDSL expects llvm.ptr -> LLVM should
                // have 1 arg
              } else if (auto ptrTy = dyn_cast<triton::PointerType>(
                             blockArg.getType())) {
                size_t expected_llvm_args = 1;

                // Verify we have enough LLVM arguments
                if (llvm_arg_idx + expected_llvm_args >
                    func.getNumArguments()) {
                  llvm::errs()
                      << "[ERROR] Not enough LLVM arguments for pointer. "
                      << "Expected " << expected_llvm_args
                      << " args starting at index " << llvm_arg_idx
                      << ", but only "
                      << (func.getNumArguments() - llvm_arg_idx)
                      << " remaining\n";
                  assert(false && "Not enough LLVM arguments for pointer");
                }

                // Get address space from LLVM argument
                uint32_t as = cast<LLVM::LLVMPointerType>(
                                  func.getArgument(llvm_arg_idx).getType())
                                  .getAddressSpace();
                Type llvmPtrTy =
                    LLVM::LLVMPointerType::get(self.getContext(), as);

                llvm::errs()
                    << "[DEBUG] Pointer case: expected_llvm_args="
                    << expected_llvm_args << ", address_space=" << as << "\n";

                // Create extract operation for pointer
                extractOps.push_back(
                    self.create<tle::ExtractPtrOp>(llvmPtrTy, blockArg));

                // Map LLVM function argument to extract operation result
                mapper.map(func.getArgument(llvm_arg_idx), extractOps.back());

                llvm_arg_idx += expected_llvm_args;

                // Case 3: Scalar (i32, i64, etc.) -> EDSL expects same type ->
                // LLVM should have 1 arg
              } else {
                size_t expected_llvm_args = 1;

                // Verify we have enough LLVM arguments
                if (llvm_arg_idx + expected_llvm_args >
                    func.getNumArguments()) {
                  llvm::errs()
                      << "[ERROR] Not enough LLVM arguments for scalar. "
                      << "Expected " << expected_llvm_args
                      << " args starting at index " << llvm_arg_idx
                      << ", but only "
                      << (func.getNumArguments() - llvm_arg_idx)
                      << " remaining\n";
                  assert(false && "Not enough LLVM arguments for scalar");
                }

                llvm::errs()
                    << "[DEBUG] Scalar case: type=" << blockArg.getType()
                    << ", expected_llvm_args=" << expected_llvm_args << "\n";

                // For scalars, use the block argument directly (no extract
                // needed)
                extractOps.push_back(blockArg);

                // Map LLVM function argument to block argument
                mapper.map(func.getArgument(llvm_arg_idx), blockArg);

                llvm_arg_idx += expected_llvm_args;
              }
            }

            // Verify we consumed all LLVM function arguments
            if (llvm_arg_idx != func.getNumArguments()) {
              llvm::errs() << "[WARNING] Mismatch in LLVM argument count. "
                           << "Consumed " << llvm_arg_idx
                           << " args, but function has "
                           << func.getNumArguments() << " args\n";
            }

            mapper.map(&oldBlock, newBlock);
          } else {
            // For other blocks, just map block arguments
            Block *newBlock = builder.createBlock(
                &body, {}, oldBlock.getArgumentTypes(),
                SmallVector<Location>(oldBlock.getNumArguments(),
                                      self.getLastLoc()));
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
                tle::PackOp packOp = builder.create<tle::PackOp>(
                    operation.getLoc(), dslRegionOp.getResult(0).getType(),
                    mapper.lookup(returnOp.getArg()));
                yields.push_back(packOp.getOutput());
              } else {
                for (auto [idx, result] :
                     llvm::enumerate(dslRegionOp.getResults())) {
                  LLVM::ExtractValueOp operand =
                      builder.create<LLVM::ExtractValueOp>(
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
      });
}

void init_tle_raw_passes(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_tle_convert_arg_to_memdesc",
                     mlir::triton::tle::createTleConvertArgToMemDesc);
  ADD_PASS_WRAPPER_0("add_tle_dsl_region_inline",
                     mlir::triton::tle::createTleDSLRegionInline);
}

void init_triton_tle(py::module &&m) {
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    // TODO: move our td defines here
    // registry.insert<mlir::triton::tle::tleDialect>();
    // context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  init_triton_tle_ir(m.def_submodule("ir"));
  init_triton_tle_passes(m.def_submodule("passes"));
  init_tle_raw_ir(m.def_submodule("raw_ir"));
  init_tle_raw_passes(m.def_submodule("raw_passes"));
}
