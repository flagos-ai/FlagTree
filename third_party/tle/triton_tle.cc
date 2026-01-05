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
         const std::vector<Value> &outputs, const std::vector<Value> &inputs) {
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
        tle::DSLRegionOp dslRegionOp =
            self.create<tle::DSLRegionOp>(outputTys, operands);
        OpBuilder::InsertionGuard guard(builder);
        Region &body = dslRegionOp.getBody();
        SmallVector<Type> operandTys = llvm::map_to_vector(
            operands, [](Value value) -> Type { return value.getType(); });
        IRMapping mapper;
        uint32_t start_idx = 0;
        for (auto [idx, oldBlock] : llvm::enumerate(func.getBlocks())) {
          if (idx == 0) {
            Block *newBlock = builder.createBlock(
                &body, {}, operandTys,
                SmallVector<Location>(operandTys.size(), self.getLastLoc()));
            SmallVector<Value> extractOps;
            for (const auto &input : body.getArguments()) {
              if (RankedTensorType tensorTy =
                      dyn_cast<RankedTensorType>(input.getType())) {
                uint32_t as = cast<LLVM::LLVMPointerType>(
                                  func.getArgument(start_idx).getType())
                                  .getAddressSpace();
                Type ty = LLVM::LLVMPointerType::get(self.getContext(), as);
                extractOps.push_back(
                    self.create<tle::ExtractAllocatedPtrOp>(ty, input));
                extractOps.push_back(
                    self.create<tle::ExtractAlignedPtrOp>(ty, input));
                extractOps.push_back(self.create<tle::ExtractOffsetOp>(input));
                const size_t rank = tensorTy.getRank();
                auto sizesOp = self.create<tle::ExtractSizesOp>(rank, input);
                auto stridesOp =
                    self.create<tle::ExtractStridesOp>(rank, input);
                for (const auto &result : sizesOp.getResults()) {
                  extractOps.push_back(result);
                }
                for (const auto &result : stridesOp.getResults()) {
                  extractOps.push_back(result);
                }
                start_idx += (3 + 2 * rank);
              } else if (auto ptrTy =
                             dyn_cast<triton::PointerType>(input.getType())) {
                uint32_t as = cast<LLVM::LLVMPointerType>(
                                  func.getArgument(start_idx).getType())
                                  .getAddressSpace();
                Type ty = LLVM::LLVMPointerType::get(self.getContext(), as);
                extractOps.push_back(self.create<tle::ExtractPtrOp>(
                    LLVM::LLVMPointerType::get(self.getContext(), as), input));
                start_idx += 1;
              } else {
                extractOps.push_back(input);
                start_idx += 1;
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
