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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "passes.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include <cstdint>

namespace py = pybind11;
using namespace mlir;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;
namespace tle = triton::tle;

extern tle::DSLRegionOp
createTLERawRegionByLLVMFunc(TritonOpBuilder &self, std::string_view text,
                             const std::vector<Value> &outputs,
                             const std::vector<Value> &inputs);

void init_triton_tle_ir(py::module &&m) {

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
              auto CTALayout = ttg::CTAEncodingAttr::fromSplitParams(
                  context, CTAsPerCGA, CTASplitNum, CTAOrder);
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
             auto CTALayout = ttg::CTAEncodingAttr::fromSplitParams(
                 context, CTAsPerCGA, CTASplitNum, CTAOrder);
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
             const unsigned colStride = unpacked ? 2 : 1;
             return mlir::cast<Attribute>(ttng::TensorMemoryEncodingAttr::get(
                 context, blockM, blockN, colStride, CTASplitM, CTASplitN,
                 /*twoCTAs=*/false));
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
      .def("create_local_pointers",
           [](TritonOpBuilder &self, Type resultTy, Value memDesc,
              py::args args) -> OpState {
             llvm::SmallVector<Value> indices;
             indices.reserve(args.size());
             for (const auto &arg : args) {
               indices.push_back(py::cast<Value>(arg));
             }
             return self.create<tle::LocalPointersOp>(resultTy, memDesc,
                                                      indices);
           })
      .def("create_distributed_barrier",
           [](TritonOpBuilder &self) -> void {
             self.create<tle::DistributedBarrierOp>(
                 StringAttr(), IntegerAttr(), DenseI32ArrayAttr(),
                 DenseI32ArrayAttr(), DenseI32ArrayAttr());
           })
      .def(
          "create_distributed_barrier",
          [](TritonOpBuilder &self, const std::string &groupKind,
             const std::vector<int32_t> &groupShape,
             const std::vector<int32_t> &groupAxes,
             const std::vector<int32_t> &groupMask) -> void {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            StringAttr kindAttr;
            IntegerAttr rankAttr;
            DenseI32ArrayAttr shapeAttr;
            DenseI32ArrayAttr axesAttr;
            DenseI32ArrayAttr maskAttr;

            if (!groupKind.empty()) {
              kindAttr = builder.getStringAttr(groupKind);
            }
            // Only materialize subgroup metadata when provided.
            // This allows kind-only barriers (e.g. group_kind="grid").
            if (!groupShape.empty() || !groupAxes.empty() ||
                !groupMask.empty()) {
              rankAttr = builder.getI32IntegerAttr(
                  static_cast<int32_t>(groupShape.size()));
              if (!groupShape.empty()) {
                shapeAttr = DenseI32ArrayAttr::get(ctx, groupShape);
              }
              if (!groupAxes.empty()) {
                axesAttr = DenseI32ArrayAttr::get(ctx, groupAxes);
              }
              if (!groupMask.empty()) {
                maskAttr = DenseI32ArrayAttr::get(ctx, groupMask);
              }
            }

            self.create<tle::DistributedBarrierOp>(
                kindAttr, rankAttr, shapeAttr, axesAttr, maskAttr);
          },
          py::arg("group_kind"), py::arg("group_shape"), py::arg("group_axes"),
          py::arg("group_mask"))
      .def("create_remote_pointers",
           [](TritonOpBuilder &self, Type resultTy, Value src,
              Value shardId) -> OpState {
             return self.create<tle::RemotePointersOp>(resultTy, src, shardId);
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
  ADD_PASS_WRAPPER_0("add_assign_local_pointers_encoding",
                     tle::createTritonTleAssignLocalPointersEncoding);
  ADD_PASS_WRAPPER_0("add_insert_local_pointer_barriers",
                     tle::createTritonTleInsertLocalPointerBarriers);
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
  builder_cls->def("create_tle_raw_region_by_llvm_func",
                   &createTLERawRegionByLLVMFunc);
  builder_cls->def("get_context", &TritonOpBuilder::getContext);
}

void init_tle_raw_passes(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_tle_convert_arg_to_memdesc",
                     mlir::triton::tle::createTleConvertArgToMemDesc);
  ADD_PASS_WRAPPER_0("add_tle_dsl_region_inline",
                     mlir::triton::tle::createTleDSLRegionInline);
}

void init_llvm(py::module &&m) {
  m.def("parse_llvm_ir",
        [](std::string_view text, llvm::LLVMContext &llvmContext,
           mlir::MLIRContext &mlirContext) -> mlir::ModuleOp {
          std::unique_ptr<llvm::MemoryBuffer> buffer =
              llvm::MemoryBuffer::getMemBuffer(text);
          llvm::SMDiagnostic error;
          std::unique_ptr<llvm::Module> llvmModule =
              llvm::parseIR(buffer->getMemBufferRef(), error, llvmContext);
          if (!llvmModule) {
            llvm::report_fatal_error(
                "failed to parse IR: " + error.getMessage() +
                "lineno: " + std::to_string(error.getLineNo()));
          }
          return mlir::translateLLVMIRToModule(std::move(llvmModule),
                                               &mlirContext)
              ->clone();
        });
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
  init_llvm(m.def_submodule("llvm"));
}
