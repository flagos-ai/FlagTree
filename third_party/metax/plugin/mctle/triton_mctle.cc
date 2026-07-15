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

#include "Python.h"
#include "ir.h" // TritonOpBuilder
#include "mctle/dialect/include/IR/Dialect.h"
#include "mctle/dialect/include/Transforms/Passes.h"
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
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include <cstdint>

namespace py = pybind11;
using namespace mlir;
namespace ttg = triton::gpu;
namespace tle = triton::mctle;

void init_triton_mctle_ir(py::module &&m) {
  auto &builder_cls = *ir::getBuilderClass();

  builder_cls
      // TLE-Lite
      .def(
          "create_extract_tile",
          [](TritonOpBuilder &self, Value &input, Value &index,
             std::vector<int64_t> &tileShape) -> Value {
            auto op = self.create<tle::ExtractTileOp>(input, index, tileShape);
            return op.getResult();
          },
          py::arg("input"), py::arg("index"), py::arg("tileShape"),
          "Create extract_tile operation")
      .def(
          "create_insert_tile",
          [](TritonOpBuilder &self, Value &input, Value &tile,
             Value &index) -> Value {
            auto op = self.create<tle::InsertTileOp>(input, tile, index);
            return op.getResult();
          },
          py::arg("input"), py::arg("tile"), py::arg("index"),
          "Create insert_tile operation")
      .def("make_swizzled_shared_encoding_attr",
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
           });
}

void init_triton_mctle_passes(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_early_assign_memory_space",
                     tle::createTritonTleEarlyAssignMemorySpace);
  ADD_PASS_WRAPPER_0("add_select_encodings",
                     tle::createTritonTleSelectEncodings);
  ADD_PASS_WRAPPER_0("add_assign_local_pointers_encoding",
                     tle::createTritonTleSelectEncodings);
  ADD_PASS_WRAPPER_0("add_insert_local_pointer_barriers",
                     tle::createTritonTleInsertLocalPointerBarriers);
  ADD_PASS_WRAPPER_0("add_lower_async_load",
                     tle::createTritonTleLowerAsyncLoad);
  ADD_PASS_WRAPPER_0("add_optimize_local_pointer_loads",
                     tle::createTritonTleOptimizeLocalPointerLoads);
  ADD_PASS_WRAPPER_0("add_optimize_local_pointer_stores",
                     tle::createTritonTleOptimizeLocalPointerStores);
  ADD_PASS_WRAPPER_0("add_lower_extract_tile",
                     tle::createTritonTleLowerExtractTile);
  ADD_PASS_WRAPPER_0("add_lower_insert_tile",
                     tle::createTritonTleLowerInsertTile);
}

void init_triton_mctle(py::module &&m) {
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::mctle::McTleDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
  init_triton_mctle_ir(m.def_submodule("ir"));
  init_triton_mctle_passes(m.def_submodule("passes"));
}
