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
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "pybind11/pybind11.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
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
  ADD_PASS_WRAPPER_0("add_lower_tma_copy", tle::createTritonTleLowerTmaCopy);
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
}
