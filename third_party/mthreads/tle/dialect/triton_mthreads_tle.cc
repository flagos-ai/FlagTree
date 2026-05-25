#ifdef __TLE__

#include "Dialect/MUSATLE/IR/Dialect.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "ir.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace ttg = mlir::triton::gpu;

// Backend-local `musa_tle` dialect adapters. Frontend marker pass wrappers
// live in tle/frontend/triton_mthreads_frontend.cc; keep them separate from
// `musa_tle.local_pointers` builder and transform bindings.

namespace {

void checkCtaRank(llvm::ArrayRef<unsigned> order,
                  llvm::ArrayRef<unsigned> ctasPerCGA,
                  llvm::ArrayRef<unsigned> ctaSplitNum,
                  llvm::ArrayRef<unsigned> ctaOrder) {
  if (order.size() != ctasPerCGA.size() || order.size() != ctaSplitNum.size() ||
      order.size() != ctaOrder.size())
    throw py::value_error("shared layout rank mismatch in CTA parameters");
}

void normalizeRank0SharedLayout(std::vector<unsigned> &order,
                                std::vector<unsigned> &ctasPerCGA,
                                std::vector<unsigned> &ctaSplitNum,
                                std::vector<unsigned> &ctaOrder) {
  if (!order.empty())
    return;
  if (!ctasPerCGA.empty() || !ctaSplitNum.empty() || !ctaOrder.empty())
    throw py::value_error("rank-0 shared layout expects empty CTA parameters");
  // TritonGPU memdesc currently rejects true rank-0 descriptors.  Mthreads TLE
  // keeps Python-visible rank-0 semantics by backing such buffers with one
  // shared element and a rank-1 shared layout.
  order = {0};
  ctasPerCGA = {1};
  ctaSplitNum = {1};
  ctaOrder = {0};
}

std::vector<int64_t> normalizeRank0MemDescShape(std::vector<int64_t> shape) {
  if (shape.empty())
    return {1};
  return shape;
}

ttg::CGAEncodingAttr makeCgaLayout(mlir::MLIRContext *context,
                                   llvm::ArrayRef<unsigned> ctasPerCGA,
                                   llvm::ArrayRef<unsigned> ctaSplitNum,
                                   llvm::ArrayRef<unsigned> ctaOrder) {
  return ttg::CGAEncodingAttr::fromSplitParams(context, ctasPerCGA, ctaSplitNum,
                                               ctaOrder);
}

mlir::Attribute getSharedMemorySpace(mlir::MLIRContext *context,
                                     const std::string &storage) {
  if (storage == "smem" || storage == "share_memory" ||
      storage == "shared_memory")
    return ttg::SharedMemorySpaceAttr::get(context);
  if (storage == "tmem" || storage == "tensor_memory")
    throw py::value_error("mthreads TLE alloc does not support tmem storage");
  throw py::value_error("mthreads TLE alloc only supports smem storage");
}

} // namespace

void init_triton_musa_tle_ir(py::module m) {
  (void)m;

  auto *builderClsPtr = ir::getBuilderClass();
  if (!builderClsPtr)
    throw std::runtime_error("triton IR builder class is not initialized");

  auto &builderCls = *builderClsPtr;
  builderCls
      .def("make_swizzled_shared_encoding_attr",
           [](TritonOpBuilder &self, unsigned vectorSize, unsigned perPhase,
              unsigned maxPhase, std::vector<unsigned> order,
              std::vector<unsigned> CTAsPerCGA,
              std::vector<unsigned> CTASplitNum,
              std::vector<unsigned> CTAOrder) -> mlir::Attribute {
             normalizeRank0SharedLayout(order, CTAsPerCGA, CTASplitNum,
                                        CTAOrder);
             checkCtaRank(order, CTAsPerCGA, CTASplitNum, CTAOrder);
             auto *context = self.getBuilder().getContext();
             auto cgaLayout =
                 makeCgaLayout(context, CTAsPerCGA, CTASplitNum, CTAOrder);
             return ttg::SwizzledSharedEncodingAttr::get(
                 context, vectorSize, perPhase, maxPhase, order, cgaLayout);
           })
      .def("make_nv_mma_shared_encoding_attr",
           [](TritonOpBuilder &, std::vector<int64_t>, std::vector<unsigned>,
              mlir::Type &, std::vector<unsigned>, std::vector<unsigned>,
              std::vector<unsigned>, bool, bool) -> mlir::Attribute {
             throw py::value_error("mthreads TLE alloc does not support "
                                   "nv_mma_shared_layout=True");
           })
      .def("make_tensor_memory_encoding_attr",
           [](TritonOpBuilder &, unsigned, unsigned, unsigned, unsigned,
              unsigned, bool) -> mlir::Attribute {
             throw py::value_error(
                 "mthreads TLE alloc does not support tmem storage");
           })
      .def("create_local_alloc",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              mlir::Type &elementType,
              mlir::Attribute &encoding) -> mlir::Value {
             auto *context = self.getBuilder().getContext();
             auto memorySpace = ttg::SharedMemorySpaceAttr::get(context);
             shape = normalizeRank0MemDescShape(std::move(shape));
             auto memDesc = ttg::MemDescType::get(shape, elementType, encoding,
                                                  memorySpace,
                                                  /*mutableMemory=*/true);
             return self.create<ttg::LocalAllocOp>(memDesc);
           })
      .def("create_local_alloc",
           [](TritonOpBuilder &self, mlir::Type resultTy,
              mlir::Value value) -> mlir::Value {
             return self.create<ttg::LocalAllocOp>(resultTy, value);
           })
      .def("get_memdesc_type",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              mlir::Type &elementType, mlir::Attribute &encoding,
              std::string storage) -> mlir::Type {
             auto *context = self.getBuilder().getContext();
             auto memorySpace = getSharedMemorySpace(context, storage);
             shape = normalizeRank0MemDescShape(std::move(shape));
             return ttg::MemDescType::get(shape, elementType, encoding,
                                          memorySpace,
                                          /*mutableMemory=*/true);
           })
      .def("get_memdesc_type",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              mlir::Type &elementType, mlir::Attribute &encoding,
              std::string storage,
              std::vector<int64_t> allocShape) -> mlir::Type {
             auto *context = self.getBuilder().getContext();
             auto memorySpace = getSharedMemorySpace(context, storage);
             shape = normalizeRank0MemDescShape(std::move(shape));
             allocShape = normalizeRank0MemDescShape(std::move(allocShape));
             return ttg::MemDescType::get(shape, elementType, encoding,
                                          memorySpace,
                                          /*mutableMemory=*/true, allocShape);
           })
      .def("create_tma_copy",
           [](TritonOpBuilder &self, mlir::Value src, mlir::Value dst,
              std::vector<mlir::Value> indices) -> void {
             self.create<ttg::TMACopyOp>(src, dst, indices);
           })
      .def("create_local_pointers",
           [](TritonOpBuilder &self, mlir::Type resultTy, mlir::Value memDesc,
              py::args args) -> mlir::OpState {
             llvm::SmallVector<mlir::Value> indices;
             indices.reserve(args.size());
             for (const auto &arg : args)
               indices.push_back(py::cast<mlir::Value>(arg));
             return self.create<mlir::triton::musa_tle::LocalPointersOp>(
                 resultTy, memDesc, indices);
           });
}

void init_triton_musa_tle_dialect_passes_ttgpuir(py::module m) {
  ADD_PASS_WRAPPER_0("add_tle_select_encodings",
                     mlir::createTritonMUSAGPUTLESelectEncodings);
  ADD_PASS_WRAPPER_0("add_tle_insert_local_pointer_barriers",
                     mlir::createTritonMUSAGPUTLEInsertLocalPointerBarriers);
  ADD_PASS_WRAPPER_0("add_tle_optimize_local_pointer_loads",
                     mlir::createTritonMUSAGPUTLEOptimizeLocalPointerLoads);
  ADD_PASS_WRAPPER_0("add_tle_optimize_local_pointer_stores",
                     mlir::createTritonMUSAGPUTLEOptimizeLocalPointerStores);
  ADD_PASS_WRAPPER_0(
      "add_tle_optimize_local_pointer_async_stores",
      mlir::createTritonMUSAGPUTLEOptimizeLocalPointerAsyncStores);
}

void register_triton_musa_tle_dialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::triton::musa_tle::MUSATLEDialect>();
}

#endif // __TLE__
