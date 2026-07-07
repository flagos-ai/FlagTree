#ifdef __ILUVATAR_TLE__

#include "IR/Dialect.h"
#include "Transforms/Passes.h"
#include "ir.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace ttg = mlir::triton::gpu;
namespace iluvatar_tle = mlir::triton::iluvatar_tle;

namespace {

void checkCtaRank(llvm::ArrayRef<unsigned> order,
                  llvm::ArrayRef<unsigned> ctasPerCGA,
                  llvm::ArrayRef<unsigned> ctaSplitNum,
                  llvm::ArrayRef<unsigned> ctaOrder) {
  if (order.size() != ctasPerCGA.size() || order.size() != ctaSplitNum.size() ||
      order.size() != ctaOrder.size())
    throw py::value_error("shared layout rank mismatch in CTA parameters");
}

mlir::Attribute getSharedMemorySpace(mlir::MLIRContext *context,
                                     const std::string &storage) {
  if (storage == "smem" || storage == "share_memory" ||
      storage == "shared_memory")
    return ttg::SharedMemorySpaceAttr::get(context);
  if (storage == "tmem" || storage == "tensor_memory")
    throw py::value_error("iluvatar TLE alloc does not support tmem storage");
  throw py::value_error("iluvatar TLE alloc only supports smem storage");
}

} // namespace

void init_triton_iluvatar_tle_ir(py::module m) {
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
             checkCtaRank(order, CTAsPerCGA, CTASplitNum, CTAOrder);
             auto *context = self.getBuilder().getContext();
             auto ctaLayout = ttg::CTAEncodingAttr::fromSplitParams(
                 context, CTAsPerCGA, CTASplitNum, CTAOrder);
             return ttg::SwizzledSharedEncodingAttr::get(
                 context, vectorSize, perPhase, maxPhase, order, ctaLayout);
           })
      .def("make_nv_mma_shared_encoding_attr",
           [](TritonOpBuilder &, std::vector<int64_t>, std::vector<unsigned>,
              mlir::Type &, std::vector<unsigned>, std::vector<unsigned>,
              std::vector<unsigned>, bool, bool) -> mlir::Attribute {
             throw py::value_error("iluvatar TLE alloc does not support "
                                   "nv_mma_shared_layout=True");
           })
      .def("make_tensor_memory_encoding_attr",
           [](TritonOpBuilder &, unsigned, unsigned, unsigned, unsigned,
              unsigned, bool) -> mlir::Attribute {
             throw py::value_error(
                 "iluvatar TLE alloc does not support tmem storage");
           })
      .def("create_local_alloc",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              mlir::Type &elementType,
              mlir::Attribute &encoding) -> mlir::Value {
             auto *context = self.getBuilder().getContext();
             auto memorySpace = ttg::SharedMemorySpaceAttr::get(context);
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
      .def("create_tma_copy",
           [](TritonOpBuilder &, mlir::Value, mlir::Value,
              std::vector<mlir::Value>) -> void {
             throw std::runtime_error("tle.gpu.copy with tensor_descriptor is "
                                      "not supported on Iluvatar TLE");
           })
      .def("create_extract_tile",
           [](TritonOpBuilder &self, mlir::Value &input, mlir::Value &index,
              std::vector<int64_t> &tileShape) -> mlir::Value {
             auto op = self.create<iluvatar_tle::ExtractTileOp>(input, index,
                                                                tileShape);
             return op.getResult();
           })
      .def("create_insert_tile",
           [](TritonOpBuilder &self, mlir::Value &input, mlir::Value &tile,
              mlir::Value &index) -> mlir::Value {
             auto op =
                 self.create<iluvatar_tle::InsertTileOp>(input, tile, index);
             return op.getResult();
           })
      .def("create_local_pointers",
           [](TritonOpBuilder &self, mlir::Type resultTy, mlir::Value memDesc,
              py::args args) -> mlir::OpState {
             llvm::SmallVector<mlir::Value> indices;
             indices.reserve(args.size());
             for (const auto &arg : args)
               indices.push_back(py::cast<mlir::Value>(arg));
             return self.create<iluvatar_tle::LocalPointersOp>(
                 resultTy, memDesc, indices);
           })
      .def("get_memdesc_type",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              mlir::Type &elementType, mlir::Attribute &encoding,
              std::string storage) -> mlir::Type {
             auto *context = self.getBuilder().getContext();
             auto memorySpace = getSharedMemorySpace(context, storage);
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
             return ttg::MemDescType::get(shape, elementType, encoding,
                                          memorySpace,
                                          /*mutableMemory=*/true, allocShape);
           });
}

void init_triton_iluvatar_tle_passes(py::module m) {
  ADD_PASS_WRAPPER_0(
      "add_insert_local_pointer_barriers",
      iluvatar_tle::createTritonIluvatarTleInsertLocalPointerBarriers);
  ADD_PASS_WRAPPER_0(
      "add_optimize_local_pointer_loads",
      iluvatar_tle::createTritonIluvatarTleOptimizeLocalPointerLoads);
  ADD_PASS_WRAPPER_0(
      "add_optimize_local_pointer_stores",
      iluvatar_tle::createTritonIluvatarTleOptimizeLocalPointerStores);
}

#endif // __ILUVATAR_TLE__
