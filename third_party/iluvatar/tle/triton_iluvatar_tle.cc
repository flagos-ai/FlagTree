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
      .def("create_warp_return",
           [](TritonOpBuilder &self) -> mlir::Operation * {
             return self.create<ttg::WarpReturnOp>();
           })
      .def("create_warp_yield",
           [](TritonOpBuilder &self,
              std::vector<mlir::Value> values) -> mlir::Operation * {
             return self.create<ttg::WarpYieldOp>(values);
           })
      .def("create_warp_specialize_partitions",
           [](TritonOpBuilder &self, int numPartitions) -> mlir::Operation * {
             return self.create<ttg::WarpSpecializePartitionsOp>(numPartitions);
           })
      .def("create_warp_specialize",
           [](TritonOpBuilder &self, std::vector<mlir::Type> resultTypes,
              std::vector<mlir::Value> explicitCaptures,
              std::vector<int> partitionNumWarps) {
             return self.create<ttg::WarpSpecializeOp>(
                 resultTypes, explicitCaptures, partitionNumWarps);
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
      .def("create_memdesc_index",
           [](TritonOpBuilder &self, mlir::Type resultType, mlir::Value src,
              mlir::Value index) -> mlir::Value {
             return self.create<ttg::MemDescIndexOp>(resultType, src, index);
           })
      .def("create_exclusive_cumsum",
           [](TritonOpBuilder &self, mlir::Type exclusiveTy, mlir::Type totalTy,
              mlir::Value src, int axis, bool reverse) -> mlir::OpState {
             auto &builder = self.getBuilder();
             return self.create<iluvatar_tle::ExclusiveCumsumOp>(
                 mlir::TypeRange{exclusiveTy, totalTy}, src,
                 builder.getI32IntegerAttr(axis), builder.getBoolAttr(reverse));
           })
      .def("create_pipe_create",
           [](TritonOpBuilder &self, std::vector<mlir::Value> fields,
              int32_t capacity, const std::string &scope,
              const std::string &pipeName, std::vector<std::string> fieldNames,
              std::vector<std::string> readerNames, bool oneShot) -> void {
             auto &builder = self.getBuilder();
             llvm::SmallVector<mlir::Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (llvm::StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             llvm::SmallVector<mlir::Attribute> readerNameAttrs;
             readerNameAttrs.reserve(readerNames.size());
             for (llvm::StringRef name : readerNames)
               readerNameAttrs.push_back(builder.getStringAttr(name));
             mlir::StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             mlir::ArrayAttr readerNamesAttr;
             if (!readerNameAttrs.empty())
               readerNamesAttr = builder.getArrayAttr(readerNameAttrs);
             mlir::BoolAttr oneShotAttr;
             if (oneShot)
               oneShotAttr = builder.getBoolAttr(true);
             self.create<iluvatar_tle::PipeCreateOp>(
                 fields, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs), readerNamesAttr,
                 oneShotAttr);
           })
      .def("create_pipe_writer_acquire",
           [](TritonOpBuilder &self, std::vector<mlir::Value> fields,
              mlir::Value stage, mlir::Value phase, int32_t capacity,
              const std::string &scope, const std::string &pipeName,
              std::vector<std::string> fieldNames) -> void {
             auto &builder = self.getBuilder();
             llvm::SmallVector<mlir::Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (llvm::StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             mlir::StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             self.create<iluvatar_tle::PipeWriterAcquireOp>(
                 fields, stage, phase, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs));
           })
      .def("create_pipe_writer_commit",
           [](TritonOpBuilder &self, std::vector<mlir::Value> fields,
              mlir::Value stage, int32_t capacity, const std::string &scope,
              const std::string &pipeName,
              std::vector<std::string> fieldNames) -> void {
             auto &builder = self.getBuilder();
             llvm::SmallVector<mlir::Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (llvm::StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             mlir::StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             self.create<iluvatar_tle::PipeWriterCommitOp>(
                 fields, stage, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs));
           })
      .def("create_pipe_writer_close",
           [](TritonOpBuilder &self, std::vector<mlir::Value> fields,
              mlir::Value stage, mlir::Value phase, int32_t capacity,
              const std::string &scope, const std::string &pipeName,
              std::vector<std::string> fieldNames) -> void {
             auto &builder = self.getBuilder();
             llvm::SmallVector<mlir::Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (llvm::StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             mlir::StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             self.create<iluvatar_tle::PipeWriterCloseOp>(
                 fields, stage, phase, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs));
           })
      .def("create_pipe_reader_wait",
           [](TritonOpBuilder &self, std::vector<mlir::Value> fields,
              mlir::Value stage, mlir::Value phase, int32_t capacity,
              const std::string &scope, const std::string &pipeName,
              std::vector<std::string> fieldNames,
              const std::string &readerName,
              std::vector<std::string>) -> mlir::Value {
             auto &builder = self.getBuilder();
             llvm::SmallVector<mlir::Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (llvm::StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             mlir::StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             mlir::StringAttr readerNameAttr;
             if (!readerName.empty())
               readerNameAttr = builder.getStringAttr(readerName);
             return self.create<iluvatar_tle::PipeReaderWaitOp>(
                 builder.getI1Type(), fields, stage, phase,
                 builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs), readerNameAttr);
           })
      .def("create_pipe_reader_release",
           [](TritonOpBuilder &self, std::vector<mlir::Value> fields,
              mlir::Value stage, int32_t capacity, const std::string &scope,
              const std::string &pipeName, std::vector<std::string> fieldNames,
              const std::string &readerName, std::vector<std::string>) -> void {
             auto &builder = self.getBuilder();
             llvm::SmallVector<mlir::Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (llvm::StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             mlir::StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             mlir::StringAttr readerNameAttr;
             if (!readerName.empty())
               readerNameAttr = builder.getStringAttr(readerName);
             self.create<iluvatar_tle::PipeReaderReleaseOp>(
                 fields, stage, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs), readerNameAttr);
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

  // Expose the ttg.warp_specialize op accessors used by the shared TLE
  // frontend (tle.gpu.warp_specialize). Registered module-local so it does not
  // clash with the (optional) Gluon binding of the same op.
  using ret = py::return_value_policy;
  py::class_<ttg::WarpSpecializeOp, mlir::OpState>(m, "WarpSpecializeOp",
                                                   py::module_local())
      .def("get_default_region", &ttg::WarpSpecializeOp::getDefaultRegion,
           ret::reference)
      .def("get_partition_op_holder",
           &ttg::WarpSpecializeOp::getPartitionOpHolder, ret::reference)
      .def("set_requested_registers", [](ttg::WarpSpecializeOp &self,
                                         std::vector<int> &requestedRegisters) {
        self.setRequestedRegisters(requestedRegisters);
      });
}

void init_triton_iluvatar_tle_passes(py::module m) {
  ADD_PASS_WRAPPER_0(
      "add_early_assign_memory_space",
      iluvatar_tle::createTritonIluvatarTleEarlyAssignMemorySpace);
  ADD_PASS_WRAPPER_0(
      "add_optimize_local_pointer_async_stores",
      iluvatar_tle::createTritonIluvatarTleOptimizeLocalPointerAsyncStores);
  ADD_PASS_WRAPPER_0(
      "add_insert_local_pointer_barriers",
      iluvatar_tle::createTritonIluvatarTleInsertLocalPointerBarriers);
  ADD_PASS_WRAPPER_0(
      "add_optimize_local_pointer_loads",
      iluvatar_tle::createTritonIluvatarTleOptimizeLocalPointerLoads);
  ADD_PASS_WRAPPER_0(
      "add_optimize_local_pointer_stores",
      iluvatar_tle::createTritonIluvatarTleOptimizeLocalPointerStores);
  ADD_PASS_WRAPPER_0("add_lower_async_load",
                     iluvatar_tle::createTritonIluvatarTleLowerAsyncLoad);
  ADD_PASS_WRAPPER_0(
      "add_optimize_exclusive_cumsum_layouts",
      iluvatar_tle::createTritonIluvatarTleOptimizeExclusiveCumsumLayouts);
  ADD_PASS_WRAPPER_0("add_lower_exclusive_cumsum",
                     iluvatar_tle::createTritonIluvatarTleLowerExclusiveCumsum);
  ADD_PASS_WRAPPER_0("add_lower_pipe_to_barriers",
                     iluvatar_tle::createTritonIluvatarTleLowerPipeToBarriers);
}

#endif // __ILUVATAR_TLE__
