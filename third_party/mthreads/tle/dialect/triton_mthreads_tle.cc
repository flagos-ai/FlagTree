#ifdef __TLE__

#include "Dialect/MUSATLE/IR/Dialect.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "ir.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
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

class TLEWarpSpecializeOp {
public:
  explicit TLEWarpSpecializeOp(ttg::WarpSpecializeOp op) : op(op) {}

  mlir::Region &getDefaultRegion() { return op.getDefaultRegion(); }
  mlir::Region &getPartitionOpHolder() { return op.getPartitionOpHolder(); }
  mlir::Operation *getOperation() { return op.getOperation(); }

  mlir::Value getResult(unsigned index) {
    if (index >= op.getNumResults())
      throw py::index_error("WarpSpecializeOp result index out of range");
    return op.getResult(index);
  }

  void setRequestedRegisters(std::vector<int32_t> requestedRegisters) {
    op.setRequestedRegisters(requestedRegisters);
  }

private:
  ttg::WarpSpecializeOp op;
};

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
  py::class_<TLEWarpSpecializeOp>(m, "WarpSpecializeOp", py::module_local())
      .def("get_default_region", &TLEWarpSpecializeOp::getDefaultRegion,
           py::return_value_policy::reference)
      .def("get_partition_op_holder",
           &TLEWarpSpecializeOp::getPartitionOpHolder,
           py::return_value_policy::reference)
      .def("get_operation", &TLEWarpSpecializeOp::getOperation,
           py::return_value_policy::reference)
      .def("get_result", &TLEWarpSpecializeOp::getResult)
      .def("set_requested_registers",
           &TLEWarpSpecializeOp::setRequestedRegisters);

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
      .def(
          "create_tma_copy",
          [](TritonOpBuilder &self, mlir::Value src, mlir::Value dst,
             std::vector<mlir::Value> indices, py::object completionBarrier,
             int32_t expectBytes) -> void {
            mlir::Value barrier;
            if (!completionBarrier.is_none())
              barrier = py::cast<mlir::Value>(completionBarrier);
            auto op = self.create<ttg::TMACopyOp>(src, dst, indices, barrier);
            if (expectBytes >= 0)
              op->setAttr("expect_bytes",
                          self.getBuilder().getI32IntegerAttr(expectBytes));
          },
          py::arg("src"), py::arg("dst"), py::arg("indices"),
          py::arg("completionBarrier") = py::none(),
          py::arg("expectBytes") = -1)
      .def("create_local_pointers",
           [](TritonOpBuilder &self, mlir::Type resultTy, mlir::Value memDesc,
              py::args args) -> mlir::OpState {
             llvm::SmallVector<mlir::Value> indices;
             indices.reserve(args.size());
             for (const auto &arg : args)
               indices.push_back(py::cast<mlir::Value>(arg));
             return self.create<mlir::triton::musa_tle::LocalPointersOp>(
                 resultTy, memDesc, indices);
           })
      .def("create_memdesc_index",
           [](TritonOpBuilder &self, mlir::Type resultType, mlir::Value src,
              mlir::Value index) -> mlir::Value {
             auto indexType =
                 mlir::dyn_cast<mlir::IntegerType>(index.getType());
             if (!indexType || !indexType.isInteger(32))
               throw py::value_error(
                   "mthreads TLE memdesc index requires an int32 index");

             if (src.getType().isInteger(32)) {
               // The public barrier object still asks for a logical slot type,
               // but mthreads barriers are hardware IDs rather than memdescs.
               (void)resultType;
               return self
                   .create<mlir::triton::musa_tle::BarrierIndexOp>(src, index)
                   .getBarId();
             }

             auto srcType = mlir::dyn_cast<ttg::MemDescType>(src.getType());
             if (!srcType || srcType.getShape().empty())
               throw py::value_error(
                   "mthreads TLE memdesc index requires a memdesc source");

             llvm::APInt constantIndex;
             if (mlir::matchPattern(index,
                                    mlir::m_ConstantInt(&constantIndex))) {
               int64_t slot = constantIndex.getSExtValue();
               int64_t leadingDimension = srcType.getShape().front();
               if (slot < 0 || slot >= leadingDimension)
                 throw py::value_error("mthreads TLE memdesc index " +
                                       std::to_string(slot) +
                                       " out of bounds for leading dimension " +
                                       std::to_string(leadingDimension));
             }

             return self.create<ttg::MemDescIndexOp>(resultType, src, index);
           })
      .def("create_barrier_alloc",
           [](TritonOpBuilder &self, mlir::Type resultType, int32_t numBarriers,
              int32_t arriveCount, int32_t initPolarity,
              int32_t expectBytes) -> mlir::Value {
             // The frontend result type is a logical barrier-array type.  The
             // backend handle is an i32 base ID and is resolved by late
             // mthreads barrier lowering.
             (void)resultType;
             if (numBarriers > 63)
               throw py::value_error(
                   "mthreads TLE barrier allocation exceeds the 63 hardware "
                   "barrier id limit");
             auto &builder = self.getBuilder();
             mlir::IntegerAttr expectBytesAttr;
             if (expectBytes > 0)
               expectBytesAttr = builder.getI32IntegerAttr(expectBytes);
             return self.create<mlir::triton::musa_tle::BarrierAllocOp>(
                 builder.getI32IntegerAttr(numBarriers),
                 builder.getI32IntegerAttr(arriveCount),
                 builder.getI32IntegerAttr(initPolarity), expectBytesAttr);
           })
      .def("create_barrier_wait_mbarrier",
           [](TritonOpBuilder &self, mlir::Value barrier,
              mlir::Value phase) -> void {
             self.create<mlir::triton::musa_tle::BarrierWaitOp>(barrier, phase);
           })
      .def("create_barrier_arrive_mbarrier",
           [](TritonOpBuilder &self, mlir::Value barrier, int32_t arriveCount,
              mlir::Value phase) -> void {
             if (arriveCount != 1)
               throw py::value_error(
                   "mthreads hardware barrier arrive requires arrive_count = "
                   "1");
             auto &builder = self.getBuilder();
             self.create<mlir::triton::musa_tle::BarrierArriveOp>(
                 barrier, phase, builder.getI32IntegerAttr(arriveCount));
           })
      .def("create_barrier_wait_named",
           [](TritonOpBuilder &, mlir::Value, int32_t, int32_t) -> void {
             throw py::value_error(
                 "mthreads TLE named barrier backend is unsupported; "
                 "phaseIdx is required");
           })
      .def("create_barrier_arrive_named",
           [](TritonOpBuilder &, mlir::Value, int32_t, int32_t) -> void {
             throw py::value_error(
                 "mthreads TLE named barrier backend is unsupported; "
                 "phaseIdx is required");
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
           [](TritonOpBuilder &self, std::vector<mlir::Value> explicitCaptures,
              int32_t numPartitions) -> mlir::Operation * {
             return self.create<ttg::WarpSpecializePartitionsOp>(
                 explicitCaptures, numPartitions);
           })
      .def("create_warp_specialize",
           [](TritonOpBuilder &self, std::vector<mlir::Type> resultTypes,
              std::vector<int32_t> partitionNumWarps) -> TLEWarpSpecializeOp {
             return TLEWarpSpecializeOp(self.create<ttg::WarpSpecializeOp>(
                 resultTypes, partitionNumWarps));
           })
      .def("create_exclusive_cumsum",
           [](TritonOpBuilder &self, mlir::Type exclusiveTy, mlir::Type totalTy,
              mlir::Value src, int axis, bool reverse) -> mlir::OpState {
             auto &builder = self.getBuilder();
             return self.create<mlir::triton::musa_tle::ExclusiveCumsumOp>(
                 mlir::TypeRange{exclusiveTy, totalTy}, src,
                 builder.getI32IntegerAttr(axis), builder.getBoolAttr(reverse));
           })
      .def("create_extract_tile",
           [](TritonOpBuilder &self, mlir::Value input, mlir::Value index,
              std::vector<int64_t> tileShape) -> mlir::Value {
             auto op = self.create<mlir::triton::musa_tle::ExtractTileOp>(
                 input, index, tileShape);
             return op.getResult();
           })
      .def("create_insert_tile",
           [](TritonOpBuilder &self, mlir::Value input, mlir::Value tile,
              mlir::Value index) -> mlir::Value {
             auto op = self.create<mlir::triton::musa_tle::InsertTileOp>(
                 input, tile, index);
             return op.getResult();
           });
}

void init_triton_musa_tle_dialect_passes_ttgpuir(py::module m) {
  ADD_PASS_WRAPPER_0("add_tle_select_encodings",
                     mlir::createTritonMUSAGPUTLESelectEncodings);
  ADD_PASS_WRAPPER_0("add_tle_lower_exclusive_cumsum",
                     mlir::createTritonMUSAGPUTLELowerExclusiveCumsum);
  ADD_PASS_WRAPPER_0("add_tle_lower_barrier_allocations",
                     mlir::createTritonMUSAGPUTLELowerBarrierAllocations);
  ADD_PASS_WRAPPER_0("add_tle_lower_tme_transactions",
                     mlir::createTritonMUSAGPUTLELowerTMETransactions);
  ADD_PASS_WRAPPER_0("add_tle_lower_barrier_operations",
                     mlir::createTritonMUSAGPUTLELowerBarrierOperations);
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
