#ifdef __ILUVATAR_TLE__

#include "IR/Dialect.h"
#include "Transforms/Passes.h"
#include "ir.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "passes.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
namespace ttg = mlir::triton::gpu;
namespace iluvatar_tle = mlir::triton::iluvatar_tle;

// Defined in triton_iluvatar_tle_raw.cc.
extern std::vector<int64_t>
computeAliasOperandIndices(TritonOpBuilder &self, std::string_view text,
                           const std::vector<mlir::Value> &args,
                           std::string_view funcName);

extern iluvatar_tle::DSLRegionOp
createTLERawRegionByLLVMFunc(TritonOpBuilder &self, std::string_view text,
                             std::string_view regionDialect,
                             std::string_view argDialect,
                             const std::vector<mlir::Value> &args,
                             const std::vector<int64_t> &aliasOperandIndices,
                             std::string_view hint, std::string_view funcName);

extern iluvatar_tle::DSLRegionOp createTLERawRegionDeferred(
    TritonOpBuilder &self, std::string_view sourceId,
    std::string_view regionDialect, std::string_view argDialect,
    const std::vector<mlir::Value> &args,
    const std::vector<int64_t> &aliasOperandIndices, std::string_view hint,
    std::string_view dsl_file_name, std::string_view extern_func_name);

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
      .def(
          "make_swizzled_shared_encoding_attr",
          [](TritonOpBuilder &self, unsigned vectorSize, unsigned perPhase,
             unsigned maxPhase, std::vector<unsigned> order,
             std::vector<unsigned> CTAsPerCGA,
             std::vector<unsigned> CTASplitNum, std::vector<unsigned> CTAOrder,
             bool useTcu) -> mlir::Attribute {
            checkCtaRank(order, CTAsPerCGA, CTASplitNum, CTAOrder);
            auto *context = self.getBuilder().getContext();
            auto ctaLayout = ttg::CTAEncodingAttr::fromSplitParams(
                context, CTAsPerCGA, CTASplitNum, CTAOrder);
            return ttg::SwizzledSharedEncodingAttr::get(
                context, vectorSize, perPhase, maxPhase, order, ctaLayout,
                useTcu);
          },
          py::arg("vectorSize"), py::arg("perPhase"), py::arg("maxPhase"),
          py::arg("order"), py::arg("CTAsPerCGA"), py::arg("CTASplitNum"),
          py::arg("CTAOrder"), py::arg("use_tcu") = false)
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
      .def(
          "create_remote_pointers",
          [](TritonOpBuilder &self, mlir::Type resultTy,
             std::optional<mlir::Value> &src, mlir::Value shardId,
             const std::string &space,
             std::optional<mlir::Value> &offset) -> mlir::OpState {
            auto &builder = self.getBuilder();
            static const std::unordered_set<std::string> valid = {
                "cluster", "device", "node"};
            if (valid.find(space) == valid.end()) {
              throw std::invalid_argument(
                  "Invalid space: " + space +
                  ". Expected one of: cluster, device, node.");
            }
            auto space_attr = builder.getStringAttr(space);

            return self.create<iluvatar_tle::RemotePointersOp>(
                resultTy, src.value_or(mlir::Value()), shardId, space_attr,
                offset.value_or(mlir::Value()));
          },
          py::arg("resultTy"), py::arg("src") = py::none(), py::arg("shardId"),
          py::arg("space"), py::arg("offset") = py::none())
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
      .def("get_device_id",
           [](TritonOpBuilder &self, mlir::Type resultTy,
              std::optional<mlir::Value> src) -> mlir::Value {
             return self.create<iluvatar_tle::GetDeviceIdOp>(
                 resultTy, src.value_or(mlir::Value()));
           })
      .def("get_n_pes",
           [](TritonOpBuilder &self, mlir::Type resultTy,
              mlir::Value src) -> mlir::Value {
             return self.create<iluvatar_tle::GetNumPesOp>(resultTy, src);
           })
      .def("create_distributed_barrier",
           [](TritonOpBuilder &self) -> void {
             self.create<iluvatar_tle::DistributedBarrierOp>(
                 mlir::Value(), mlir::StringAttr(), mlir::StringAttr(),
                 iluvatar_tle::MemoryOrderAttr(), mlir::StringAttr(),
                 mlir::IntegerAttr(), mlir::IntegerAttr(),
                 iluvatar_tle::SyncScopeAttr(), mlir::IntegerAttr(),
                 mlir::DenseI32ArrayAttr(), mlir::DenseI32ArrayAttr(),
                 mlir::DenseI32ArrayAttr());
           })
      .def(
          "create_distributed_barrier",
          [](TritonOpBuilder &self, std::optional<mlir::Value> src,
             size_t barrier_index = 0, const std::string &space = "device",
             const std::string &group_kind = "block",
             iluvatar_tle::MemoryOrder order =
                 iluvatar_tle::MemoryOrder::ACQ_REL,
             const std::string &barrier_kind = "sync", size_t context_id = 0,
             iluvatar_tle::SyncScope memory_scope =
                 iluvatar_tle::SyncScope::SYSTEM) -> void {
            auto &builder = self.getBuilder();
            auto getOptStrAttr = [&](const std::string &s) -> mlir::StringAttr {
              return s.empty() ? mlir::StringAttr() : builder.getStringAttr(s);
            };
            auto spaceAttr = getOptStrAttr(space);
            auto kindAttr = getOptStrAttr(group_kind);
            auto orderAttr =
                builder.getAttr<iluvatar_tle::MemoryOrderAttr>(order);
            auto barrierTypeAttr = getOptStrAttr(barrier_kind);
            auto memoryScopeAttr =
                builder.getAttr<iluvatar_tle::SyncScopeAttr>(memory_scope);
            auto barrierIndexAttr =
                builder.getI32IntegerAttr(static_cast<int32_t>(barrier_index));
            auto contextIdAttr =
                builder.getI32IntegerAttr(static_cast<int32_t>(context_id));

            self.create<iluvatar_tle::DistributedBarrierOp>(
                src.value_or(mlir::Value()), spaceAttr, barrierTypeAttr,
                orderAttr, kindAttr, barrierIndexAttr, contextIdAttr,
                memoryScopeAttr, mlir::IntegerAttr(), mlir::DenseI32ArrayAttr(),
                mlir::DenseI32ArrayAttr(), mlir::DenseI32ArrayAttr());
          },
          py::arg("src") = py::none(), py::arg("barrier_index"),
          py::arg("space"), py::arg("group_kind"), py::arg("order"),
          py::arg("barrier_kind"), py::arg("context_id") = 0,
          py::arg("memory_scope") = "system")
      .def(
          "create_distributed_barrier",
          [](TritonOpBuilder &self, const std::string &groupKind,
             const std::vector<int32_t> &groupShape,
             const std::vector<int32_t> &groupAxes,
             const std::vector<int32_t> &groupMask) -> void {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            mlir::StringAttr kindAttr;
            mlir::IntegerAttr rankAttr;
            mlir::DenseI32ArrayAttr shapeAttr;
            mlir::DenseI32ArrayAttr axesAttr;
            mlir::DenseI32ArrayAttr maskAttr;

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
                shapeAttr = mlir::DenseI32ArrayAttr::get(ctx, groupShape);
              }
              if (!groupAxes.empty()) {
                axesAttr = mlir::DenseI32ArrayAttr::get(ctx, groupAxes);
              }
              if (!groupMask.empty()) {
                maskAttr = mlir::DenseI32ArrayAttr::get(ctx, groupMask);
              }
            }

            self.create<iluvatar_tle::DistributedBarrierOp>(
                mlir::Value(), mlir::StringAttr(), mlir::StringAttr(),
                iluvatar_tle::MemoryOrderAttr(), kindAttr, mlir::IntegerAttr(),
                mlir::IntegerAttr(), iluvatar_tle::SyncScopeAttr(), rankAttr,
                shapeAttr, axesAttr, maskAttr);
          },
          py::arg("group_kind"), py::arg("group_shape"), py::arg("group_axes"),
          py::arg("group_mask"))
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
}

void init_triton_iluvatar_tle_raw_ir(py::module m) {
  using ret = py::return_value_policy;

  py::class_<iluvatar_tle::DSLRegionOp>(m, "DSLRegionOp", py::module_local(),
                                        py::dynamic_attr())
      .def(
          "get_results",
          [](iluvatar_tle::DSLRegionOp &op) -> std::vector<mlir::OpResult> {
            auto results_range = op->getResults();
            return std::vector<mlir::OpResult>(results_range.begin(),
                                               results_range.end());
          },
          ret::reference)
      .def("dump", &iluvatar_tle::DSLRegionOp::dump);

  py::class_<iluvatar_tle::YieldOp>(m, "YieldOp", py::module_local(),
                                    py::dynamic_attr())
      .def("dump", &iluvatar_tle::YieldOp::dump);

  auto *builder_cls = ir::getBuilderClass();
  if (!builder_cls)
    throw std::runtime_error("triton IR builder class is not initialized");
  builder_cls->def("compute_alias_operand_indices", &computeAliasOperandIndices,
                   py::arg("text"), py::arg("args"), py::arg("func_name") = "");
  builder_cls->def("create_tle_raw_region_by_llvm_func",
                   &createTLERawRegionByLLVMFunc, py::arg("text"),
                   py::arg("region_dialect"), py::arg("arg_dialect"),
                   py::arg("args"), py::arg("output_operand_indices"),
                   py::arg("hint") = "", py::arg("func_name") = "");
  builder_cls->def(
      "create_tle_raw_region_deferred", &createTLERawRegionDeferred,
      py::arg("source_id"), py::arg("region_dialect"), py::arg("arg_dialect"),
      py::arg("args"), py::arg("output_operand_indices"), py::arg("hint") = "",
      py::arg("dsl_file_name") = "", py::arg("extern_func_name") = "");
  builder_cls->def("get_context", &TritonOpBuilder::getContext);
}

// The shared TLE frontend (python/triton/experimental/tle/language/
// distributed.py) opens with `from triton._C.libtriton.tle import attr, utils`.
// That module comes from the trunk TLE plugin, which an iluvatar build does not
// compile (FLAGTREE_TLE is OFF, FLAGTREE_ILUVATAR_TLE is ON), so the import
// silently fails and every `attr` use raises NameError at kernel compile time.
// Registering the module here keeps the fix inside third_party/iluvatar instead
// of patching the shared frontend.
//
// Only the enums the barrier path needs are exposed. The signal enums
// (SignalOpKind, SignalWaitKind) are deliberately absent: iluvatar implements
// no signal op, so binding them would be dead code. `utils` holds the trunk
// signal verifiers and is empty here for the same reason; it exists because the
// import above names it.
void init_triton_iluvatar_tle_compat(py::module &&m) {
  auto attr = m.def_submodule("attr");

  py::enum_<iluvatar_tle::FlagCXTeamKind>(attr, "FlagCXTeamKind")
      .value("Intra", iluvatar_tle::FlagCXTeamKind::INTRA)
      .value("Inter", iluvatar_tle::FlagCXTeamKind::INTER)
      .value("World", iluvatar_tle::FlagCXTeamKind::WORLD)
      .def_static(
          "from_str",
          [](std::string name) {
            return iluvatar_tle::symbolizeFlagCXTeamKind(name);
          },
          py::arg("name"))
      .def_static(
          "from_int",
          [](int value) -> std::optional<iluvatar_tle::FlagCXTeamKind> {
            if (value < 0 ||
                value > iluvatar_tle::getMaxEnumValForFlagCXTeamKind())
              return std::nullopt;
            return static_cast<iluvatar_tle::FlagCXTeamKind>(value);
          },
          py::arg("value"));
  py::enum_<iluvatar_tle::FlagCXCoopKind>(attr, "FlagCXCoopKind")
      .value("Thread", iluvatar_tle::FlagCXCoopKind::THREAD)
      .value("Warp", iluvatar_tle::FlagCXCoopKind::WARP)
      .value("Block", iluvatar_tle::FlagCXCoopKind::BLOCK)
      .def_static(
          "from_str",
          [](std::string name) {
            return iluvatar_tle::symbolizeFlagCXCoopKind(name);
          },
          py::arg("name"));

  py::enum_<iluvatar_tle::SyncScope>(attr, "SyncScope")
      .value("System", iluvatar_tle::SyncScope::SYSTEM)
      .value("Device", iluvatar_tle::SyncScope::DEVICE)
      .value("Block", iluvatar_tle::SyncScope::BLOCK)
      .value("Thread", iluvatar_tle::SyncScope::THREAD)
      .def_static(
          "from_str",
          [](std::string name) {
            return iluvatar_tle::symbolizeSyncScope(name);
          },
          py::arg("name"));
  py::enum_<iluvatar_tle::MemoryOrder>(attr, "MemoryOrder")
      .value("Relaxed", iluvatar_tle::MemoryOrder::RELAXED)
      .value("Acquire", iluvatar_tle::MemoryOrder::ACQUIRE)
      .value("Release", iluvatar_tle::MemoryOrder::RELEASE)
      .value("AcqRel", iluvatar_tle::MemoryOrder::ACQ_REL)
      .def_static(
          "from_str",
          [](std::string name) { return iluvatar_tle::parseMemoryOrder(name); },
          py::arg("name"));

  m.def_submodule("utils");
}

void init_triton_iluvatar_tle_llvm(py::module m) {
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

void init_triton_iluvatar_tle_raw_passes(py::module m) {
  ADD_PASS_WRAPPER_0("add_tle_convert_arg_to_memdesc",
                     iluvatar_tle::createTritonIluvatarTleConvertArgToMemDesc);
  ADD_PASS_WRAPPER_0("add_tle_remove_redundant_copy",
                     iluvatar_tle::createTritonIluvatarTleRemoveRedundantCopy);
  ADD_PASS_WRAPPER_0("add_tle_dsl_region_inline",
                     iluvatar_tle::createTritonIluvatarTleDSLRegionInline);
}

void init_triton_iluvatar_tle_passes(py::module m) {
  ADD_PASS_WRAPPER_0("add_params_for_distribution",
                     iluvatar_tle::createTritonIluvatarTleAddDistributedParams);
  ADD_PASS_WRAPPER_0(
      "add_early_assign_memory_space",
      iluvatar_tle::createTritonIluvatarTleEarlyAssignMemorySpace);
  ADD_PASS_OPTION_WRAPPER_2(
      "add_optimize_local_pointer_async_stores",
      iluvatar_tle::createTritonIluvatarTleOptimizeLocalPointerAsyncStores,
      unsigned, int64_t);
  ADD_PASS_OPTION_WRAPPER_1(
      "add_mark_sme_dot_operands",
      iluvatar_tle::createTritonIluvatarTleMarkSmeDotOperands, unsigned);
  ADD_PASS_OPTION_WRAPPER_1(
      "add_promote_local_store_staging",
      iluvatar_tle::createTritonIluvatarTlePromoteLocalStoreStaging, int64_t);
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
