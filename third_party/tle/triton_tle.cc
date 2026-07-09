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
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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

extern std::vector<int64_t>
computeAliasOperandIndices(TritonOpBuilder &self, std::string_view text,
                           const std::vector<Value> &args);

extern tle::DSLRegionOp createTLERawRegionByLLVMFunc(
    TritonOpBuilder &self, std::string_view text,
    std::string_view regionDialect, std::string_view argDialect,
    const std::vector<Value> &args,
    const std::vector<int64_t> &aliasOperandIndices, std::string_view hint);

extern tle::DSLRegionOp createTLERawRegionDeferred(
    TritonOpBuilder &self, std::string_view sourceId,
    std::string_view regionDialect, std::string_view argDialect,
    const std::vector<Value> &args,
    const std::vector<int64_t> &aliasOperandIndices, std::string_view hint);

void init_triton_tle_ir(py::module &&m) {

  // Get the existing builder class from the main ir module (TLX style)
  auto &builder_cls = *ir::getBuilderClass();

  // Add TLE extensions to the existing TritonOpBuilder class
  builder_cls
      // TLE-Lite
      .def(
          "create_extract_tile",
          [](TritonOpBuilder &self, Value &input,
             // std::vector<int64_t> &offsets,
             Value &index, std::vector<int64_t> &tileShape) -> Value {
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
      // TLE-Struct
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
      .def("make_nv_mma_encoding_attr",
           [](TritonOpBuilder &self, Value opndA, Value opndAcc,
              unsigned versionMajor, unsigned versionMinor,
              unsigned moduleNumWarps) {
             auto context = self.getBuilder().getContext();
             auto dtypeA =
                 cast<ttg::TensorOrMemDesc>(opndA.getType()).getElementType();
             auto retType = cast<RankedTensorType>(opndAcc.getType());
             Operation *parentOp =
                 self.getBuilder().getInsertionBlock()->getParentOp();
             unsigned numWarps =
                 ttg::maybeLookupNumWarps(parentOp).value_or(moduleNumWarps);
             auto instrShape = mmaVersionToInstrShape(
                 versionMajor, retType.getShape(), dtypeA, numWarps);

             // Match the current Hopper WGMMA lowering convention: partition
             // the accumulator rows across the warp group.
             SmallVector<unsigned, 2> warpsPerCTA = {numWarps, 1};
             SmallVector<unsigned, 2> CTAsPerCGA = {1, 1};
             SmallVector<unsigned, 2> CTASplitNum = {1, 1};
             SmallVector<unsigned, 2> CTAOrder = {1, 0};
             auto CTALayout = ttg::CTAEncodingAttr::fromSplitParams(
                 context, CTAsPerCGA, CTASplitNum, CTAOrder);
             return mlir::cast<Attribute>(ttg::NvidiaMmaEncodingAttr::get(
                 context, versionMajor, versionMinor, warpsPerCTA, CTALayout,
                 instrShape));
           })
      .def("make_dot_operand_encoding_attr",
           [](TritonOpBuilder &self, Value opnd, unsigned opIdx,
              Attribute parentEnc) -> Attribute {
             auto context = self.getBuilder().getContext();
             auto eltType =
                 cast<RankedTensorType>(opnd.getType()).getElementType();
             return ttg::DotOperandEncodingAttr::get(context, opIdx, parentEnc,
                                                     eltType);
           })
      .def("get_block_ty_with_encoding",
           [](TritonOpBuilder &self, Type &elementType,
              std::vector<int64_t> &shape, Attribute &encoding) -> Type {
             return RankedTensorType::get(shape, elementType, encoding);
           })
      .def("create_convert_layout",
           [](TritonOpBuilder &self, Type resultTy, Value value) -> Value {
             return self.create<ttg::ConvertLayoutOp>(resultTy, value);
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
#ifdef __HCU__
             self.create<ttg::TMACopyOp>(src, dst, indices);
#else
             self.create<ttg::TMACopyOp>(src, dst, indices, Value(),
                                         IntegerAttr());
#endif
             return;
           })
      .def(
          "create_tma_copy",
          [](TritonOpBuilder &self, Value src, Value dst,
             std::vector<Value> &indices, py::object barrier,
             int32_t expectBytes) {
#ifdef __HCU__
            if (!barrier.is_none() || expectBytes > 0)
              throw py::value_error(
                  "TMA completion barrier is only supported on NVIDIA backend");
            self.create<ttg::TMACopyOp>(src, dst, indices);
#else
             auto &builder = self.getBuilder();
             Value barrierValue;
             if (!barrier.is_none())
               barrierValue = py::cast<Value>(barrier);
             IntegerAttr expectBytesAttr;
             if (expectBytes > 0)
               expectBytesAttr = builder.getI32IntegerAttr(expectBytes);
             self.create<ttg::TMACopyOp>(src, dst, indices, barrierValue,
                                         expectBytesAttr);
#endif
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
      .def("create_tle_wgmma",
           [](TritonOpBuilder &self, mlir::Value &a, mlir::Value &b,
              mlir::Value &c, triton::InputPrecision inputPrecision,
              int maxNumImpreciseAcc, bool isAsync) -> mlir::Value {
             return self.create<tle::WGMMAOp>(c.getType(), a, b, c,
                                              inputPrecision,
                                              maxNumImpreciseAcc, isAsync);
           })
      .def("create_tle_wgmma_wait",
           [](TritonOpBuilder &self, mlir::Value &input,
              unsigned pendings) -> mlir::Value {
             auto pendingsAttr = self.getBuilder().getI32IntegerAttr(pendings);
             return self
                 .create<tle::WGMMAWaitOp>(input.getType(), input, pendingsAttr)
                 .getOutput();
           })
      .def("create_warp_group_dot",
           [](TritonOpBuilder &self, mlir::Value &a, mlir::Value &b,
              mlir::Value &c, triton::InputPrecision inputPrecision,
              int maxNumImpreciseAcc, bool isAsync) -> mlir::Value {
             return self.create<ttng::WarpGroupDotOp>(
                 c.getType(), a, b, c, Value(), inputPrecision,
                 maxNumImpreciseAcc, isAsync);
           })
      .def("create_warp_group_dot_wait",
           [](TritonOpBuilder &self, std::vector<Value> inputs,
              unsigned pendings) -> std::vector<Value> {
             auto waitOp =
                 self.create<ttng::WarpGroupDotWaitOp>(inputs, pendings);
             std::vector<Value> outputs;
             outputs.reserve(waitOp->getNumResults());
             for (Value result : waitOp->getResults())
               outputs.push_back(result);
             return outputs;
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
      .def("create_memdesc_index",
           [](TritonOpBuilder &self, Type resultType, Value src,
              Value index) -> Value {
             return self.create<ttg::MemDescIndexOp>(resultType, src, index);
           })
      .def("create_memdesc_trans",
           [](TritonOpBuilder &self, Value src,
              std::vector<int> &order) -> Value {
             return self.create<ttg::MemDescTransOp>(src, order);
           })
      .def("create_barrier_alloc",
           [](TritonOpBuilder &self, Type resultType, int32_t numBarriers,
              int32_t arriveCount, int32_t initPolarity,
              int32_t expectBytes) -> Value {
             auto &builder = self.getBuilder();
             IntegerAttr expectBytesAttr;
             if (expectBytes > 0)
               expectBytesAttr = builder.getI32IntegerAttr(expectBytes);
             return self.create<tle::BarrierAllocOp>(
                 resultType, builder.getI32IntegerAttr(numBarriers),
                 builder.getI32IntegerAttr(arriveCount),
                 builder.getI32IntegerAttr(initPolarity), expectBytesAttr);
           })
      .def("create_barrier_wait_mbarrier",
           [](TritonOpBuilder &self, Value barrier, Value phase) -> void {
             auto &builder = self.getBuilder();
             self.create<tle::BarrierWaitOp>(
                 barrier, phase, builder.getStringAttr("mbarrier"),
                 builder.getI32IntegerAttr(0), builder.getI32IntegerAttr(0));
           })
      .def("create_barrier_wait_named",
           [](TritonOpBuilder &self, Value barrier, int32_t namedId,
              int32_t numThreads) -> void {
             auto &builder = self.getBuilder();
             self.create<tle::BarrierWaitOp>(
                 barrier, Value(), builder.getStringAttr("named"),
                 builder.getI32IntegerAttr(namedId),
                 builder.getI32IntegerAttr(numThreads));
           })
      .def("create_barrier_arrive_mbarrier",
           [](TritonOpBuilder &self, Value barrier, int32_t arriveCount,
              py::object phase) -> void {
             auto &builder = self.getBuilder();
             Value phaseValue;
             if (!phase.is_none())
               phaseValue = py::cast<Value>(phase);
             self.create<tle::BarrierArriveOp>(
                 barrier, phaseValue, builder.getStringAttr("mbarrier"),
                 builder.getI32IntegerAttr(arriveCount),
                 builder.getI32IntegerAttr(0), builder.getI32IntegerAttr(0));
           })
      .def("create_barrier_arrive_named",
           [](TritonOpBuilder &self, Value barrier, int32_t namedId,
              int32_t numThreads) -> void {
             auto &builder = self.getBuilder();
             self.create<tle::BarrierArriveOp>(
                 barrier, Value(), builder.getStringAttr("named"),
                 builder.getI32IntegerAttr(1),
                 builder.getI32IntegerAttr(namedId),
                 builder.getI32IntegerAttr(numThreads));
           })
      .def("create_memdesc_subslice",
           [](TritonOpBuilder &self, Type resultType, Value src,
              std::vector<int32_t> &offsets) -> Value {
             return self.create<ttg::MemDescSubsliceOp>(resultType, src,
                                                        offsets);
           })
      .def("create_warp_return",
           [](TritonOpBuilder &self) -> Operation * {
             return self.create<ttg::WarpReturnOp>();
           })
      .def("create_warp_yield",
           [](TritonOpBuilder &self, std::vector<Value> values) -> Operation * {
             return self.create<ttg::WarpYieldOp>(values);
           })
      .def("create_warp_specialize_partitions",
           [](TritonOpBuilder &self, int numPartitions) -> Operation * {
             return self.create<ttg::WarpSpecializePartitionsOp>(numPartitions);
           })
      .def("create_warp_specialize",
           [](TritonOpBuilder &self, std::vector<Type> resultTypes,
              std::vector<Value> explicitCaptures,
              std::vector<int> partitionNumWarps) {
             return self.create<ttg::WarpSpecializeOp>(
                 resultTypes, explicitCaptures, partitionNumWarps);
           })
      .def("create_pipe_create",
           [](TritonOpBuilder &self, std::vector<Value> fields,
              int32_t capacity, const std::string &scope,
              const std::string &pipeName, std::vector<std::string> fieldNames,
              std::vector<std::string> readerNames, bool oneShot) -> void {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             SmallVector<Attribute> readerNameAttrs;
             readerNameAttrs.reserve(readerNames.size());
             for (StringRef name : readerNames)
               readerNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             ArrayAttr readerNamesAttr;
             if (!readerNameAttrs.empty())
               readerNamesAttr = builder.getArrayAttr(readerNameAttrs);
             BoolAttr oneShotAttr;
             if (oneShot)
               oneShotAttr = builder.getBoolAttr(true);
             self.create<tle::PipeCreateOp>(
                 fields, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs), readerNamesAttr,
                 oneShotAttr);
           })
      .def("create_pipe_writer_acquire",
           [](TritonOpBuilder &self, std::vector<Value> fields, Value stage,
              Value phase, int32_t capacity, const std::string &scope,
              const std::string &pipeName,
              std::vector<std::string> fieldNames) -> void {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             self.create<tle::PipeWriterAcquireOp>(
                 fields, stage, phase, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs));
           })
      .def("create_pipe_writer_commit",
           [](TritonOpBuilder &self, std::vector<Value> fields, Value stage,
              int32_t capacity, const std::string &scope,
              const std::string &pipeName,
              std::vector<std::string> fieldNames) -> void {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             self.create<tle::PipeWriterCommitOp>(
                 fields, stage, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs));
           })
      .def("create_pipe_writer_close",
           [](TritonOpBuilder &self, std::vector<Value> fields, Value stage,
              Value phase, int32_t capacity, const std::string &scope,
              const std::string &pipeName,
              std::vector<std::string> fieldNames) -> void {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             self.create<tle::PipeWriterCloseOp>(
                 fields, stage, phase, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs));
           })
      .def("create_pipe_reader_wait",
           [](TritonOpBuilder &self, std::vector<Value> fields, Value stage,
              Value phase, int32_t capacity, const std::string &scope,
              const std::string &pipeName, std::vector<std::string> fieldNames,
              const std::string &readerName,
              std::vector<std::string>) -> Value {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             StringAttr readerNameAttr;
             if (!readerName.empty())
               readerNameAttr = builder.getStringAttr(readerName);
             return self.create<tle::PipeReaderWaitOp>(
                 builder.getI1Type(), fields, stage, phase,
                 builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs), readerNameAttr);
           })
      .def("create_pipe_reader_release",
           [](TritonOpBuilder &self, std::vector<Value> fields, Value stage,
              int32_t capacity, const std::string &scope,
              const std::string &pipeName, std::vector<std::string> fieldNames,
              const std::string &readerName, std::vector<std::string>) -> void {
             auto &builder = self.getBuilder();
             SmallVector<Attribute> fieldNameAttrs;
             fieldNameAttrs.reserve(fieldNames.size());
             for (StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             StringAttr readerNameAttr;
             if (!readerName.empty())
               readerNameAttr = builder.getStringAttr(readerName);
             self.create<tle::PipeReaderReleaseOp>(
                 fields, stage, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs), readerNameAttr);
           })
      .def("create_exclusive_cumsum",
           [](TritonOpBuilder &self, Type exclusiveTy, Type totalTy, Value src,
              int axis, bool reverse) -> OpState {
             auto &builder = self.getBuilder();
             return self.create<tle::ExclusiveCumsumOp>(
                 TypeRange{exclusiveTy, totalTy}, src,
                 builder.getI32IntegerAttr(axis), builder.getBoolAttr(reverse));
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
      .def(
          "create_remote_pointers",
          [](TritonOpBuilder &self, Type resultTy, Value src, Value shardId,
             const std::string &space,
             std::optional<Value> &offset) -> OpState {
            auto &builder = self.getBuilder();
            static const std::unordered_set<std::string> valid = {
                "cluster", "device", "node"};
            if (valid.find(space) == valid.end()) {
              throw std::invalid_argument(
                  "Invalid space: " + space +
                  ". Expected one of: cluster, device, node.");
            }
            auto space_attr = builder.getStringAttr(space);
            return self.create<tle::RemotePointersOp>(
                resultTy, src, shardId, space_attr, offset.value_or(Value()));
          },
          py::arg("resultTy"), py::arg("src"), py::arg("shardId"),
          py::arg("space"), py::arg("offset") = py::none())
      .def("get_device_id",
           [](TritonOpBuilder &self, Type resultTy, Value src) -> Value {
             auto &builder = self.getBuilder();
             return self.create<tle::GetDeviceIdOp>(resultTy, src);
           })
      .def("get_n_pes",
           [](TritonOpBuilder &self, Type resultTy, Value src) -> Value {
             auto &builder = self.getBuilder();
             return self.create<tle::GetNumPesOp>(resultTy, src);
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
           })
      .def("get_memdesc_type",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              Type &elementType, Attribute &encoding, std::string storage,
              std::vector<int64_t> allocShape) -> Type {
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
                                          memorySpace, /*mutableMemory=*/true,
                                          allocShape);
           });
}

void init_triton_tle_passes(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_early_assign_memory_space",
                     tle::createTritonTleEarlyAssignMemorySpace);
  ADD_PASS_WRAPPER_0("add_select_encodings",
                     tle::createTritonTleSelectEncodings);
  // Backward-compatible alias.
  ADD_PASS_WRAPPER_0("add_assign_local_pointers_encoding",
                     tle::createTritonTleSelectEncodings);
  ADD_PASS_WRAPPER_0("add_insert_local_pointer_barriers",
                     tle::createTritonTleInsertLocalPointerBarriers);
  ADD_PASS_WRAPPER_0("add_optimize_local_pointer_loads",
                     tle::createTritonTleOptimizeLocalPointerLoads);
  ADD_PASS_WRAPPER_0("add_optimize_local_pointer_stores",
                     tle::createTritonTleOptimizeLocalPointerStores);
  ADD_PASS_WRAPPER_0("add_optimize_local_pointer_async_stores",
                     tle::createTritonTleOptimizeLocalPointerAsyncStores);
  ADD_PASS_WRAPPER_0("add_promote_local_store_staging",
                     tle::createTritonTlePromoteLocalStoreStaging);
  ADD_PASS_WRAPPER_0("add_tile_style_pipeline_schedule",
                     tle::createTritonTleTileStylePipelineSchedule);
  ADD_PASS_WRAPPER_0("add_materialize_tile_style_pipeline",
                     tle::createTritonTleMaterializeTileStylePipeline);
  ADD_PASS_WRAPPER_0("add_downgrade_invalid_async_copy",
                     tle::createTritonTleDowngradeInvalidAsyncCopy);
  ADD_PASS_WRAPPER_0("add_optimize_exclusive_cumsum_layouts",
                     tle::createTritonTleOptimizeExclusiveCumsumLayouts);
  ADD_PASS_WRAPPER_0("add_lower_exclusive_cumsum",
                     tle::createTritonTleLowerExclusiveCumsum);
  ADD_PASS_WRAPPER_0("add_lower_async_load",
                     tle::createTritonTleLowerAsyncLoad);
  ADD_PASS_WRAPPER_0("add_lower_wgmma", tle::createTritonTleLowerWGMMA);
  ADD_PASS_WRAPPER_0("add_lower_pipe_to_nvws",
                     tle::createTritonTleLowerPipeToNvws);
  ADD_PASS_WRAPPER_0("add_lower_barriers", tle::createTritonTleLowerBarriers);
  ADD_PASS_WRAPPER_0("add_allocate_named_barriers",
                     tle::createTritonTleAllocateNamedBarriers);
  ADD_PASS_WRAPPER_0("add_lower_tma_copy", tle::createTritonTleLowerTmaCopy);
  ADD_PASS_WRAPPER_0("add_schedule_tma_store_sync",
                     tle::createTritonTleScheduleTmaStoreSync);

  ADD_PASS_WRAPPER_0("add_lower_extract_tile",
                     tle::createTritonTleLowerExtractTile);

  ADD_PASS_WRAPPER_0("add_lower_insert_tile",
                     tle::createTritonTleLowerInsertTile);
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
  builder_cls->def("compute_alias_operand_indices",
                   &computeAliasOperandIndices);
  builder_cls->def(
      "create_tle_raw_region_by_llvm_func", &createTLERawRegionByLLVMFunc,
      py::arg("text"), py::arg("region_dialect"), py::arg("arg_dialect"),
      py::arg("args"), py::arg("output_operand_indices"), py::arg("hint") = "");
  builder_cls->def(
      "create_tle_raw_region_deferred", &createTLERawRegionDeferred,
      py::arg("source_id"), py::arg("region_dialect"), py::arg("arg_dialect"),
      py::arg("args"), py::arg("output_operand_indices"), py::arg("hint") = "");
  builder_cls->def("get_context", &TritonOpBuilder::getContext);
}

void init_tle_raw_passes(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_tle_convert_arg_to_memdesc",
                     mlir::triton::tle::createTleConvertArgToMemDesc);
  ADD_PASS_WRAPPER_0("add_tle_remove_redundant_copy",
                     mlir::triton::tle::createTleRemoveRedundantCopy);
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
