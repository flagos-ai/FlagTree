#ifdef __TLE__

#include "Dialect/MUSATLE/IR/Dialect.h"
#include "TritonMUSACommon/MMAContractUtils.h"
#include "TritonMUSACommon/MMAEncodingUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "ir.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace musa = mlir::triton::musa;
namespace ttg = mlir::triton::gpu;
namespace tle = mlir::triton::tle;

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

ttg::CGAEncodingAttr
buildMthreadsCgaLayout(mlir::MLIRContext *context,
                       llvm::ArrayRef<std::vector<int32_t>> cgaBases,
                       unsigned rank) {
  for (auto [index, basis] : llvm::enumerate(cgaBases)) {
    if (basis.size() != rank)
      throw py::value_error("mthreads TLE CGA layout basis " +
                            std::to_string(index) + " has rank " +
                            std::to_string(basis.size()) + ", expected " +
                            std::to_string(rank));
    if (llvm::any_of(basis, [](int32_t value) { return value < 0; }))
      throw py::value_error("mthreads TLE CGA layout basis " +
                            std::to_string(index) +
                            " contains a negative value");
  }

  auto block = mlir::StringAttr::get(context, "block");
  mlir::triton::LinearLayout::BasesT bases;
  bases[block] =
      std::vector<std::vector<int32_t>>(cgaBases.begin(), cgaBases.end());
  auto outDims = mlir::triton::standardOutDimNames(context, rank);
  return ttg::CGAEncodingAttr::get(
      context,
      mlir::triton::LinearLayout(std::move(bases), std::move(outDims)));
}

bool isPositivePowerOfTwo(unsigned value) {
  return value != 0 && (value & (value - 1)) == 0;
}

void validateBlockedEncodingArguments(llvm::ArrayRef<unsigned> sizePerThread,
                                      llvm::ArrayRef<unsigned> threadsPerWarp,
                                      llvm::ArrayRef<unsigned> warpsPerCTA,
                                      llvm::ArrayRef<unsigned> order) {
  unsigned rank = order.size();
  if (rank == 0)
    throw py::value_error(
        "mthreads TLE blocked encoding rank must be positive");
  if (sizePerThread.size() != rank || threadsPerWarp.size() != rank ||
      warpsPerCTA.size() != rank)
    throw py::value_error(
        "mthreads TLE blocked encoding fields must have the same rank");

  llvm::SmallVector<bool> seen(rank, false);
  for (unsigned value : order) {
    if (value >= rank || seen[value])
      throw py::value_error(
          "mthreads TLE blocked encoding order must be a permutation of "
          "0..rank-1");
    seen[value] = true;
  }

  struct PowerOfTwoField {
    llvm::StringLiteral name;
    llvm::ArrayRef<unsigned> values;
  };
  const PowerOfTwoField fields[] = {
      {"size_per_thread", sizePerThread},
      {"threads_per_warp", threadsPerWarp},
      {"warps_per_cta", warpsPerCTA},
  };
  for (const auto &field : fields) {
    if (llvm::any_of(field.values, [](unsigned value) {
          return !isPositivePowerOfTwo(value);
        }))
      throw py::value_error("mthreads TLE blocked encoding " +
                            field.name.str() +
                            " entries must be positive powers of two");
  }

  uint64_t totalThreads = 1;
  for (unsigned value : threadsPerWarp)
    totalThreads *= value;
  if (totalThreads != 32)
    throw py::value_error("mthreads TLE PH1 blocked encoding requires "
                          "product(threads_per_warp) == 32");
}

mlir::Attribute buildMthreadsBlockedLayout(
    TritonOpBuilder &self, llvm::ArrayRef<unsigned> sizePerThread,
    llvm::ArrayRef<unsigned> threadsPerWarp,
    llvm::ArrayRef<unsigned> warpsPerCTA, llvm::ArrayRef<unsigned> order,
    llvm::ArrayRef<std::vector<int32_t>> cgaBases) {
  validateBlockedEncodingArguments(sizePerThread, threadsPerWarp, warpsPerCTA,
                                   order);
  auto *context = self.getContext();
  auto cgaLayout = buildMthreadsCgaLayout(context, cgaBases, order.size());
  auto encoding = ttg::BlockedEncodingAttr::getChecked(
      [&]() { return mlir::emitError(self.getLastLoc()); }, context,
      sizePerThread, threadsPerWarp, warpsPerCTA, order, cgaLayout);
  if (!encoding)
    throw py::value_error(
        "mthreads TLE blocked encoding failed TTGPU verification");
  return encoding;
}

mlir::Attribute buildMthreadsSlicedLayout(TritonOpBuilder &self, unsigned dim,
                                          mlir::Attribute parent) {
  auto distributed = mlir::dyn_cast<ttg::DistributedEncodingTrait>(parent);
  if (!distributed)
    throw py::value_error(
        "mthreads TLE sliced encoding parent must be a distributed encoding");
  unsigned parentRank = mlir::cast<ttg::LayoutEncodingTrait>(parent).getRank();
  if (parentRank < 2)
    throw py::value_error(
        "mthreads TLE sliced encoding parent rank must be at least 2");
  if (dim >= parentRank)
    throw py::value_error(
        "mthreads TLE sliced encoding dim must be less than parent rank");

  auto encoding = ttg::SliceEncodingAttr::getChecked(
      [&]() { return mlir::emitError(self.getLastLoc()); }, self.getContext(),
      dim, distributed);
  if (!encoding)
    throw py::value_error(
        "mthreads TLE sliced encoding failed TTGPU verification");
  return encoding;
}

mlir::Attribute buildMthreadsDotOperandLayout(TritonOpBuilder &self,
                                              unsigned operandIndex,
                                              mlir::Attribute parent,
                                              unsigned kWidth) {
  if (operandIndex > 1)
    throw py::value_error("mthreads TLE dot operand index must be 0 or 1");
  if (!mlir::isa<ttg::MUSAWmmaEncodingAttr, ttg::MUSASqmmaEncodingAttr>(parent))
    throw py::value_error(
        "mthreads TLE dot operand parent must be a MUSA WMMA or SQMMA "
        "encoding");
  if (kWidth != 0)
    throw py::value_error("mthreads TLE MUSA dot operand requires k_width=0");

  auto encoding = ttg::DotOperandEncodingAttr::getChecked(
      [&]() { return mlir::emitError(self.getLastLoc()); }, self.getContext(),
      operandIndex, parent, kWidth);
  if (!encoding)
    throw py::value_error(
        "mthreads TLE dot operand encoding failed TTGPU verification");
  return encoding;
}

mlir::Type cloneMthreadsTensorTypeWithEncoding(TritonOpBuilder &self,
                                               mlir::Type type,
                                               mlir::Attribute encoding) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType)
    throw py::type_error(
        "mthreads TLE can only clone a ranked tensor type with an encoding");
  auto distributed = mlir::dyn_cast<ttg::DistributedEncodingTrait>(encoding);
  if (!distributed)
    throw py::type_error(
        "mthreads TLE tensor encoding must be a distributed encoding");
  unsigned encodingRank =
      mlir::cast<ttg::LayoutEncodingTrait>(encoding).getRank();
  if (static_cast<unsigned>(tensorType.getRank()) != encodingRank)
    throw py::value_error(
        "mthreads TLE tensor rank must match distributed encoding rank");
  return tensorType.cloneWithEncoding(encoding);
}

void validateMthreadsSetLayoutArguments(mlir::Value value,
                                        mlir::Attribute targetEncoding) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!tensorType)
    throw py::type_error(
        "mthreads TLE set_layout source must be a ranked tensor");

  if (!mlir::dyn_cast<ttg::DistributedEncodingTrait>(targetEncoding))
    throw py::type_error(
        "mthreads TLE set_layout target_encoding must be a distributed "
        "encoding");

  auto layoutEncoding =
      mlir::dyn_cast<ttg::LayoutEncodingTrait>(targetEncoding);
  if (!layoutEncoding)
    throw py::type_error(
        "mthreads TLE set_layout distributed target_encoding must expose a "
        "layout rank");

  unsigned targetRank = layoutEncoding.getRank();
  if (targetRank != static_cast<unsigned>(tensorType.getRank()))
    throw py::value_error("mthreads TLE set_layout target encoding rank " +
                          std::to_string(targetRank) +
                          " must match source tensor rank " +
                          std::to_string(tensorType.getRank()));
}

void validateMusaMmaLayoutArguments(
    llvm::StringRef layoutName, llvm::ArrayRef<unsigned> version,
    llvm::ArrayRef<unsigned> warpsPerCTA,
    llvm::ArrayRef<std::vector<int32_t>> cgaBases,
    llvm::ArrayRef<unsigned> instrShape) {
  if (version.size() != 2)
    throw py::value_error("mthreads TLE " + layoutName.str() +
                          " version must contain major and minor");
  const auto &ph1Traits = musa::getMusaWmmaArchTraits(musa::MusaArch::PH1);
  if (version[0] != ph1Traits.versionMajor ||
      version[1] != ph1Traits.versionMinor)
    throw py::value_error("mthreads TLE " + layoutName.str() +
                          " currently supports only MUSA PH1 version [3, 1]");
  if (warpsPerCTA.size() != 2 && warpsPerCTA.size() != 3)
    throw py::value_error("mthreads TLE " + layoutName.str() +
                          " warps_per_cta rank must be 2 or 3");
  if (llvm::any_of(warpsPerCTA,
                   [](unsigned value) { return !isPositivePowerOfTwo(value); }))
    throw py::value_error("mthreads TLE " + layoutName.str() +
                          " warps_per_cta entries must be positive powers of "
                          "two");
  if (warpsPerCTA.size() == 3 && warpsPerCTA[2] != 1)
    throw py::value_error("mthreads TLE " + layoutName.str() +
                          " rank-3 warps_per_cta must end in 1");
  if (instrShape.size() != 3)
    throw py::value_error("mthreads TLE " + layoutName.str() +
                          " instr_shape must contain logical (M, N, K)");

  // Validate the CGA bases before constructing a LinearLayout so malformed
  // input is reported at the Python/native boundary.
  for (auto [index, basis] : llvm::enumerate(cgaBases)) {
    if (basis.size() != warpsPerCTA.size())
      throw py::value_error("mthreads TLE " + layoutName.str() +
                            " CGA layout basis " + std::to_string(index) +
                            " has rank " + std::to_string(basis.size()) +
                            ", expected " + std::to_string(warpsPerCTA.size()));
    if (llvm::any_of(basis, [](int32_t value) { return value < 0; }))
      throw py::value_error("mthreads TLE " + layoutName.str() +
                            " CGA layout basis " + std::to_string(index) +
                            " contains a negative value");
  }
}

bool isSupportedWmmaInstructionShape(llvm::ArrayRef<unsigned> instrShape) {
  return llvm::any_of(musa::kWmmaIntrinsics, [&](const auto &intrinsic) {
    return intrinsic.m == instrShape[0] && intrinsic.n == instrShape[1] &&
           intrinsic.k == instrShape[2];
  });
}

bool isSupportedSqmmaInstructionShape(llvm::ArrayRef<unsigned> instrShape) {
  const unsigned m = instrShape[0];
  const unsigned n = instrShape[1];
  const unsigned k = instrShape[2];
  const auto &sqTraits = *musa::getMusaSqmmaArchTraits(musa::MusaArch::PH1);
  return musa::isSupportedSqmma(musa::SQMMAEltType::f16,
                                musa::SQMMAEltType::f16,
                                musa::SQMMAEltType::f32, m, n, k, sqTraits) ||
         musa::isSupportedSqmma(musa::SQMMAEltType::bf16,
                                musa::SQMMAEltType::bf16,
                                musa::SQMMAEltType::f32, m, n, k, sqTraits) ||
         musa::isSupportedSqmma(musa::SQMMAEltType::tf32,
                                musa::SQMMAEltType::tf32,
                                musa::SQMMAEltType::f32, m, n, k, sqTraits) ||
         musa::isSupportedSqmma(musa::SQMMAEltType::s8, musa::SQMMAEltType::s8,
                                musa::SQMMAEltType::s32, m, n, k, sqTraits) ||
         musa::isSupportedSqmma(musa::SQMMAEltType::e4m3,
                                musa::SQMMAEltType::e4m3,
                                musa::SQMMAEltType::f32, m, n, k, sqTraits) ||
         musa::isSupportedSqmma(musa::SQMMAEltType::e5m2,
                                musa::SQMMAEltType::e5m2,
                                musa::SQMMAEltType::f32, m, n, k, sqTraits);
}

mlir::Attribute
buildMusaWmmaLayout(TritonOpBuilder &self, llvm::ArrayRef<unsigned> version,
                    llvm::ArrayRef<unsigned> warpsPerCTA,
                    llvm::ArrayRef<std::vector<int32_t>> cgaBases,
                    llvm::ArrayRef<unsigned> instrShape) {
  validateMusaMmaLayoutArguments("WMMA", version, warpsPerCTA, cgaBases,
                                 instrShape);
  if (!isSupportedWmmaInstructionShape(instrShape))
    throw py::value_error(
        "mthreads TLE WMMA instr_shape is not supported by any PH1 WMMA "
        "intrinsic");

  auto *context = self.getContext();
  auto cgaLayout =
      buildMthreadsCgaLayout(context, cgaBases, warpsPerCTA.size());
  auto encoding = ttg::MUSAWmmaEncodingAttr::getChecked(
      [&]() { return mlir::emitError(self.getLastLoc()); }, context, version[0],
      version[1], warpsPerCTA, cgaLayout, instrShape);
  if (!encoding)
    throw py::value_error(
        "mthreads TLE WMMA encoding failed TTGPU verification");
  return encoding;
}

mlir::Attribute
buildMusaSqmmaLayout(TritonOpBuilder &self, llvm::ArrayRef<unsigned> version,
                     llvm::ArrayRef<unsigned> warpsPerCTA,
                     llvm::ArrayRef<std::vector<int32_t>> cgaBases,
                     llvm::ArrayRef<unsigned> instrShape) {
  validateMusaMmaLayoutArguments("SQMMA", version, warpsPerCTA, cgaBases,
                                 instrShape);
  if (warpsPerCTA[0] % 4 != 0)
    throw py::value_error(
        "mthreads TLE SQMMA warps_per_cta[0] must be a multiple of 4");
  if (!isSupportedSqmmaInstructionShape(instrShape))
    throw py::value_error(
        "mthreads TLE SQMMA instr_shape is not supported by any PH1 SQMMA "
        "type contract");

  auto *context = self.getContext();
  auto cgaLayout =
      buildMthreadsCgaLayout(context, cgaBases, warpsPerCTA.size());
  auto encoding = ttg::MUSASqmmaEncodingAttr::getChecked(
      [&]() { return mlir::emitError(self.getLastLoc()); }, context, version[0],
      version[1], warpsPerCTA, cgaLayout, instrShape);
  if (!encoding)
    throw py::value_error(
        "mthreads TLE SQMMA encoding failed TTGPU verification");
  return encoding;
}

mlir::ModuleOp findParentModule(TritonOpBuilder &self) {
  auto *block = self.getBuilder().getInsertionBlock();
  if (!block)
    throw py::value_error(
        "mthreads TLE cannot set ttg layout attributes without an insertion "
        "block");

  mlir::Operation *op = block->getParentOp();
  while (op && !mlir::isa<mlir::ModuleOp>(op))
    op = op->getParentOp();
  auto module = mlir::dyn_cast_or_null<mlir::ModuleOp>(op);
  if (!module)
    throw py::value_error(
        "mthreads TLE cannot find parent module for ttg layout attributes");
  return module;
}

std::string printAttribute(mlir::Attribute attr) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  attr.print(stream);
  return stream.str();
}

void ensureTtgLayoutAttrs(TritonOpBuilder &self, int numWarps,
                          int threadsPerWarp, int numCtas) {
  if (numWarps <= 0 || (static_cast<unsigned>(numWarps) &
                        (static_cast<unsigned>(numWarps) - 1)) != 0)
    throw py::value_error(
        "mthreads TLE num_warps must be a positive power of two");
  if (threadsPerWarp != 32)
    throw py::value_error(
        "mthreads TLE PH1 requires warp_size (threads per warp) to be 32");
  if (numCtas <= 0)
    throw py::value_error("mthreads TLE num_ctas must be positive");

  auto module = findParentModule(self);
  struct LayoutAttrSpec {
    llvm::StringLiteral name;
    int value;
  };
  const LayoutAttrSpec specs[] = {
      {"ttg.num-warps", numWarps},
      {"ttg.threads-per-warp", threadsPerWarp},
      {"ttg.num-ctas", numCtas},
  };

  // Validate all existing attributes before adding any missing ones. This
  // keeps the module unchanged when one of the requested values conflicts.
  for (const auto &spec : specs) {
    auto attr = module->getAttr(spec.name);
    if (!attr)
      continue;
    auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attr);
    if (!integer || !integer.getType().isInteger(32) ||
        integer.getInt() != spec.value) {
      throw py::value_error("mthreads TLE layout attribute '" +
                            spec.name.str() + "' mismatch: module has " +
                            printAttribute(attr) + ", requested " +
                            std::to_string(spec.value) + " : i32");
    }
  }

  auto i32 = mlir::IntegerType::get(self.getContext(), 32);
  for (const auto &spec : specs) {
    if (!module->hasAttr(spec.name))
      module->setAttr(spec.name, mlir::IntegerAttr::get(i32, spec.value));
  }
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
      .def("ensure_ttg_layout_attrs",
           [](TritonOpBuilder &self, int numWarps, int threadsPerWarp,
              int numCtas) {
             ensureTtgLayoutAttrs(self, numWarps, threadsPerWarp, numCtas);
           })
      .def("get_blocked_encoding",
           [](TritonOpBuilder &self, std::vector<unsigned> sizePerThread,
              std::vector<unsigned> threadsPerWarp,
              std::vector<unsigned> warpsPerCTA, std::vector<unsigned> order,
              std::vector<std::vector<int32_t>> cgaBases) -> mlir::Attribute {
             return buildMthreadsBlockedLayout(self, sizePerThread,
                                               threadsPerWarp, warpsPerCTA,
                                               order, cgaBases);
           })
      .def("get_sliced_encoding",
           [](TritonOpBuilder &self, unsigned dim,
              mlir::Attribute parent) -> mlir::Attribute {
             return buildMthreadsSlicedLayout(self, dim, parent);
           })
      .def("get_dot_operand_layout",
           [](TritonOpBuilder &self, unsigned operandIndex,
              mlir::Attribute parent, unsigned kWidth) -> mlir::Attribute {
             return buildMthreadsDotOperandLayout(self, operandIndex, parent,
                                                  kWidth);
           })
      .def("clone_tensor_type_with_encoding",
           [](TritonOpBuilder &self, mlir::Type type,
              mlir::Attribute encoding) -> mlir::Type {
             return cloneMthreadsTensorTypeWithEncoding(self, type, encoding);
           })
      .def("get_musa_wmma_layout",
           [](TritonOpBuilder &self, std::vector<unsigned> version,
              std::vector<unsigned> warpsPerCTA,
              std::vector<std::vector<int32_t>> cgaBases,
              std::vector<unsigned> instrShape) -> mlir::Attribute {
             return buildMusaWmmaLayout(self, version, warpsPerCTA, cgaBases,
                                        instrShape);
           })
      .def("get_musa_sqmma_layout",
           [](TritonOpBuilder &self, std::vector<unsigned> version,
              std::vector<unsigned> warpsPerCTA,
              std::vector<std::vector<int32_t>> cgaBases,
              std::vector<unsigned> instrShape) -> mlir::Attribute {
             return buildMusaSqmmaLayout(self, version, warpsPerCTA, cgaBases,
                                         instrShape);
           })
      .def("create_tle_gpu_set_layout",
           [](TritonOpBuilder &self, mlir::Value value,
              mlir::Attribute targetEncoding) -> mlir::Value {
             validateMthreadsSetLayoutArguments(value, targetEncoding);
             return self.create<mlir::triton::musa_tle::SetLayoutOp>(
                 value.getType(), value, targetEncoding);
           })
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
      .def("mark_musa_tle_auto_shared_layout",
           [](TritonOpBuilder &self, mlir::Value value) -> void {
             mlir::Operation *def = value.getDefiningOp();
             if (!def || !mlir::isa<ttg::LocalAllocOp>(def))
               throw py::value_error(
                   "mthreads TLE auto shared layout marker requires a "
                   "ttg.local_alloc result");
             def->setAttr("musa_tle.auto_shared_layout",
                          self.getBuilder().getUnitAttr());
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
      .def("create_pipe_create",
           [](TritonOpBuilder &self, std::vector<mlir::Value> fields,
              int32_t capacity, const std::string &scope,
              const std::string &pipeName, std::vector<std::string> fieldNames,
              std::vector<std::string> readerNames, bool oneShot) -> void {
             auto &builder = self.getBuilder();
             llvm::SmallVector<mlir::Attribute> fieldNameAttrs;
             for (llvm::StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             llvm::SmallVector<mlir::Attribute> readerNameAttrs;
             for (llvm::StringRef name : readerNames)
               readerNameAttrs.push_back(builder.getStringAttr(name));
             mlir::StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             mlir::ArrayAttr readersAttr;
             if (!readerNameAttrs.empty())
               readersAttr = builder.getArrayAttr(readerNameAttrs);
             mlir::BoolAttr oneShotAttr;
             if (oneShot)
               oneShotAttr = builder.getBoolAttr(true);
             self.create<tle::PipeCreateOp>(
                 fields, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs), readersAttr,
                 oneShotAttr);
           })
      .def("create_pipe_writer_acquire",
           [](TritonOpBuilder &self, std::vector<mlir::Value> fields,
              mlir::Value stage, mlir::Value phase, int32_t capacity,
              const std::string &scope, const std::string &pipeName,
              std::vector<std::string> fieldNames) -> void {
             auto &builder = self.getBuilder();
             llvm::SmallVector<mlir::Attribute> fieldNameAttrs;
             for (llvm::StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             mlir::StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             self.create<tle::PipeWriterAcquireOp>(
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
             for (llvm::StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             mlir::StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             self.create<tle::PipeWriterCommitOp>(
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
             for (llvm::StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             mlir::StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             self.create<tle::PipeWriterCloseOp>(
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
             for (llvm::StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             mlir::StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             mlir::StringAttr readerNameAttr;
             if (!readerName.empty())
               readerNameAttr = builder.getStringAttr(readerName);
             return self.create<tle::PipeReaderWaitOp>(
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
             for (llvm::StringRef name : fieldNames)
               fieldNameAttrs.push_back(builder.getStringAttr(name));
             mlir::StringAttr pipeNameAttr;
             if (!pipeName.empty())
               pipeNameAttr = builder.getStringAttr(pipeName);
             mlir::StringAttr readerNameAttr;
             if (!readerName.empty())
               readerNameAttr = builder.getStringAttr(readerName);
             self.create<tle::PipeReaderReleaseOp>(
                 fields, stage, builder.getI32IntegerAttr(capacity),
                 builder.getStringAttr(scope), pipeNameAttr,
                 builder.getArrayAttr(fieldNameAttrs), readerNameAttr);
           })
      .def("create_tle_wgmma",
           [](TritonOpBuilder &self, mlir::Value a, mlir::Value b,
              mlir::Value c, mlir::triton::InputPrecision inputPrecision,
              int maxNumImpreciseAcc, bool isAsync) -> mlir::Value {
             return self
                 .create<mlir::triton::musa_tle::SqmmaOp>(
                     c.getType(), a, b, c, inputPrecision, maxNumImpreciseAcc,
                     isAsync)
                 .getD();
           })
      .def("create_tle_wgmma_wait",
           [](TritonOpBuilder &self, mlir::Value input,
              unsigned pendings) -> mlir::Value {
             auto attr = self.getBuilder().getI32IntegerAttr(pendings);
             return self
                 .create<mlir::triton::musa_tle::SqmmaWaitOp>(input.getType(),
                                                              input, attr)
                 .getOutput();
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
      .def("create_memdesc_trans",
           [](TritonOpBuilder &self, mlir::Value src,
              std::vector<int> order) -> mlir::Value {
             return self.create<ttg::MemDescTransOp>(src, order);
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
             auto &builder = self.getBuilder();
             auto op = self.create<ttg::WarpSpecializeOp>(resultTypes,
                                                          partitionNumWarps);
             op->setAttr("musa_tle.static_warp_specialize",
                         builder.getUnitAttr());
             return TLEWarpSpecializeOp(op);
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
  ADD_PASS_WRAPPER_0("add_tle_lower_sqmma",
                     mlir::createTritonMUSAGPUTLELowerSqmma);
  ADD_PASS_WRAPPER_0("add_tle_lower_pipe",
                     mlir::createTritonMUSAGPUTLELowerPipe);
  ADD_PASS_WRAPPER_0("add_tle_prepare_warp_specialize",
                     mlir::createTritonMUSAGPUTLEPrepareWarpSpecialize);
  ADD_PASS_WRAPPER_0("add_tle_lower_warp_specialize",
                     mlir::createTritonMUSAGPUTLELowerWarpSpecialize);
  ADD_PASS_WRAPPER_0("add_tle_lower_barrier_allocations",
                     mlir::createTritonMUSAGPUTLELowerBarrierAllocations);
  ADD_PASS_WRAPPER_0("add_tle_lower_tme_transactions",
                     mlir::createTritonMUSAGPUTLELowerTMETransactions);
  ADD_PASS_WRAPPER_0("add_tle_lower_barrier_operations",
                     mlir::createTritonMUSAGPUTLELowerBarrierOperations);
  ADD_PASS_WRAPPER_0("add_tle_finalize_explicit_layouts",
                     mlir::createTritonMUSAGPUTLEFinalizeExplicitLayouts);
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
  registry.insert<mlir::triton::musa_tle::MUSATLEDialect,
                  mlir::triton::tle::TleDialect>();
}

#endif // __TLE__
