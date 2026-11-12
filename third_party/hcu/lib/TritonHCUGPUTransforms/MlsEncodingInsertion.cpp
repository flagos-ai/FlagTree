#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "TritonHCUGPUToLLVM/TargetUtils.h"
#include "TritonHCUGPUTransforms/MfmaGroup.h"
#include "TritonHCUGPUTransforms/MlsGroup.h"
#include "Utility.h"
#include "MatrixDescriptorLayout.h"
#include "../TritonHCUGPUToLLVM/ScalarMemoryAnalysis.h"
#include "../TritonHCUGPUToLLVM/WarpSpecializeUtils.h"
#include "Analysis/AxisInfoExt.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

// Scalar type alone is not a CTA-uniformity proof. Counted recurrences need
// uniform entry/backedge data and control; loads and lane queries stay unknown.
static bool registerDescriptorUniform(Value value, DenseSet<Value> &seen,
                                      unsigned &budget) {
  if (!budget || !value.getType().isIntOrIndex())
    return false;
  --budget;
  value = triton::HCU::resolveWarpSpecializeCapture(value);
  if (!seen.insert(value).second)
    return true;
  auto uniform = [&](Value v) {
    return registerDescriptorUniform(v, seen, budget);
  };
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (auto fn = dyn_cast<tt::FuncOp>(arg.getOwner()->getParentOp()))
      return arg.getOwner() == &fn.getBody().front();
    if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
      if (arg.getOwner() != loop.getBody() ||
          !uniform(loop.getLowerBound()) || !uniform(loop.getUpperBound()) ||
          !uniform(loop.getStep()))
        return false;
      unsigned n = arg.getArgNumber();
      return n == 0 ||
             (uniform(loop.getInitArgs()[n - 1]) &&
              uniform(cast<scf::YieldOp>(loop.getBody()->getTerminator())
                          .getOperand(n - 1)));
    }
    return false;
  }
  Operation *op = value.getDefiningOp();
  if (!op)
    return false;
  if (isa<tt::GetProgramIdOp, tt::GetNumProgramsOp>(op))
    return true;
  if (auto branch = dyn_cast<scf::IfOp>(op)) {
    unsigned n = cast<OpResult>(value).getResultNumber();
    return uniform(branch.getCondition()) &&
           uniform(branch.thenYield().getOperand(n)) &&
           uniform(branch.elseYield().getOperand(n));
  }
  return op->getName().getStringRef().starts_with("arith.") &&
         op->getNumRegions() == 0 &&
         llvm::all_of(op->getOperands(), uniform);
}

static bool registerDescriptorUniform(Value value) {
  DenseSet<Value> seen;
  unsigned budget = 1024;
  return registerDescriptorUniform(value, seen, budget);
}

static bool supportsRegisterDescriptor(tt::MatrixLoadOp op,
                                     triton::HCU::ModuleAxisInfoAnalysis &axis) {
  auto type = op.getType();
  auto mod = op->getParentOfType<ModuleOp>();
  auto target = mod->getAttrOfType<StringAttr>("ttg.target");
  if (!target || target.getValue() != "hip:gfx936" || type.getRank() != 2 ||
      op.getShape().size() != 2 || op.getStrides().size() != 2 ||
      op.getIndices().size() != 2 || op.getCache() != tt::CacheModifier::NONE ||
      op.getEvict() != tt::EvictionPolicy::NORMAL || op.getIsVolatile() ||
      op.getBoundaryCheck().size() != 2 ||
      !llvm::is_contained(op.getBoundaryCheck(), 0) ||
      !llvm::is_contained(op.getBoundaryCheck(), 1) ||
      !(type.getElementType().isF16() || type.getElementType().isBF16()) ||
      !isa<ttg::DistributedEncodingTrait>(type.getEncoding()) ||
      ttg::TritonGPUDialect::getNumCTAs(mod) != 1 ||
      ttg::TritonGPUDialect::getThreadsPerWarp(mod) != 64)
    return false;
  APInt k, rowStride, columnStride;
  if (!matchPattern(op.getShape()[1], m_ConstantInt(&k)) || !k.isSignedIntN(32) ||
      k.getSExtValue() <= 0 || k.getSExtValue() >= 0x2000 ||
      !matchPattern(op.getStrides()[0], m_ConstantInt(&rowStride)) ||
      rowStride.getSExtValue() != k.getSExtValue() ||
      !matchPattern(op.getStrides()[1], m_ConstantInt(&columnStride)) ||
      !columnStride.isOne() || type.getShape()[1] > k.getSExtValue() ||
      !llvm::isPowerOf2_64(type.getShape()[1]) ||
      k.getSExtValue() % type.getShape()[1])
    return false;
  // The descriptor has a whole-window K guard. Only complete aligned windows
  // have the same semantics as the ordinary per-element boundary masks.
  APInt origin;
  bool aligned = matchPattern(op.getIndices()[1], m_ConstantInt(&origin)) &&
                 origin.getSExtValue() % type.getShape()[1] == 0;
  if (!aligned) {
    auto *info = axis.getAxisInfo(op.getIndices()[1]);
    aligned = info && info->getRank() &&
              info->getDivisibility(0) % type.getShape()[1] == 0;
  }
  if (!aligned)
    return false;
  Value base = triton::HCU::resolveWarpSpecializeCapture(op.getBase());
  while (auto add = base.getDefiningOp<tt::AddPtrOp>()) {
    if (!registerDescriptorUniform(add.getOffset()))
      return false;
    base = triton::HCU::resolveWarpSpecializeCapture(add.getPtr());
  }
  auto pointer = dyn_cast<tt::PointerType>(base.getType());
  auto arg = dyn_cast<BlockArgument>(base);
  auto fn = op->getParentOfType<tt::FuncOp>();
  if (!pointer || pointer.getAddressSpace() != 1 || !arg || !fn ||
      arg.getOwner() != &fn.getBody().front())
    return false;
  for (auto values : {op.getShape(), op.getStrides(), op.getIndices()})
    for (Value value : values)
      if (!registerDescriptorUniform(value))
        return false;
  bool control = true;
  fn.walk([&](Operation *nested) {
    if (auto branch = dyn_cast<scf::IfOp>(nested))
      control &= registerDescriptorUniform(branch.getCondition());
    else if (auto loop = dyn_cast<scf::ForOp>(nested))
      control &= registerDescriptorUniform(loop.getLowerBound()) &&
                 registerDescriptorUniform(loop.getUpperBound()) &&
                 registerDescriptorUniform(loop.getStep());
    else if (isa<scf::WhileOp, ttg::MaskOp>(nested) ||
             nested->getNumSuccessors() > 1)
      control = false;
  });
  auto ll = ttg::toLinearLayout(type);
  auto *ctx = op.getContext();
  return control && ll.getInDimSize(StringAttr::get(ctx, "block")) == 1 &&
         uint64_t(ll.getInDimSize(StringAttr::get(ctx, "register"))) * 64 *
                 ll.getInDimSize(StringAttr::get(ctx, "warp")) <= (1u << 20);
}

// The matrix-copy instruction handles complete 16B vectors of a statically
// described, contiguous source. Keep ordinary loads for other legal inputs,
// including dynamic descriptors and dots assigned to the scalar FMA path.
bool supportsDescriptorCopy(triton::MatrixLoadOp op) {
  auto mod = op->getParentOfType<ModuleOp>();
  auto target = mod->getAttrOfType<StringAttr>("ttg.target");
  auto type = op.getType();
  if (!target || target.getValue() != "hip:gfx936" ||
      ttg::TritonGPUDialect::getNumCTAs(mod) != 1 ||
      ttg::TritonGPUDialect::getThreadsPerWarp(mod) != 64 ||
      !(type.getElementType().isF16() || type.getElementType().isBF16()) ||
      op.getShape().size() != 2 || op.getStrides().size() != 2 ||
      op.getIndices().size() != 2 || op.getCache() != tt::CacheModifier::NONE ||
      op.getEvict() != tt::EvictionPolicy::NORMAL || op.getIsVolatile())
    return false;
  SmallVector<int64_t> sizes, strides;
  for (auto values : {op.getShape(), op.getStrides()}) {
    auto &result = sizes.empty() ? sizes : strides;
    for (Value value : values) {
      APInt number;
      if (!matchPattern(value, m_ConstantInt(&number)) || number.isNegative() ||
          number.isZero() || !number.isSignedIntN(32))
        return false;
      result.push_back(number.getSExtValue());
    }
  }
  unsigned column = strides[1] == 1 ? 1 : 0, row = 1 - column;
  unsigned waves = ttg::lookupNumWarps(op);
  uint64_t elements = type.getNumElements();
  if (strides[column] != 1 || strides[row] < sizes[column] ||
      strides[row] % 8 || sizes[column] % 8 || !waves || waves > 8 ||
      !llvm::isPowerOf2_32(waves) || elements < 512 * waves ||
      elements > (1u << 20) || elements % (512 * waves) ||
      !llvm::all_of(op.getIndices(), [](Value value) {
        return registerDescriptorUniform(value);
      }))
    return false;
  if (!triton::HCU::memory_analysis::multipleOf(op.getIndices()[column], 8))
    return false;
  uint64_t tileSpan =
      2 * (uint64_t(type.getShape()[row] - 1) * strides[row] +
           type.getShape()[column]);
  if (strides[row] >= 0x2000 && tileSpan >= (uint64_t(1) << 31))
    return false;
  auto arg = dyn_cast<BlockArgument>(op.getBase());
  auto fn = op->getParentOfType<tt::FuncOp>();
  auto alignment = arg && fn && arg.getOwner() == &fn.getBody().front()
                       ? fn.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),
                                                         "tt.divisibility")
                       : IntegerAttr{};
  return alignment && alignment.getInt() >= 16 && alignment.getInt() % 16 == 0 &&
         triton::HCU::getMatrixDescriptorSharedLayout(op);
}

Value expandDescriptorLoad(triton::MatrixLoadOp op, PatternRewriter &rw) {
  auto type = op.getType();
  auto loc = op.getLoc();
  auto context = op.getContext();
  auto encoding = cast<ttg::DistributedEncodingTrait>(type.getEncoding());
  auto integer = RankedTensorType::get(type.getShape(), rw.getI64Type(), encoding);
  auto boolean = RankedTensorType::get(type.getShape(), rw.getI1Type(), encoding);
  Value mask, offset;
  for (unsigned axis = 0; axis < 2; ++axis) {
    auto slice = ttg::SliceEncodingAttr::get(context, 1 - axis, encoding);
    auto rangeType = RankedTensorType::get({type.getShape()[axis]}, rw.getI32Type(), slice);
    auto vectorType = rangeType.clone(rw.getI64Type());
    Value range = rw.create<tt::MakeRangeOp>(loc, rangeType, 0, type.getShape()[axis]);
    range = rw.create<arith::ExtSIOp>(loc, vectorType, range);
    Value origin = rw.create<arith::ExtSIOp>(loc, rw.getI64Type(), op.getIndices()[axis]);
    origin = rw.create<tt::SplatOp>(loc, vectorType, origin);
    Value index = rw.create<arith::AddIOp>(loc, range, origin);
    if (llvm::is_contained(op.getBoundaryCheck(), axis)) {
      Value size = rw.create<arith::ExtSIOp>(loc, rw.getI64Type(), op.getShape()[axis]);
      size = rw.create<tt::SplatOp>(loc, vectorType, size);
      Value zero = rw.create<arith::ConstantIntOp>(loc, 0, 64);
      zero = rw.create<tt::SplatOp>(loc, vectorType, zero);
      Value lower = rw.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, index, zero);
      Value upper = rw.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, index, size);
      Value valid = rw.create<arith::AndIOp>(loc, lower, upper);
      valid = rw.create<tt::ExpandDimsOp>(loc, valid, 1 - axis);
      valid = rw.create<tt::BroadcastOp>(loc, boolean, valid);
      mask = mask ? Value(rw.create<arith::AndIOp>(loc, mask, valid)) : valid;
    }
    Value stride = rw.create<tt::SplatOp>(loc, vectorType, op.getStrides()[axis]);
    Value term = rw.create<arith::MulIOp>(loc, index, stride);
    term = rw.create<tt::ExpandDimsOp>(loc, term, 1 - axis);
    term = rw.create<tt::BroadcastOp>(loc, integer, term);
    offset = offset ? Value(rw.create<arith::AddIOp>(loc, offset, term)) : term;
  }
  auto pointers = type.clone(op.getBase().getType());
  Value base = rw.create<tt::SplatOp>(loc, pointers, op.getBase());
  Value ptr = rw.create<tt::AddPtrOp>(loc, pointers, base, offset);
  Value other;
  if (mask) {
    Value zero = rw.create<arith::ConstantOp>(loc, rw.getZeroAttr(type.getElementType()));
    other = rw.create<tt::SplatOp>(loc, type, zero);
  }
  auto load = rw.create<tt::LoadOp>(loc, ptr, mask, other, op.getCache(),
                                  op.getEvict(), op.getIsVolatile(), StringAttr{});
  load->setAttr("tt.load.async", op->getAttr("tt.load.async"));
  return load.getResult();
}

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

SmallVector<unsigned> getMatrixLoadTensorOrder(triton::MatrixLoadOp matrixOp,
                                               int opIdx) {
  auto strides = matrixOp.getStrides();
  assert(strides.size() == 2);

  auto strideKIdx = opIdx == 0 ? 1 : 0;
  auto strideK = strides[strideKIdx];

  bool kMajor = false;
  if (auto constantOp = strideK.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
      kMajor = attr.getValue().isOne();
    }
  }

  if (opIdx == 0) {
    return {kMajor, !kMajor};
  } else {
    return {!kMajor, kMajor};
  }
}

// Chooses a proper MLS instruction
FailureOr<MlsInsn> chooseMlsInstruction(tt::DotOp dot, int opIdx,
                                        triton::MatrixLoadOp matrixOp,
                                        bool kMajor, int version,
                                        int numWarps) {
  // get mfma encoding info
  auto aType = dot.getA().getType();
  auto bType = dot.getB().getType();
  auto dType = dot.getResult().getType();
  auto mfmaEncoding = dyn_cast<ttg::AMDMfmaEncodingAttr>(dType.getEncoding());

  // get the mls shape info
  auto rank = matrixOp.getType().getRank();
  auto elemType = matrixOp.getType().getElementType();
  auto bitwidth = elemType.getIntOrFloatBitWidth();
  auto blockK = matrixOp.getType().getShape()[(rank - 1) - opIdx];
  auto blockNonK = matrixOp.getType().getShape()[(rank - 2) + opIdx];

  // Determine the kDim based on the matrix load and numWarps Info.
  unsigned kTile = 0;
  unsigned nonKTile = opIdx == 0 ? mfmaEncoding.getMfmaTile()[0]
                                 : mfmaEncoding.getMfmaTile()[1];
  if (bitwidth == 16) {
    kTile = kMajor ? (blockK >= 64 ? 64 : 32) : 16;
    if (kTile == 64 && kTile * nonKTile * numWarps > blockK * blockNonK)
      kTile = 32;
  } else if (bitwidth == 8) {
    kTile = kMajor ? (blockK >= 128 ? 128 : 64) : 32;
    if (kTile == 128 && kTile * nonKTile * numWarps > blockK * blockNonK)
      kTile = 64;
  }
  auto altKind = MlsInterleaveKind::InterleaveNone;

  // select the mls insn
  auto maybeMlsInsn = MlsInsn::selectOrGetMlsInsn(
      nonKTile, kTile, bitwidth, opIdx, kMajor, altKind, version);
  if (failed(maybeMlsInsn))
    llvm::report_fatal_error("No match found in MLS database\n");

  return maybeMlsInsn;
}

SmallVector<unsigned> warpsPerCTAMatrixLoad(ArrayRef<int64_t> shape,
                                            ArrayRef<unsigned> shapePerWarp,
                                            ArrayRef<unsigned> order,
                                            int numWarps) {
  unsigned rank = shape.size();
  SmallVector<unsigned> warpsPerCTA(rank);

  unsigned remainingWarps = numWarps;
  unsigned prevWarps = 1;

  // starting from the contiguous dimension
  for (unsigned d = 0; d < rank - 1; ++d) {
    unsigned i = order[d];
    unsigned maxWarpsInDim = std::max<unsigned>(
        1, static_cast<unsigned>(shape[i]) / shapePerWarp[i]);
    warpsPerCTA[i] = std::clamp<unsigned>(remainingWarps, 1, maxWarpsInDim);
    remainingWarps /= warpsPerCTA[i];
    prevWarps *= warpsPerCTA[i];
  }

  // Expand the last dimension to fill the remaining lanes and warps
  warpsPerCTA[order[rank - 1]] = numWarps / prevWarps;

  return warpsPerCTA;
}

Value convertAndCastTensor(OpBuilder &builder, Value value,
                           Attribute newEncoding) {
  auto loc = value.getLoc();
  auto oldType = cast<RankedTensorType>(value.getType());
  auto oldElemType = oldType.getElementType();

  assert(oldElemType.isIntOrFloat());
  auto convertedType =
      RankedTensorType::get(oldType.getShape(), oldElemType, newEncoding);

  Value convertedTensor =
      builder.create<ttg::ConvertLayoutOp>(loc, convertedType, value);

  return convertedTensor;
}

struct MlsEncodingInsertion : public OpRewritePattern<triton::MatrixLoadOp> {
public:
  MlsEncodingInsertion(MLIRContext *context,
                       triton::HCU::ModuleAxisInfoAnalysis &axis)
      : OpRewritePattern(context, 1), axis(axis) {}

  LogicalResult matchAndRewrite(triton::MatrixLoadOp matrixOp,
                                PatternRewriter &rewriter) const override {
    if (matrixOp->hasAttr("hcu.structured_register"))
      return failure();
    if (triton::HCU::isSyncMatrixDescriptor(matrixOp)) {
      if (!supportsRegisterDescriptor(matrixOp, axis)) {
        rewriter.replaceOp(matrixOp, expandDescriptorLoad(matrixOp, rewriter));
        return success();
      }
      Value enabled = rewriter.create<arith::ConstantIntOp>(
          matrixOp.getLoc(), 1, 32);
      rewriter.modifyOpInPlace(matrixOp, [&] {
        matrixOp.getIndicesMutable().append(enabled);
        matrixOp.setBoundaryCheckAttr(rewriter.getDenseI32ArrayAttr({}));
        matrixOp->setAttr("hcu.structured_register", rewriter.getUnitAttr());
      });
      return success();
    }
    if (triton::HCU::isAsyncMatrixDescriptor(matrixOp)) {
      if (matrixOp.getType().getRank() != 2 || matrixOp.getShape().size() != 2 ||
          matrixOp.getStrides().size() != 2 || matrixOp.getIndices().size() != 2)
        return failure();
      if (supportsDescriptorCopy(matrixOp))
        return failure();
      rewriter.replaceOp(matrixOp, expandDescriptorLoad(matrixOp, rewriter));
      return success();
    }
    RankedTensorType oldRetType = matrixOp.getType();
    if (!oldRetType.getEncoding() ||
        !isa<ttg::BlockedEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    auto maybeDotOpIdxPair = getDotOpIdxFromMatrixLoad(matrixOp);
    if (succeeded(maybeDotOpIdxPair)) {
      auto dotOp = maybeDotOpIdxPair.value().first;
      auto opIdx = maybeDotOpIdxPair.value().second;

      // 1. choose the mls instruction for matrixOp
      assert(isa<ttg::AMDMfmaEncodingAttr>(dotOp.getType().getEncoding()));
      auto order = getMatrixLoadTensorOrder(matrixOp, opIdx);
      bool kMajor = opIdx == 0 ? order[0] == 1 : order[0] == 0;
      auto numWarps = triton::gpu::lookupNumWarps(matrixOp);
      auto mlsInsn =
          chooseMlsInstruction(dotOp, opIdx, matrixOp, kMajor, 1, numWarps);

      // 2. warps per tile calculate
      auto mlsTile = mlsInsn->getMlsTile();
      auto shape = matrixOp.getResult().getType().getShape();
      auto elemBitWidth = mlsInsn->getElemBitWidth();
      auto altKind = mlsInsn->getAlt2Kind();
      auto version = mlsInsn->getMlsVersion();
      auto warpsPerCTA = warpsPerCTAMatrixLoad(shape, mlsTile, order, numWarps);

      // 3. create new matrix load op with new encoding and mls attr
      auto newEncoding = opIdx == 0 ? dotOp.getA().getType().getEncoding()
                                    : dotOp.getB().getType().getEncoding();
      if (dyn_cast<ttg::DotOperandEncodingAttr>(newEncoding).getKWidth() !=
          mlsInsn->getDotLayoutKWidth()) {
        newEncoding = ttg::DotOperandEncodingAttr::get(
            matrixOp.getContext(), opIdx,
            dyn_cast<ttg::DotOperandEncodingAttr>(newEncoding).getParent(),
            mlsInsn->getDotLayoutKWidth());
      }

      auto oldResultType =
          cast<RankedTensorType>(matrixOp.getResult().getType());
      auto newResultType =
          RankedTensorType::get(oldResultType.getShape(),
                                oldResultType.getElementType(), newEncoding);

      auto newMatrixOp = rewriter.create<triton::MatrixLoadOp>(
          matrixOp.getLoc(), newResultType, matrixOp.getBase(),
          matrixOp.getShape(), matrixOp.getStrides(), matrixOp.getTensorShape(),
          matrixOp.getIndices(), matrixOp.getBoundaryCheck(),
          matrixOp.getCache(), matrixOp.getEvict(), matrixOp.getIsVolatile());
      auto mlsEncoding = triton::hcugpu::MlsEncodingAttr::get(
          matrixOp.getContext(), opIdx, mlsTile, elemBitWidth, altKind, version,
          order, warpsPerCTA);
      newMatrixOp->setAttr(triton::hcugpu::MlsEncodingAttr::getMnemonic(),
                           mlsEncoding);

      // 4. replace the original op
      auto convertedTensor = convertAndCastTensor(
          rewriter, newMatrixOp.getResult(), matrixOp.getType().getEncoding());
      rewriter.replaceOp(matrixOp, convertedTensor);
    } else { // non-dot case, try to construct a mfma encoding for the matrix
             // load
      // 1. choose the mls instruction for matrixOp
      unsigned opIdx = 0;
      auto order = getMatrixLoadTensorOrder(matrixOp, opIdx);
      bool kMajor = opIdx == 0 ? order[0] == 1 : order[0] == 0;

      auto rank = matrixOp.getType().getRank();
      auto elemType = matrixOp.getType().getElementType();
      auto bitwidth = elemType.getIntOrFloatBitWidth();
      auto blockK = matrixOp.getType().getShape()[(rank - 1) - opIdx];
      auto blockNonK = matrixOp.getType().getShape()[(rank - 2) + opIdx];
      auto numWarps = triton::gpu::lookupNumWarps(matrixOp);

      unsigned kTile = 0;
      unsigned nonKTile = 0;
      if (bitwidth == 16) {
        nonKTile = kMajor ? 16 : (blockNonK >= 64 ? 64 : 32);
        kTile = kMajor ? (blockK >= 64 ? 64 : 32) : 16;
        if (kTile == 64 && kTile * nonKTile * numWarps > blockK * blockNonK)
          kTile = 32;
      } else if (bitwidth == 8) {
        nonKTile = kMajor ? 16 : (blockNonK >= 128 ? 128 : 64);
        kTile = kMajor ? (blockK >= 128 ? 128 : 64) : 32;
        if (kTile == 128 && kTile * nonKTile * numWarps > blockK * blockNonK)
          kTile = 64;
      }

      auto altKind = MlsInterleaveKind::InterleaveNone;
      auto mlsInsn = MlsInsn::selectOrGetMlsInsn(nonKTile, kTile, bitwidth,
                                                 opIdx, kMajor, altKind, 1);
      if (failed(mlsInsn))
        llvm::report_fatal_error("No match found in MLS database\n");

      assert(blockK % kTile == 0 && blockNonK % nonKTile == 0 &&
             "M or N should be divisible by mDim or nDim");

      // 3. warps per tile calucate
      auto mlsTile = mlsInsn->getMlsTile();
      auto shape = matrixOp.getResult().getType().getShape();
      auto elemBitWidth = mlsInsn->getElemBitWidth();
      auto version = mlsInsn->getMlsVersion();
      auto warpsPerCTA = warpsPerCTAMatrixLoad(shape, mlsTile, order, numWarps);

      // 3. try to construct a mfma encoding for the matrix load
      constexpr unsigned mfmaVersion = 3;
      auto maybeMfmaInsn = MfmaIntrinsic::selectFor(
          matrixOp.getLoc(), mfmaVersion, 16, 16, blockK, elemType, elemType,
          false, false, HCUISAFeature::MAMC_FP8);
      if (failed(maybeMfmaInsn))
        llvm::report_fatal_error("No match found in MFMA database\n");

      SmallVector<unsigned> tilesPerWarp = {1, 1};
      SmallVector<unsigned> mfmaTiles = {16, 16};
      SmallVector<unsigned> mfmaOrder = {1, 0};
      if (rank == 3) {
        tilesPerWarp.insert(tilesPerWarp.begin(), 1);
        mfmaOrder.insert(mfmaOrder.begin(), 2);
        mfmaTiles.insert(mfmaTiles.begin(), 1);
      }

      auto tileM = nonKTile;
      auto tileN = kTile;

      bool hasBatchDim = rank == 3;
      int mIndex = 0 + hasBatchDim;
      int nIndex = 1 + hasBatchDim;
      tilesPerWarp[mIndex] = tileM / maybeMfmaInsn->mDim;
      tilesPerWarp[nIndex] = tileN / maybeMfmaInsn->nDim;

      SmallVector<unsigned> warpsPerCTAMfma =
          warpsPerCTAMatrixLoad(shape, mfmaTiles, mfmaOrder, numWarps);
      unsigned mfmaElementBitWidth = elemType.isF64() ? 64 : 32;
      auto mfmaEnc = ttg::AMDMfmaEncodingAttr::get(
          matrixOp.getContext(),
          /*versionMajor*/ mfmaVersion, warpsPerCTAMfma,
          /*instrShape=*/
          SmallVector<unsigned>{maybeMfmaInsn->mDim, maybeMfmaInsn->nDim,
                                maybeMfmaInsn->kDim},
          /*isTransposed=*/false,
          ttg::getCTALayout(matrixOp.getType().getEncoding()), tilesPerWarp,
          mfmaElementBitWidth, ttg::MmacLayout::INTERLEAVE_TRANSPOSE);
      auto dotAEncoding = ttg::DotOperandEncodingAttr::get(
          matrixOp.getContext(), 0, mfmaEnc, maybeMfmaInsn->kBase);

      // 4. create new matrix load op with new encoding and mls attr
      auto newEncoding = dotAEncoding;

      auto oldResultType =
          cast<RankedTensorType>(matrixOp.getResult().getType());
      auto newResultType =
          RankedTensorType::get(oldResultType.getShape(),
                                oldResultType.getElementType(), newEncoding);

      auto newMatrixOp = rewriter.create<triton::MatrixLoadOp>(
          matrixOp.getLoc(), newResultType, matrixOp.getBase(),
          matrixOp.getShape(), matrixOp.getStrides(), matrixOp.getTensorShape(),
          matrixOp.getIndices(), matrixOp.getBoundaryCheck(),
          matrixOp.getCache(), matrixOp.getEvict(), matrixOp.getIsVolatile());
      auto mlsEncoding = triton::hcugpu::MlsEncodingAttr::get(
          matrixOp.getContext(), opIdx, mlsTile, elemBitWidth,
          static_cast<unsigned>(altKind), version, order, warpsPerCTA);
      newMatrixOp->setAttr(triton::hcugpu::MlsEncodingAttr::getMnemonic(),
                           mlsEncoding);

      // 5. replace the original op
      auto convertedTensor = convertAndCastTensor(
          rewriter, newMatrixOp.getResult(), matrixOp.getType().getEncoding());
      rewriter.replaceOp(matrixOp, convertedTensor);
    }

    return success();
  }

private:
  triton::HCU::ModuleAxisInfoAnalysis &axis;
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace mlir {

#define GEN_PASS_DEF_TRITONHCUGPUMLSENCODINGINSERTION
#include "TritonHCUGPUTransforms/Passes.h.inc"

class TritonHCUGPUMlsEncodingInsertionPass
    : public impl::TritonHCUGPUMlsEncodingInsertionBase<
          TritonHCUGPUMlsEncodingInsertionPass> {
public:
  using impl::TritonHCUGPUMlsEncodingInsertionBase<
      TritonHCUGPUMlsEncodingInsertionPass>::
      TritonHCUGPUMlsEncodingInsertionBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    mlir::RewritePatternSet patterns(context);
    triton::HCU::ModuleAxisInfoAnalysis axis(mod);
    patterns.add<MlsEncodingInsertion>(context, axis);
    if (mlir::applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
