// Copyright 2026- Xcoresigma Technology Co., Ltd
//
// TLE distributed op -> hivm.custom conversion. Completed against the
// upstream Triton-distributed-ascend reference implementation
// (lib/Conversion/TritonDistributedToHIVM/ASCEND/DistributedOpToHIVM.cpp):
//   * notify: the device symbol is derived from the SIGNAL POINTER pointee
//     type and the signal operation (int32 -> aclshmemx_signal_op;
//     int64/uint64 SET -> aclshmem_{u}int64_p; int64/uint64 ADD ->
//     aclshmem_atomic_add_{u}int64), and the operand list is rebuilt to
//     match the device ABI (int32 truncates the signal value and appends an
//     explicit ACLSHMEM signal-op code operand).
//   * notify: unsupported signal pointer types fail the match (before any
//     IR mutation) instead of asserting.
//   * get_rank / get_num_ranks: the aclshmem device symbols take no
//     arguments, so op operands (e.g. the mesh axis) are dropped.
//   * extern_call: an optional "distributed.core_type" attribute
//     (cube / vector / cube_and_vector / cube_or_vector) overrides the
//     core type; when absent, fall back to the existDot-derived type
//     (FlagTree's Python side does not emit the attribute, unlike upstream
//     which hard-requires it).
//   * unsupported types now yield match failures instead of asserts so a
//     single bad op does not abort the whole pass.

#include "tle/dsa/dialect/include/Conversion/TleToHIVM/TleDistributedOpToHIVMConverter.h"

#if __has_include("bishengir/Dialect/HIVM/IR/HIVM.h")
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#endif
#if __has_include("bishengir/Dialect/Annotation/IR/Annotation.h") &&           \
                  __has_include("ascend/include/Utils/Utils.h")
#include "ascend/include/Utils/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#define TLE_HAVE_ANNOTATION_UTILS
#endif
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tle/dsa/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include <optional>
#include <string>
#include <type_traits>

using namespace mlir;

namespace {

// ACLSHMEM signal-op codes expected by the aclshmemx_signal_op device
// symbol: ACLSHMEM_SIGNAL_SET = 0, ACLSHMEM_SIGNAL_ADD = 1.
constexpr int32_t kAclshmemSignalSet = 0;
constexpr int32_t kAclshmemSignalAdd = 1;

// Optional attribute overriding the executing core of extern_call ops
// (mirrors the upstream "distributed.core_type" convention).
constexpr StringRef kCoreTypeAttrName = "distributed.core_type";

static LogicalResult getTypeName(Type type, std::string &typeName) {
  if (auto fpTy = llvm::dyn_cast<FloatType>(type)) {
    if (fpTy.isBF16()) {
      typeName = "bfloat16";
      return success();
    } else if (fpTy.isF16()) {
      typeName = "half";
      return success();
    } else if (fpTy.isF32()) {
      typeName = "float";
      return success();
    }
  }
  if (auto intTy = llvm::dyn_cast<IntegerType>(type)) {
    switch (intTy.getWidth()) {
    case 8:
      typeName = "int8";
      return success();
    case 16:
      typeName = "int16";
      return success();
    case 32:
      typeName = "int32";
      return success();
    case 64:
      typeName = "int64";
      return success();
    }
  }
  if (auto pointerTy = llvm::dyn_cast<triton::PointerType>(type)) {
    std::string pointeeTypeName;
    if (failed(getTypeName(pointerTy.getPointeeType(), pointeeTypeName)))
      return failure();
    typeName = pointeeTypeName + "_ptr_1d";
    return success();
  }
  if (auto tensorTy = llvm::dyn_cast<RankedTensorType>(type)) {
    auto pointerTy =
        llvm::dyn_cast<triton::PointerType>(tensorTy.getElementType());
    if (!pointerTy)
      return failure();
    std::string pointeeTypeName;
    if (failed(getTypeName(pointerTy.getPointeeType(), pointeeTypeName)))
      return failure();
    typeName =
        pointeeTypeName + "_ptr_" + std::to_string(tensorTy.getRank()) + "d";
    return success();
  }
  return failure();
}

/// Generic template to convert TLE distributed ops to hivm.custom
template <typename TleOp>
class TleDistributedOpToHIVM : public OpRewritePattern<TleOp> {
public:
  TleDistributedOpToHIVM(MLIRContext *context, bool existDot = false,
                         PatternBenefit benefit = PatternBenefit(1))
      : OpRewritePattern<TleOp>(context, benefit), existDotFlag(existDot) {}

  LogicalResult inferCoreType(TleOp op) const {
    // extern_call ops may carry an optional "distributed.core_type"
    // attribute selecting the executing core (upstream convention).
    if constexpr (std::is_same_v<TleOp, triton::tle::ExternCallOp>) {
      if (auto attr =
              op->template getAttrOfType<StringAttr>(kCoreTypeAttrName)) {
        std::optional<hivm::TCoreType> coreType;
        if (attr.getValue() == "cube") {
          coreType = hivm::TCoreType::CUBE;
        } else if (attr.getValue() == "vector") {
          coreType = hivm::TCoreType::VECTOR;
        } else if (attr.getValue() == "cube_and_vector") {
          coreType = hivm::TCoreType::CUBE_AND_VECTOR;
        } else if (attr.getValue() == "cube_or_vector") {
          // TODO: pass TCoreType::CUBE_OR_VECTOR and rewrite it in npu-ir;
          // currently inferred from existDotFlag (upstream behaviour).
        } else {
          return op.emitError("invalid distributed.core_type '")
                 << attr.getValue()
                 << "'; expected one of: cube, vector, cube_and_vector, "
                    "cube_or_vector";
        }
        if (coreType) {
          op->setAttr(hivm::TCoreTypeAttr::name,
                      hivm::TCoreTypeAttr::get(op->getContext(), *coreType));
          return success();
        }
      }
    }
    // If kernel contains tt.dot, need both CUBE and VECTOR; otherwise VECTOR
    // only.
    auto coreType = existDotFlag ? hivm::TCoreType::CUBE_AND_VECTOR
                                 : hivm::TCoreType::VECTOR;
    op->setAttr(hivm::TCoreTypeAttr::name,
                hivm::TCoreTypeAttr::get(op->getContext(), coreType));
    return success();
  }

  LogicalResult matchAndRewrite(TleOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Validate before mutating anything so a failed match leaves the IR
    // intact (mirrors the upstream converter).
    if constexpr (std::is_same_v<TleOp, triton::tle::NotifyOp>) {
      auto notifyOp = llvm::cast<triton::tle::NotifyOp>(op);
      auto sigPtrTy =
          llvm::dyn_cast<triton::PointerType>(notifyOp.getSigAddr().getType());
      auto sigElemTy =
          sigPtrTy ? llvm::dyn_cast<IntegerType>(sigPtrTy.getPointeeType())
                   : nullptr;
      bool isSupportedSignalType =
          sigElemTy &&
          ((sigElemTy.getWidth() == 32 && !sigElemTy.isUnsigned()) ||
           sigElemTy.getWidth() == 64);
      if (!isSupportedSignalType) {
        return rewriter.notifyMatchFailure(
            op, "notify requires an i32, i64, or ui64 signal pointer");
      }
    }

    if (failed(inferCoreType(op)))
      return failure();

    // Determine symbol name
    std::string symbolName;
    if (auto sym = op->template getAttrOfType<StringAttr>("symbol")) {
      symbolName = sym.str();
    }
    bool hasSideEffect = !mlir::isMemoryEffectFree(op.getOperation());
    bool unsupportedTypeForSymbol = false;

    llvm::TypeSwitch<Operation *, void>(op)
        .Case([&](triton::tle::SymmAtOp symmOp) {
          auto elemTy = llvm::dyn_cast<triton::PointerType>(
              symmOp.getSymmAddr().getType());
          std::string typeName;
          if (!elemTy ||
              failed(getTypeName(elemTy.getPointeeType(), typeName))) {
            unsupportedTypeForSymbol = true;
            return;
          }
          symbolName = "aclshmem_ptr_" + typeName;
        })
        .Case([&](triton::tle::NotifyOp notifyOp) {
          // Derive the device symbol from the SIGNAL POINTER pointee type and
          // the signal operation (mirrors the upstream converter).
          auto sigPtrTy =
              llvm::cast<triton::PointerType>(notifyOp.getSigAddr().getType());
          auto sigElemTy = llvm::cast<IntegerType>(sigPtrTy.getPointeeType());
          if (sigElemTy.getWidth() == 32) {
            symbolName = "aclshmemx_signal_op";
          } else if (notifyOp.getSigOp() ==
                     triton::distributed::SignalOp::ADD) {
            symbolName = sigElemTy.isUnsigned() ? "aclshmem_atomic_add_uint64"
                                                : "aclshmem_atomic_add_int64";
          } else {
            symbolName = sigElemTy.isUnsigned() ? "aclshmem_uint64_p"
                                                : "aclshmem_int64_p";
          }
        })
        .Case([&](triton::tle::ConsumeTokenOp consumeTokenOp) {
          std::string typeName;
          if (failed(
                  getTypeName(consumeTokenOp.getInput().getType(), typeName))) {
            unsupportedTypeForSymbol = true;
            return;
          }
          symbolName = "aclshmem_consume_token_" + typeName;
        })
        .Case([&](triton::tle::WaitOp waitOp) {
          auto ptrTy = llvm::dyn_cast<triton::PointerType>(
              waitOp.getBarrierPtr().getType());
          std::string typeName;
          if (!ptrTy || failed(getTypeName(ptrTy.getPointeeType(), typeName))) {
            unsupportedTypeForSymbol = true;
            return;
          }
          symbolName = "aclshmem_wait_" + typeName;
        })
        .Case(
            [&](triton::tle::GetNumRanksOp) { symbolName = "aclshmem_n_pes"; })
        .Case([&](triton::tle::GetRankOp) { symbolName = "aclshmem_my_pe"; });

    if (unsupportedTypeForSymbol) {
      return rewriter.notifyMatchFailure(
          op, "unsupported type when deriving distributed symbol");
    }

    if (symbolName == "aclshmem_ptr") {
      auto elemTy = llvm::cast<triton::PointerType>(op->getOperand(0).getType())
                        .getPointeeType();
      std::string typeName;
      if (failed(getTypeName(elemTy, typeName))) {
        return rewriter.notifyMatchFailure(
            op, "failed to derive aclshmem_ptr type suffix");
      }
      symbolName = "aclshmem_ptr_" + typeName;
    }

    if (symbolName.empty()) {
      symbolName = op->getName().stripDialect().str();
    }

    std::string customName = "dist." + symbolName;

    // Build the operand list; some device symbols have a narrower signature
    // than the TLE op (mirrors the upstream converter):
    //   * aclshmem_my_pe / aclshmem_n_pes take no arguments, so the op
    //     operands (e.g. the mesh axis of get_rank) are dropped.
    //   * aclshmem_{u}int64_p / aclshmem_atomic_add_{u}int64 take
    //     (sigAddr, signalVal, rank); the sigOp/commScope enum attrs are
    //     already encoded in the symbol choice.
    //   * aclshmemx_signal_op additionally takes an explicit ACLSHMEM
    //     signal-op code and a truncated (i32) signal value.
    ValueRange operands = op->getOperands();
    llvm::SmallVector<Value> operandStorage;
    if constexpr (std::is_same_v<TleOp, triton::tle::GetRankOp> ||
                  std::is_same_v<TleOp, triton::tle::GetNumRanksOp>) {
      operands = ValueRange();
    } else if constexpr (std::is_same_v<TleOp, triton::tle::NotifyOp>) {
      auto notifyOp = llvm::cast<triton::tle::NotifyOp>(op);
      auto sigPtrTy =
          llvm::cast<triton::PointerType>(notifyOp.getSigAddr().getType());
      auto sigElemTy = llvm::cast<IntegerType>(sigPtrTy.getPointeeType());
      if (sigElemTy.getWidth() == 64) {
        operandStorage = {notifyOp.getSigAddr(), notifyOp.getSignalVal(),
                          notifyOp.getRank()};
      } else {
        auto i32Ty = rewriter.getI32Type();
        Value signal = rewriter.create<arith::TruncIOp>(
            loc, i32Ty, notifyOp.getSignalVal());
        // ACLSHMEM_SIGNAL_SET = 0, ACLSHMEM_SIGNAL_ADD = 1
        int32_t aclSigOp =
            notifyOp.getSigOp() == triton::distributed::SignalOp::ADD
                ? kAclshmemSignalAdd
                : kAclshmemSignalSet;
        Value sigOpVal = rewriter.create<arith::ConstantOp>(
            loc, i32Ty, rewriter.getI32IntegerAttr(aclSigOp));
        operandStorage = {notifyOp.getSigAddr(), signal, sigOpVal,
                          notifyOp.getRank()};
      }
      operands = ValueRange(operandStorage);
    }

    // Build custom results (tensor outputs need tensor::EmptyOp)
    llvm::SmallVector<Value> customResults;
    for (auto res : op->getResults()) {
      if (auto tensorTy = llvm::dyn_cast<RankedTensorType>(res.getType())) {
        auto emptyOp = rewriter.create<tensor::EmptyOp>(
            loc, tensorTy.getShape(), tensorTy.getElementType());
        customResults.emplace_back(emptyOp);
      }
    }

    auto customOp =
        rewriter.create<hivm::CustomOp>(loc, op->getResultTypes(), customName,
                                        operands, customResults, ValueRange{});

    // Copy original attrs first, then override with controlled values
    customOp->setAttrs(op->getAttrs());
    // The core-type override is a conversion hint and must not leak into the
    // HIVM op (mirrors the upstream removeAttr).
    customOp->removeAttr(kCoreTypeAttrName);
    customOp->setAttr("hivm.is_distributed", rewriter.getUnitAttr());
    customOp.setPipe(hivm::PIPE::PIPE_S);
    customOp.setVFMode(hivm::VFMode::SIMD);
    customOp->setAttr("symbol", rewriter.getStringAttr(symbolName));

// For consume_token with a descriptor tensor result, mark the memory as
// continuous and record the source pointer index (mirrors the original
// Ascend TritonDistributedToHIVM conversion).
#ifdef TLE_HAVE_ANNOTATION_UTILS
    if (llvm::isa<triton::tle::ConsumeTokenOp>(op) &&
        llvm::isa<RankedTensorType>(customOp->getResult(0).getType())) {
      auto annotationOp =
          rewriter.create<annotation::MarkOp>(loc, customOp->getResult(0));
      annotationOp->setAttr(ConverterUtils::continuousAttrName,
                            rewriter.getUnitAttr());
      customOp->setAttr(ConverterUtils::customSrcPtrIndexAttrName,
                        rewriter.getDenseI32ArrayAttr({0}));
    }
#endif

    if (!hasSideEffect) {
      customOp->setAttr("no_side_effect", rewriter.getUnitAttr());
    }

    // Record GM addr argument indices for downstream passes
    llvm::SmallVector<int> gmAddrArgsIndices;
    auto funcOp = customOp->template getParentOfType<triton::FuncOp>();
    if (funcOp) {
      for (auto &&[idx, operand] : llvm::enumerate(customOp->getOperands())) {
        if (!llvm::isa<triton::PointerType>(operand.getType()))
          continue;
        if (auto arg = llvm::dyn_cast<BlockArgument>(operand)) {
          if (arg.getOwner() == &funcOp.getFunctionBody().front()) {
            gmAddrArgsIndices.emplace_back(idx);
          }
        }
      }
    }
    customOp->setAttr("gm_addr_args_indices",
                      rewriter.getDenseI32ArrayAttr(gmAddrArgsIndices));

    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, customOp);
    }

    return success();
  }

private:
  bool existDotFlag;
};

/// Helper to register patterns with existDot flag
template <typename... Args>
void registerTleDistributedOpToHIVM(
    RewritePatternSet &patterns, bool existDot,
    PatternBenefit benefit = PatternBenefit(1)) {
  patterns.add<TleDistributedOpToHIVM<Args>...>(patterns.getContext(), existDot,
                                                benefit);
}

} // namespace

void mlir::triton::tle::populateTleDistributedOpToHIVMConversionPatterns(
    mlir::RewritePatternSet &patterns, PatternBenefit benefit, bool existDot) {
  registerTleDistributedOpToHIVM<
      triton::tle::SymmAtOp, triton::tle::ExternCallOp, triton::tle::GetRankOp,
      triton::tle::NotifyOp, triton::tle::ConsumeTokenOp, triton::tle::WaitOp,
      triton::tle::GetNumRanksOp>(patterns, existDot, benefit);
}
