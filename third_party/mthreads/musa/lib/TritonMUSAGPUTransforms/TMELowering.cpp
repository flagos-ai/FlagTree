#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/BarrierUtils.h"
#include "TritonMUSACommon/MemDescUtils.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

static Value materializeStaticTMETransactionBytes(Location loc,
                                                  ArrayRef<int64_t> shape,
                                                  Type elemType,
                                                  RewriterBase &rewriter) {
  int64_t totalElements = 1;
  for (int64_t dim : shape) {
    if (dim <= 0)
      return {};
    totalElements *= dim;
  }
  int64_t elemBits = elemType.getIntOrFloatBitWidth();
  if (elemBits <= 0 || (elemBits % 8) != 0)
    return {};
  return arith::ConstantIntOp::create(rewriter, loc,
                                      totalElements * (elemBits / 8), 32);
}

static LogicalResult lowerDescriptorLoad(tt::DescriptorLoadOp op,
                                         RewriterBase &rewriter) {
  auto loc = op.getLoc();
  auto descTy = op.getDesc().getType();
  auto descBlockTy = descTy.getSignlessBlockType();
  auto memDescTy = triton::musa::resolveDescriptorLoadLandingMemDescType(op);
  if (failed(memDescTy))
    return op.emitOpError("descriptor load requires normalized canonical "
                          "landing memdesc encoding");

  rewriter.setInsertionPoint(op);
  auto coord =
      triton::musa::materializeTMECoordValues(loc, op.getIndices(), rewriter);
  if (failed(coord))
    return op.emitOpError("unsupported descriptor block rank for TME load");

  auto alloc = ttg::LocalAllocOp::create(rewriter, loc, *memDescTy);
  triton::musa::copyCanonicalLandingSqmmaAttrs(op, alloc.getOperation());
  auto config = triton::musa::resolveFinalTMECopyConfig(
      *memDescTy, descBlockTy.getShape(),
      triton::musa::TMECopyKind::GlobalToLocal);
  if (failed(config))
    return op.emitOpError("unable to resolve final TME load config");

  Value pred = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
  auto barId = triton::musa::reserveFreshBarrierId(op);
  if (failed(barId))
    return op.emitOpError("exhausted MUSA async barrier ids");
  Value barIdValue = arith::ConstantIntOp::create(rewriter, loc, *barId, 32);

  Value phaseInit = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  Value arriveCnt = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
  Value alwaysIssue = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
  Value totalBytes = materializeStaticTMETransactionBytes(
      loc, descBlockTy.getShape(), op.getType().getElementType(), rewriter);
  if (!totalBytes)
    return op.emitOpError("unable to materialize descriptor load transaction "
                          "bytes");

  triton::musa::InitArrivalOp::create(rewriter, loc, barIdValue, arriveCnt,
                                      phaseInit);
  triton::musa::BarrierAddTransOp::create(rewriter, loc, barIdValue, totalBytes,
                                          alwaysIssue);
  triton::musa::createAsyncTMECopyGlobalToLocal(
      rewriter, loc, op.getDesc(), *coord, barIdValue, alloc, pred, *config);
  triton::musa::ArriveBarrierNoRetOp::create(rewriter, loc, barIdValue,
                                             alwaysIssue);
  triton::musa::WaitBarrierOp::create(rewriter, loc, barIdValue, phaseInit);

  SmallVector<Operation *> users(op->getUsers().begin(), op->getUsers().end());
  Value tensorValue = op.getResult();
  Value localLoadValue;
  auto getLocalLoadValue = [&]() -> Value {
    if (!localLoadValue) {
      localLoadValue =
          ttg::LocalLoadOp::create(rewriter, loc, op.getType(), alloc)
              .getResult();
    }
    return localLoadValue;
  };

  for (Operation *user : users) {
    bool isTensorViewUser = isa<tt::TransOp, tt::ReshapeOp>(user);
    if (triton::musa::tryReplaceTensorUserWithMemDesc(
            rewriter, tensorValue, alloc.getResult(), user) &&
        !isTensorViewUser)
      continue;
    if (isTensorViewUser && user->use_empty()) {
      rewriter.eraseOp(user);
      continue;
    }
    rewriter.setInsertionPoint(user);
    user->replaceUsesOfWith(tensorValue, getLocalLoadValue());
    if (isTensorViewUser && user->use_empty())
      rewriter.eraseOp(user);
  }

  rewriter.eraseOp(op);
  return success();
}

static LogicalResult lowerDescriptorStore(tt::DescriptorStoreOp op,
                                          RewriterBase &rewriter) {
  auto loc = op.getLoc();
  auto descTy = op.getDesc().getType();
  auto descBlockTy = descTy.getSignlessBlockType();
  auto memDescTy = triton::musa::resolveDescriptorStoreLandingMemDescType(
      op, /*mutableMemory=*/false);
  if (failed(memDescTy))
    return op.emitOpError("descriptor store requires normalized canonical "
                          "landing memdesc encoding");

  rewriter.setInsertionPoint(op);
  auto coord =
      triton::musa::materializeTMECoordValues(loc, op.getIndices(), rewriter);
  if (failed(coord))
    return op.emitOpError("unsupported descriptor block rank for TME store");

  auto alloc =
      ttg::LocalAllocOp::create(rewriter, loc, *memDescTy, op.getSrc());
  auto config = triton::musa::resolveFinalTMECopyConfig(
      *memDescTy, descBlockTy.getShape(),
      triton::musa::TMECopyKind::LocalToGlobal);
  if (failed(config))
    return op.emitOpError("unable to resolve final TME store config");

  Value pred = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
  triton::musa::createAsyncTMECopyLocalToGlobal(rewriter, loc, op.getDesc(),
                                                *coord, alloc, pred, *config);
  triton::musa::TMEStoreCommitOp::create(rewriter, loc);
  triton::musa::TMEStoreReadWaitOp::create(rewriter, loc);

  rewriter.eraseOp(op);
  return success();
}

} // namespace

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTMELOWERING
#include "TritonMUSAGPUTransforms/Passes.h.inc"

struct TritonMUSAGPUTMELoweringPass
    : impl::TritonMUSAGPUTMELoweringBase<TritonMUSAGPUTMELoweringPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    getContext().getOrLoadDialect<vector::VectorDialect>();
    IRRewriter rewriter(&getContext());

    for (tt::FuncOp func : mod.getOps<tt::FuncOp>()) {
      SmallVector<tt::DescriptorLoadOp> loadOps;
      func.walk([&](tt::DescriptorLoadOp op) { loadOps.push_back(op); });
      for (tt::DescriptorLoadOp op : loadOps) {
        if (!op->getBlock())
          continue;
        if (failed(lowerDescriptorLoad(op, rewriter))) {
          signalPassFailure();
          return;
        }
      }

      SmallVector<tt::DescriptorStoreOp> storeOps;
      func.walk([&](tt::DescriptorStoreOp op) { storeOps.push_back(op); });
      for (tt::DescriptorStoreOp op : storeOps) {
        if (!op->getBlock())
          continue;
        if (failed(lowerDescriptorStore(op, rewriter))) {
          signalPassFailure();
          return;
        }
      }
    }
  }
};

} // namespace mlir
