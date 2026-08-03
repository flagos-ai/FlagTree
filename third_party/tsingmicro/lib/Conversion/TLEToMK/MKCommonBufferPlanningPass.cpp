//===---------------- MKCommBufferPlanningPass.cpp ---------------------===//
//
// Insert local SPM buffers for paired mk.recv/mk.send that share the same
// placeholder base address. The placeholder base is represented by the i64
// src_addr/dst_addr operands.
//
// Strategy:
// - Find pairs where recv.src_addr and send.dst_addr are the same SSA value.
// - Replace the shared placeholder "addr" with two distinct buffers:
//   one buffer for send's remote dst, one buffer for recv's remote src.
// - The buffers are created as tensor.empty (or memref.alloc if already
// bufferized).
//
//===--------------------------------------------------------------------===//

#include "magic-kernel/Conversion/TLEToMK/TLEToMK.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <optional>

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "magic-kernel/Conversion/TLEToMK/Passes.h.inc"

namespace {

static int64_t alignUp(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

static std::optional<int64_t> getStaticBytes(ShapedType ty) {
  if (!ty || !ty.hasStaticShape())
    return std::nullopt;
  auto elemTy = ty.getElementType();
  if (!elemTy.isIntOrFloat())
    return std::nullopt;
  int64_t elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
  if (elemBytes <= 0)
    return std::nullopt;
  return ty.getNumElements() * elemBytes;
}

/// Return a canonical "placeholder root" for an addr-like value.
/// This lets us match send/recv pairs even if the i64 addr was computed by
/// distinct ptr_to_int/extract ops.
static Value getPlaceholderRoot(Value addrLike) {
  Value v = addrLike;
  // Peel trivial casts (best-effort).
  if (auto cast = v.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (!cast.getOperands().empty())
      v = cast.getOperands().front();
  }

  // If it's an integer address derived from a triton ptr, use the ptr source.
  if (v.getType().isInteger(64)) {
    if (auto p2i = v.getDefiningOp<triton::PtrToIntOp>())
      v = p2i.getSrc();
  }

  // If it comes from extracting element [0,0,...] from a tensor of ptrs, use
  // the tensor-of-ptrs as the root.
  if (auto ex = v.getDefiningOp<tensor::ExtractOp>()) {
    // Only treat it as placeholder root if indices are all constants.
    // (We expect [0,0,...] here.)
    v = ex.getTensor();
  }

  return v;
}

static Value createEmptyLikeShaped(OpBuilder &b, Location loc, ShapedType ty) {
  if (auto t = dyn_cast<RankedTensorType>(ty)) {
    return b.create<tensor::EmptyOp>(loc, t.getShape(), t.getElementType());
  }
  if (auto m = dyn_cast<MemRefType>(ty)) {
    return b.create<memref::AllocOp>(loc, m);
  }
  return Value();
}

struct MKCommBufferPlanningPass
    : public MKCommBufferPlanningBase<MKCommBufferPlanningPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<mk::MagicKernelDialect, triton::TritonDialect,
                arith::ArithDialect, memref::MemRefDialect,
                bufferization::BufferizationDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](triton::FuncOp func) {
      // Map base -> first send/recv.
      DenseMap<Value, mk::SendOp> rootToSend;
      DenseMap<Value, mk::RecvOp> rootToRecv;

      func.walk([&](Operation *op) {
        if (auto send = dyn_cast<mk::SendOp>(op)) {
          Value root = getPlaceholderRoot(send.getDstAddr());
          rootToSend.try_emplace(root, send);
        } else if (auto recv = dyn_cast<mk::RecvOp>(op)) {
          Value root = getPlaceholderRoot(recv.getDst());
          rootToRecv.try_emplace(root, recv);
        }
      });

      OpBuilder b(func.getContext());
      for (auto &it : rootToRecv) {
        Value root = it.first;
        auto recv = it.second;
        auto sendIt = rootToSend.find(root);
        if (sendIt == rootToSend.end())
          continue;
        auto send = sendIt->second;

        Location loc = recv.getLoc();

        // Create a single shared buffer as close as possible while still
        // dominating both send and recv.
        DominanceInfo dom(func);
        Block *sendBlock = send->getBlock();
        Block *recvBlock = recv->getBlock();
        Block *insBlock = dom.findNearestCommonDominator(sendBlock, recvBlock);
        if (!insBlock)
          insBlock = &func.getBody().front();

        if (insBlock == sendBlock && insBlock == recvBlock) {
          // Same block: insert before the earlier op.
          Operation *insPt = send->isBeforeInBlock(recv) ? send.getOperation()
                                                         : recv.getOperation();
          b.setInsertionPoint(insPt);
        } else {
          // Different blocks: insert at end of common dominator block
          // (before terminator if any).
          b.setInsertionPointToEnd(insBlock);
          if (!insBlock->empty() &&
              insBlock->back().hasTrait<OpTrait::IsTerminator>())
            b.setInsertionPoint(&insBlock->back());
        }

        auto sendSrcTy = dyn_cast<ShapedType>(send.getSrc().getType());
        auto recvTy = recv.getNumResults() > 0
                          ? dyn_cast<ShapedType>(recv->getResult(0).getType())
                          : dyn_cast<ShapedType>(recv.getDst().getType());
        if (!sendSrcTy || !recvTy)
          continue;

        // For scheme C we expect send/recv to communicate same shape/type.
        if (sendSrcTy.getElementType() != recvTy.getElementType() ||
            sendSrcTy.getShape() != recvTy.getShape())
          continue;

        Value sharedBuf = createEmptyLikeShaped(b, loc, sendSrcTy);
        if (!sharedBuf)
          continue;

        // Replace the shared placeholder root with the shared buffer.
        // Operand layout:
        //   mk.send: 4 coords + dst_addr + src
        //   mk.recv: 4 coords + dst_key + dst
        send->setOperand(4, sharedBuf);
        recv->setOperand(4, sharedBuf); // dst_key
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> triton::createMKCommBufferPlanningPass() {
  return std::make_unique<MKCommBufferPlanningPass>();
}
