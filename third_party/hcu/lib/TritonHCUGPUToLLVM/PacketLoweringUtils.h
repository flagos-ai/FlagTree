#ifndef TRITON_HCU_PACKET_LOWERING_UTILS_H
#define TRITON_HCU_PACKET_LOWERING_UTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include <iterator>
#include <optional>

// Shared by WholePacket, RawWindow, structured register loads and legacy MLS.
// Coordinate proofs here do not authorize memory motion or barrier removal.
namespace mlir::triton::HCU::packet_lowering {
inline LLVM::LLVMFuncOp getOrInsertHCUIntrinsic(ModuleOp module, Location loc,
    ConversionPatternRewriter &rewriter, StringRef name, Type result,
    ArrayRef<Type> args) {
  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto fn = LLVM::LLVMFuncOp::create(rewriter, loc, name,
      LLVM::LLVMFunctionType::get(result, args, false));
  fn.setConvergent(true);
  fn.setPassthroughAttr(rewriter.getArrayAttr({rewriter.getStringAttr("nounwind")}));
  return fn;
}

// Closed static-loop source-coordinate validation.
// This validates coordinates only. The separate closed protocol verifier must
// prove all memory effects, exact accumulator chain, view ownership and barriers.
// No certificate attributes, new pass or generic range-analysis infrastructure.

inline std::optional<int64_t> getSignedI64Constant(Value v) {
  auto c = v.getDefiningOp<arith::ConstantOp>();
  auto a = c ? dyn_cast<IntegerAttr>(c.getValue()) : IntegerAttr{};
  if (!a || !a.getValue().isSignedIntN(64))
    return std::nullopt;
  return a.getInt();
}

struct CountedLoopBody {
  Block *body;
  Value iv;
  int64_t trips;
};

// Actual compiler_hcu.make_llir converts SCF to CF before this preflight.
// Reuse the existing MembarUtility exact-four-block counted-loop proof shape,
// specialized to constant positive trip/step1. Default arity2 remains the
// original stage1 contract; arity0 is used only by prefix2's local SSA queues.
inline std::optional<CountedLoopBody>
getCanonicalCountedLoopBody(Operation *anchor, int64_t maxTrips, unsigned arguments = 2) {
  auto fn = anchor->getParentOfType<triton::FuncOp>();
  if (!fn || fn.getBody().getBlocks().size() != 4)
    return std::nullopt;
  Block *entry = &fn.getBody().front();
  auto enter = dyn_cast<cf::BranchOp>(entry->getTerminator());
  if (!enter)
    return std::nullopt;
  Block *header = enter.getDest();
  auto cond = dyn_cast<cf::CondBranchOp>(header->getTerminator());
  unsigned arity = header->getNumArguments();
  if (!cond || (arguments ? arity != arguments : arity < 2) ||
      !header->getArgument(0).getType().isInteger(32) ||
      enter.getDestOperands().size() != arity ||
      getSignedI64Constant(enter.getDestOperands()[0]) != 0 ||
      std::distance(header->begin(), header->end()) != 2)
    return std::nullopt;
  Block *body = cond.getTrueDest(), *exit = cond.getFalseDest();
  llvm::SmallPtrSet<Block *, 4> blocks{entry, header, body, exit};
  if (blocks.size() != 4 || anchor->getBlock() != body ||
      body->getNumArguments() || exit->getNumArguments() != 0 ||
      !cond.getTrueDestOperands().empty() ||
      !cond.getFalseDestOperands().empty() ||
      body->getSinglePredecessor() != header ||
      exit->getSinglePredecessor() != header ||
      !entry->hasNoPredecessors())
    return std::nullopt;
  auto predecessors = llvm::to_vector(header->getPredecessors());
  if (predecessors.size() != 2 ||
      !llvm::is_contained(predecessors, entry) ||
      !llvm::is_contained(predecessors, body))
    return std::nullopt;
  auto cmp = cond.getCondition().getDefiningOp<arith::CmpIOp>();
  auto trips = cmp ? getSignedI64Constant(cmp.getRhs()) : std::nullopt;
  Value iv = header->getArgument(0);
  if (!cmp || cmp->getBlock() != header ||
      cmp.getPredicate() != arith::CmpIPredicate::slt || cmp.getLhs() != iv ||
      !trips || *trips < 1 || *trips > maxTrips)
    return std::nullopt;
  auto back = dyn_cast<cf::BranchOp>(body->getTerminator());
  if (!back || back.getDest() != header || back.getDestOperands().size() != arity)
    return std::nullopt;
  auto inc = back.getDestOperands()[0].getDefiningOp<arith::AddIOp>();
  if (!inc || inc->getBlock() != body || inc.getLhs() != iv ||
      getSignedI64Constant(inc.getRhs()) != 1)
    return std::nullopt;
  // Initial0, step1, sltTrip, unique entry/backedge prove 0<=IV<Trip.
  // Caller bounds Trip by its validated source extent before using IV*BK.
  return CountedLoopBody{body, iv, *trips};
}


} // namespace mlir::triton::HCU::packet_lowering
#endif
