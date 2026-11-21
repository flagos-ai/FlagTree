#include "AsyncUtility.h"
#include "TritonHCUGPUToLLVM/StaticPipe.h"

#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "TargetInfo.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir::triton::HCU {
namespace {
constexpr const char *syncedViaAsyncWaitAttrName =
    "ttg.hcug.syncedViaAsyncWait";
// Traverses the def-chain including control flow of the token and returns true
// if all defining operations are an AsyncWait
bool comesFromAsyncWait(Value token) {
  if (auto defOp = token.getDefiningOp()) {
    return isa<triton::gpu::AsyncWaitOp, hcugpu::AsyncWaitOp>(defOp);
  }

  auto blockArg = dyn_cast<BlockArgument>(token);
  // If the token has no defining op and is not an BlockArgument bail out
  if (!blockArg) {
    return false;
  }

  auto block = blockArg.getOwner();
  auto argId = blockArg.getArgNumber();

  auto destOperandFromAsyncWait = [argId](auto &&operands) {
    assert(argId < operands.size());
    return comesFromAsyncWait(operands[argId]);
  };

  // Check all predecessor block's terminator and follow the passed value at
  // argId to see if they are immediately an AsyncWait.
  for (auto *pred : block->getPredecessors()) {
    auto terminator = pred->getTerminator();
    if (auto br = dyn_cast<BranchOpInterface>(terminator)) {
      for (auto successor : llvm::enumerate(br->getSuccessors())) {
        if (block != successor.value())
          continue;
        auto operands = br.getSuccessorOperands(successor.index());
        if (!destOperandFromAsyncWait(operands))
          return false;
      }
    } else {
      return false;
    }
  }
  return true;
}
} // namespace

namespace {
constexpr const char *issuerVerified = "tle.hcu.issuer_own_rows_verified";

bool constInt(Value value, int64_t expected) {
  APInt integer;
  return matchPattern(value, m_ConstantInt(&integer)) &&
         integer.getSExtValue() == expected;
}

// This first internal form has no buffer reuse, aliases or cross-wave reader.
// The public descriptor/pipe lowering must supply its own lifetime proof later.
bool proveIssuerOnly(hcugpu::AsyncWaitOp wait,
                     hcugpu::MatrixLoadToLocalOp &issue,
                     gpu::LocalLoadOp &load) {
  if (wait.getNumInst() != 0 || wait->getNumOperands() != 1 ||
      !wait.getResult().hasOneUse()) return false;
  auto commit = wait->getOperand(0).getDefiningOp<gpu::AsyncCommitGroupOp>();
  if (!commit || commit->getNumOperands() != 1 ||
      !commit.getResult().hasOneUse()) return false;
  issue = commit->getOperand(0).getDefiningOp<hcugpu::MatrixLoadToLocalOp>();
  if (!issue || !issue->hasAttr("tle.hcu.issuer_only_copy") ||
      !issue.getResult().hasOneUse()) return false;
  load = dyn_cast<gpu::LocalLoadOp>(*wait.getResult().getUsers().begin());
  if (!load || load.getToken() != wait.getResult() ||
      load.getSrc() != issue.getDest()) return false;
  auto ws = issue->getParentOfType<gpu::WarpSpecializeOp>();
  auto mod = issue->getParentOfType<ModuleOp>();
  auto arch = mod->getAttrOfType<StringAttr>("ttg.target");
  if (!ws || !mod->hasAttr("tle.hcu.software_ws_2x2") || !arch ||
      arch.getValue() != "hip:gfx938" || gpu::TritonGPUDialect::getNumCTAs(mod) != 1 ||
      gpu::lookupNumWarps(issue) != 2 ||
      gpu::TritonGPUDialect::getThreadsPerWarp(mod) != 64 ||
      issue->getBlock() != &ws.getDefaultRegion().front() ||
      wait->getBlock() != issue->getBlock() || load->getBlock() != issue->getBlock() ||
      !issue->isBeforeInBlock(wait) || !wait->isBeforeInBlock(load)) return false;
  auto allocation = issue.getDest().getDefiningOp<gpu::LocalAllocOp>();
  if (!allocation || allocation->getBlock() != issue->getBlock() ||
      allocation.getAlignmentOrDefault() < 256 ||
      !allocation->isBeforeInBlock(issue)) return false;
  // No aliases, second issue, pointer extraction, escape or reuse are admitted.
  unsigned uses = 0;
  for (Operation *user : allocation.getResult().getUsers()) {
    if (user != issue.getOperation() && user != load.getOperation()) return false;
    ++uses;
  }
  if (uses != 2) return false;
  auto dst = cast<gpu::MemDescType>(issue.getDest().getType());
  auto shared = dyn_cast<gpu::SwizzledSharedEncodingAttr>(dst.getEncoding());
  auto tensor = cast<RankedTensorType>(load.getType());
  auto blocked = dyn_cast<gpu::BlockedEncodingAttr>(tensor.getEncoding());
  if (dst.getShape() != ArrayRef<int64_t>{64,64} || dst.getAllocShape() != dst.getShape() ||
      !dst.getElementType().isInteger(8) || !shared || shared.getVec() != 32 ||
      shared.getPerPhase() != 2 || shared.getMaxPhase() != 2 ||
      shared.getOrder() != ArrayRef<unsigned>{1,0} || !blocked ||
      blocked.getSizePerThread() != ArrayRef<unsigned>{32,1} ||
      blocked.getThreadsPerWarp() != ArrayRef<unsigned>{1,64} ||
      blocked.getWarpsPerCTA() != ArrayRef<unsigned>{2,1} ||
      blocked.getOrder() != ArrayRef<unsigned>{1,0}) return false;
  if (!issue.getBoundaryCheck().empty() || issue.getTensorShape() != ArrayRef<int32_t>{64,64} || issue.getShape().size() != 2 ||
      issue.getStrides().size() != 2 || issue.getIndices().size() != 2 ||
      !constInt(issue.getShape()[0],64) || !constInt(issue.getShape()[1],64) ||
      !constInt(issue.getStrides()[0],64) || !constInt(issue.getStrides()[1],1) ||
      !constInt(issue.getIndices()[0],0) || !constInt(issue.getIndices()[1],0)) return false;
  return true;
}
} // namespace

bool isVerifiedIssuerOnlyPair(Operation *a, Operation *b) {
  if (isa<gpu::LocalLoadOp>(a)) std::swap(a,b);
  auto issue = dyn_cast<hcugpu::MatrixLoadToLocalOp>(a);
  auto load = dyn_cast<gpu::LocalLoadOp>(b);
  return issue && load && issue->hasAttr(issuerVerified) &&
         load->hasAttr(issuerVerified) && load.getSrc() == issue.getDest();
}

void addLocalBarrierAfterAmdGpuAsyncWait(ModuleOp mod) {
  auto *ctx = mod->getContext();

  mod->removeAttr("tle.hcu.invalid_issuer_copy");
  // Do not trust caller-provided proof markers from input IR.
  mod.walk([](Operation *op) { op->removeAttr(issuerVerified); });
  SmallVector<hcugpu::AsyncWaitOp> waits;
  mod->walk([&waits](hcugpu::AsyncWaitOp waitOp) { waits.push_back(waitOp); });

  IRRewriter builder(mod.getContext());
  for (auto waitOp : waits) {
    hcugpu::MatrixLoadToLocalOp issue;
    gpu::LocalLoadOp load;
    if (static_pipe::isTransportWait(waitOp,issue)) {
      issue->setAttr(issuerVerified,UnitAttr::get(ctx));
      continue;
    }
    if (proveIssuerOnly(waitOp, issue, load)) {
      issue->setAttr(issuerVerified, UnitAttr::get(ctx));
      load->setAttr(issuerVerified, UnitAttr::get(ctx));
      continue;
    }
    if (isa<mlir::gpu::BarrierOp, gpu::LocalBarrierOp>(waitOp->getNextNode()))
      continue;

    builder.setInsertionPointAfter(waitOp);
    builder.create<triton::gpu::LocalBarrierOp>(waitOp->getLoc());
  }
  // Fail before type conversion, including orphan issues or malformed tokens.
  mod.walk([&](hcugpu::MatrixLoadToLocalOp issue) {
    if (issue->hasAttr("tle.hcu.issuer_only_copy") &&
        !issue->hasAttr(issuerVerified)) {
      issue.emitOpError("unproved issuer-only MLS ownership/lifecycle");
      mod->setAttr("tle.hcu.invalid_issuer_copy", UnitAttr::get(ctx));
    }
  });
}

void annotateLocalLoadsSyncedViaAsyncWait(ModuleOp mod) {
  auto *ctx = mod->getContext();

  mod->walk([&](Operation *op) {
    TypeSwitch<Operation *, void>(op)
        .Case<triton::gpu::LocalLoadOp,
              triton::hcugpu::LocalLoadPackedTransposedOp>([&](auto loadOp) {
          if (loadOp->hasAttr(syncedViaAsyncWaitAttrName))
            return;
          Value token = loadOp.getToken();
          bool isSyncedViaAsyncWait = token && comesFromAsyncWait(token);
          loadOp->setAttr(syncedViaAsyncWaitAttrName,
                          BoolAttr::get(ctx, isSyncedViaAsyncWait));
        });
  });
}

bool isSyncedViaAsyncWait(Operation *op) {
  assert(op);

  auto attr = op->getAttr(syncedViaAsyncWaitAttrName);
  if (!attr) {
    op->emitRemark("has no async sync information attached to it which "
                   "might negatively affect performance. Run "
                   "annotateLocalLoadSyncedViaAsyncWait first");
    return false;
  }
  return cast<BoolAttr>(attr).getValue();
}

namespace {
LLVM::AliasScopeDomainAttr getLoadScopeDomain(MLIRContext *ctx) {
  Builder b(ctx);
  return b.getAttr<LLVM::AliasScopeDomainAttr>(
      b.getStringAttr("hcug.AsyncOps"),
      b.getStringAttr(
          "Domain to hold alias scopes to specify aliasing information between "
          "AsyncCopyGlobalToLocal, BufferLoadToLocal and LocalLoad ops"));
}

LLVM::AliasScopeAttr getAsyncCopyScope(MLIRContext *ctx) {
  Builder b(ctx);
  auto name = b.getStringAttr("hcug.AsyncCopies");
  auto desc = b.getStringAttr(
      "Scope containing all AsyncCopyGlobalToLocal and BufferLoadToLocal ops");
  return b.getAttr<LLVM::AliasScopeAttr>(name, getLoadScopeDomain(ctx), desc);
}

LLVM::AliasScopeAttr getLoadCopyScope(MLIRContext *ctx) {
  Builder b(ctx);
  auto name = b.getStringAttr("hcug.LocalLoads");
  auto desc = b.getStringAttr("Scope containing all LocalLoad ops");
  return b.getAttr<LLVM::AliasScopeAttr>(name, getLoadScopeDomain(ctx), desc);
}
} // namespace

void addAsyncCopyAliasScope(LLVM::AliasAnalysisOpInterface directToLdsOp) {
  auto ctx = directToLdsOp->getContext();
  Builder b(ctx);
  directToLdsOp.setAliasScopes(b.getArrayAttr(getAsyncCopyScope(ctx)));
}

void addLocalLoadNoAliasScope(Operation *localLoadOp,
                              LLVM::AliasAnalysisOpInterface llLoadOp) {
  if (!localLoadOp->hasTrait<OpTrait::LocalLoadTrait>())
    return;
  if (!isSyncedViaAsyncWait(localLoadOp))
    return;

  return addLocalLoadNoAliasScope(llLoadOp);
}

void addLocalLoadNoAliasScope(LLVM::AliasAnalysisOpInterface llLoadOp) {
  auto ctx = llLoadOp->getContext();

  // Do not alias with AsyncCopies
  auto noAliasScopes = ArrayAttr::get(ctx, getAsyncCopyScope(ctx));
  llLoadOp.setNoAliasScopes(noAliasScopes);

  // Add to different scope as ops without any scope alias with everything
  auto aliasScopes = ArrayAttr::get(ctx, getLoadCopyScope(ctx));
  llLoadOp.setAliasScopes(aliasScopes);
}

unsigned
fitToValidDirectToLdsVecSize(unsigned maxVecSize, unsigned elemBitwidth,
                             const triton::HCU::TargetInfo &targetInfo) {
  while (maxVecSize > 0 && !targetInfo.supportsDirectToLdsLoadBitWidth(
                               maxVecSize * elemBitwidth)) {
    maxVecSize /= 2;
  }
  return maxVecSize;
}

} // namespace mlir::triton::HCU
