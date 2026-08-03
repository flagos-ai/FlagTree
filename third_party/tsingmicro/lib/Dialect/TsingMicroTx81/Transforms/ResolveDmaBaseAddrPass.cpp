//===-- ResolveDmaBaseAddrPass.cpp - DMA Base Address Resolution ----------===//
//
// Backward-slice tx.rdma/tx.wdma DDR address operands to find the originating
// func.func memref argument, then append the base DDR address as a new operand.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"
#include "tsingmicro-tx81/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdlib>

#define DEBUG_TYPE "tx81-resolve-dma-base-addr"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TX81RESOLVEDMABASEADDR
#include "tsingmicro-tx81/Transforms/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

/// Trace an i64/index DDR address value back to its base memref.
/// Returns the base memref value (BlockArgument or alloc), or Value() if
/// unanalyzable.
static Value traceToBaseMemRef(Value v, unsigned maxDepth = 16) {
  if (maxDepth == 0)
    return {};

  // Terminal: direct BlockArgument of memref type
  if (auto ba = dyn_cast<BlockArgument>(v)) {
    if (isa<MemRefType>(ba.getType()) || isa<UnrankedMemRefType>(ba.getType()))
      return ba;

    llvm::errs() << "ba type: ";
    ba.getType().dump();
    llvm_unreachable("Unexpected mem address");
    return {};
  }

  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return {};

  // Cast ops: penetrate through
  if (auto op = dyn_cast<arith::IndexCastOp>(defOp))
    return traceToBaseMemRef(op.getIn(), maxDepth - 1);
  if (auto op = dyn_cast<arith::TruncIOp>(defOp))
    return traceToBaseMemRef(op.getIn(), maxDepth - 1);
  if (auto op = dyn_cast<arith::ExtSIOp>(defOp))
    return traceToBaseMemRef(op.getIn(), maxDepth - 1);
  if (auto op = dyn_cast<arith::ExtUIOp>(defOp))
    return traceToBaseMemRef(op.getIn(), maxDepth - 1);

  // Arith add/sub: try both operands, return the first that finds a memref base
  if (auto op = dyn_cast<arith::AddIOp>(defOp)) {
    if (Value r = traceToBaseMemRef(op.getLhs(), maxDepth - 1))
      return r;
    return traceToBaseMemRef(op.getRhs(), maxDepth - 1);
  }
  if (auto op = dyn_cast<arith::SubIOp>(defOp)) {
    if (Value r = traceToBaseMemRef(op.getLhs(), maxDepth - 1))
      return r;
    return traceToBaseMemRef(op.getRhs(), maxDepth - 1);
  }

  // Mul: penetrate non-constant branch
  if (auto op = dyn_cast<arith::MulIOp>(defOp)) {
    auto lhsConst = op.getLhs().getDefiningOp<arith::ConstantOp>();
    auto rhsConst = op.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!lhsConst && rhsConst)
      return traceToBaseMemRef(op.getLhs(), maxDepth - 1);
    if (lhsConst && !rhsConst)
      return traceToBaseMemRef(op.getRhs(), maxDepth - 1);
    if (Value r = traceToBaseMemRef(op.getLhs(), maxDepth - 1))
      return r;
    return traceToBaseMemRef(op.getRhs(), maxDepth - 1);
  }

  // Memref ops: key connection points from index/i64 back to memref world
  if (auto op = dyn_cast<memref::ExtractAlignedPointerAsIndexOp>(defOp))
    return traceToBaseMemRef(op.getSource(), maxDepth - 1);

  if (auto op = dyn_cast<memref::ExtractStridedMetadataOp>(defOp)) {
    // result 0 is the base buffer
    if (v == op->getResult(0))
      return traceToBaseMemRef(op.getSource(), maxDepth - 1);
    return {};
  }

  if (auto op = dyn_cast<memref::ReinterpretCastOp>(defOp))
    return traceToBaseMemRef(op.getSource(), maxDepth - 1);

  if (auto op = dyn_cast<memref::SubViewOp>(defOp))
    return traceToBaseMemRef(op.getSource(), maxDepth - 1);

  return {};
}

/// Generate a base i64 DDR address from a base memref value.
static Value generateBaseAddr(Value baseMemRef, Operation *insertBefore,
                              OpBuilder &builder) {
  auto loc = insertBefore->getLoc();
  auto i64Ty = builder.getI64Type();

  // Extract base buffer from memref chain (through reinterpret_cast etc.)
  Value current = baseMemRef;
  while (true) {
    if (auto op = current.getDefiningOp<memref::ReinterpretCastOp>()) {
      current = op.getSource();
      continue;
    }
    if (auto op = current.getDefiningOp<memref::SubViewOp>()) {
      current = op.getSource();
      continue;
    }
    break;
  }

  // If the root is unranked, we cannot extract the pointer directly — its
  // descriptor layout ({rank, ptr}) differs from ranked ({allocated, aligned,
  // offset, ...}).  Find a reinterpret_cast user that ranks it.  The base may
  // have multiple reinterpret_cast users in different regions (e.g. scf.if vs
  // scf.for); use DominanceInfo to pick one that dominates insertBefore.
  Value extractFrom = current;
  if (isa<UnrankedMemRefType>(current.getType())) {
    DominanceInfo domInfo;
    for (auto *user : current.getUsers()) {
      if (auto cast = dyn_cast<memref::ReinterpretCastOp>(user)) {
        if (domInfo.properlyDominates(cast.getResult(), insertBefore)) {
          extractFrom = cast.getResult();
          break;
        }
      }
    }
    if (extractFrom == current) {
      llvm::report_fatal_error(
          "tx81-resolve-dma-base-addr: no reinterpret_cast of unranked base "
          "dominates the DMA op");
    }
  }

  Value extracted =
      builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, extractFrom);
  Value baseI64 = builder.create<arith::IndexCastOp>(loc, i64Ty, extracted);
  return baseI64;
}

/// Get or create an LLVM global string constant, returning a GEP pointer to it.
static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     ModuleOp mod) {
  LLVM::GlobalOp global;
  if (!(global = mod.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(mod.getBody());
    auto type = LLVM::LLVMArrayType::get(
        IntegerType::get(builder.getContext(), 8), value.size());
    global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, name,
                                            builder.getStringAttr(value), 0);
  }
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                builder.getIndexAttr(0));
  return builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
      globalPtr, ArrayRef<Value>({cst0, cst0}));
}

/// Declare rcs_ep_log in the module if not already present.
static FlatSymbolRefAttr getOrInsertRcsEpLog(OpBuilder &builder, ModuleOp mod) {
  auto *ctx = mod.getContext();
  StringRef funcName = "rcs_ep_log";
  if (mod.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
    return SymbolRefAttr::get(ctx, funcName);
  auto llvmPtr = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto funcType = LLVM::LLVMFunctionType::get(
      voidTy, {llvmPtr, llvmPtr, i32Ty, i32Ty, llvmPtr}, /*isVarArg=*/true);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(mod.getBody());
  builder.create<LLVM::LLVMFuncOp>(mod.getLoc(), funcName, funcType);
  return SymbolRefAttr::get(ctx, funcName);
}

/// Insert rcs_ep_log(__FILE__, __func__, 0, 3, fmt, pid...) after the last
/// tt.get_program_id in funcOp.  Only axes that actually appear are printed:
///   x only        → "pid = %d\n"
///   x, y          → "pid = (%d, %d)\n"
///   x, y, z       → "pid = (%d, %d, %d)\n"
static void insertEpLogForFunc(func::FuncOp funcOp, ModuleOp mod) {
  Value pidVals[3] = {};
  bool hasAxis[3] = {};
  int axisCount = 0;
  Operation *lastPidOp = nullptr;

  funcOp.walk([&](triton::GetProgramIdOp op) {
    lastPidOp = op.getOperation();
    int ax = static_cast<int>(op.getAxis());
    if (!hasAxis[ax]) {
      hasAxis[ax] = true;
      axisCount++;
    }
    pidVals[ax] = op.getResult();
  });
  if (!lastPidOp)
    return;

  auto loc = lastPidOp->getLoc();
  auto *ctx = mod.getContext();
  OpBuilder builder(lastPidOp->getNextNode());
  auto i32Ty = IntegerType::get(ctx, 32);

  // Build format string dynamically based on which axes exist:
  //   1 axis  → "pid = %d\n"
  //   2 axes  → "pid = (%d, %d)\n"
  //   3 axes  → "pid = (%d, %d, %d)\n"
  std::string fmtStr;
  if (axisCount == 1) {
    fmtStr = "pid = %d\n";
  } else {
    fmtStr = "pid = (";
    bool first = true;
    for (int i = 0; i < 3; ++i) {
      if (!hasAxis[i])
        continue;
      if (!first)
        fmtStr += ", ";
      first = false;
      fmtStr += "%d";
    }
    fmtStr += ")\n";
  }
  fmtStr += '\0';

  // __FILE__: try the op's source location, fall back to "kernel.mlir"
  std::string fileStr;
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    fileStr = fileLoc.getFilename().str();
  else
    fileStr = "kernel.mlir";
  fileStr += '\0';

  std::string funcStr = funcOp.getSymName().str();
  funcStr += '\0';

  std::string fileGlobal = "rcs_ep_log_file_" + funcOp.getSymName().str();
  std::string funcGlobal = "rcs_ep_log_func_" + funcOp.getSymName().str();

  Value filePtr =
      getOrCreateGlobalString(loc, builder, fileGlobal, fileStr, mod);
  Value funcPtr =
      getOrCreateGlobalString(loc, builder, funcGlobal, funcStr, mod);
  Value fmtPtr =
      getOrCreateGlobalString(loc, builder, "rcs_ep_log_fmt_pid", fmtStr, mod);

  auto llvmPtr = LLVM::LLVMPointerType::get(ctx);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto callType = LLVM::LLVMFunctionType::get(
      voidTy, {llvmPtr, llvmPtr, i32Ty, i32Ty, llvmPtr}, /*isVarArg=*/true);

  Value line0 = builder.create<LLVM::ConstantOp>(loc, i32Ty,
                                                 builder.getI32IntegerAttr(0));
  Value level3 = builder.create<LLVM::ConstantOp>(loc, i32Ty,
                                                  builder.getI32IntegerAttr(3));

  SmallVector<Value, 8> callArgs = {filePtr, funcPtr, line0, level3, fmtPtr};
  for (int i = 0; i < 3; ++i)
    if (hasAxis[i])
      callArgs.push_back(pidVals[i]);

  auto rcsRef = getOrInsertRcsEpLog(builder, mod);
  builder.create<LLVM::CallOp>(loc, callType, rcsRef, callArgs);
}

/// Rewrite a tx.rdma or tx.wdma op by appending the resolved base DDR address.
template <typename TxDmaOp> LogicalResult resolveOp(TxDmaOp op) {
  // Determine which operand is the DDR address
  Value ddrAddr;
  if constexpr (std::is_same_v<TxDmaOp, tx::RdmaOp> ||
                std::is_same_v<TxDmaOp, tx::Rdma1dOp>) {
    ddrAddr = op.getSource(); // rdma: source is DDR
  } else {
    ddrAddr = op.getTarget(); // wdma: target is DDR
  }

  // Trace back to base memref
  Value baseMemRef = traceToBaseMemRef(ddrAddr);
  Value baseAddr;
  OpBuilder builder(op);
  if (baseMemRef) {
    baseAddr = generateBaseAddr(baseMemRef, op.getOperation(), builder);
  } else {
    // Fallback: use the original DDR address itself as base
    baseAddr = ddrAddr;
  }

  // Replace the placeholder base_ddr_addr (last operand) with the resolved one.
  // Keep the same number of operands so the existing operandSegmentSizes is
  // valid.
  SmallVector<Value, 20> newOperands;
  newOperands.append(op->operand_begin(), std::prev(op->operand_end()));
  newOperands.push_back(baseAddr);

  // Create new op with the resolved base_ddr_addr operand
  OperationState state(op.getLoc(), op->getName());
  state.addOperands(newOperands);
  for (auto namedAttr : op->getAttrs()) {
    state.addAttribute(namedAttr.getName(), namedAttr.getValue());
  }

  // 1D ops have no results; N-D ops have one result
  if constexpr (std::is_same_v<TxDmaOp, tx::Rdma1dOp> ||
                std::is_same_v<TxDmaOp, tx::Wdma1dOp>) {
    Operation *newOp = builder.create(state);
    op->erase();
  } else {
    Type resultType = op->getResult(0).getType();
    state.addTypes(resultType);
    Operation *newOp = builder.create(state);
    Value newResult = newOp->getResult(0);
    op->getResult(0).replaceAllUsesWith(newResult);
    op->erase();
  }

  return success();
}

class Tx81ResolveDmaBaseAddrPass
    : public triton::impl::Tx81ResolveDmaBaseAddrBase<
          Tx81ResolveDmaBaseAddrPass> {
  using Tx81ResolveDmaBaseAddrBase<
      Tx81ResolveDmaBaseAddrPass>::Tx81ResolveDmaBaseAddrBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tx::Tx81Dialect, func::FuncDialect, memref::MemRefDialect,
                    arith::ArithDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    const char *checkingEnv = std::getenv("TRITON_DMA_CHECKING");
    if (checkingEnv && StringRef(checkingEnv) == "1") {
      mod.walk([&](func::FuncOp funcOp) { insertEpLogForFunc(funcOp, mod); });
    }

    mod.walk([&](Operation *op) {
      if (auto rdmaOp = dyn_cast<tx::RdmaOp>(op)) {
        if (failed(resolveOp(rdmaOp)))
          signalPassFailure();
      } else if (auto wdmaOp = dyn_cast<tx::WdmaOp>(op)) {
        if (failed(resolveOp(wdmaOp)))
          signalPassFailure();
      } else if (auto rdma1dOp = dyn_cast<tx::Rdma1dOp>(op)) {
        if (failed(resolveOp(rdma1dOp)))
          signalPassFailure();
      } else if (auto wdma1dOp = dyn_cast<tx::Wdma1dOp>(op)) {
        if (failed(resolveOp(wdma1dOp)))
          signalPassFailure();
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTx81ResolveDmaBaseAddrPass() {
  return std::make_unique<Tx81ResolveDmaBaseAddrPass>();
}
