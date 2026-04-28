#include "TargetInfo.h"
#include <climits>
#include <cstdlib>
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "third_party/hcu/include/TritonHCUGPUToLLVM/Passes.h"
#include "third_party/hcu/include/Dialect/TritonHCUGPU/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_HCUGPUCONVERTWARPSPECIALIZETOLLVM
#include "TritonHCUGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// convertOpTypes
//===----------------------------------------------------------------------===//

static void convertOpTypes(Operation *op, const TypeConverter &typeConverter) {
  ImplicitLocOpBuilder b(op->getLoc(), op);
  SmallVector<Value> operands = llvm::to_vector(op->getOperands());
  for (Value &operand : operands) {
    Type type = typeConverter.convertType(operand.getType());
    if (type != operand.getType()) {
      operand =
          b.create<UnrealizedConversionCastOp>(type, operand).getResult(0);
    }
  }
  op->setOperands(operands);

  for (Region &region : op->getRegions()) {
    b.setInsertionPointToStart(&region.front());
    for (BlockArgument arg : llvm::to_vector(region.getArguments())) {
      Type type = typeConverter.convertType(arg.getType());
      BlockArgument newArg = region.addArgument(type, arg.getLoc());
      auto cast = b.create<UnrealizedConversionCastOp>(arg.getType(), newArg);
      arg.replaceAllUsesWith(cast.getResult(0));
      region.eraseArgument(0);
    }
  }

  SmallVector<Type> resultTypes;
  (void)typeConverter.convertTypes(op->getResultTypes(), resultTypes);
  if (TypeRange(resultTypes) == op->getResultTypes())
    return;
  OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                       resultTypes, op->getAttrs());
  for (Region &region : op->getRegions())
    state.addRegion()->takeBody(region);
  b.setInsertionPoint(op);
  Operation *newOp = b.create(state);

  SmallVector<Value> results;
  for (auto [i, result, type] :
       llvm::enumerate(newOp->getResults(), op->getResultTypes())) {
    auto cast = b.create<UnrealizedConversionCastOp>(type, result);
    op->getResult(i).replaceAllUsesWith(cast.getResult(0));
  }
  op->erase();
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Reserve one barrier for the default warp group, one for the start barrier,
// and one for the end barrier.
enum BarrierIndex {
  kSwitchLoopBarrierIdx,
  kPartitionStartBarrierIdx,
  kAbarrierInvBarrierIdx,
  kPartitionEndBarrierIdx,
  kDefaultWarpGroupBarrierIdx,
  kReturnBarrierIdx,
  kNumReservedBarriers,
  kNumBarriers = 16
};

static void createBarrier(TritonLLVMIRRewriter &b, unsigned barIdx,
                          std::optional<unsigned> numWarps) {
  assert(barIdx < 16 && "not enough barriers");
  if (numWarps.has_value()) {
    b.create<ROCDL::HCUEbarrierSyncOp>(barIdx, numWarps.value());
  } else {
    b.create<ROCDL::HCUEbarrierSyncOp>(barIdx, 0);
  }
}

static void lowerHCUAbarrierArriveOp(ROCDL::HCUAbarrierArriveOp bar, unsigned barIdx, unsigned warpStartId, unsigned numWarps) {
  TritonLLVMIRRewriter b(bar.getLoc(), bar);
  Value warpId = b.create<ROCDL::HCUGetWaveIdOp>();
  Value zero = b.i32_val(warpStartId);
  Value isWarpZero = b.icmp_eq(warpId, zero);
  Block *currentBlock = bar->getBlock();
  Block *afterBlock = currentBlock->splitBlock(bar);
  Block *barrierBlock = b.createBlock(afterBlock);
  b.setInsertionPointToEnd(currentBlock);
  createBarrier(b, barIdx, numWarps);
  b.create<LLVM::CondBrOp>(isWarpZero, barrierBlock, afterBlock);
  bar->moveBefore(barrierBlock, barrierBlock->begin());
  b.setInsertionPointToEnd(barrierBlock);
  b.create<LLVM::BrOp>(afterBlock);
}

//===----------------------------------------------------------------------===//
// elideTrivialCaptures
//===----------------------------------------------------------------------===//

static LogicalResult findTrivialSubcomputation(LLVM::LLVMFuncOp func,
                                               Value capture,
                                               SetVector<Operation *> &ops) {
  SetVector<Value> worklist;
  worklist.insert(capture);
  for (unsigned i = 0; i != worklist.size(); ++i) {
    Value capture = worklist[i];
    // Check for a kernel argument.
    if (auto arg = dyn_cast<BlockArgument>(capture)) {
      if (arg.getOwner() == &func.getBody().front())
        continue;
      // Otherwise, this is some other block argument that cannot be elided.
      return failure();
    }

    Operation *op = capture.getDefiningOp();
    // Check if the defining op can be rematerialized. At the LLVM level,
    // checking for pure is probably a good enough heuristic.
    if (isPure(op)) {
      ops.insert(op);
      worklist.insert(op->operand_begin(), op->operand_end());
      continue;
    }
    // The op cannot be rematerialized.
    return failure();
  }

  // Cap the number of ops that can be rematerialized.
  // FIXME: This is arbitrary. Maybe use as a autotune parameter.
  //return success(ops.size() <= 16);
  return success();
}

static void elideTrivialCaptures(LLVM::LLVMFuncOp func,
                                 ArrayRef<WarpSpecializeOp> wsOps) {
  // The goal is to completely eliminate captures by hoisting or rematerializing
  // computations. We could minimize captures by rematerializing
  // subcomputations, but that is much more complicated. Prefer rematerializing
  // because that reduces liveranges. If subgraphs are duplicated more than
  // once, we will rely on CSE to clean them up.
  SetVector<Operation *> subgraph;
  for (WarpSpecializeOp wsOp : wsOps) {
    llvm::BitVector toErase(wsOp.getNumOperands());
    for (auto [i, capture] : llvm::enumerate(wsOp.getExplicitCaptures())) {
      subgraph.clear();
      if (failed(findTrivialSubcomputation(func, capture, subgraph)))
        continue;
      toErase.set(i);
      subgraph = topologicalSort(subgraph);

      for (Region *region : wsOp.getPartitionRegions()) {
        OpBuilder b(region);
        IRMapping mapping;
        for (Operation *op : subgraph) {
          b.clone(*op, mapping);
        }
        Value remat = capture;
        if (!subgraph.empty()) {
          unsigned resultIdx = cast<OpResult>(capture).getResultNumber();
          remat = mapping.lookup(subgraph.back())->getResult(resultIdx);
        }
        region->getArgument(i).replaceAllUsesWith(remat);
      }
    }

    wsOp->eraseOperands(toErase);
    for (Region *region : wsOp.getPartitionRegions()) {
      region->front().eraseArguments(toErase);
    }
  }
}

//===----------------------------------------------------------------------===//
// lowerWarpSpecialize
//===----------------------------------------------------------------------===//

// Assign hardware barriers to each warp group and rewrite warp group barriers
// into `ebarrier` instructions. There is a maximum number of barriers.
static LogicalResult rewriteWarpGroupBarriers(LLVM::LLVMFuncOp func,
                                              ArrayRef<WarpSpecializeOp> wsOps,
                                              unsigned defaultNumWarps) {
  func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Walk into default regions but not partition regions.
    if (isa<WarpSpecializePartitionsOp>(op))
      return WalkResult::skip();

    if (auto bar = dyn_cast<ROCDL::BarrierOp>(op)) {
      TritonLLVMIRRewriter b(bar.getLoc(), bar);
      createBarrier(b, kDefaultWarpGroupBarrierIdx, defaultNumWarps);
      bar.erase();
      return WalkResult::skip();
    }

    return WalkResult::advance();
  });

  // Each partition executes simultaneously, so each will get a different
  // barrier ID, but note this means there is a maximum of 16 barriers.
  for (WarpSpecializeOp op : wsOps) {
    for (auto [idx, partition] : llvm::enumerate(op.getPartitionRegions())) {
      unsigned barIdx = idx + kNumReservedBarriers;
      if (barIdx >= kNumBarriers) {
        return func.emitError("cannot support more than ")
               << (kNumBarriers - kNumReservedBarriers)
               << " warp group partitions";
      }
      unsigned warpGroupSize = op.getPartitionNumWarps()[idx];
      unsigned warpStartId = (*op.getWarpGroupStartIds())[idx];
      partition->walk([&](ROCDL::BarrierOp bar) {
        TritonLLVMIRRewriter b(bar.getLoc(), bar);
        createBarrier(b, barIdx, warpGroupSize);
        bar.erase();
      });
    }
  }

  return success();
}

static void rewritePartitionRegions(WarpSpecializeOp ws, SmallVector<Block *> &sinkBlocks,
                                    const HCU::TargetInfo &targetInfo) {
  TritonLLVMIRRewriter b(ws.getLoc(), ws.getContext());

  for (auto [idx, partition] : llvm::enumerate(ws.getPartitionRegions())) {
    Block *sinkBlock = sinkBlocks[idx];
    // Load the explicit captures from shared memory and replace the block args
    // if there are any.
    b.setInsertionPointToStart(&partition->front());
    createBarrier(b, kSwitchLoopBarrierIdx, /*numWarps=*/std::nullopt);

    if (partition->getNumArguments()) {
      auto captureType = LLVM::LLVMStructType::getLiteral(
          b.getContext(), llvm::to_vector(partition->getArgumentTypes()),
          /*isPacked=*/true);
      Value capturePtr =
          LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, ws);
      LLVM::LLVMPointerType ptrTy = ptr_ty(b.getContext(), 3);
      for (auto [i, arg] :
           llvm::zip(llvm::seq<int32_t>(partition->getNumArguments()),
                     partition->getArguments())) {
        Value ptr =
            b.gep(ptrTy, captureType, capturePtr, ArrayRef<LLVM::GEPArg>{0, i});
        // Each thread in the warp group needs a copy of the value.
        Value value = b.load(arg.getType(), ptr, /*align=*/1);
        arg.replaceAllUsesWith(value);
      }
      partition->front().eraseArguments([](auto) { return true; });
    }

    // The shared memory is only live for the entry into the region, so put
    // another barrier here.
    //createBarrier(b, kPartitionStartBarrierIdx, /*numWarps=*/std::nullopt);

    // For the first partition, insert an ebarrier with id kAbarrierInvBarrierIdx
    // before the earliest HCUAbarrierInvOp in the region, if any.
    if (idx == 0 || idx == 2) {
      ROCDL::HCUAbarrierInvOp earliestInvOp = nullptr;
      partition->walk<mlir::WalkOrder::PreOrder>(
          [&](ROCDL::HCUAbarrierInvOp op) {
            earliestInvOp = op;
            return WalkResult::interrupt();
          });
      if (earliestInvOp) {
        TritonLLVMIRRewriter invBuilder(earliestInvOp.getLoc(), earliestInvOp);
        createBarrier(invBuilder, kAbarrierInvBarrierIdx,
                      /*numWarps=*/std::nullopt);
      }
    }

    // Rewrite all warp returns.
    Block *sink = sinkBlock;
    bool isSecondPartition = (idx == 1);
    partition->walk([&, sink, isSecondPartition](WarpReturnOp op) {
      TritonLLVMIRRewriter b(op.getLoc(), op);
      if (sink != nullptr) {
        // For the second partition, insert an ebarrier with id
        // kAbarrierInvBarrierIdx before the partition end barrier.
        if (isSecondPartition)
          createBarrier(b, kAbarrierInvBarrierIdx,
                        /*numWarps=*/std::nullopt);
        //createBarrier(b, kPartitionEndBarrierIdx, /*numWarps=*/std::nullopt);
        //b.replaceOpWithNewOp<LLVM::BrOp>(op, sink);
        createBarrier(b, kReturnBarrierIdx, /*numWarps=*/std::nullopt);
        b.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange{});
      } else {
        if (isSecondPartition)
          createBarrier(b, kAbarrierInvBarrierIdx,
                        /*numWarps=*/std::nullopt);
        //createBarrier(b, kPartitionEndBarrierIdx, /*numWarps=*/std::nullopt);
        createBarrier(b, kReturnBarrierIdx, /*numWarps=*/std::nullopt);
        b.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange{});
      }
      });
    }
  }

  // LLVM's LICM will be tempted to hoist code out of the switch loop generated by
  // the `ttg.warp_specialize` lowering. However, neither NVPTX or `ptxas` will
  // rematerialize this code back in to the partition regions, resulting in long
  // liveranges for an arbitrary number of registers.
  //
  // Due to reduced warp group registers, these live values can induce spilling
  // in the partition regions. Prevent this by disabling LICM on the switch loop.
  static void disableLICM(LLVM::BrOp latchBr) {
    Builder b(latchBr.getContext());
    MLIRContext *ctx = b.getContext();
    auto licmMD = LLVM::LoopLICMAttr::get(ctx, b.getBoolAttr(true), {});
    auto loopMD =
        LLVM::LoopAnnotationAttr::get(b.getContext(), {}, {}, {}, {}, {}, licmMD,
                                      {}, {}, {}, {}, {}, {}, {}, {}, {});
    latchBr.setLoopAnnotationAttr(loopMD);
  }

static LogicalResult lowerWarpSpecialize(LLVM::LLVMFuncOp func,
                                         const HCU::TargetInfo &targetInfo,
                                         int waspNumLoadWarps,
                                         int waspNumMmaWarps,
                                         bool wdraEnabled, 
                                         int wdraNumLoadRegs, 
                                         int wdraNumMmaRegsMain,
                                         int wdraNumMmaRegsTail) {
  SmallVector<WarpSpecializeOp> wsOps;
  func.walk([&](WarpSpecializeOp op) { wsOps.push_back(op); });
  // Nothing to do. This kernel is not warp specialized.
  if (wsOps.empty())
    return success();

  // wdra requires all warpgroups to synchronize at the end of the kernel.
  func.walk([&](LLVM::ReturnOp op) {
    TritonLLVMIRRewriter b(op.getLoc(), op);
    createBarrier(b, kReturnBarrierIdx, /*numWarps=*/std::nullopt);
  });

  // Before lowering away `ttg.warp_specialize`, lower warp group barriers.
  auto module = cast<ModuleOp>(func->getParentOp());
  unsigned threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(module);
  // The default partition is executed by the first mma partition.
  // When wdra is enabled, the num of warps of all partition is 4.
  unsigned defaultNumWarps = wdraEnabled ? 4 : waspNumMmaWarps;

  // Replace the s_barrier in each partition with ebarrier.
  if (failed(rewriteWarpGroupBarriers(func, wsOps, defaultNumWarps)))
    return failure();

  // Attempt to elide captures of trivial computations by hoisting them into the
  // header or rematerializing them into each partition.
  elideTrivialCaptures(func, wsOps);

  MLIRContext *ctx = func.getContext();
  TritonLLVMIRRewriter b(func.getLoc(), ctx);
  Builder rewriter(ctx);
  LLVM::LLVMPointerType ptrTy = ptr_ty(b.getContext(), 3);

  // Generate the function header.
  Block *entry = &func.getBody().front();
  SmallVector<Location> argLocs = llvm::to_vector(llvm::map_range(
      func.getArguments(), [](BlockArgument arg) { return arg.getLoc(); }));
  Block *header = b.createBlock(entry, func.getArgumentTypes(), argLocs);

  // The default partition is executed by the first mma partition,
  // Allocate vgprs for the first mma partition.
  b.setInsertionPointToStart(entry);
  if (wdraEnabled)
    b.create<ROCDL::HCUSetVgprSizeOp>(wdraNumMmaRegsMain);

  // Forward arguments from the header into the old entry block.
  for (auto [arg, oldArg] :
       llvm::zip(header->getArguments(), entry->getArguments()))
    oldArg.replaceAllUsesWith(arg);
  entry->eraseArguments([](auto) { return true; });

  // This is the default destination for the switch op.
  // Does nothing except arriving all the barriers and then return.
  Region::BlockListType &funcBlocks = func.getBody().getBlocks();
  Block *defaultBlock = new Block;
  auto firstWsOp = wsOps.front();
  Block *firstWsOpBlock = firstWsOp.getOperation()->getBlock();
  funcBlocks.insert(firstWsOpBlock->getIterator(), defaultBlock);
  b.setInsertionPointToStart(defaultBlock);
  createBarrier(b, kSwitchLoopBarrierIdx, /*numWarps=*/std::nullopt);
  createBarrier(b, kPartitionEndBarrierIdx, /*numWarps=*/std::nullopt);
  createBarrier(b, kReturnBarrierIdx, /*numWarps=*/std::nullopt);
  b.create<LLVM::ReturnOp>(ValueRange());

  // Create the switch op in the first block.
  b.setInsertionPointToStart(header);
  Value wid = b.create<ROCDL::HCUGetWaveIdOp>();

  // The number of warps in each partition is limited to 4 if wdra is enabled.
  auto mmaNumWarps = wdraEnabled ? 4 : waspNumMmaWarps;
  SmallVector<Block *> mmaPartitionVgprSettingBlocks;
  SmallVector<Block *> loadPartitionVgprSettingBlocks;
  SmallVector<Block *> switchBlocks(mmaNumWarps, entry);

  // Cases values initialized to warp ids from 0 to the number of warps in the first mma partition.
  auto caseValues = llvm::to_vector(llvm::map_range(
      llvm::seq<int8_t>(mmaNumWarps),
      [](int8_t i) { return APInt(8, i); }));

  // Set the rest target blocks for the switch op.
  {
    TritonLLVMIRRewriter b(firstWsOp.getLoc(), firstWsOp);
    // Counter to generate unique block identifiers to prevent block merging
    unsigned blockCounter = 0;
    unsigned nextCaseValue = mmaNumWarps;
    for (auto [idx, tuple] : llvm::enumerate(llvm::zip(
      *firstWsOp.getWarpGroupStartIds(), firstWsOp.getPartitionNumWarps(), firstWsOp.getPartitionRegions()))) {
      auto [startId, numWarps, partition] = tuple;
      // idx 0: first MMA partition, already reached via entry block; skip.
      if (idx == 0)
        continue;
      Block *vgprSettingBlock = b.createBlock(entry);
      b.setInsertionPointToStart(vgprSettingBlock);
      int numRegs = (idx == 1) ? wdraNumLoadRegs : wdraNumMmaRegsTail;
      auto vgprOp = b.create<ROCDL::HCUSetVgprSizeOp>(numRegs);
      vgprOp->setAttr("block_id", b.getI32IntegerAttr(blockCounter));
      auto brOp = b.create<LLVM::BrOp>(&partition->front());
      brOp->setAttr("block_id", b.getI32IntegerAttr(blockCounter));
      blockCounter++;
      // idx 1 = load partition; idx 2+ = additional MMA partitions
      if (idx == 1)
        loadPartitionVgprSettingBlocks.push_back(vgprSettingBlock);
      else
        mmaPartitionVgprSettingBlocks.push_back(vgprSettingBlock);
      for (auto i : llvm::seq(numWarps)) {
        Block *targetBlock = wdraEnabled ? vgprSettingBlock : &partition->front();
        caseValues.push_back(APInt(8, nextCaseValue + i));
        switchBlocks.push_back(targetBlock);
      }
      nextCaseValue += numWarps;
    }
  }

  b.create<LLVM::SwitchOp>(wid, defaultBlock, ValueRange(), caseValues, switchBlocks,
                           SmallVector<ValueRange>(switchBlocks.size()));

  for (auto [idx, ws] : llvm::enumerate(wsOps)) {
    bool isFirst = (idx == 0);
    bool isLast = (idx == wsOps.size() - 1);

    Block *before = ws->getBlock();
    Block *after = b.splitBlock(before, ws->getIterator());
    TritonLLVMIRRewriter b(ws.getLoc(), OpBuilder::atBlockEnd(before));

    // Store the captures if there are any.
    if (ws.getNumOperands()) {
      auto captureType = LLVM::LLVMStructType::getLiteral(
          b.getContext(), llvm::to_vector(ws.getOperandTypes()),
          /*isPacked=*/true);
      Value capturePtr =
          LLVM::getSharedMemoryBase(b.getLoc(), b, targetInfo, ws);
      for (auto [i, arg] : llvm::zip(llvm::seq<int32_t>(ws.getNumOperands()),
                                     ws.getOperands())) {
        Value ptr =
            b.gep(ptrTy, captureType, capturePtr, ArrayRef<LLVM::GEPArg>{0, i});
        b.store(arg, ptr, /*align=*/1);
      }
    }

    b.create<LLVM::BrOp>(&ws.getPartitionRegions()[0]->front());

    ws.getDefaultRegion().walk([&, ws = ws](WarpYieldOp op) mutable {
      TritonLLVMIRRewriter b(op.getLoc(), op);
      createBarrier(b, kPartitionEndBarrierIdx, /*numWarps=*/std::nullopt);
      b.replaceOpWithNewOp<LLVM::BrOp>(op, op.getOperands(), after);
    });
    after->getParent()->getBlocks().splice(after->getIterator(),
                                           ws.getDefaultRegion().getBlocks());

    // Replace the results.
    auto outputs = after->addArguments(
        ws.getResultTypes(),
        SmallVector<Location>(ws.getNumResults(), ws.getLoc()));
    ws.replaceAllUsesWith(outputs);

    // Rewrite the partition regions
    SmallVector<Block *> sinkBlocks = {nullptr, nullptr, nullptr};
    rewritePartitionRegions(ws, sinkBlocks, targetInfo);
  }

  // Reorder basic blocks so that blocks belonging to different branches
  // (partitions) stay grouped together without interleaving, and tag MMA
  // partitions with distinct attributes on their epilogue blocks so that later
  // optimization passes do not merge them.
  //
  // Layout:
  //   - first MMA partition (idx 0) blocks
  //   - additional MMA partitions (idx >= 2): for each, its vgpr-setting blocks
  //     followed immediately by its region blocks
  //   - load partition (idx 1): vgpr-setting blocks followed by region blocks
  //
  // This ensures that for each branch, all of its basic blocks are laid out
  // contiguously in the function.
  auto partitionNumWarps = firstWsOp.getPartitionNumWarps();
  unsigned numPartitions = partitionNumWarps.size();

  auto tagMmaPartitionBlocks = [&](Region *region, unsigned partitionId) {
    // Insert a tiny partition-specific inline asm marker in the epilogue blocks
    // (right before returns). This makes the epilogues of different MMA
    // partitions structurally different so that the backend won't merge them,
    // while having no semantic effect.
    for (Block &b : region->getBlocks()) {
      for (Operation &op : b) {
        if (auto ret = dyn_cast<LLVM::ReturnOp>(&op)) {
          OpBuilder markerBuilder(ret);
          auto *ctx = markerBuilder.getContext();
          std::string asmStr = "; mma_epilogue_partition_" + std::to_string(partitionId);
          auto asmDialect = LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT);
          markerBuilder.create<LLVM::InlineAsmOp>(
              ret.getLoc(),
              TypeRange(),          // no results
              ValueRange(),         // no operands
              asmStr,               // asm string
              "",                   // constraints
              /*hasSideEffects=*/true,
              /*isAlignStack=*/false,
              LLVM::TailCallKind::None,
              asmDialect,
              ArrayAttr::get(ctx, {}));
          // Only need one marker per epilogue block.
          break;
        }
      }
    }
  };

  // 1) First MMA partition (idx 0) – no dedicated VGPR-setting blocks.
  for (auto ws : wsOps) {
    auto mmaPartition = ws.getPartitionRegions()[0];
    tagMmaPartitionBlocks(mmaPartition, /*partitionId=*/0);
    funcBlocks.splice(funcBlocks.end(), mmaPartition->getBlocks());
  }

  // 2) Additional MMA partitions (idx >= 2).
  for (unsigned p = 2; p < numPartitions; ++p) {
    unsigned numWarps = partitionNumWarps[p];
    Block *block = mmaPartitionVgprSettingBlocks[0];
    block->getParent()->getBlocks().remove(block);
    funcBlocks.insert(funcBlocks.end(), block);

    // Region blocks for this MMA partition from all warp-specialize ops.
    for (auto ws : wsOps) {
      if (p >= ws.getPartitionRegions().size())
        continue;
      auto region = ws.getPartitionRegions()[p];
      tagMmaPartitionBlocks(region, /*partitionId=*/p);
      funcBlocks.splice(funcBlocks.end(), region->getBlocks());
    }
  }

  // 3) Load partition (idx 1).
  if (numPartitions > 1) {
    unsigned numWarps = partitionNumWarps[1];
    Block *block = loadPartitionVgprSettingBlocks[0];
    block->getParent()->getBlocks().remove(block);
    funcBlocks.insert(funcBlocks.end(), block);

    // Region blocks for the load partition from all warp-specialize ops.
    for (auto ws : wsOps) {
      if (ws.getPartitionRegions().size() <= 1)
        continue;
      auto loadPartition = ws.getPartitionRegions()[1];
      funcBlocks.splice(funcBlocks.end(), loadPartition->getBlocks());
    }
  }

  for (auto ws : wsOps) {
    ws.erase();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct HCUGPUConvertWarpSpecializeToLLVM
    : public mlir::triton::impl::HCUGPUConvertWarpSpecializeToLLVMBase<
          HCUGPUConvertWarpSpecializeToLLVM> {
  explicit HCUGPUConvertWarpSpecializeToLLVM(StringRef targetArch, int waspNumLoadWarps, 
      int waspNumMmaWarps, bool wdraEnabled, int wdraNumLoadRegs, 
      int wdraNumMmaRegsMain, int wdraNumMmaRegsTail) {

    this->arch = targetArch.str();
    this->waspNumLoadWarps = waspNumLoadWarps;
    this->waspNumMmaWarps = waspNumMmaWarps;
    this->wdraEnabled = wdraEnabled;
    this->wdraNumLoadRegs = wdraNumLoadRegs;
    this->wdraNumMmaRegsMain = wdraNumMmaRegsMain;
    this->wdraNumMmaRegsTail = wdraNumMmaRegsTail;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    HCU::TargetInfo targetInfo(this->arch.getValue());
    if (targetInfo.getGPUKind() != llvm::AMDGPU::GK_GFX946) {
      mod.emitError("wasp unsupported target: '") << this->arch.getValue() << "'";
      return signalPassFailure();
    }

    // Convert types and cleanup unrealized conversions.
    mlir::LowerToLLVMOptions option(&getContext());
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(&getContext(), option, targetInfo);
    mod.walk([&](Operation *op) {
      if (isa<WarpSpecializeOp, WarpSpecializePartitionsOp, WarpYieldOp>(op))
        convertOpTypes(op, typeConverter);
    });
    RewritePatternSet patterns(&getContext());
    UnrealizedConversionCastOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(applyPatternsGreedily(mod, std::move(patterns))))
      return signalPassFailure();

    SmallVector<LLVM::LLVMFuncOp> kernels;
    for (auto func : mod.getOps<LLVM::LLVMFuncOp>()) {
      if (func.isPublic())
        kernels.push_back(func);
    }

    for (LLVM::LLVMFuncOp kernel : kernels)
      if (failed(lowerWarpSpecialize(kernel, targetInfo, 
        this->waspNumLoadWarps, this->waspNumMmaWarps, this->wdraEnabled,
        this->wdraNumLoadRegs, this->wdraNumMmaRegsMain, this->wdraNumMmaRegsTail)))
        return signalPassFailure();

    // Convert GPU dialect to ROCDL dialect
    FailureOr<mlir::amdgpu::Chipset> maybeChipset = mlir::amdgpu::Chipset::parse(this->arch.getValue());
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(&getContext()),
                "Invalid HCUGPU chipset name: " + this->arch.getValue());
      return signalPassFailure();
    }

    // Create conversion target for GPU to ROCDL conversion
    ConversionTarget gpuTarget(getContext());
    gpuTarget.addLegalDialect<LLVM::LLVMDialect>();
    gpuTarget.addLegalDialect<ROCDL::ROCDLDialect>();
    gpuTarget.addIllegalDialect<mlir::gpu::GPUDialect>();

    // Create patterns for GPU to ROCDL conversion
    RewritePatternSet gpuPatterns(&getContext());
    mlir::populateGpuToROCDLConversionPatterns(
        typeConverter, gpuPatterns, mlir::gpu::amd::HIP, *maybeChipset);

    // Apply GPU to ROCDL conversion
    if (failed(applyPartialConversion(mod, gpuTarget, std::move(gpuPatterns)))) {
      return signalPassFailure();
    }

    // Elide canceling casts: unrealized_conversion_cast (A -> B) followed by
    // arith.index_cast (B -> A) can be removed.
    SmallVector<std::pair<UnrealizedConversionCastOp, arith::IndexCastOp>> cancelingPairs;
    mod.walk([&](UnrealizedConversionCastOp unrealizedCast) {
      if (unrealizedCast.getNumOperands() != 1 || unrealizedCast.getNumResults() != 1)
        return;
      Value original = unrealizedCast.getOperand(0);
      Value mid = unrealizedCast.getResult(0);
      if (!mid.hasOneUse())
        return;
      Operation *user = *mid.getUsers().begin();
      auto indexCast = dyn_cast<arith::IndexCastOp>(user);
      if (!indexCast)
        return;
      // Check types form A->B then B->A
      if (indexCast.getType() != original.getType())
        return;
      if (indexCast.getIn().getType() != mid.getType())
        return;
      cancelingPairs.emplace_back(unrealizedCast, indexCast);
    });
    for (auto [unrealizedCast, indexCast] : cancelingPairs) {
      Value original = unrealizedCast.getOperand(0);
      indexCast.getResult().replaceAllUsesWith(original);
      indexCast.erase();
      unrealizedCast.erase();
    }
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Factory Function
//===----------------------------------------------------------------------===//
namespace mlir::triton {

  std::unique_ptr<OperationPass<ModuleOp>>
  createHCUGPUConvertWarpSpecializeToLLVM(StringRef targetArch, int waspNumLoadWarps, int waspNumMmaWarps, 
     bool wdraEnabled, int wdraNumLoadRegs, int wdraNumMmaRegsMain, int wdraNumMmaRegsTail) {
    return std::make_unique<HCUGPUConvertWarpSpecializeToLLVM>(targetArch, waspNumLoadWarps, waspNumMmaWarps,
      wdraEnabled, wdraNumLoadRegs, wdraNumMmaRegsMain, wdraNumMmaRegsTail);
  }
  
  } // namespace mlir::triton