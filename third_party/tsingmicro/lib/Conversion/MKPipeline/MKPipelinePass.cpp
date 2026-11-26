//===----------------------------------------------------------------------===//
// MKPipelinePass — software-pipeline scf.for loops with mk.dot
//
//===----------------------------------------------------------------------===/

#include "magic-kernel/Conversion/MKPipeline/Passes.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <cassert>
#include <functional>
#include <optional>

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_MKPIPELINEPASS
#include "magic-kernel/Conversion/MKPipeline/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {
using namespace mlir;
using namespace mlir::scf;

// ─────────────────────────────────────────────────────────────────────────────
// 工具函数
// ─────────────────────────────────────────────────────────────────────────────

static bool sanitizeScfIfs(Operation *container) {
  if (!container)
    return true;

  bool ok = true;
  container->walk([&](scf::IfOp ifOp) {
    Location loc = ifOp.getLoc();
    for (Region &region : ifOp->getRegions()) {
      if (region.empty()) {
        OpBuilder b(ifOp.getContext());
        b.createBlock(&region);
      }

      Block &block = region.front();

      // dedup yields：一次性删除"非最后一个" scf.yield。
      SmallVector<scf::YieldOp> yields;
      for (auto y : block.getOps<scf::YieldOp>())
        yields.push_back(y);
      for (size_t i = 0; i + 1 < yields.size(); ++i)
        yields[i]->erase();

      // 补齐 terminator。
      if (block.empty() || !isa<scf::YieldOp>(block.back())) {
        OpBuilder b(&block, block.end());
        if (ifOp.getNumResults() == 0) {
          scf::YieldOp::create(b, loc);
        } else {
          // 带结果的 ifOp 缺 yield 是上游构造错误。无法凭空补齐合法
          // 默认值；标记失败，让 caller 触发 signalPassFailure。
          ifOp.emitError(
              "scf.if with results is missing a terminating scf.yield; "
              "MKPipelinePass produced malformed IR");
          ok = false;
        }
      }
    }
  });
  return ok;
}

// Returns the effective pipeline depth for `forOp`.
//   n <= 1  → caller should skip pipelining (return 1).
//   n >= 2  → clamp to `max-stages` (default 4).
//
// Any depth in [2, maxStages] is supported by the rewrite itself: prologue,
// kernel loop and epilogue are all written in terms of (numStages - 1). The
// number of in-flight DDR->SPM buffers is numStages - 1, so a deeper pipeline
// costs (numStages - 1) tile buffers of SPM per pipelined load. Triton's
// OutOfResources gate rejects a kernel whose total SPM use exceeds the per-core
// capacity, so an over-deep request fails at compile time instead of on the
// device.
static int getEffectiveNumStages(scf::ForOp forOp, int globalDefault,
                                 int maxStages) {
  int n = globalDefault; // default 2
  if (forOp->hasAttr("tt.num_stages"))
    n = mlir::cast<IntegerAttr>(forOp->getAttr("tt.num_stages")).getInt();

  if (n > maxStages) {
    llvm::errs() << "warning: tt.num_stages (" << n
                 << ") is greater than max-stages, clamping to " << maxStages
                 << "\n";
    n = maxStages;
  } else if (n <= 1) {
    n = 1;
  }
  return n;
}

static bool isAllocBacked(Value v);
static void remapMixedFoldResults(ArrayRef<OpFoldResult> in,
                                  SmallVectorImpl<OpFoldResult> &out,
                                  const IRMapping &rm);
static MemRefType inferSubviewResultType(memref::SubViewOp sv,
                                         MemRefType srcType,
                                         ArrayRef<OpFoldResult> offsets,
                                         ArrayRef<OpFoldResult> sizes,
                                         ArrayRef<OpFoldResult> strides);

// A loop is treated as "pipelineable" (effectively innermost) even if it has
// nested scf.for loops, provided those nested loops are "trivial" — they
// contain no mk::DotOp and no alloc-backed memref::CopyOp. Trivial inner
// loops (e.g., the scalar row-fill loops that broadcast softmax row-max/sum
// into a 2D buffer) are cloned intact into the compute phase by
// cloneComputeOps and do not interfere with multi-buffering.
static bool isInnermostForOp(scf::ForOp forOp) {
  bool hasSignificantNestedFor = false;
  forOp.walk([&](scf::ForOp inner) -> WalkResult {
    if (inner == forOp)
      return WalkResult::advance();
    bool significant = false;
    inner.walk([&](Operation *op) -> WalkResult {
      if (isa<mlir::mk::DotOp>(op)) {
        significant = true;
        return WalkResult::interrupt();
      }
      if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
        if (isAllocBacked(copyOp.getTarget())) {
          significant = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (significant) {
      hasSignificantNestedFor = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !hasSignificantNestedFor;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fill-once-per-bank：把 OOB-zero 的 linalg.fill 提到 K-loop 之外
//
// 动机
// ----
// 对未对齐的 GEMM，stage1 IR 形如：
//   scf.for %k = ... {
//     %a = memref.alloc() : memref<1024x64xf16>            // base alloc
//     %sv_in = subview %a[0,0][%M_tail, 64]                // in-bound view
//     scf.if %needFillA { linalg.fill ins(0.0) outs(%a) }  // zero整 base
//     memref.copy %ddr_sv, %sv_in                           // 仅写 in-bound
//     mk.dot ... reads %a ...                              // 读整 slot
//   }
// 多 buffer 化之后，base alloc 被替换成 `memref<2x1024x64xf16>` 的两个 slot；
// 原 IR 的 fill 被搬进 (a) prologue scf.if、(b) kernel-load scf.for 的每轮
// iter，导致 K-loop 每轮都对整 slot 写 0（mm_kernel 1024×64 单 fill 就是
// 1024×64×2B = 128KB SPM 写）。
//
// 关键观察：%needFillA 是 per-PID 常量（M_tail 由 PID 决定，K-loop 内不
// 变），fill 的写区域 = 整 base alloc，copy 的写区域 ⊆ in-bound 视图，
// **K-loop 期间无任何其它 op 写 OOB 区域**。所以"每个 bank 的 OOB 只需
// 在循环外被零化一次，之后所有 K-iter 的 copy 都不会污染它"。
//
// 实现
// ----
// 1) collectPipelinedLoads 给 LoadGroup 增加 3 条字段：
//      matchedZeroFillIf  —— 直接持有匹配到的 scf.if；
//      fillCoversBase     —— fill outs 直接是 baseAlloc 时为 true（结构
//                            合格性，是把 fill 提到外面的【必要】前提）；
//      preInitEmitted     —— emitPerBankPreInits 真正成功发射 pre-init
//                            后才置 true。clonePreludeIntoLoad 据此决定
//                            是否跳过原 fill-if（不能用 fillCoversBase，
//                            否则 fallback 路径会漏发 fill —— 见 addmm
//                            K-tail 案例的修复说明）。
// 2) PipelineRewriter::run 在 prologue 之前调用 emitPerBankPreInits：
//      对每个合格 multi-buffer 发射一条 scf.if (cond) { fill 全部 N 个 slot }，
//      把"每轮 K-iter 一次 fill"压缩为"K-loop 之外仅 N 次 fill"。
// 3) clonePreludeIntoLoad 在 preInitEmitted 时跳过 matchedZeroFillIf，
//      使 prologue / kernel-load 路径都不再发射 fill；compute 路径仍由
//      skipInComputeSet 跳过同一条 if。
//
// 安全性
// ------
//   * fillCoversBase 只在 fill 写满整 base alloc 时为 true：保证"slot \ copy
//     的 OOB 区域 ⊆ fill 区域"，预填充能完全覆盖。fill 写 partial subview
//     的 group 一律 fallback 到原 per-iter 行为。
//   * 条件值必须由 forOp 之外定义（改造点 A 的 LICM 应已实现）。若个别条件
//     未被提走（典型：addmm 的 ori(M_tail, K_tail) 因 K_tail 依赖 K-IV
//     而仍在 forOp 内），emitPerBankPreInits 跳过该 group 不发射 pre-init，
//     preInitEmitted 保持 false，clonePreludeIntoLoad 自然 fallback 仍发
//     原 per-iter fill。这一双闸结构是改造点 E 的关键安全保证：
//       - 第一闸（fillCoversBase）：fill 写区域是否能整 slot 覆盖 OOB；
//       - 第二闸（cond 是否循环不变 → preInitEmitted）：fill 触发频度
//         是否能被"一次性"等价替代。两闸都通过才允许 hoist。
//   * 多个 LoadGroup 共享同一 base alloc 时，按 multi-buffer 聚合所有
//     conditions 取 OR：因为每个原 fill 都写满整 slot，OR 等价于"任一
//     condition 真则该 slot 整个置零"。
//
// 收益（mm_kernel 1024×64×64, K=8, numStages=2 为例）：
//   K-loop 内 fill 数：8 → 0
//   loop 外一次性 fill 数：0 → 2  (slot 0 + slot 1，guarded by %needFill)
//   总 fill 写量：8×128KB = 1MB → 2×128KB = 256KB（4× 减少）
//   且 OOB-fill 与所有 K-iter 的 RDMA/dot 解耦，可被流水掩盖。
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Pre-LICM：把 `forOp` body 中循环不变的纯 op 提到 forOp 之前
//
// 动机
// ----
// MKPipeline 后端会把"边界 mask 的 condition 链"（典型形态：
//   %a = arith.addi  %M_base, %c1024 : index
//   %b = arith.index_cast %arg_M  : i32 to index
//   %c = arith.minsi %a, %b       : index
//   %d = arith.maxsi %c, %M_base  : index
//   %e = arith.subi  %d, %M_base  : index
//   %f = arith.cmpi  slt, %e, %c1024 : index
//   scf.if %f { linalg.fill ... }
// ）连同 zero-fill scf.if 一起作为 prelude，独立 clone 进
//   (a) prologue scf.if
//   (b) kernel-load 路径（clonePreludeIntoLoad）
// 而 cloneComputeOps 走原 forOp body 时，由于 subview 的动态尺寸操作数
// （上链中的 sub 结果 %e）并非 load-only（它同时被 copy 的 subview 和
// fill 的 scf.if 使用），整条链在 compute 路径上又会再被 clone 一次。
// 三次发射的同一组 6 个 arith op 在 LL IR 阶段会真实变成多份
// `add/index_cast/minsi/maxsi/sub/cmpi`，并各自驱动一份 alloca/store
// 描述符链（详见 stage2_ok/ll_0.mlir 主循环 bb8/bb10/bb12 重复段）。
//
// 但这条链的全部输入（M_base/N_base 来自 PID、arg_M/arg_N 来自 func
// 形参、常数）在整个 K 维 scf.for 里都是循环不变量。把它们提到 forOp
// 之前一次性算完，prologue / kernel-load / kernel-compute 三处 clone
// 时直接引用同一个循环外 SSA value，副本就只剩一份。
//
// 实现
// ----
// 迭代式 LICM：每轮挑出 body 内"纯 + 无 region + 所有 operand 均定义
// 在 body 之外（含来自 forOp 之外的 BlockArgument / 函数 arg / 常量
// op）"的 op，整体 moveBefore(forOp)；下一轮再扫描，处理刚被解锁的
// 上层依赖（比如 cmpi 在 minsi/maxsi 提走之后才被识别为不变）。
//
// 与 PipelineRewriter 的协作
// --------------------------
// 提之后：
//   * collectIfConditionDefChain：def->getBlock() != forOp.getBody()
//     立即 return，preludeOps 仅含 zero-fill scf.if 自身；
//   * cloneDefChainInLoopBody：同样早返，subview 动态 size operand
//     直接复用循环外 SSA；
//   * clonePreludeIntoLoad / cloneComputeOps 不需要任何改动。
//
// 安全性
// ------
// 只搬 isMemoryEffectFree 且无 region 的 op；这覆盖了边界 mask 链涉及
// 的 arith.addi/subi/minsi/maxsi/cmpi/index_cast，并自动排除：
//   - memref.alloc / memref.copy / linalg.fill（有内存副作用）
//   - scf.if / scf.while（有 region）
//   - mk.dot（写 acc，有副作用）
// 这些 op 必须留在 body 内才能被 PipelineRewriter 正确流水化。
//
// 提升不会越过 forOp 的支配关系（forOp 之前的位置严格支配 forOp body
// 内任何 use），所以 SSA 合法性自然保留。
// ─────────────────────────────────────────────────────────────────────────────
static size_t hoistLoopInvariantPureOps(scf::ForOp forOp) {
  Block *body = forOp.getBody();
  size_t moved = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> hoistList;
    for (Operation &op : body->without_terminator()) {
      // 仅搬纯 op（无内存副作用 + 不会 trap 的算术）。memref.alloc /
      // memref.copy / linalg.fill / mk.dot 等都不满足。
      if (!isMemoryEffectFree(&op))
        continue;
      // 含 region 的 op（scf.if / scf.while / scf.for / linalg.generic
      // 等）不在这里搬：它们的 body 内可能引用循环 IV / iter_args，
      // 整体上提会破坏 SSA。需要搬只能逐个分析其 region。
      if (op.getNumRegions() != 0)
        continue;
      // 没有 result 的纯 op 在 LICM 视角下没有"被使用 = 必须重算"的
      // 价值，跳过避免无谓搬移。
      if (op.getNumResults() == 0)
        continue;

      bool invariant = true;
      for (Value v : op.getOperands()) {
        if (Operation *def = v.getDefiningOp()) {
          if (def->getBlock() == body) {
            invariant = false;
            break;
          }
        } else if (auto ba = dyn_cast<BlockArgument>(v)) {
          // forOp 自身的 BlockArgument（IV / iter_args）一定挂在 body
          // 上，命中此分支即视为"循环依赖"。其它来源的 BlockArgument
          // （函数参数、外层 region 的入参）owner != body，落到 else
          // 之后被认为是不变的输入，符合预期。
          if (ba.getOwner() == body) {
            invariant = false;
            break;
          }
        }
      }
      if (invariant)
        hoistList.push_back(&op);
    }
    for (Operation *op : hoistList) {
      op->moveBefore(forOp);
      ++moved;
      changed = true;
    }
  }
  return moved;
}

// ─────────────────────────────────────────────────────────────────────────────
// Bank index 推进：把 `(x + 1) % numStages` 这条
//   index → i64 → addi → remsi → index
// round-trip 折成位运算 / 单条算子，且全程留在 index 域。
//
// 现状（pre-fix）：每个流水化 for 都会发射两组（写槽 + 读槽）形如
//   %a = arith.index_cast %x : index to i64
//   %b = arith.addi %a, %c1_i64
//   %c = arith.remsi %b, %c2_i64
//   %d = arith.index_cast %c : i64 to index
// 主循环每轮 8 个 op、epilogue 每个 stage 4 个 op，连同 epilogue 的
// `%c1_i64 / %c2_i64 / nVal_i64` 函数级常量与 `arith.index_cast`
// 一起把 LL IR 描述符链拉长 + 把 srem 这种除法器单元留在主循环里。
//
// 观察：
//   1) numStages 可以是 [2, max-stages] 的任意值；`n <= 1` 在
//      runOnOperation 里被早 return。
//   2) `arith.xori` / `arith.andi` 直接接受 index 操作数，不必绕 i64。
//   3) 非 2 的幂（如 3）不要用 remsi：整数除法会留在主循环里。
//
// 按 numStages 取最便宜的实现：
//   * numStages == 2          →  arith.xori   %x, %c1_idx
//   * numStages 为 2 的幂 (>2) →  arith.andi (addi %x %c1) (N-1)
//   * 其它任意 N              →  cmp + select 回绕：
//                                 %n = addi %x, %c1
//                                 %w = cmpi sge %n, %N
//                                 %r = select %w, %c0, %n
//
// `oneIdx` 由 caller 传入（复用 PipelineRewriter::run 里已经构造好
// 的 `c1 : index` 常量），避免重复发射常量 op。
// ─────────────────────────────────────────────────────────────────────────────
static Value makeBankAdvance(IRRewriter &rewriter, Location loc, Value cur,
                             int numStages, Value oneIdx) {
  assert(numStages >= 2 && "bank advance only meaningful for >=2 stages");
  if (numStages == 2) {
    // 0 ↔ 1 toggle：单条 xori，无加法、无除法、无 cast。
    return arith::XOrIOp::create(rewriter, loc, cur, oneIdx);
  }
  Value next = arith::AddIOp::create(rewriter, loc, cur, oneIdx);
  if (llvm::isPowerOf2_64(static_cast<uint64_t>(numStages))) {
    Value mask = arith::ConstantIndexOp::create(rewriter, loc, numStages - 1);
    return arith::AndIOp::create(rewriter, loc, next, mask);
  }
  // 非 2 的幂：next ∈ [1, N]，只有 next == N 需要回绕到 0。
  Value limit = arith::ConstantIndexOp::create(rewriter, loc, numStages);
  Value wrapped = arith::CmpIOp::create(rewriter, loc,
                                        arith::CmpIPredicate::sge, next, limit);
  Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  return arith::SelectOp::create(rewriter, loc, wrapped, zeroIdx, next);
}

// 判断一个 memref 值是否最终来自 memref::AllocOp（SPM 缓冲）。
// 通过递归穿透所有 view op（不改数据、只改类型/形状/布局的操作）。
// ──────────────────────────────────────────────────────────────────────────
// ⚠️ 风险：此函数必须覆盖 ALL view op 类型。如果遗漏某个 view op：
//   - SPM→SPM 的 copy 会被误判为 DDR→SPM → 错误多缓冲化 → 精度错误
//   - DDR→SPM 的 copy 可能找不到 base alloc → pipeline 跳过 → 依赖顺序错误
// 所有 memref 的 ViewLikeOpInterface 实现均在此覆盖。新增 view op
// 时务必同步更新。
// ──────────────────────────────────────────────────────────────────────────
static bool isAllocBacked(Value v) {
  if (!v)
    return false;
  if (v.getDefiningOp<memref::AllocOp>())
    return true;
  if (auto op = v.getDefiningOp<memref::SubViewOp>())
    return isAllocBacked(op.getSource());
  if (auto op = v.getDefiningOp<memref::ReinterpretCastOp>())
    return isAllocBacked(op.getSource());
  if (auto op = v.getDefiningOp<memref::CollapseShapeOp>())
    return isAllocBacked(op.getSrc());
  if (auto op = v.getDefiningOp<memref::ExpandShapeOp>())
    return isAllocBacked(op.getSrc());
  if (auto op = v.getDefiningOp<memref::CastOp>())
    return isAllocBacked(op.getSource());
  if (auto op = v.getDefiningOp<memref::ReshapeOp>())
    return isAllocBacked(op.getSource());
  if (auto op = v.getDefiningOp<memref::MemorySpaceCastOp>())
    return isAllocBacked(op.getSource());
  if (auto op = v.getDefiningOp<memref::ViewOp>())
    return isAllocBacked(op.getSource());
  if (auto op = v.getDefiningOp<memref::TransposeOp>())
    return isAllocBacked(op.getIn());
  return false;
}

// 与 isAllocBacked 配套：沿 view 链找到最终的 memref::AllocOp。
// 覆盖范围必须与 isAllocBacked 保持同步。
static memref::AllocOp findBaseAlloc(Value v) {
  if (!v)
    return {};
  if (auto a = v.getDefiningOp<memref::AllocOp>())
    return a;
  if (auto op = v.getDefiningOp<memref::SubViewOp>())
    return findBaseAlloc(op.getSource());
  if (auto op = v.getDefiningOp<memref::ReinterpretCastOp>())
    return findBaseAlloc(op.getSource());
  if (auto op = v.getDefiningOp<memref::CollapseShapeOp>())
    return findBaseAlloc(op.getSrc());
  if (auto op = v.getDefiningOp<memref::ExpandShapeOp>())
    return findBaseAlloc(op.getSrc());
  if (auto op = v.getDefiningOp<memref::CastOp>())
    return findBaseAlloc(op.getSource());
  if (auto op = v.getDefiningOp<memref::ReshapeOp>())
    return findBaseAlloc(op.getSource());
  if (auto op = v.getDefiningOp<memref::MemorySpaceCastOp>())
    return findBaseAlloc(op.getSource());
  if (auto op = v.getDefiningOp<memref::ViewOp>())
    return findBaseAlloc(op.getSource());
  if (auto op = v.getDefiningOp<memref::TransposeOp>())
    return findBaseAlloc(op.getIn());
  return {};
}

// 一组与流水线化 copy 配对的"前置准备 op"：
//   - copy 本身
//   - 紧邻 copy 之前、对同一 base alloc 做"清零"的 scf.if (含其 fill)，
//     以及该 if 的 condition 仅由该 copy 之前的 cmp/ori 链构成
// 这些 op 在生成 prologue/kernel-load 时必须与 copy 一起迁移到 load 分支，
// 并把 fill 的 outs 重映射到 multi-buffer slot；compute 阶段则跳过它们，
// 否则会在 load 之后把已经载入的数据清零，破坏 OOB 区域的 0 语义。
struct LoadGroup {
  Operation *copy; // 可以是 memref::CopyOp 或 linalg::CopyOp
  // 标记该copy的目标alloc是否在循环外分配（如cu_seqlens偏移缓冲）。
  // 循环外分配的缓冲不需要多缓冲，但copy本身仍需从compute移到load阶段。
  bool isOuterAlloc = false;
  // prelude 中的所有 op（含 scf.if、其内部 fill、condition 计算的 cmp/ori 等）
  // 全部位于 forOp body 顶层、copy 之前。collectPipelinedLoads 一并填充。
  // 在 load 阶段需要按顺序 clone 全部 preludeOps 到 multi-buffer slot
  // 上下文中。
  SmallVector<Operation *> preludeOps;
  // preludeOps 中"必须在 compute 阶段被跳过"的 op 子集。
  // 当前仅含 zero-fill scf.if 自身：它的副作用 (linalg.fill) 在 load 阶段已
  // 写入到 slot 的 OOB 区域，若 compute 阶段重新执行同一 if（指向原始
  // baseAlloc 的 view 在 compute 时已被 mapping 指向 extract slot），将
  // 把刚 load 的有效数据再次清零。
  //
  // condition 计算链 (cmp/min/max/ori 等) 是纯 op，且经常在 compute 体内
  // 仍有非 prelude 的 use（例如作为 subview 的动态尺寸/偏移）。它们必须
  // 保留在 compute 阶段，否则原 forOp 被 erase 后那些 use 会指向已销毁的
  // SSA value，触发 "operation destroyed but still has uses" crash。
  llvm::SmallPtrSet<Operation *, 4> skipInCompute;

  // 直接持有匹配到的 zero-fill scf.if（preludeOps 末尾那条），
  // 用于在 PipelineRewriter::run 中识别"可一次性预填充"的 group：
  //   - matchedZeroFillIf 非空 = 找到了一组 fill+copy；
  //   - fillCoversBase = true  = fill 的 outs 直接是 baseAlloc 本身（无
  //     subview 链），意味着 fill 写满整个 base alloc。这是"一次性预填充"
  //     的【结构性】前提（详见文件级注释 [改造点 E]），但仅此还不够。
  //   - preInitEmitted = true  = emitPerBankPreInits 真的为该 group 在
  //     K-loop 之外发射了 pre-init scf.if。只有这一标志为 true，
  //     clonePreludeIntoLoad 才允许跳过原 fill-if；否则必须 fallback 回
  //     per-iter fill 行为。
  //
  // 解耦动机（addmm-495-5333-71 精度回归）：fillCoversBase 只看 fill 写
  // 区域，不看 fill condition 是否循环不变。当 condition 依赖 K-tail
  // （如 addmm 中的 ori(M_tail, K_tail)，K_tail = arg8 - arg19*32 不变量
  // 不成立），emitPerBankPreInits 会安全跳过；但若 clonePreludeIntoLoad
  // 仍按 fillCoversBase 跳过原 fill，则 prologue / kernel-load 两条
  // load 路径都不再发 fill，OOB 区永久未清零，mk.dot 读到 SPM 残留
  // 数据 → 精度错误。
  scf::IfOp matchedZeroFillIf;
  bool fillCoversBase = false;
  bool preInitEmitted = false;
};

// 判定 ifOp 是否仅做 "fill base 0"，且填充的 base 与 copy 的 target 同源。
// scf.if %58 {
//    linalg.fill ins(%cst : f16) outs(%alloc_9 : memref<128x1024xf16>)
//  }
//  fill 的 input 是 constant，且值为 0。
//  fill 的 output 是 memref::AllocOp。
//  fill 的 output 与 copy 的 target 同源。
static bool isZeroFillIfFor(scf::IfOp ifOp, memref::AllocOp baseAlloc) {
  if (!ifOp || !baseAlloc)
    return false;
  if (ifOp.getNumResults() != 0)
    return false;
  // 只接受 else 区域为空，或 else 区域仅含 scf.yield 的 if。
  // 带副作用的 else 被迁移到 load 阶段会改变 OOB 之外的 slot 内容，
  // 故直接放弃匹配此 if，保留原 IR 行为。
  if (!ifOp.getElseRegion().empty()) {
    Block &elseBlock = ifOp.getElseRegion().front();
    for (Operation &op : elseBlock) {
      if (!isa<scf::YieldOp>(op))
        return false;
    }
  }
  // 仅允许 then 区域非空、else 区域为空或仅 yield。
  Region &thenRegion = ifOp.getThenRegion();
  if (thenRegion.empty())
    return false;
  Block &thenBlock = thenRegion.front();
  // 找到唯一一个 linalg.fill；其它非 yield op 视为有副作用，拒绝。
  linalg::FillOp fillOp;
  for (Operation &op : thenBlock) {
    if (isa<scf::YieldOp>(op))
      continue;
    auto f = dyn_cast<linalg::FillOp>(&op);
    if (!f)
      return false;
    if (fillOp)
      return false;
    fillOp = f;
  }
  if (!fillOp)
    return false;
  if (fillOp.getOutputs().size() != 1)
    return false;

  Value fillVal = fillOp.getInputs()[0];
  auto cstOp = fillVal.getDefiningOp<arith::ConstantOp>();
  if (!cstOp)
    return false;

  Attribute attr = cstOp.getValue();
  bool isZero = false;
  if (auto fa = dyn_cast<FloatAttr>(attr))
    isZero = fa.getValue().isZero();
  else if (auto ia = dyn_cast<IntegerAttr>(attr))
    isZero = ia.getValue().isZero();
  if (!isZero)
    return false;

  Value out = fillOp.getOutputs()[0];
  // 允许 out 直接是 base，或经过 subview/reinterpret_cast 链回到 base。
  return findBaseAlloc(out) == baseAlloc;
}

// 沿 condition 反向收集仅由 cmp/ori/and/xor/constant 等纯计算构成的 def-chain，
// 且所有 op 都位于 forOp body 顶层。失败时返回 false（不当作可迁移 prelude）。
static bool
collectIfConditionDefChain(Value cond, scf::ForOp forOp, Operation *boundary,
                           SmallVectorImpl<Operation *> &out,
                           llvm::SmallPtrSetImpl<Operation *> &seen) {
  Operation *def = cond.getDefiningOp();
  if (!def)
    return true; // BlockArg / 常量外引用，认为无须迁移
  if (def->getBlock() != forOp.getBody())
    return true; // 来自循环外，无需克隆
  if (def->isBeforeInBlock(boundary) == false)
    return false;
  if (!seen.insert(def).second)
    return true;
  // 只允许这些"纯计算" op 作为 if condition 的来源。
  if (!isa<arith::CmpIOp, arith::OrIOp, arith::AndIOp, arith::XOrIOp,
           arith::ConstantOp, arith::ConstantIntOp, arith::ConstantIndexOp,
           arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::MinSIOp,
           arith::MaxSIOp, arith::IndexCastOp>(def))
    return false;
  for (Value opnd : def->getOperands())
    if (!collectIfConditionDefChain(opnd, forOp, boundary, out, seen))
      return false;
  out.push_back(def);
  return true;
}

static Value getCopyTarget(Operation *copyOp) {
  if (auto mcpy = dyn_cast<memref::CopyOp>(copyOp))
    return mcpy.getTarget();
  if (auto lcpy = dyn_cast<linalg::CopyOp>(copyOp))
    return lcpy.getOutputs()[0];
  return Value();
}
static Value getCopySource(Operation *copyOp) {
  if (auto mcpy = dyn_cast<memref::CopyOp>(copyOp))
    return mcpy.getSource();
  if (auto lcpy = dyn_cast<linalg::CopyOp>(copyOp))
    return lcpy.getInputs()[0];
  return Value();
}

static SmallVector<LoadGroup> collectPipelinedLoads(scf::ForOp forOp) {
  SmallVector<LoadGroup> groups;
  // 标记被收集为 prelude 的 op，避免不同 copy 之间错误共享。
  llvm::SmallPtrSet<Operation *, 16> claimed;

  for (Operation &op : forOp.getBody()->without_terminator()) {
    // 识别 memref.copy 或 linalg.copy 作为 DDR→SPM 加载
    Operation *copyOp = nullptr;
    Value target, source;
    if (auto mcpy = dyn_cast<memref::CopyOp>(&op)) {
      copyOp = mcpy.getOperation();
      target = mcpy.getTarget();
      source = mcpy.getSource();
    } else if (auto lcpy = dyn_cast<linalg::CopyOp>(&op)) {
      copyOp = lcpy.getOperation();
      target = lcpy.getOutputs()[0];
      source = lcpy.getInputs()[0];
    }
    if (!copyOp)
      continue;
    if (!isAllocBacked(target))
      continue;
    if (isAllocBacked(source))
      continue; // skip SPM→SPM copies; only DDR→SPM loads are pipelined
    memref::AllocOp baseAlloc = findBaseAlloc(target);
    if (!baseAlloc)
      continue;

    LoadGroup grp;
    grp.copy = copyOp;
    grp.isOuterAlloc = (baseAlloc->getBlock() != forOp.getBody());

    // 在 copy 之前向上扫描，找到第一个对 baseAlloc 做清零的 scf.if。
    // 一旦遇到任何对 baseAlloc 有写副作用的其他 op，立刻停止匹配（保守）。
    scf::IfOp matchedIf;
    for (Operation *cur = copyOp->getPrevNode(); cur != nullptr;
         cur = cur->getPrevNode()) {
      if (claimed.contains(cur))
        continue;
      if (auto ifOp = dyn_cast<scf::IfOp>(cur)) {
        if (isZeroFillIfFor(ifOp, baseAlloc)) {
          matchedIf = ifOp;
          break;
        }
      }
      // 任何对 baseAlloc 的潜在写入（除了我们正在找的 fill-if）都断开匹配。
      bool writesBase = false;
      cur->walk([&](Operation *inner) {
        if (writesBase)
          return;

        if (auto store = dyn_cast<memref::StoreOp>(inner))
          if (findBaseAlloc(store.getMemRef()) == baseAlloc)
            writesBase = true;
        if (auto cpy = dyn_cast<memref::CopyOp>(inner))
          if (cpy.getOperation() != copyOp &&
              findBaseAlloc(cpy.getTarget()) == baseAlloc)
            writesBase = true;
        if (auto lcpy = dyn_cast<linalg::CopyOp>(inner))
          if (lcpy.getOperation() != copyOp &&
              findBaseAlloc(lcpy.getOutputs()[0]) == baseAlloc)
            writesBase = true;
        if (auto fill = dyn_cast<linalg::FillOp>(inner)) {
          for (Value o : fill.getOutputs())
            if (findBaseAlloc(o) == baseAlloc)
              writesBase = true;
        }
      });
      if (writesBase)
        break;
    }

    if (matchedIf) {
      // 收集 if condition 的 def-chain（必须在 matchedIf 之前的同一 block
      // 内）。
      SmallVector<Operation *> condChain;
      llvm::SmallPtrSet<Operation *, 8> seen;
      if (collectIfConditionDefChain(matchedIf.getCondition(), forOp, matchedIf,
                                     condChain, seen)) {
        for (Operation *c : condChain) {
          grp.preludeOps.push_back(c);
          // 注意：condition 链是纯计算 op，不放入 claimed —— 同一组 cmp/
          // min/max 链可能被多个 copy 共用（例如 alloc_3 与 alloc_7 共用
          // 同一个 OOB mask 的 cmpi/maxsi/minsi）。允许重复 clone 到各
          // load 分支（pure op，无副作用）。同样不放入 skipInCompute：
          // 它们必须在 compute 阶段保留，因为 compute 体内其它 op 可能
          // 仍引用它们的结果（典型：作为 dynamic subview size）。
        }
        grp.preludeOps.push_back(matchedIf);
        // 仅 scf.if (含 fill 的副作用) 才需要在 compute 阶段被跳过 +
        // 在 copy 之间唯一归属。
        claimed.insert(matchedIf);
        grp.skipInCompute.insert(matchedIf);

        // 记录 matchedIf + 判定 fillCoversBase。
        // fillCoversBase 仅在 fill 的 outs 直接是 baseAlloc 本身（无任何
        // subview / reinterpret_cast 包装）时为 true：因为后续我们要把
        // fill 一次性提到 K-loop 之外、按 bank 写满整个 slot —— 这要求
        // 原 fill 也是写满整个 base alloc。若 fill 只覆盖 base alloc 的
        // 一部分，则 "OOB(slot) ⊆ fill region" 不成立，按 bank 预填充
        // 会漏写 fill 之外、copy 之外的那部分，破坏 OOB-0 不变量。
        grp.matchedZeroFillIf = matchedIf;
        for (Operation &innerOp :
             matchedIf.getThenRegion().front().getOperations()) {
          if (auto fillOp = dyn_cast<linalg::FillOp>(&innerOp)) {
            if (fillOp.getOutputs().size() == 1 &&
                fillOp.getOutputs()[0] == baseAlloc.getResult()) {
              grp.fillCoversBase = true;
            }
            break;
          }
        }
      }
      // 若 condition chain 不可识别，则放弃迁移此 fill：保留原 IR 行为
      // （在 OOB 边界 tile 下仍可能错，但至少不会引入更糟的 IR）。
    }

    groups.push_back(std::move(grp));
  }
  return groups;
}

// 兼容旧接口：返回所有 copies。
static SmallVector<memref::CopyOp> collectDDRToSPMCopies(scf::ForOp forOp) {
  SmallVector<memref::CopyOp> copies;
  for (auto copyOp : forOp.getOps<memref::CopyOp>()) {
    if (isAllocBacked(copyOp.getTarget()))
      copies.push_back(copyOp);
  }
  return copies;
}

static std::optional<int64_t> tryGetMemRefStaticBytes(MemRefType ty) {
  if (!ty.hasStaticShape())
    return std::nullopt;
  Type elemTy = ty.getElementType();
  if (!elemTy.isIntOrFloat())
    return std::nullopt;
  unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
  int64_t numElems = 1;
  for (int64_t d : ty.getShape())
    numElems *= d;
  int64_t bitTotal = numElems * static_cast<int64_t>(bitWidth);
  return (bitTotal + 7) / 8;
}

static void
cloneDefChainInLoopBody(Operation *root, scf::ForOp forOp, IRMapping &mapping,
                        OpBuilder &builder,
                        llvm::DenseSet<Operation *> &alreadyCloned,
                        llvm::DenseMap<Operation *, Operation *> &cloneOf);

static void dedupIfYields(scf::IfOp ifOp, OpBuilder &builder) {
  Location ifLoc = ifOp.getLoc();
  for (Region &region : ifOp->getRegions()) {
    if (region.empty())
      continue;
    for (Block &block : region) {
      // 一次性收集并删除所有"非最后一个" scf.yield。
      // 删除后剩余至多 1 个 yield，无需重复扫描。
      SmallVector<scf::YieldOp> yields;
      for (Operation &op : block)
        if (auto y = dyn_cast<scf::YieldOp>(&op))
          yields.push_back(y);
      for (size_t i = 0; i + 1 < yields.size(); ++i)
        yields[i]->erase();
    }
    bool needTerminator = false;
    for (Block &block : region) {
      if (!block.mightHaveTerminator()) {
        needTerminator = true;
        break;
      }
    }
    if (needTerminator)
      scf::IfOp::ensureTerminator(region, builder, ifLoc);
  }
  for (Region &region : ifOp->getRegions()) {
    if (region.empty())
      continue;
    region.walk([&](scf::IfOp nested) {
      if (nested != ifOp)
        dedupIfYields(nested, builder);
    });
  }
}

static void
cloneDefChainInLoopBody(Operation *root, scf::ForOp forOp, IRMapping &mapping,
                        OpBuilder &builder,
                        llvm::DenseSet<Operation *> &alreadyCloned,
                        llvm::DenseMap<Operation *, Operation *> &cloneOf);

// 在 generic clone(`op`) 之前，把 op region 内部所引用的外层 forOp body
// 顶层 SSA value 提前 clone 到当前 builder/mapping 中。
//
// 对 region-bearing op（scf.for / scf.while / scf.if / linalg.generic …）：
//   builder.clone(op, mapping) 会原样复制 region 中的子 op，并在子 op 的
//   operand 上做 mapping.lookupOrDefault。如果某外层 value 没在 mapping，
//   clone 出来的子 op 会继续指向原始 SSA value——等到原 forOp 被 erase 时，
//   "operation destroyed but still has uses" 立即触发。
//
// 因此对所有 region-bearing op，clone 之前必须先把 region 中"defined above
// 且属于 forOp body 顶层"的 def 全部 cloneDefChainInLoopBody 进来。
static void
ensureRegionExternalsCloned(Operation *op, scf::ForOp forOp, IRMapping &mapping,
                            OpBuilder &builder,
                            llvm::DenseSet<Operation *> &alreadyCloned,
                            llvm::DenseMap<Operation *, Operation *> &cloneOf) {
  if (!op || op->getNumRegions() == 0)
    return;
  llvm::SetVector<Value> outerValues;
  for (Region &region : op->getRegions())
    mlir::getUsedValuesDefinedAbove(region, outerValues);

  SmallVector<Operation *> outerDefs;
  llvm::SmallPtrSet<Operation *, 8> outerDefsSeen;
  for (Value v : outerValues) {
    Operation *def = v.getDefiningOp();
    if (!def)
      continue;
    if (def->getBlock() != forOp.getBody())
      continue;
    if (def == op)
      continue;
    if (mapping.contains(v))
      continue;
    if (outerDefsSeen.insert(def).second)
      outerDefs.push_back(def);
  }
  for (Operation *def : outerDefs)
    cloneDefChainInLoopBody(def, forOp, mapping, builder, alreadyCloned,
                            cloneOf);
}

static void
cloneDefChainInLoopBody(Operation *root, scf::ForOp forOp, IRMapping &mapping,
                        OpBuilder &builder,
                        llvm::DenseSet<Operation *> &alreadyCloned,
                        llvm::DenseMap<Operation *, Operation *> &cloneOf) {
  if (!root || root->getBlock() != forOp.getBody())
    return;
  if (alreadyCloned.contains(root)) {
    Operation *cached = cloneOf.lookup(root);
    if (cached == root)
      return;
    // 同一个 stage 内的克隆缓存（alreadyCloned/cloneOf）会在**不同的
    // IRMapping 上下文**之间共享：kernel body 里"下一轮次态"的克隆用
    // computeMapping，而 load 侧的 prelude / copy operands 用 copyMapping。
    // 后者对原 IV 的代入值比前者大一个 step，同一个原 op 在两套上下文里
    // 会算出不同的值（典型：`%ub = arith.addi %iv, %step` 既是 loop-carried
    // 的下一轮 K 偏移，又是 `min(%ub, K)` 的**上界**操作数）。
    // 因此：若当前 mapping 已经绑定了 root 的全部结果，说明本上下文已有正确
    // 的克隆（例如 clonePreludeIntoLoad 刚 clone 过的 if-condition 链），
    // 必须保留；用另一个上下文的缓存覆盖它等价于把按另一套 IV 代入算出的值
    // 塞进动态尺寸，会让 copy 尺寸算错。
    bool allResultsBound = true;
    for (Value r : root->getResults()) {
      if (!mapping.contains(r)) {
        allResultsBound = false;
        break;
      }
    }
    if (allResultsBound)
      return;
    assert(cached && "clone cache missing for already-cloned op");
    for (auto [from, to] : llvm::zip(root->getResults(), cached->getResults()))
      mapping.map(from, to);
    return;
  }
  if (auto allocOp = dyn_cast<memref::AllocOp>(root)) {
    if (mapping.contains(allocOp.getResult())) {
      alreadyCloned.insert(root);
      cloneOf[root] = root;
      return;
    }
  }
  for (Value opnd : root->getOperands()) {
    if (Operation *def = opnd.getDefiningOp()) {
      cloneDefChainInLoopBody(def, forOp, mapping, builder, alreadyCloned,
                              cloneOf);
    } else if (auto ba = dyn_cast<BlockArgument>(opnd)) {
      // forOp body 的 BlockArg（induction var / iter_args）必须由 caller
      // 在 mapping 里预先映射；否则 generic clone 会引用原 forOp 的
      // BlockArg，导致跨 region 的 SSA 非法。这里同时保留 assert
      // （debug 构建快速发现）和运行时 emit error（release 构建可见）。
      if (ba.getOwner() == forOp.getBody() && !mapping.contains(opnd)) {
        assert(false && "IRMapping must cover loop IV / iter_args before "
                        "cloning op into pipeline stage");
        root->emitError("MKPipelinePass: block-argument operand of ")
            << root->getName()
            << " is not present in IRMapping; pipeline "
               "transform produced invalid IR";
        return;
      }
    }
  }

  // 对 region-bearing root，clone 前先把 region 内引用的外层 forOp body
  // 顶层 def 提前 clone 到 mapping 中，避免 generic clone 出来的子 op
  // 仍然指向原 forOp 内即将被 erase 的 SSA value。详见
  // ensureRegionExternalsCloned 的注释。
  ensureRegionExternalsCloned(root, forOp, mapping, builder, alreadyCloned,
                              cloneOf);

  // 在 generic clone 之前，确保 mapping 中没有任何针对 root
  // 自身 region 内 block-arg 的预映射。如果存在（例如其他路径误把这些
  // BlockArgument 当作"已映射"加入），Region::cloneInto 会跳过
  // addArgument，导致克隆后的 region 入口 block 没有 block args，
  // 触发 "region control flow edge from parent operands to Region #N:
  // source has K operands, but target successor needs 0" verifier 错误。
  for (Region &region : root->getRegions())
    for (Block &block : region)
      for (BlockArgument arg : block.getArguments())
        if (mapping.contains(arg))
          mapping.erase(arg);

  // SubViewOp/ExpandShapeOp/CollapseShapeOp：当 source 的类型因 mapping 而改变
  // （例如 alloc → multi-buffer slot，引入 dynamic offset/stride），结果类型
  // 必须重新推导，否则 builder.clone 会原样保留旧 result type，产生
  // "mismatch of result layout" 验证错误。
  Operation *cloned = nullptr;
  if (auto sv = dyn_cast<memref::SubViewOp>(root)) {
    SmallVector<OpFoldResult> offs, sizes, strides;
    remapMixedFoldResults(sv.getMixedOffsets(), offs, mapping);
    remapMixedFoldResults(sv.getMixedSizes(), sizes, mapping);
    remapMixedFoldResults(sv.getMixedStrides(), strides, mapping);
    Value newSrc = mapping.lookupOrDefault(sv.getSource());
    auto newSrcTy = cast<MemRefType>(newSrc.getType());
    MemRefType resultTy =
        inferSubviewResultType(sv, newSrcTy, offs, sizes, strides);
    cloned = memref::SubViewOp::create(builder, sv.getLoc(), resultTy, newSrc,
                                       offs, sizes, strides);
  } else if (auto expand = dyn_cast<memref::ExpandShapeOp>(root)) {
    Value newSrc = mapping.lookupOrDefault(expand.getSrc());
    auto newSrcTy = cast<MemRefType>(newSrc.getType());
    auto reassoc = expand.getReassociationIndices();
    FailureOr<MemRefType> resultTy = memref::ExpandShapeOp::computeExpandedType(
        newSrcTy, expand.getResultType().getShape(), reassoc);
    if (failed(resultTy)) {
      cloned = builder.clone(*root, mapping);
    } else {
      OpBuilder shapeBuilder(builder.getContext());
      SmallVector<OpFoldResult> mixedOut = getMixedValues(
          expand.getStaticOutputShape(), expand.getOutputShape(), shapeBuilder);
      SmallVector<OpFoldResult> remappedMixed;
      remapMixedFoldResults(mixedOut, remappedMixed, mapping);
      cloned = memref::ExpandShapeOp::create(
          builder, expand.getLoc(), *resultTy, newSrc, reassoc, remappedMixed);
    }
  } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(root)) {
    Value newSrc = mapping.lookupOrDefault(collapse.getSrc());
    auto newSrcTy = cast<MemRefType>(newSrc.getType());
    auto reassoc = collapse.getReassociationIndices();
    MemRefType resultTy =
        memref::CollapseShapeOp::computeCollapsedType(newSrcTy, reassoc);
    cloned = memref::CollapseShapeOp::create(builder, collapse.getLoc(),
                                             resultTy, newSrc, reassoc);
  } else {
    cloned = builder.clone(*root, mapping);
    if (auto clonedIf = dyn_cast<scf::IfOp>(cloned))
      dedupIfYields(clonedIf, builder);
  }

  cloneOf[root] = cloned;
  for (auto [from, to] : llvm::zip(root->getResults(), cloned->getResults()))
    mapping.map(from, to);
  alreadyCloned.insert(root);
}

static void remapCopySourceProducersInLoop(
    memref::CopyOp copyOp, scf::ForOp forOp, IRMapping &mapping,
    OpBuilder &builder, llvm::DenseSet<Operation *> &alreadyCloned,
    llvm::DenseMap<Operation *, Operation *> &cloneOf) {
  Value src = copyOp.getSource();
  if (!src)
    return;
  if (Operation *def = src.getDefiningOp())
    cloneDefChainInLoopBody(def, forOp, mapping, builder, alreadyCloned,
                            cloneOf);
}

static MemRefType inferSubviewResultType(memref::SubViewOp sv,
                                         MemRefType srcType,
                                         ArrayRef<OpFoldResult> offsets,
                                         ArrayRef<OpFoldResult> sizes,
                                         ArrayRef<OpFoldResult> strides) {
  if (sv.getType().getRank() < srcType.getRank())
    return memref::SubViewOp::inferRankReducedResultType(
        sv.getType().getShape(), srcType, offsets, sizes, strides);
  return cast<MemRefType>(
      memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides));
}

static OpFoldResult remapOfr(OpFoldResult ofr, const IRMapping &rm) {
  if (Value v = llvm::dyn_cast_if_present<Value>(ofr))
    return rm.lookupOrDefault(v);
  return ofr;
}

static void remapMixedFoldResults(ArrayRef<OpFoldResult> in,
                                  SmallVectorImpl<OpFoldResult> &out,
                                  const IRMapping &rm) {
  out.clear();
  for (OpFoldResult ofr : in)
    out.push_back(remapOfr(ofr, rm));
}

static Value remapDestToSlot(OpBuilder &b, Location loc,
                             scf::ForOp pipelinedLoop, IRMapping &mapping,
                             llvm::DenseSet<Operation *> &producerDone,
                             llvm::DenseMap<Operation *, Operation *> &cloneOf,
                             Value origDest, Value slot,
                             memref::AllocOp baseAlloc) {
  if (!baseAlloc || origDest == baseAlloc.getResult())
    return slot;

  SmallVector<Operation *, 4> path;
  Value cur = origDest;
  Value baseVal = baseAlloc.getResult();
  while (cur != baseVal) {
    Operation *def = cur.getDefiningOp();
    if (!def)
      return slot;
    path.push_back(def);
    if (auto sv = dyn_cast<memref::SubViewOp>(def)) {
      cur = sv.getSource();
      continue;
    }
    if (auto rc = dyn_cast<memref::ReinterpretCastOp>(def)) {
      cur = rc.getSource();
      continue;
    }
    return slot;
  }

  Value mapped = slot;
  for (Operation *op : llvm::reverse(path)) {
    Value srcV;
    if (auto sv = dyn_cast<memref::SubViewOp>(op))
      srcV = sv.getSource();
    else if (auto rc = dyn_cast<memref::ReinterpretCastOp>(op))
      srcV = rc.getSource();
    else
      continue;

    for (Value oper : op->getOperands()) {
      if (oper == srcV)
        continue;
      if (Operation *def = oper.getDefiningOp()) {
        if (def->getBlock() == pipelinedLoop.getBody())
          cloneDefChainInLoopBody(def, pipelinedLoop, mapping, b, producerDone,
                                  cloneOf);
      }
    }

    IRMapping rm(mapping);
    rm.map(srcV, mapped);

    if (auto sv = dyn_cast<memref::SubViewOp>(op)) {
      SmallVector<OpFoldResult> offs, sizes, strides;
      remapMixedFoldResults(sv.getMixedOffsets(), offs, rm);
      remapMixedFoldResults(sv.getMixedSizes(), sizes, rm);
      remapMixedFoldResults(sv.getMixedStrides(), strides, rm);
      Value newSrc = rm.lookupOrDefault(sv.getSource());
      auto newSrcTy = cast<MemRefType>(newSrc.getType());
      MemRefType resultTy =
          inferSubviewResultType(sv, newSrcTy, offs, sizes, strides);
      mapped = memref::SubViewOp::create(b, loc, resultTy, newSrc, offs, sizes,
                                         strides);
    } else if (isa<memref::ReinterpretCastOp>(op)) {
      mapped = b.clone(*op, rm)->getResult(0);
    }
  }
  return mapped;
}

// ─────────────────────────────────────────────────────────────────────────────
// Loop-carried 状态的前瞻推进
//
// 流水线把 load 提前 (numStages-1) 轮：kernel 第 k 轮 load 的是原循环第
// k+(numStages-1) 轮的数据。原循环里由 scf.yield 携带的值（典型：K 方向的
// 指针 / 偏移）在第 k+(numStages-1) 轮的取值，等于把"更新链"连续应用
// (numStages-1) 次。
//
//   * numStages == 2 → 只应用 1 次，即老代码里的 `nextState`；
//   * numStages  > 2 → 必须应用 N-1 次。只应用 1 次会让 load 落后 N-2 块：
//     重复读同一块、漏读后面的块，且不会出 NaN，只表现为数值错误。
//
// 安全前提：被克隆的更新链必须是纯计算。链里一旦出现 mk.dot / memref.store /
// memref.copy 等有副作用的 op，重复求值会改变程序语义；此时
// canAdvanceCarriedState 返回 false，调用方必须放弃更深的流水线。
// ─────────────────────────────────────────────────────────────────────────────

// 判断 value v 的 def-chain（只看 forOp body 内定义的 op）是否全部无副作用。
static bool isPureCarriedChain(scf::ForOp forOp, Value v,
                               llvm::SmallPtrSetImpl<Operation *> &seen) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return true; // BlockArg / 循环外值
  if (def->getBlock() != forOp.getBody())
    return true; // 循环外定义，无需克隆
  if (!seen.insert(def).second)
    return true;
  if (!isMemoryEffectFree(def))
    return false;
  for (Value operand : def->getOperands())
    if (!isPureCarriedChain(forOp, operand, seen))
      return false;
  return true;
}

// yield 链里是否真的用到了 induction var。大多数 kernel 的 K 偏移是
// `off += step`，与 IV 无关；此时不必为每一步发射 `iv + t*step`（否则主循环里
// 会多出两条死 arith op，虽然能被后续 canonicalize 删掉）。
static bool carriedChainUsesInductionVar(scf::ForOp forOp) {
  if (forOp.getNumRegionIterArgs() == 0)
    return false;
  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  llvm::SmallPtrSet<Operation *, 16> seen;
  bool uses = false;
  std::function<void(Value)> visit = [&](Value v) {
    if (uses)
      return;
    if (v == forOp.getInductionVar()) {
      uses = true;
      return;
    }
    Operation *def = v.getDefiningOp();
    if (!def || def->getBlock() != forOp.getBody())
      return;
    if (!seen.insert(def).second)
      return;
    for (Value operand : def->getOperands())
      visit(operand);
  };
  for (Value v : yield.getOperands())
    visit(v);
  return uses;
}

// 判断该循环的 carried-state 是否可以被安全地向前推进多次。
static bool canAdvanceCarriedState(scf::ForOp forOp) {
  if (forOp.getNumRegionIterArgs() == 0)
    return true;
  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  for (Value v : yield.getOperands()) {
    llvm::SmallPtrSet<Operation *, 16> seen;
    if (!isPureCarriedChain(forOp, v, seen))
      return false;
  }
  return true;
}

// 把 forOp 的 loop-carried 状态从第 k 轮推进到第 k+steps 轮。
//   curState : 第 k 轮各 region iter_arg 的取值（长度 == getNumRegionIterArgs）
//   baseIv   : 第 k 轮的 induction var
//   mapping  : 调用方上下文。只有 t == 0 这一步会写回它；t >= 1 使用全新
//              IRMapping，避免命中 t == 0 在另一套上下文里生成的克隆（见
//              dcd52c46d 的同类陷阱）。
//              [必须独立] 调用方不能传入"后续还要给别的 IV 上下文用"的
//              mapping/缓存（computeMapping、copyMapping 都不行）：克隆缓存
//              只按原始 op 索引，不区分 IV 代入值。prologue 与 kernel-load
//              两处都用独立 mapping + 独立缓存，见各自调用点的注释。
//   out      : 第 k+steps 轮各 iter_arg 的取值
static void
advanceCarriedState(scf::ForOp forOp, ArrayRef<Value> curState, unsigned steps,
                    Value baseIv, IRMapping &mapping, OpBuilder &builder,
                    llvm::DenseSet<Operation *> &producerDone,
                    llvm::DenseMap<Operation *, Operation *> &cloneOf,
                    SmallVectorImpl<Value> &out) {
  out.assign(curState.begin(), curState.end());
  if (steps == 0 || forOp.getNumRegionIterArgs() == 0)
    return;

  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  Value step = forOp.getStep();
  Type ivType = step.getType();
  Location loc = forOp.getLoc();
  const bool chainUsesIv = carriedChainUsesInductionVar(forOp);

  // 第 t 步对应的 induction var：baseIv + t * step。
  auto makeIvAt = [&](unsigned t) -> Value {
    if (t == 0)
      return baseIv;
    Value tConst;
    if (ivType.isIndex())
      tConst =
          arith::ConstantIndexOp::create(builder, loc, static_cast<int64_t>(t));
    else
      tConst =
          arith::ConstantIntOp::create(builder, loc, static_cast<int64_t>(t),
                                       cast<IntegerType>(ivType).getWidth());
    Value delta = arith::MulIOp::create(builder, loc, tConst, step);
    return arith::AddIOp::create(builder, loc, baseIv, delta);
  };

  for (unsigned t = 0; t < steps; ++t) {
    IRMapping stepMapping;
    IRMapping *m = (t == 0) ? &mapping : &stepMapping;
    m->map(forOp.getInductionVar(), chainUsesIv ? makeIvAt(t) : baseIv);
    for (unsigned j = 0; j < forOp.getNumRegionIterArgs(); ++j)
      m->map(forOp.getRegionIterArgs()[j], out[j]);

    // t == 0 复用调用方的克隆缓存（与老路径完全一致）；t >= 1 用独立缓存，
    // 否则会命中上一步的克隆，算出上一轮的值。
    llvm::DenseSet<Operation *> localDone;
    llvm::DenseMap<Operation *, Operation *> localCloneOf;
    auto &done = (t == 0) ? producerDone : localDone;
    auto &clones = (t == 0) ? cloneOf : localCloneOf;

    SmallVector<Value> next;
    for (unsigned j = 0; j < forOp.getNumRegionIterArgs(); ++j) {
      Value y = yield.getOperand(j);
      if (Operation *def = y.getDefiningOp())
        cloneDefChainInLoopBody(def, forOp, *m, builder, done, clones);
      next.push_back(m->lookupOrDefault(y));
    }
    out = std::move(next);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 流水线重写器
// ─────────────────────────────────────────────────────────────────────────────

struct PipelineRewriter {
  scf::ForOp forOp;
  SmallVector<Operation *> copies; // 可以是 memref::CopyOp 或 linalg::CopyOp
  // 与 copies 一一对应，保存每个 copy 的 prelude（含 fill-if + 其 cmp/ori
  // 链）。
  SmallVector<LoadGroup> loadGroups;
  // compute 阶段必须跳过的 op 集合 —— 仅含 zero-fill 的 scf.if 自身。
  // 详见 LoadGroup::skipInCompute 的注释。
  llvm::SmallPtrSet<Operation *, 8> skipInComputeSet;
  // compute 阶段要跳过的"仅服务于 load"的 op 集合。详见 buildLoadOnlyOps
  // 的注释。超集包含 copies、skipInComputeSet 以及它们上游的计算/
  // scf.while 等；核心目的是避免把 scf.while 内部的 memref.copy 在 compute
  // 阶段重新执行一次（本已在 load 阶段写入 slot[insertIdx]，compute 再写
  // slot[extractIdx] 是纯浪费 DMA）。
  llvm::SmallPtrSet<Operation *, 32> loadOnlyOps;
  // 本 pass 真正创建的多缓冲 alloc（`memref<numStages x ...>`）。getSlot 用它
  // 区分"多缓冲"与"循环外分配、原样使用的普通缓冲"，不再用 shape[0] 嗅探：
  // 3 级流水线后，shape[0] == 3 的外层 alloc 会被误判成多缓冲。
  llvm::DenseSet<Value> multiBufferValues;
  int numStages;
  IRRewriter &rewriter;
  Location loc;

  PipelineRewriter(scf::ForOp forOp, SmallVector<LoadGroup> groups,
                   int numStages, IRRewriter &rewriter)
      : forOp(forOp), loadGroups(std::move(groups)), numStages(numStages),
        rewriter(rewriter), loc(forOp.getLoc()) {
    for (auto &g : loadGroups) {
      copies.push_back(g.copy);
      for (Operation *op : g.skipInCompute)
        skipInComputeSet.insert(op);
    }
    buildLoadOnlyOps();
  }

  // Classify forOp.getBody() top-level ops as "load-only": those whose
  // every direct top-level user is also load-only. Seeds:
  //   - Pipelined memref.copy ops.
  //   - skipInComputeSet (zero-fill scf.if whose side effects on
  //     pipelined alloc slots are already done in the load phase).
  //
  // Then iterate until fixed point: an op with at least one use, whose
  // users all end at load-only ops, is load-only too.
  //
  // Motivation: an inner memref.copy inside scf.while (e.g. for partial
  // row-wraparound reads from DDR into SPM) would otherwise be cloned
  // twice — once into load phase via the pipelined copy's source
  // def-chain, and once into compute phase via cloneComputeOps's
  // top-level walk. The second clone issues a redundant DMA into the
  // extract slot that was already correctly populated by the earlier
  // load phase. Skipping the whole load-only sub-graph in compute kills
  // the duplicate DMA and the wasted arith that feeds it.
  void buildLoadOnlyOps() {
    for (Operation *c : copies)
      loadOnlyOps.insert(c);
    for (Operation *op : skipInComputeSet)
      loadOnlyOps.insert(op);

    // Pipelined alloc bases are NOT load-only (compute reads them via
    // the extract slot), but they must not block propagation either.
    llvm::SmallPtrSet<Operation *, 4> pipelinedAllocBases;
    for (Operation *c : copies)
      if (auto base = findBaseAlloc(getCopyTarget(c)))
        pipelinedAllocBases.insert(base.getOperation());

    Block *body = forOp.getBody();
    bool changed = true;
    while (changed) {
      changed = false;
      for (Operation &op : body->without_terminator()) {
        if (loadOnlyOps.contains(&op))
          continue;
        if (pipelinedAllocBases.contains(&op))
          continue;
        // Side-effect-only ops (no results) cannot be classified via
        // user-propagation — they're seeded explicitly through
        // skipInComputeSet when appropriate.
        if (op.getNumResults() == 0)
          continue;
        bool hasUser = false;
        bool allUsersLoadOnly = true;
        for (Operation *user : op.getUsers()) {
          Operation *topUser = user;
          while (topUser && topUser->getBlock() != body)
            topUser = topUser->getParentOp();
          if (!topUser || topUser->getBlock() != body) {
            allUsersLoadOnly = false;
            break;
          }
          hasUser = true;
          if (!loadOnlyOps.contains(topUser)) {
            allUsersLoadOnly = false;
            break;
          }
        }
        if (hasUser && allUsersLoadOnly) {
          loadOnlyOps.insert(&op);
          changed = true;
        }
      }
    }
  }

  // ───────────────────────────────────────────────────────────────────────
  // 多缓冲的槽间距（bank stride）必须满足 mk.dot 的 256B 对齐要求。
  //
  // 槽基址 = alloc 基址 + i * 每槽字节数；而 AllocateSharedMemory 只保证
  // alloc 的 *基址* 256B 对齐（handleTargetDependentAlignmentRequirements）。
  // 每槽字节数不是 256 的整数倍时（典型：8x8xf16 = 128B 这类小 tile），
  // slot 1..N-1 的 gemm 操作数就会错位。
  //
  // 补 padding 的做法：分配一维 flat buffer（总大小 = N * 补齐后的槽字节数），
  // 再用 memref.reinterpret_cast 得到
  //     memref<N x d0 x d1, strided<[P, d1, 1]>>
  // 槽本身仍是连续 tile，subview 出来的类型与不补 padding 时完全一致
  // （`memref<d0xd1, strided<[d1,1], offset:?>>`），下游无需感知。
  // 不需要补齐时（绝大多数 tile）仍走原来的 identity layout，IR 不变。
  // ───────────────────────────────────────────────────────────────────────
  static constexpr int64_t kSpmOperandAlignment = 256;

  Value createMultiBuffer(memref::AllocOp baseAlloc) {
    auto baseType = mlir::cast<MemRefType>(baseAlloc.getResult().getType());
    Type elemTy = baseType.getElementType();
    IntegerAttr alignment = baseAlloc.getAlignmentAttr();

    SmallVector<int64_t> newShape;
    newShape.push_back(numStages);
    for (int64_t d : baseType.getShape())
      newShape.push_back(d);

    auto makeIdentityMulti = [&]() -> Value {
      auto multiBufType =
          MemRefType::get(newShape, elemTy, MemRefLayoutAttrInterface{},
                          baseType.getMemorySpace());
      return memref::AllocOp::create(rewriter, loc, multiBufType,
                                     baseAlloc.getDynamicSizes(), alignment)
          .getResult();
    };

    const unsigned elemBits = elemTy.getIntOrFloatBitWidth();
    const bool fixedWidth =
        elemBits == 8 || elemBits == 16 || elemBits == 32 || elemBits == 64;
    if (!baseType.hasStaticShape() || !fixedWidth) {
      // 动态尺寸 / 非常规元素类型：无法在编译期算槽间距。
      // 保持旧行为（identity layout），并提示对齐约束无法保证。
      llvm::errs() << "warning: multi-buffer has a dynamic tile shape; "
                      "256B SPM operand alignment of slots is not guaranteed\n";
      return makeIdentityMulti();
    }

    const int64_t elemsPerBank = baseType.getNumElements();
    const int64_t bytesPerBank =
        elemsPerBank * static_cast<int64_t>(elemBits) / 8;
    if (elemsPerBank <= 0 || bytesPerBank % kSpmOperandAlignment == 0)
      return makeIdentityMulti();

    // 槽间距补齐到 256B 的整数倍（按元素计）。
    const int64_t padElems = llvm::alignTo(bytesPerBank, kSpmOperandAlignment) *
                             8 / static_cast<int64_t>(elemBits);

    auto flatTy =
        MemRefType::get({static_cast<int64_t>(numStages) * padElems}, elemTy,
                        MemRefLayoutAttrInterface{}, baseType.getMemorySpace());
    auto flat =
        memref::AllocOp::create(rewriter, loc, flatTy, ValueRange{}, alignment);

    // 维度 stride：第 0 维是补齐后的槽间距，其余维保持 tile 内的连续布局。
    SmallVector<int64_t> strides(newShape.size(), 1);
    for (int i = static_cast<int>(newShape.size()) - 2; i >= 1; --i)
      strides[i] = strides[i + 1] * newShape[i + 1];
    strides[0] = padElems;

    auto stridedTy = MemRefType::get(
        newShape, elemTy,
        StridedLayoutAttr::get(rewriter.getContext(), /*offset=*/0, strides),
        baseType.getMemorySpace());
    SmallVector<OpFoldResult> sizes, strideAttrs;
    for (int64_t d : newShape)
      sizes.push_back(rewriter.getIndexAttr(d));
    for (int64_t s : strides)
      strideAttrs.push_back(rewriter.getIndexAttr(s));
    return memref::ReinterpretCastOp::create(
               rewriter, loc, stridedTy, flat.getResult(),
               rewriter.getIndexAttr(0), sizes, strideAttrs)
        .getResult();
  }

  // Returns true on success. On failure, `destToMultiBuf` may be partially
  // populated but the caller MUST treat the rewrite as failed — downstream
  // code looks up every `copyOp.getTarget()` and would dereference null if
  // any entry is missing.
  bool createMultiBuffers(DenseMap<Value, Value> &destToMultiBuf) {
    DenseMap<Value, Value> baseMemToMulti;
    rewriter.setInsertionPoint(forOp);
    for (Operation *copyOp : copies) {
      Value origDest = getCopyTarget(copyOp);
      memref::AllocOp baseAlloc = findBaseAlloc(origDest);
      if (!baseAlloc) {
        mlir::emitError(loc,
                        "copy target has no memref.alloc base (isAllocBacked "
                        "mismatch?)");
        return false;
      }
      Value baseMem = baseAlloc.getResult();
      // 循环外分配的缓冲（如cu_seqlens偏移缓冲1xi32）不需要多缓冲，
      // 只需把copy移到load阶段即可。后续compute/load阶段直接用原始缓冲。
      if (baseAlloc->getBlock() != forOp.getBody()) {
        destToMultiBuf[origDest] = baseMem;
        continue;
      }
      Value multi = baseMemToMulti.lookup(baseMem);
      if (!multi) {
        multi = createMultiBuffer(baseAlloc);
        multiBufferValues.insert(multi);
        baseMemToMulti[baseMem] = multi;
      }
      destToMultiBuf[origDest] = multi;
    }
    return true;
  }

  // 将某个 copy 的 prelude（fill-if + 其 condition 的
  // cmp/ori 链）按原顺序 clone 到当前 builder 的位置。调用方需要保证：
  //   - mapping 已经把 baseAlloc.getResult() 映射到当前 stage 的 slot；
  //   - mapping 已经覆盖 forOp 的 induction var / iter_args。
  // 注意：fill 的 outs 有可能是 baseAlloc 的派生 view；那些派生 view 在
  // 原循环体内由本身就属于 forOp body 顶层的 op 产生，会被外层
  // cloneDefChainInLoopBody 自动拉入。这里只负责按顺序 clone prelude 自身。
  // 注：参数取非 const ref 是因为 mlir 的 Op 包装类型（scf::IfOp 等）
  // 的 accessor 没有 const 重载，访问 grp.matchedZeroFillIf.getOperation()
  // 需要非 const 路径。grp 内部的 preludeOps / 其它字段不会被本函数修改。
  void clonePreludeIntoLoad(LoadGroup &grp, OpBuilder &b, IRMapping &mapping) {
    Operation *zeroFillIfOp =
        grp.matchedZeroFillIf ? grp.matchedZeroFillIf.getOperation() : nullptr;
    for (Operation *p : grp.preludeOps) {
      // 仅当 emitPerBankPreInits 已经为本 group 在 K-loop
      // 之外发射了 pre-init scf.if（preInitEmitted == true）才跳过原
      // fill-if。注意不能改用 fillCoversBase：那只是结构合格性，与
      // "pre-init 是否真发射"无关；emitPerBankPreInits 还会用 condition
      // 是否循环不变量做二次过滤（addmm K-tail 类 kernel 在那一步会被
      // 安全跳过），此时 prologue / kernel-load 必须保留原 per-iter
      // fill，否则 OOB 区不会被清零，mk.dot 读到 SPM 残留垃圾。
      if (grp.preInitEmitted && p == zeroFillIfOp)
        continue;

      // 多个 LoadGroup 可能共享同一段 condition 链（例如 alloc_3 与 alloc_7
      // 共用一个 OOB mask）。若该 op 的所有 result 已经在 mapping 中（即上
      // 一组 prelude 或同 stage 内的前置已经 clone 过），就不再重复 clone，
      // 也不要覆盖 mapping —— 否则后续使用者指向新克隆的副本，旧引用
      // 仍然散落在已生成的 IR 中，行为虽然合法但会产生死代码。
      bool allMapped = !p->getResults().empty();
      for (Value r : p->getResults()) {
        if (!mapping.contains(r)) {
          allMapped = false;
          break;
        }
      }
      if (allMapped)
        continue;

      // 任一 prelude op 的 operand 若来自 forOp body 的非 prelude/
      // 非 copies 部分，则已经被 cloneDefChainInLoopBody 处理；
      // 这里只需顺序 clone。
      // [关键修复] 同样在 generic clone 前清理 mapping 中针对 op 自身
      // region block-arg 的旧映射，避免 Region::cloneInto 跳过 addArgument。
      for (Region &region : p->getRegions())
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments())
            if (mapping.contains(arg))
              mapping.erase(arg);
      Operation *cloned = b.clone(*p, mapping);
      // 对 scf.if 的 fill-then-yield 子块，clone 后必须确保 terminator 合法。
      if (auto clonedIf = dyn_cast<scf::IfOp>(cloned))
        dedupIfYields(clonedIf, b);
    }
  }

  Value getSlot(Value multiBuf, Value slotIdx) {
    auto bufType = mlir::cast<MemRefType>(multiBuf.getType());
    // 非多缓冲（循环外分配的原始缓冲）直接返回本身
    // 只有本 pass 创建的多缓冲才需要切 slot；外层 alloc 原样返回。
    if (!multiBufferValues.contains(multiBuf) || bufType.getShape().empty())
      return multiBuf;
    SmallVector<int64_t> subShape(bufType.getShape().drop_front());

    SmallVector<OpFoldResult> offsets, sizes, strides;
    offsets.push_back(slotIdx);
    sizes.push_back(rewriter.getIndexAttr(1));
    strides.push_back(rewriter.getIndexAttr(1));
    for (int64_t d : subShape) {
      offsets.push_back(rewriter.getIndexAttr(0));
      sizes.push_back(rewriter.getIndexAttr(d));
      strides.push_back(rewriter.getIndexAttr(1));
    }
    MemRefType subType = memref::SubViewOp::inferRankReducedResultType(
        subShape, bufType, offsets, sizes, strides);
    assert(subType &&
           "inferRankReducedResultType failed for multi-buffer slot");
    return memref::SubViewOp::create(rewriter, loc, subType, multiBuf, offsets,
                                     sizes, strides);
  }

  SmallVector<Value> cloneComputeOps(IRMapping &mapping) {
    llvm::DenseSet<Operation *> producerDone;
    llvm::DenseMap<Operation *, Operation *> computeCloneOf;
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (llvm::is_contained(copies, &op))
        continue;
      if (auto allocOp = dyn_cast<memref::AllocOp>(&op)) {
        bool isPipelinedSpmBase = false;
        for (Operation *c : copies) {
          if (findBaseAlloc(getCopyTarget(c)) == allocOp) {
            isPipelinedSpmBase = true;
            break;
          }
        }
        if (isPipelinedSpmBase)
          continue;
      }
      // 跳过已在 load 阶段执行过的、带副作用的 prelude
      // (zero-fill scf.if)。否则 compute 阶段会再次执行 fill，把刚
      // load 进 slot 的数据清零。
      //
      // 注意：condition 链 (cmp/min/max/ori 等) 不放进 skipInComputeSet，
      // 因为它们是纯计算 op，compute 体内（例如 scf.while 内的 subview
      // 动态尺寸）可能仍有 use；只有整个子图确实"只服务于 load"时才能
      // 跳过它们。这由下面的 loadOnlyOps 全图传播来保证——其他使用者
      // 如果落在 compute 侧，子图里的任何 op 都不会被标成 load-only。
      if (skipInComputeSet.contains(&op))
        continue;
      // [Perf] 跳过"仅服务于 load"的 op。这类 op 的所有 top-level 使用者
      // 要么是 pipelined copy，要么是 skipInCompute 的 zero-fill，要么是
      // 它们的上游链（例如 scf.while 计算 DDR 偏移给 copy）。在 load 阶
      // 段已被 cloneDefChainInLoopBody 克隆过，compute 阶段再克隆一次会
      // 让 scf.while 里的 memref.copy 发起一次重复 DMA 打到 extract slot。
      // 见 buildLoadOnlyOps 的注释。
      if (loadOnlyOps.contains(&op))
        continue;
      for (Value opnd : op.getOperands()) {
        if (Operation *def = opnd.getDefiningOp())
          cloneDefChainInLoopBody(def, forOp, mapping, rewriter, producerDone,
                                  computeCloneOf);
      }
      // 若 op 自身带 region（scf.for / scf.while / scf.if 等），还要把
      // region 内部引用的外层 forOp body 顶层 def 也提前 clone，否则
      // 下面 rewriter.clone(op, mapping) 出来的 IR 仍会指向原 forOp 中
      // 即将被 erase 的 SSA value，触发 "operation destroyed but still has
      // uses"。
      ensureRegionExternalsCloned(&op, forOp, mapping, rewriter, producerDone,
                                  computeCloneOf);
      Operation *cloned = nullptr;
      if (auto sv = dyn_cast<memref::SubViewOp>(&op)) {
        SmallVector<OpFoldResult> offs, sizes, strides;
        remapMixedFoldResults(sv.getMixedOffsets(), offs, mapping);
        remapMixedFoldResults(sv.getMixedSizes(), sizes, mapping);
        remapMixedFoldResults(sv.getMixedStrides(), strides, mapping);
        Value newSrc = mapping.lookupOrDefault(sv.getSource());
        auto newSrcTy = cast<MemRefType>(newSrc.getType());
        MemRefType resultTy =
            inferSubviewResultType(sv, newSrcTy, offs, sizes, strides);
        cloned = memref::SubViewOp::create(rewriter, sv.getLoc(), resultTy,
                                           newSrc, offs, sizes, strides);
      } else if (auto expand = dyn_cast<memref::ExpandShapeOp>(&op)) {
        Value newSrc = mapping.lookupOrDefault(expand.getSrc());
        auto newSrcTy = cast<MemRefType>(newSrc.getType());
        auto reassoc = expand.getReassociationIndices();
        FailureOr<MemRefType> resultTy =
            memref::ExpandShapeOp::computeExpandedType(
                newSrcTy, expand.getResultType().getShape(), reassoc);
        if (failed(resultTy)) {
          cloned = rewriter.clone(op, mapping);
        } else {
          OpBuilder shapeBuilder(rewriter.getContext());
          SmallVector<OpFoldResult> mixedOut =
              getMixedValues(expand.getStaticOutputShape(),
                             expand.getOutputShape(), shapeBuilder);
          SmallVector<OpFoldResult> remappedMixed;
          remapMixedFoldResults(mixedOut, remappedMixed, mapping);
          cloned = memref::ExpandShapeOp::create(rewriter, expand.getLoc(),
                                                 *resultTy, newSrc, reassoc,
                                                 remappedMixed);
        }
      } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(&op)) {
        Value newSrc = mapping.lookupOrDefault(collapse.getSrc());
        auto newSrcTy = cast<MemRefType>(newSrc.getType());
        auto reassoc = collapse.getReassociationIndices();
        MemRefType resultTy =
            memref::CollapseShapeOp::computeCollapsedType(newSrcTy, reassoc);
        cloned = memref::CollapseShapeOp::create(rewriter, collapse.getLoc(),
                                                 resultTy, newSrc, reassoc);
      } else {
        // 与 cloneDefChainInLoopBody 中保持一致：在 generic
        // clone 之前清掉 mapping 里针对 op 自身 region 内 block-arg 的
        // 旧映射；否则 Region::cloneInto 会跳过 addArgument，导致克隆
        // 后的入口 block 没有 block args，触发 verifier 错误
        // ("region control flow edge ... target successor needs 0")。
        for (Region &region : op.getRegions())
          for (Block &block : region)
            for (BlockArgument arg : block.getArguments())
              if (mapping.contains(arg))
                mapping.erase(arg);

        cloned = rewriter.clone(op, mapping);
        // cloneComputeOps 的 generic clone 路径：同样需要 dedupIfYields。
        // 注意：若此 IfOp 已被 cloneDefChainInLoopBody 提前克隆并写入
        // producerDone，则上面 cloneDefChainInLoopBody 调用会走 cached 路径
        // 不再 clone，此处的 rewriter.clone 不会执行到。
        // 若确实走到此处（顶层直接遇到 IfOp），仍需修复。
        if (auto clonedIf = dyn_cast<scf::IfOp>(cloned))
          dedupIfYields(clonedIf, rewriter);
      }
      computeCloneOf[&op] = cloned;
      for (auto [from, to] : llvm::zip(op.getResults(), cloned->getResults()))
        mapping.map(from, to);
      producerDone.insert(&op);
    }
    auto origYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    SmallVector<Value> yieldedVals;
    for (Value v : origYield.getOperands())
      yieldedVals.push_back(mapping.lookupOrDefault(v));
    return yieldedVals;
  }

  // 在 prologue 之前、按 (multi-buffer × condition) 一次性
  // 预填充 OOB 区域。返回值：被成功"提到 K-loop 之外"的 LoadGroup 数量
  // （主要用于日志/调试，目前未使用，保留 size_t 以备将来扩展）。
  //
  // 安全前提（参见文件级注释 [改造点 E]）：
  //   * grp.fillCoversBase == true：fill 写满整个 baseAlloc；
  //   * grp.matchedZeroFillIf.getCondition() 在 forOp 之外定义（改造点 A 的
  //     LICM 应已把整条 cmp 链提走；若未提走则保守跳过该 group，保持
  //     per-iter prelude 行为，正确性不受影响）。
  //
  // 输出 IR 形态（每个 multi-buffer 一条 scf.if）：
  //   scf.if (%cond_or_combined) {
  //     for s = 0 .. numStages-1:
  //       %slot_s = subview %multi[s, 0, 0][1, full_dims...]
  //       linalg.fill ins(%cst) outs(%slot_s)
  //   }
  //
  // 与同一 multi-buffer 关联的多个 group 共用一条 scf.if，conditions 之间
  // 取 OR：因为每个原 fill 都写满整 base alloc，等效于"任一 condition 为
  // 真时整 slot 都该被零初始化"。
  size_t emitPerBankPreInits(const DenseMap<Value, Value> &destToMultiBuf) {
    auto condDominatesForOp = [&](Value c) -> bool {
      if (!c)
        return false;
      Operation *def = c.getDefiningOp();
      if (!def)
        return true; // BlockArg / func arg / 常量外引用：默认认为支配
      return !forOp->isProperAncestor(def);
    };

    // 收集 (multi -> { (groupIdx, cond, fillConstantOp), ... })，按 multi
    // 聚合，发射时合并 conditions（OR 语义），最后把 preInitEmitted=true
    // 写回所有共享该 multi 的合格 group。
    struct PerMultiItem {
      size_t groupIdx;
      Value cond;
      arith::ConstantOp cstOp;
    };
    SmallVector<Value> orderedMulti;
    DenseMap<Value, SmallVector<PerMultiItem>> perMulti;

    // 注：grp 取非 const ref 是因为 mlir 的 Op 包装类型 accessor 没有
    // const 重载（getOperation / getCondition / getThenRegion / getTarget
    // 都不是 const-qualified）。
    for (size_t i = 0; i < loadGroups.size(); ++i) {
      LoadGroup &grp = loadGroups[i];
      if (!grp.fillCoversBase || !grp.matchedZeroFillIf)
        continue;
      Value multi = destToMultiBuf.lookup(getCopyTarget(grp.copy));
      if (!multi)
        continue;
      // [关键安全检查] 必须确认 condition 是循环不变量。否则 fill 与
      // 其触发条件本就是 per-iter 行为（典型：addmm 的 K-tail 让
      // ori(M_tail, K_tail) 含 arg19），任何"提到外面做一次"的尝试都
      // 不能复刻原 per-iter 语义 —— 只能 fallback 保留原 fill。
      Value cond = grp.matchedZeroFillIf.getCondition();
      if (!condDominatesForOp(cond))
        continue;

      // 提取 then-region 内的 linalg.fill 常量。
      arith::ConstantOp cstOp;
      for (Operation &innerOp :
           grp.matchedZeroFillIf.getThenRegion().front().getOperations()) {
        if (auto fillOp = dyn_cast<linalg::FillOp>(&innerOp)) {
          cstOp = fillOp.getInputs()[0].getDefiningOp<arith::ConstantOp>();
          break;
        }
      }
      if (!cstOp)
        continue;

      if (perMulti.find(multi) == perMulti.end())
        orderedMulti.push_back(multi);
      perMulti[multi].push_back({i, cond, cstOp});
    }

    if (perMulti.empty())
      return 0;

    rewriter.setInsertionPoint(forOp);
    size_t emitted = 0;
    for (Value multi : orderedMulti) {
      const auto &items = perMulti[multi];
      assert(!items.empty());

      // 合并所有 condition：cond_0 OR cond_1 OR ...
      Value mergedCond = items[0].cond;
      for (size_t i = 1; i < items.size(); ++i)
        mergedCond =
            arith::OrIOp::create(rewriter, loc, mergedCond, items[i].cond);

      // 任选一个常量作为 fill 值（同 multi 上的 fill 都是同一个零常量）。
      arith::ConstantOp anyCstOp = items[0].cstOp;

      auto bufType = mlir::cast<MemRefType>(multi.getType());
      SmallVector<int64_t> subShape(bufType.getShape().drop_front());

      auto preIf = scf::IfOp::create(
          rewriter, loc, mergedCond,
          [&](OpBuilder &b, Location l) {
            // 在 then-block 中本地克隆常量，避免依赖 anyCstOp 的位置假设
            // （scf.if 的 then-region 与 anyCstOp 通常都在 func 顶层，但
            // 局部克隆使得后续 IR 重写不会被远程引用打扰）。
            Value localCst = b.clone(*anyCstOp)->getResult(0);
            for (int s = 0; s < numStages; ++s) {
              Value slotIdx = arith::ConstantIndexOp::create(b, l, s);
              SmallVector<OpFoldResult> offsets, sizes, strides;
              offsets.push_back(slotIdx);
              sizes.push_back(b.getIndexAttr(1));
              strides.push_back(b.getIndexAttr(1));
              for (int64_t d : subShape) {
                offsets.push_back(b.getIndexAttr(0));
                sizes.push_back(b.getIndexAttr(d));
                strides.push_back(b.getIndexAttr(1));
              }
              MemRefType subType =
                  memref::SubViewOp::inferRankReducedResultType(
                      subShape, bufType, offsets, sizes, strides);
              assert(subType &&
                     "inferRankReducedResultType failed (pre-init slot)");
              Value slot = memref::SubViewOp::create(b, l, subType, multi,
                                                     offsets, sizes, strides);
              linalg::FillOp::create(b, l, ValueRange{localCst},
                                     ValueRange{slot});
            }
            scf::YieldOp::create(b, l);
          },
          [&](OpBuilder &b, Location l) { scf::YieldOp::create(b, l); });
      dedupIfYields(preIf, rewriter);
      // 发射成功后回写到 group：clonePreludeIntoLoad 据此跳过原 fill-if。
      for (const PerMultiItem &it : items)
        loadGroups[it.groupIdx].preInitEmitted = true;
      ++emitted;
    }
    return emitted;
  }

  LogicalResult run() {
    // numStages > 2 需要把 load 侧的 carried-state 连续推进 N-1 次。若更新链
    // 含副作用（mk.dot / store / copy），重复求值会改变程序语义，此时退回
    // 2 级流水线；正确性优先于深度。
    if (numStages > 2 && !canAdvanceCarriedState(forOp)) {
      llvm::errs() << "warning: loop-carried update chain is not pure, "
                      "falling back to 2 pipeline stages\n";
      numStages = 2;
    }

    DenseMap<Value, Value> destToMultiBuf;
    if (!createMultiBuffers(destToMultiBuf) || destToMultiBuf.empty()) {
      mlir::emitError(loc, "create multi buffer failed for dest mem");
      return failure();
    }

    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();
    Value step = forOp.getStep();
    Type ivType = step.getType();

    rewriter.setInsertionPoint(forOp);
    auto makeIvConst = [&](int64_t v) -> Value {
      if (ivType.isIndex())
        return arith::ConstantIndexOp::create(rewriter, loc, v);
      auto intTy = dyn_cast<IntegerType>(ivType);
      assert(intTy && "scf.for induction type must be index or integer");
      return arith::ConstantIntOp::create(rewriter, loc, v, intTy.getWidth());
    };
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    // 不再发射 `nVal / i64Type / nVal_i64`：bank rotation
    // 的 `(x + 1) % numStages` 全部改走 makeBankAdvance（index 域 +
    // 位运算优先），无 i64 round-trip，函数级也少 3 条常量/cast。

    // 在 prologue 之前对每个合格 multi-buffer 一次性预填充
    // 所有 bank slot 的 OOB 区域。命中后，对应 LoadGroup 的 zero-fill
    // scf.if 在 prologue / kernel-load 路径上都会被 clonePreludeIntoLoad
    // 跳过；compute 路径仍由 skipInComputeSet 跳过。语义对等于"OOB 区
    // 一旦被零化、K-loop 内任何 op 都不会再触碰它"——参见 collectPipelined
    // Loads 中的 fillCoversBase 注释 + run() 函数体顶部的整体说明。
    (void)emitPerBankPreInits(destToMultiBuf);

    // ══════════════════════════════════════════════════════════════════════════
    // 1. Prologue
    // ══════════════════════════════════════════════════════════════════════════
    rewriter.setInsertionPoint(forOp);

    for (int i = 0; i < numStages - 1; ++i) {
      Value prologueIv;
      if (i == 0) {
        prologueIv = lb;
      } else {
        Value iVal = makeIvConst(i);
        prologueIv = arith::AddIOp::create(
            rewriter, loc, lb,
            arith::MulIOp::create(rewriter, loc, iVal, step));
      }
      Value slotIdx = arith::ConstantIndexOp::create(rewriter, loc, i);

      Value inBound = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::slt, prologueIv, ub);

      auto prologueIf = scf::IfOp::create(
          rewriter, loc, inBound,
          [&](OpBuilder &b, Location l) {
            llvm::DenseSet<Operation *> copyProducersDone;
            llvm::DenseMap<Operation *, Operation *> copyCloneOf;
            IRMapping copyMapping;
            copyMapping.map(forOp.getInductionVar(), prologueIv);
            // prologue 第 i 级 load 的是原循环第 i 轮，其 carried-state 是
            // "初始状态推进 i 次"。numStages == 2 时 i == 0，等价于老代码
            // 直接用 initArgs；numStages > 2 时 i > 0，必须真正推进。
            // 注意 baseIv 必须是"第 0 轮"的 IV（lb），不是本级的
            // prologueIv：状态推进 i 步要用 IV_0 .. IV_{i-1}。
            //
            // [关键] 推进必须用独立的 mapping 与克隆缓存。若直接复用
            // 下面的 copyMapping，advanceCarriedState 会把"推进前状态"下
            // 克隆出的值（例如 `iv + step` 在 iv=0 时的 64）绑定到原始
            // body op 上；随后 load 路径克隆同一批尺寸链时命中该缓存，
            // 于是 K 块尺寸算成 0，整块数据不被搬运（linear/baddbmm 在
            // numStages=3 下的错误即由此而来）。
            SmallVector<Value> prologueInitArgs(forOp.getInitArgs());
            SmallVector<Value> prologueState;
            {
              IRMapping stateMapping;
              llvm::DenseSet<Operation *> stateProducersDone;
              llvm::DenseMap<Operation *, Operation *> stateCloneOf;
              advanceCarriedState(forOp, prologueInitArgs,
                                  static_cast<unsigned>(i), lb, stateMapping, b,
                                  stateProducersDone, stateCloneOf,
                                  prologueState);
            }
            for (unsigned j = 0; j < forOp.getNumRegionIterArgs(); ++j)
              copyMapping.map(forOp.getRegionIterArgs()[j], prologueState[j]);
            for (auto &grp : loadGroups) {
              Operation *copyOp = grp.copy;
              Value origTarget = getCopyTarget(copyOp);
              Value multi = destToMultiBuf.lookup(origTarget);
              Value slot;
              memref::AllocOp baseAlloc = findBaseAlloc(origTarget);
              if (grp.isOuterAlloc) {
                // 循环外分配的缓冲直接用原缓冲
                slot = multi;
              } else {
                auto bufType = mlir::cast<MemRefType>(multi.getType());
                SmallVector<int64_t> subShape(bufType.getShape().drop_front());
                SmallVector<OpFoldResult> offsets, sizes, strides;
                offsets.push_back(slotIdx);
                sizes.push_back(b.getIndexAttr(1));
                strides.push_back(b.getIndexAttr(1));
                for (int64_t d : subShape) {
                  offsets.push_back(b.getIndexAttr(0));
                  sizes.push_back(b.getIndexAttr(d));
                  strides.push_back(b.getIndexAttr(1));
                }
                MemRefType subType =
                    memref::SubViewOp::inferRankReducedResultType(
                        subShape, bufType, offsets, sizes, strides);
                assert(subType &&
                       "inferRankReducedResultType failed (prologue slot)");
                slot = memref::SubViewOp::create(b, l, subType, multi, offsets,
                                                 sizes, strides);
              }

              if (baseAlloc)
                copyMapping.map(baseAlloc.getResult(), slot);
              clonePreludeIntoLoad(grp, b, copyMapping);

              Value mappedTarget =
                  remapDestToSlot(b, l, forOp, copyMapping, copyProducersDone,
                                  copyCloneOf, origTarget, slot, baseAlloc);
              copyMapping.map(origTarget, mappedTarget);
              Value origSource = getCopySource(copyOp);
              if (Operation *def = origSource.getDefiningOp())
                cloneDefChainInLoopBody(def, forOp, copyMapping, b,
                                        copyProducersDone, copyCloneOf);
              b.clone(*copyOp, copyMapping);
            }
            scf::YieldOp::create(b, l);
          },
          [&](OpBuilder &b, Location l) { scf::YieldOp::create(b, l); });
      dedupIfYields(prologueIf, rewriter);
    }

    rewriter.setInsertionPoint(forOp);
    // [DISABLED] post-prologue barrier.
    //
    // `mk.barrier` 只有一个来源：`dsa.DistributedBarrierOp`，语义是跨 tile
    // 的全局同步（"Synchronizes all work items"）。单 tile 的 matmul 调用它
    // 会等所有协作 tile 到达，但实际 grid 里根本没有其他参与者，于是
    // `__Barrier()` 永远不返回——栈跟踪 `txStreamSynchronize →
    // Stream::finish → Event::awaitCompletion` 死循环即由此而来。
    //
    // 正常（未流水）版本的 tx/LLVM IR 里一次 `tx.barrier` 都没有，完全靠
    // 硬件 stream 的 FIFO 顺序保证 `__Rdma`/`__Gemm`/`__Memset` 之间的
    // program-order 依赖。流水化改写只是 SSA 层面的多缓冲切分，同样不需要
    // 显式 barrier。
    //
    // 在 mk 方言补齐 per-tile stream fence 之前，这里不再插入 barrier。
    // mlir::mk::BarrierOp::create(rewriter, loc);

    // ══════════════════════════════════════════════════════════════════════════
    // 2. Kernel Loop
    // ══════════════════════════════════════════════════════════════════════════
    Value kernelLb = arith::AddIOp::create(
        rewriter, loc, lb,
        arith::MulIOp::create(rewriter, loc, makeIvConst(numStages - 1), step));

    SmallVector<Value> initArgs;
    initArgs.push_back(
        arith::ConstantIndexOp::create(rewriter, loc, numStages - 1));
    initArgs.push_back(zero);
    for (Value v : forOp.getInitArgs())
      initArgs.push_back(v);

    auto newForOp = scf::ForOp::create(
        rewriter, loc, kernelLb, ub, step, initArgs,
        [&](OpBuilder &b, Location nestedLoc, Value, ValueRange iterArgs) {
          scf::YieldOp::create(b, nestedLoc, iterArgs);
        });
    auto oldYieldInKernel =
        cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    rewriter.setInsertionPoint(oldYieldInKernel);

    Value insertIdx = newForOp.getBody()->getArgument(1);
    Value extractIdx = newForOp.getBody()->getArgument(2);

    Value kernelIv = newForOp.getInductionVar();
    Value computeIv = arith::SubIOp::create(
        rewriter, loc, kernelIv,
        arith::MulIOp::create(rewriter, loc,
                              makeIvConst(static_cast<int64_t>(numStages - 1)),
                              step));
    IRMapping computeMapping;
    computeMapping.map(forOp.getInductionVar(), computeIv);
    for (int i = 0; i < forOp.getNumRegionIterArgs(); ++i)
      computeMapping.map(forOp.getRegionIterArgs()[i],
                         newForOp.getBody()->getArgument(3 + i));

    llvm::DenseSet<Operation *> kernelCopyProducersDone;
    llvm::DenseMap<Operation *, Operation *> kernelCopyCloneOf;
    IRMapping copyMapping;
    copyMapping.map(forOp.getInductionVar(), kernelIv);
    // kernel 第 k 轮 load 的是原循环第 k+(numStages-1) 轮，因此 carried-state
    // 要推进 numStages-1 步（numStages == 2 时即老代码的 nextState）。
    SmallVector<Value> kernelCurState;
    for (unsigned j = 0; j < forOp.getNumRegionIterArgs(); ++j)
      kernelCurState.push_back(
          computeMapping.lookupOrDefault(forOp.getRegionIterArgs()[j]));
    SmallVector<Value> kernelLoadState;
    {
      // [关键] 与 prologue 的推进完全对称：使用独立的 IRMapping 与克隆
      // 缓存，绝不能复用 computeMapping / kernelCopy*。
      //
      // 克隆缓存只以原始 op 为 key，不区分 IV 代入值。若这里用
      // computeMapping（t == 0 的 IV = computeIv）并把结果记进
      // kernelCopyCloneOf，随后 load 路径用 copyMapping（IV = kernelIv，
      // 比 computeIv 大 numStages-1 步）克隆同一份偏移/尺寸链时，会命中
      // 该缓存并把"只推进 1 步"的值塞进 load（numStages > 2 时 load 落后
      // N-2 块：重复读同一块并漏读后面的块，不会出 NaN，只表现为数值
      // 错误）。prologue 侧曾踩过同一陷阱，当时没有同步修 kernel 侧。
      IRMapping stateMapping;
      llvm::DenseSet<Operation *> stateProducersDone;
      llvm::DenseMap<Operation *, Operation *> stateCloneOf;
      advanceCarriedState(forOp, kernelCurState,
                          static_cast<unsigned>(numStages - 1), computeIv,
                          stateMapping, rewriter, stateProducersDone,
                          stateCloneOf, kernelLoadState);
    }
    for (unsigned j = 0; j < forOp.getNumRegionIterArgs(); ++j)
      copyMapping.map(forOp.getRegionIterArgs()[j], kernelLoadState[j]);
    for (auto &grp : loadGroups) {
      Operation *copyOp = grp.copy;
      Value origTarget = getCopyTarget(copyOp);
      Value multi = destToMultiBuf.lookup(origTarget);
      Value slot = getSlot(multi, insertIdx);
      memref::AllocOp baseAlloc = findBaseAlloc(origTarget);

      if (baseAlloc)
        copyMapping.map(baseAlloc.getResult(), slot);
      clonePreludeIntoLoad(grp, rewriter, copyMapping);

      Value mappedTarget = remapDestToSlot(
          rewriter, loc, forOp, copyMapping, kernelCopyProducersDone,
          kernelCopyCloneOf, origTarget, slot, baseAlloc);
      copyMapping.map(origTarget, mappedTarget);
      Value origSource = getCopySource(copyOp);
      if (Operation *def = origSource.getDefiningOp())
        cloneDefChainInLoopBody(def, forOp, copyMapping, rewriter,
                                kernelCopyProducersDone, kernelCopyCloneOf);
      rewriter.clone(*copyOp, copyMapping);
    }

    llvm::DenseSet<Operation *> computeDestProducersDone;
    llvm::DenseMap<Operation *, Operation *> computeDestCloneOf;
    for (auto [origDest, multi] : destToMultiBuf) {
      memref::AllocOp baseAlloc = findBaseAlloc(origDest);
      Value extSlot = getSlot(multi, extractIdx);
      if (baseAlloc)
        computeMapping.map(baseAlloc.getResult(), extSlot);
      Value remapped = remapDestToSlot(
          rewriter, loc, forOp, computeMapping, computeDestProducersDone,
          computeDestCloneOf, origDest, extSlot, baseAlloc);
      if (baseAlloc && origDest != baseAlloc.getResult())
        computeMapping.map(origDest, remapped);
    }

    SmallVector<Value> clonedYieldVals = cloneComputeOps(computeMapping);

    // [DISABLED] kernel-body barrier。`mk.barrier` 是 distributed barrier，
    // 单 tile 调用会 hang。详见 run() 上方 post-prologue barrier 处的说明。
    // mlir::mk::BarrierOp::create(rewriter, loc);

    // index 域单 op rotation；numStages == 2 时落到一条
    // arith.xori，主循环每轮少 8 个 op（详见 makeBankAdvance 注释）。
    Value nextInsert = makeBankAdvance(rewriter, loc, insertIdx, numStages,
                                       /*oneIdx=*/one);
    Value nextExtract = makeBankAdvance(rewriter, loc, extractIdx, numStages,
                                        /*oneIdx=*/one);

    SmallVector<Value> yieldVals = {nextInsert, nextExtract};
    for (Value v : clonedYieldVals)
      yieldVals.push_back(v);
    scf::YieldOp::create(rewriter, loc, yieldVals);
    rewriter.eraseOp(oldYieldInKernel);

    // 再收敛 kernel 内所有 scf.if（含嵌套）：clone 顺序下仍可能残留双 yield。
    newForOp.walk([&](scf::IfOp op) { dedupIfYields(op, rewriter); });

    // ══════════════════════════════════════════════════════════════════════════
    // 3. Epilogue
    // ══════════════════════════════════════════════════════════════════════════
    rewriter.setInsertionPointAfter(newForOp);
    Value hasAnyWork =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt, lb, ub);

    auto guardedEpilogue =
        scf::IfOp::create(rewriter, loc, forOp.getResultTypes(), hasAnyWork,
                          /*withElseRegion=*/true);

    {
      OpBuilder::InsertionGuard g(rewriter);
      OpBuilder thenB = guardedEpilogue.getThenBodyBuilder();
      rewriter.setInsertionPoint(thenB.getInsertionBlock(),
                                 thenB.getInsertionPoint());

      Value curExtractIdx = newForOp.getResult(1);

      SmallVector<Value> epilogueAccs;
      for (unsigned i = 2; i < newForOp.getNumResults(); ++i)
        epilogueAccs.push_back(newForOp.getResult(i));

      // ── epilogue 的 induction value ──────────────────────────────────
      // 原循环 `scf.for %iv = lb to ub step s` 一共有
      //     n      = ceil((ub - lb) / s)
      // 轮，最后一个 IV 是
      //     lastIv = lb + (n - 1) * s
      //
      // 令 r = ub - lastIv，即最后一个 step 实际走过的长度，1 <= r <= s。
      // 展开 ub - 1 - lb：
      //     ub - 1 - lb = (n - 1) * s + (r - 1)
      // 因为 0 <= r - 1 < s，所以 (ub - 1 - lb) % s == r - 1，于是
      //     lastIv = ub - 1 - ((ub - 1 - lb) % s) = ub - r
      //
      // 旧实现直接写 ub - step，两者相差：
      //     ub - step = (lastIv + r) - s = lastIv - (s - r)
      // 只有 r == s（即 (ub - lb) % s == 0，最后一个 step 刚好走满）时
      // 才等于 lastIv；否则 IV 会偏小 s - r。两个真实反例：
      //   * non-causal（lb=0, ub=1030, s=64）：
      //     n=17, lastIv=1024, r=6，旧值 1030-64=966 = 1024-58；
      //     K-tail mask 把 58 个零填充列当成有效 key（真实只有 6 列）。
      //   * causal on-band（lb=0, ub=min((start_m+1)*BM, KV)=32,
      //     s=BLOCK_N=64）：
      //     n=1, lastIv=0, r=32，旧值 32-64=-32；`epilogueIv >= lb`
      //     为假，整块 compute 被跳过，输出接近全 0。
      //
      // 此处已在外层 scf.if(lb < ub) 的 then-region 内，故 ub-1-lb >= 0，
      // 可以用无符号取余。详细推导与数字见
      // docs/backend/tsingmicro-mk-pipeline-epilogue-iv.md。
      Value ubMinusOne =
          arith::SubIOp::create(rewriter, loc, ub, makeIvConst(1));
      Value tailRem = arith::RemUIOp::create(
          rewriter, loc, arith::SubIOp::create(rewriter, loc, ubMinusOne, lb),
          step);
      Value lastIv = arith::SubIOp::create(rewriter, loc, ubMinusOne, tailRem);
      // alignedUb 是"对齐后的 ub"，即 lastIv + step。代回原来的
      // `ub - (numStages-1-e)*step` 即可保持 epilogue 的升序顺序：
      //   numStages == 2: e=0 -> lastIv
      //   numStages == 3: e=0 -> lastIv-step, e=1 -> lastIv
      // （getEffectiveNumStages 目前只返回 1 或 2，这里保留通用形式。）
      Value alignedUb = arith::AddIOp::create(rewriter, loc, lastIv, step);

      for (int e = 0; e < numStages - 1; ++e) {
        Value eOffsetVal = makeIvConst(static_cast<int64_t>(numStages - 1 - e));
        Value epilogueIv = arith::SubIOp::create(
            rewriter, loc, alignedUb,
            arith::MulIOp::create(rewriter, loc, eOffsetVal, step));

        // 当原循环实际迭代数 N_orig < numStages - 1 时，
        // 部分 epilogue stage 对应的 epilogueIv 会 < lb（即原循环根本没有
        // 跑到这一轮）。此时不能执行该 stage 的 compute（slot 未被
        // load 过），也不能用其结果污染 acc。
        // 用 scf.if (epilogueIv >= lb) 包裹本轮：then 走真实 compute 并
        // yield 新 acc；else 直接透传当前 acc 与 curExtractIdx。
        Value isValid = arith::CmpIOp::create(
            rewriter, loc, arith::CmpIPredicate::sge, epilogueIv, lb);

        SmallVector<Type> stageResultTypes;
        for (Value v : epilogueAccs)
          stageResultTypes.push_back(v.getType());
        // 同时在 if 里前进 extract 指针，避免后续轮使用了无效的 slot。
        stageResultTypes.push_back(rewriter.getIndexType());

        auto stageIf = scf::IfOp::create(rewriter, loc, stageResultTypes,
                                         isValid, /*withElseRegion=*/true);

        // ── then: 真实 compute ─────────────────────────────────
        {
          OpBuilder::InsertionGuard g(rewriter);
          OpBuilder thenB2 = stageIf.getThenBodyBuilder();
          rewriter.setInsertionPoint(thenB2.getInsertionBlock(),
                                     thenB2.getInsertionPoint());

          IRMapping epilogueMapping;
          epilogueMapping.map(forOp.getInductionVar(), epilogueIv);
          for (int i = 0; i < forOp.getNumRegionIterArgs(); ++i)
            epilogueMapping.map(forOp.getRegionIterArgs()[i], epilogueAccs[i]);

          llvm::DenseSet<Operation *> epilogueDestProducersDone;
          llvm::DenseMap<Operation *, Operation *> epilogueDestCloneOf;
          for (auto [origDest, multi] : destToMultiBuf) {
            memref::AllocOp baseAlloc = findBaseAlloc(origDest);
            Value extSlot = getSlot(multi, curExtractIdx);
            if (baseAlloc)
              epilogueMapping.map(baseAlloc.getResult(), extSlot);
            Value remapped =
                remapDestToSlot(rewriter, loc, forOp, epilogueMapping,
                                epilogueDestProducersDone, epilogueDestCloneOf,
                                origDest, extSlot, baseAlloc);
            if (baseAlloc && origDest != baseAlloc.getResult())
              epilogueMapping.map(origDest, remapped);
          }

          SmallVector<Value> newAccs = cloneComputeOps(epilogueMapping);
          // [DISABLED] epilogue barrier。`mk.barrier` 是 distributed barrier，
          // 单 tile 调用会 hang。之前加这一处是想修 "epilogue 后 dealloc/
          // truncf/Wdma 早于 Gemm 完成" 的问题，但实际正常（未流水）版本
          // 根本没有 barrier，靠 stream FIFO 顺序就能保证最后的
          // `tx.fp32_fp16`/`tx.wdma` 在 `tx.gemm` 之后执行。详见 run()
          // 上方 post-prologue barrier 处的完整说明。
          // mlir::mk::BarrierOp::create(rewriter, loc);

          // 同主循环：index 域单 op，numStages==2 → xori。
          Value advancedExtract = makeBankAdvance(rewriter, loc, curExtractIdx,
                                                  numStages, /*oneIdx=*/one);

          SmallVector<Value> thenYield(newAccs.begin(), newAccs.end());
          thenYield.push_back(advancedExtract);
          scf::YieldOp::create(rewriter, loc, thenYield);
        }

        // ── else: 透传当前 acc 与 extract idx ─────────────────
        {
          OpBuilder elseB2 = stageIf.getElseBodyBuilder();
          SmallVector<Value> elseYield(epilogueAccs.begin(),
                                       epilogueAccs.end());
          elseYield.push_back(curExtractIdx);
          scf::YieldOp::create(elseB2, loc, elseYield);
        }

        // 更新 epilogueAccs / curExtractIdx 为 if 的结果，下一轮使用。
        epilogueAccs.clear();
        for (unsigned i = 0; i + 1 < stageIf.getNumResults(); ++i)
          epilogueAccs.push_back(stageIf.getResult(i));
        curExtractIdx = stageIf.getResult(stageIf.getNumResults() - 1);
      }
      scf::YieldOp::create(thenB, loc, epilogueAccs);
    }

    {
      OpBuilder elseB = guardedEpilogue.getElseBodyBuilder();
      scf::YieldOp::create(elseB, loc, forOp.getInitArgs());
    }

    guardedEpilogue.walk([&](scf::IfOp op) { dedupIfYields(op, rewriter); });

    // The buffer deallocation pass has been deprecated in favor of the
    // ownership-based buffer deallocation pipeline. The deprecated pass has
    // some limitations that may cause memory leaks in the resulting IR.
    // llvm::DenseSet<Value> deallocatedMultis;
    // for (auto &entry : destToMultiBuf) {
    //   if (deallocatedMultis.insert(entry.second).second)
    //     memref::DeallocOp::create(rewriter, loc, entry.second);
    // }

    for (unsigned i = 0; i < forOp.getNumResults(); ++i)
      forOp.getResult(i).replaceAllUsesWith(guardedEpilogue.getResult(i));

    rewriter.eraseOp(forOp);
    return success();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pass 入口
// ─────────────────────────────────────────────────────────────────────────────

struct MKPipelinePass
    : public mlir::triton::impl::MKPipelinePassBase<MKPipelinePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(module.getContext());

    SmallVector<scf::ForOp> candidates;
    module.walk([&](scf::ForOp forOp) {
      if (!isInnermostForOp(forOp))
        return;
      int n = getEffectiveNumStages(forOp, this->numStages, this->maxStages);
      if (n <= 1)
        return;
      auto groups = collectPipelinedLoads(forOp);
      if (groups.empty())
        return;
      // Spec requires at least one mk.dot (compute) in a pipelined loop —
      // without it, multi-buffering just adds overhead.
      bool hasDot = false;
      forOp.walk([&](mlir::mk::DotOp) {
        hasDot = true;
        return WalkResult::interrupt();
      });
      if (!hasDot)
        return;

      candidates.push_back(forOp);
    });

    for (auto forOp : candidates) {
      // 在拆 prologue / kernel / epilogue 之前先做一次本地
      // LICM。把循环内"边界 mask 的 condition 链"等纯不变 op 上提到
      // forOp 之前，使得 collectPipelinedLoads 收到的 condChain 为空，
      // 后续 prologue / kernel-load / kernel-compute 三处 clone 都直接
      // 引用同一份循环外 SSA value，不再重复发射 6 个 arith op + 一串
      // 描述符 alloca/store。详细动机见 hoistLoopInvariantPureOps 注释。
      (void)hoistLoopInvariantPureOps(forOp);

      int n = getEffectiveNumStages(forOp, this->numStages, this->maxStages);
      auto groups = collectPipelinedLoads(forOp);
      if (failed(PipelineRewriter(forOp, groups, n, rewriter).run()))
        signalPassFailure();
    }

    if (!sanitizeScfIfs(module))
      signalPassFailure();
  }
};

} // namespace
