#include "BufferOpsEmitter.h"
#include "Dialect/TritonHCUGPU/IR/WholePacketLayout.h"
#include "triton/Analysis/Allocation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "TritonHCUGPUToLLVM/MembarUtility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include <optional>
#include <functional>
#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "Dialect/TritonHCUGPU/IR/LegacyV0Packet.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonHCUGPUTransforms/MlsGroup.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton::gpu;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::HCUMlsSharedEncodingAttr;
using ::mlir::triton::hcugpu::MlsEncodingAttr;
using ::mlir::triton::hcugpu::LegacyV0Packet;
using ::mlir::triton::hcugpu::getLegacyV0Packet;

// Proposed closed gfx936 v0 experiment. This is not the v1 MLS instruction ABI.
// v0: opIdx=1, mlsTile={K,N} is the whole logical B packet, 16-bit,
// alt2Kind=0 means normal legacy reader, order={1,0}, single CTA. Only the
// measured {32,128}/4wave and {16,256}/8wave packets are admitted. No toLL is
// defined for this opaque shared encoding: canonical-u below is only an
// address-calculation chart; R(u) is the actual, nonlinear physical mapping.
namespace {
static std::optional<int64_t> legacyV0Constant(Value value) {
  if (auto op = value.getDefiningOp<arith::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(op.getValue()))
      return attr.getInt();
  return std::nullopt;
}

// One finite full-square source protocol, shared by preflight and emission.
// Position is constant and full-tile only: four corners plus the old col8
// top-row control. The descriptor covers the actual whole G-by-G allocation;
// validity of that NUM_RECORDS interpretation on gfx936 remains a GPU gate.
struct LegacyV0SourceTuple {
  unsigned row, column, records;
};

static std::optional<LegacyV0SourceTuple> getLegacyV0SourceTuple(
    triton::hcugpu::MatrixLoadToLocalOp copy, const LegacyV0Packet &p) {
  if (copy.getShape().size() != 2 || copy.getStrides().size() != 2 ||
      copy.getIndices().size() != 2)
    return std::nullopt;
  for (unsigned i = 0; i < 2; ++i)
    if (legacyV0Constant(copy.getShape()[i]) != p.global ||
        legacyV0Constant(copy.getStrides()[i]) != (i == 0 ? p.global : 1))
      return std::nullopt;
  auto row = legacyV0Constant(copy.getIndices()[0]);
  auto column = legacyV0Constant(copy.getIndices()[1]);
  if (!row || !column || (*row != 0 && *row != p.global - p.k) ||
      (*column != 0 && *column != p.global - p.n &&
       !(*row == 0 && *column == 8)))
    return std::nullopt;
  // All arithmetic below is unsigned64 after the finite nonnegative checks.
  // A lane reads8 contiguous16-bit elements. Check the final byte of the
  // last16B vector and signed-i32 address arithmetic, without pointer promises.
  uint64_t r = uint64_t(*row), c = uint64_t(*column);
  uint64_t records = 2ull * p.global * p.global;
  uint64_t lastVector = 2ull * ((r + p.k - 1) * p.global + c + p.n - 8);
  if (r + p.k > p.global || c + p.n > p.global || c % 8 ||
      lastVector % 16 || lastVector + 15 >= records ||
      records > 0x7fffffffull || lastVector > 0x7fffffffull - 15)
    return std::nullopt;
  return LegacyV0SourceTuple{unsigned(r), unsigned(c), unsigned(records)};
}

// Inverse of the measured producer's canonical source-coordinate chart.
// It is not the physical layout; arithmetic rotation is applied separately.
static unsigned legacyV0Canonical(unsigned k, unsigned n, unsigned tileK) {
  if (tileK == 32)
    return n | ((k & 1) << 9) | ((k & 2) << 6) | ((k & 4) << 9) |
           ((k & 8) << 5) | ((k & 16) << 6);
  return n | ((k & 1) << 9) | ((k & 2) << 7) | ((k & 4) << 8) |
         ((k & 8) << 8);
}

static Value emitLegacyV0Canonical(TritonLLVMOpBuilder &b, Value k, Value n,
                                   unsigned tileK) {
  auto insert = [&](unsigned mask, unsigned shift) {
    n = b.or_(n, b.shl(b.and_(k, b.i32_val(mask)), b.i32_val(shift)));
  };
  insert(1, 9);
  insert(2, tileK == 32 ? 6 : 7);
  insert(4, tileK == 32 ? 9 : 8);
  insert(8, tileK == 32 ? 5 : 8);
  if (tileK == 32)
    insert(16, 6);
  return n;
}

static Value emitLegacyV0Physical(TritonLLVMOpBuilder &b, Value u) {
  Value oddProducerPacket = b.and_(b.lshr(u, b.i32_val(9)), b.i32_val(1));
  Value low = b.add(b.and_(u, b.i32_val(511)),
                    b.shl(oddProducerPacket, b.i32_val(5)));
  return b.or_(b.and_(u, b.i32_val(0xfffffe00u)),
                 b.and_(low, b.i32_val(511)));
}

static bool isLegacyV0Dot(RankedTensorType dst, const LegacyV0Packet &p) {
  if (dst.getRank() != 2 || dst.getShape()[0] != p.k ||
      dst.getShape()[1] != p.n ||
      !(dst.getElementType().isF16() || dst.getElementType().isBF16()))
    return false;
  auto dot = dyn_cast<DotOperandEncodingAttr>(dst.getEncoding());
  auto mma = dot ? dyn_cast<AMDMfmaEncodingAttr>(dot.getParent())
                 : AMDMfmaEncodingAttr{};
  if (!mma || dot.getOpIdx() != 1 || dot.getKWidth() != 4 ||
      mma.getMmacLayout() != MmacLayout::LEGACY ||
      mma.getInstrShape() != ArrayRef<unsigned>({16, 16, 16}) ||
      mma.getTilesPerWarp() != ArrayRef<unsigned>({1, 1}) ||
      mma.getWarpsPerCTA() != ArrayRef<unsigned>({p.warps / 2, 2}))
    return false;
  // Legal normal-reader identity tile in canonical-u, not a fake physical LL.
  auto ll = triton::gpu::toLinearLayout(dst);
  auto *ctx = dst.getContext();
  auto reg = StringAttr::get(ctx, "register"), lane = StringAttr::get(ctx, "lane");
  auto warp = StringAttr::get(ctx, "warp"), block = StringAttr::get(ctx, "block");
  // This is a raw logical map (outputs dim0/dim1), not an endomorphism
  // with a block output. Single-CTA ownership is a size-one input domain.
  if (ll.getInDimSize(block) != 1 || ll.getInDimSize(reg) != 32 ||
      ll.getInDimSize(lane) != 64 || ll.getInDimSize(warp) != p.warps)
    return false;
  auto basis = [&](StringAttr dim, unsigned bit) {
    auto xy = ll.getBasis(dim, bit);
    return legacyV0Canonical(xy[0], xy[1], p.k);
  };
  for (unsigned bit = 0; bit < 4; ++bit)
    if (basis(lane, bit) != (1u << bit))
      return false;
  for (auto dim : {reg, warp})
    for (unsigned bit = 0; bit < ll.getInDimSizeLog2(dim); ++bit)
      if (basis(dim, bit) & 7)
        return false;
  for (unsigned bit = 4; bit < 6; ++bit)
    if (basis(lane, bit) & 7)
      return false;
  return true;
}


// Private opaque-v0 single-packet consumer experiment. This is the exact
// existing Legacy A/B/C ownership exported for s25 KW8/TPW22; it is not a
// new matrix-read instruction or a widening of the prefix2/LP1 contracts.
static bool isLegacyV0S25Kw8Dot(RankedTensorType dst,
                               const LegacyV0Packet &p) {
  if (p.k != 32 || p.n != 128 || p.warps != 4 || p.global != 1024 ||
      dst.getRank() != 2 || dst.getShape() != ArrayRef<int64_t>({32, 128}) ||
      !(dst.getElementType().isF16() || dst.getElementType().isBF16()))
    return false;
  auto dot = dyn_cast<DotOperandEncodingAttr>(dst.getEncoding());
  auto mma = dot ? dyn_cast<AMDMfmaEncodingAttr>(dot.getParent())
                 : AMDMfmaEncodingAttr{};
  if (!mma || dot.getOpIdx() != 1 || dot.getKWidth() != 8 ||
      mma.getVersion() != 3 || mma.getElementBitWidth() != 32 ||
      mma.getMmacLayout() != MmacLayout::LEGACY ||
      !mma.getIsTransposed() ||
      mma.getInstrShape() != ArrayRef<unsigned>({16, 16, 16}) ||
      mma.getTilesPerWarp() != ArrayRef<unsigned>({2, 2}) ||
      mma.getWarpsPerCTA() != ArrayRef<unsigned>({2, 2}) ||
      !llvm::equal(mma.getCTALayout().getCTAsPerCGA(),
                   ArrayRef<unsigned>({1, 1})) ||
      !llvm::equal(mma.getCTALayout().getCTASplitNum(),
                   ArrayRef<unsigned>({1, 1})))
    return false;
  auto ll = triton::gpu::toLinearLayout(dst);
  auto *ctx = dst.getContext();
  auto reg = StringAttr::get(ctx, "register");
  auto lane = StringAttr::get(ctx, "lane");
  auto warp = StringAttr::get(ctx, "warp");
  auto block = StringAttr::get(ctx, "block");
  if (!ll.hasInDim(reg) || !ll.hasInDim(lane) || !ll.hasInDim(warp) ||
      !ll.hasInDim(block) || ll.getInDimSize(reg) != 32 ||
      ll.getInDimSize(lane) != 64 || ll.getInDimSize(warp) != 4 ||
      ll.getInDimSize(block) != 1)
    return false;
  auto matches = [&](StringAttr dim,
                     ArrayRef<std::pair<int32_t, int32_t>> expected) {
    if (ll.getInDimSizeLog2(dim) != expected.size())
      return false;
    for (unsigned bit = 0; bit < expected.size(); ++bit) {
      auto actual = ll.getBasis(dim, bit);
      if (actual.size() != 2 || actual[0] != expected[bit].first ||
          actual[1] != expected[bit].second)
        return false;
    }
    return true;
  };
  return matches(reg, {{1, 0}, {2, 0}, {4, 0}, {0, 16}, {0, 64}}) &&
         matches(lane, {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {8, 0}, {16, 0}}) &&
         matches(warp, {{0, 32}, {0, 0}});
}

static LLVM::LLVMFuncOp getLegacyV0Function(ModuleOp module, Location loc,
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

static std::optional<int64_t> legacyV0SafeInteger(Value v) {
  auto c = v.getDefiningOp<arith::ConstantOp>();
  auto a = c ? dyn_cast<IntegerAttr>(c.getValue()) : IntegerAttr{};
  if (!a || !a.getValue().isSignedIntN(64))
    return std::nullopt;
  return a.getInt();
}

struct LegacyV0CountedBody {
  Block *body;
  Value iv;
  int64_t trips;
};

// Actual compiler_hcu.make_llir converts SCF to CF before this preflight.
// Reuse the existing MembarUtility exact-four-block counted-loop proof shape,
// specialized to constant positive trip/step1. Default arity2 remains the
// original stage1 contract; arity0 is used only by prefix2's local SSA queues.
static std::optional<LegacyV0CountedBody>
getLegacyV0CountedBody(Operation *anchor, int64_t maxTrips, unsigned arguments = 2) {
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
      legacyV0SafeInteger(enter.getDestOperands()[0]) != 0 ||
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
  auto trips = cmp ? legacyV0SafeInteger(cmp.getRhs()) : std::nullopt;
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
      legacyV0SafeInteger(inc.getRhs()) != 1)
    return std::nullopt;
  // Initial0, step1, sltTrip, unique entry/backedge prove 0<=IV<Trip.
  // Caller bounds Trip by its validated source extent before using IV*BK.
  return LegacyV0CountedBody{body, iv, *trips};
}

// Bound the entire full BN column tile without trusting launch metadata.
// Only one uniform CTA program-id axis1, unsigned remainder (or exact power2
// mask), then BN scale is accepted. It does not alter any pointer/descriptor.
static bool isLegacyV0BoundedPIDColumn(Value value, int64_t totalN, unsigned bn) {
  if (totalN < bn || totalN % bn)
    return false;
  auto constant = [](Value v) -> std::optional<int64_t> {
    auto op = v.getDefiningOp<arith::ConstantOp>();
    auto a = op ? dyn_cast<IntegerAttr>(op.getValue()) : IntegerAttr{};
    if (!a || !a.getValue().isSignedIntN(64)) return std::nullopt;
    return a.getValue().getSExtValue();
  };
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>()) {
    if (!ext.getIn().getType().isInteger(32) || !value.getType().isInteger(64)) return false;
    value = ext.getIn();
  } else if (auto ext = value.getDefiningOp<arith::ExtUIOp>()) {
    if (!ext.getIn().getType().isInteger(32) || !value.getType().isInteger(64)) return false;
    value = ext.getIn();
  }
  if (!value.getType().isInteger(32)) return false;
  Value tile;
  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    if (constant(mul.getRhs()) != bn) return false;
    tile = mul.getLhs();
  } else if (auto shift = value.getDefiningOp<arith::ShLIOp>()) {
    unsigned log = bn == 128 ? 7 : bn == 256 ? 8 : 32;
    if (log == 32 || constant(shift.getRhs()) != log) return false;
    tile = shift.getLhs();
  } else return false;
  int64_t tiles = totalN / bn;
  Value pid;
  if (auto rem = tile.getDefiningOp<arith::RemUIOp>()) {
    if (constant(rem.getRhs()) != tiles) return false;
    pid = rem.getLhs();
  } else if (auto mask = tile.getDefiningOp<arith::AndIOp>()) {
    if (tiles < 1 || (uint64_t(tiles) & (uint64_t(tiles) - 1)) ||
        constant(mask.getRhs()) != tiles - 1) return false;
    pid = mask.getLhs();
  } else return false;
  auto op = pid.getDefiningOp<triton::GetProgramIdOp>();
  return op && op.getAxisAsInt() == 1;
}

struct LegacyV0SourceWindow {
  int64_t minRow, maxRow;
  unsigned column, records, totalK;
};

static std::optional<LegacyV0SourceWindow> getLegacyV0SourceWindow(
    triton::hcugpu::MatrixLoadToLocalOp copy, const LegacyV0Packet &p,
    bool prefix2 = false) {
  if (copy.getShape().size() != 2 || copy.getStrides().size() != 2 ||
      copy.getIndices().size() != 2)
    return std::nullopt;
  auto totalK = legacyV0SafeInteger(copy.getShape()[0]);
  auto totalN = legacyV0SafeInteger(copy.getShape()[1]);
  if (!totalK || !totalN || *totalK < p.k || *totalN < p.n ||
      *totalK > 0x7fffffffll || *totalN > 0x7fffffffll ||
      *totalK % p.k || *totalN % p.n ||
      legacyV0SafeInteger(copy.getStrides()[0]) != *totalN ||
      legacyV0SafeInteger(copy.getStrides()[1]) != 1)
    return std::nullopt;
  // Check before multiplication, including the descriptor's signed-i32 bound.
  // totalN is a multiple of BN128/256, so every row is16B aligned.
  if (uint64_t(*totalK) > 0x7fffffffull / (2ull * uint64_t(*totalN)))
    return std::nullopt;
  // Keep column0, or prove a bounded CTA tile column independently of grid.
  // Non-loop one-shot corners/col8 still use their unchanged old verifier.
  int64_t maxColumn = 0;
  if (legacyV0SafeInteger(copy.getIndices()[1]) != 0) {
    if (!isLegacyV0BoundedPIDColumn(copy.getIndices()[1], *totalN, p.n))
      return std::nullopt;
    maxColumn = *totalN - p.n;
  }
  int64_t lo = 0, hi = 0;
  if (auto row = legacyV0SafeInteger(copy.getIndices()[0])) {
    if (*row < 0 || *row % p.k || *row > *totalK - p.k)
      return std::nullopt;
    lo = hi = *row; // optional static-unrolled2/4 diagnostic
  } else {
    auto loop = getLegacyV0CountedBody(copy, *totalK / p.k - int(prefix2), prefix2 ? 0 : 2);
    auto dynamicRow = copy.getIndices()[0].getDefiningOp<arith::MulIOp>();
    if (!loop || !dynamicRow || dynamicRow->getBlock() != loop->body ||
        (prefix2 ? !dynamicRow.getLhs().getDefiningOp<arith::AddIOp>()
                 : dynamicRow.getLhs() != loop->iv) ||
        legacyV0SafeInteger(dynamicRow.getRhs()) != p.k)
      return std::nullopt;
    if (prefix2) {
      auto next = dynamicRow.getLhs().getDefiningOp<arith::AddIOp>();
      if (next.getLhs() != loop->iv || legacyV0SafeInteger(next.getRhs()) != 1)
        return std::nullopt;
    }
    lo = prefix2 ? p.k : 0;
    hi = (loop->trips - 1 + int(prefix2)) * p.k;
  }
  uint64_t records = 2ull * uint64_t(*totalK) * uint64_t(*totalN);
  uint64_t lastVector =
      2ull * ((uint64_t(hi) + p.k - 1) * uint64_t(*totalN) + uint64_t(maxColumn) + p.n - 8);
  if (uint64_t(hi) + p.k > uint64_t(*totalK) || lastVector % 16 ||
      lastVector + 15 >= records || records > 0x7fffffffull ||
      lastVector > 0x7fffffffull - 15)
    return std::nullopt;
  return LegacyV0SourceWindow{lo, hi, unsigned(maxColumn), unsigned(records),
                              unsigned(*totalK)};
}


// LP1's separate lookahead2/4 wrapper. The old +1 prefix2 SourceWindow
// branch and its clients remain unchanged. Prefix constants reuse its bounds.
static std::optional<LegacyV0SourceWindow>
getLegacyV0LocalPrefetchSourceWindow(
    ArrayRef<triton::hcugpu::MatrixLoadToLocalOp> prefix,
    triton::hcugpu::MatrixLoadToLocalOp future, const LegacyV0Packet &p) {
  unsigned slots = prefix.size();
  if (slots != 2 && slots != 4) return std::nullopt;
  auto first = prefix.front();
  auto initial = getLegacyV0SourceWindow(first, p);
  if (!initial || initial->totalK < (slots + 2) * p.k ||
      future.getIndices().size() != 2)
    return std::nullopt;
  for (unsigned j = 0; j < slots; ++j) {
    auto source = getLegacyV0SourceWindow(prefix[j], p);
    if (!source || source->minRow != int64_t(j * p.k) ||
        source->maxRow != source->minRow || source->records != initial->records ||
        source->totalK != initial->totalK || source->column != initial->column)
      return std::nullopt;
  }
  SmallVector<triton::hcugpu::MatrixLoadToLocalOp> all(prefix.begin(), prefix.end());
  all.push_back(future);
  for (auto copy : all)
    if (copy.getBase() != first.getBase() ||
        !llvm::equal(copy.getShape(), first.getShape()) ||
        !llvm::equal(copy.getStrides(), first.getStrides()) ||
        copy.getIndices().size() != 2 || copy.getIndices()[1] != first.getIndices()[1])
      return std::nullopt;
  auto loop = getLegacyV0CountedBody(future, initial->totalK / p.k - slots, 0);
  auto row = future.getIndices()[0].getDefiningOp<arith::MulIOp>();
  auto tile = row ? row.getLhs().getDefiningOp<arith::AddIOp>() : arith::AddIOp{};
  if (!loop || loop->trips != initial->totalK / p.k - slots ||
      !row || !tile || row->getBlock() != loop->body || tile->getBlock() != loop->body ||
      tile.getLhs() != loop->iv || legacyV0SafeInteger(tile.getRhs()) != slots ||
      legacyV0SafeInteger(row.getRhs()) != p.k)
    return std::nullopt;
  int64_t hi = (loop->trips + slots - 1) * int64_t(p.k);
  auto totalN = legacyV0SafeInteger(first.getShape()[1]);
  if (!totalN) return std::nullopt; // established by the prefix SourceWindows
  uint64_t lastVector = 2ull * ((uint64_t(hi) + p.k - 1) * uint64_t(*totalN) +
                               initial->column + p.n - 8);
  if (uint64_t(hi) + p.k > initial->totalK || lastVector % 16 ||
      lastVector + 15 >= initial->records || lastVector > 0x7fffffffull - 15)
    return std::nullopt;
  return LegacyV0SourceWindow{int64_t(slots * p.k), hi, initial->column,
                              initial->records, initial->totalK};
}


// Isolated gfx936 indexed-userA packet experiment. The opaque encoding has
// no GF(2) physical layout: the measured arithmetic rotation stays explicit.
// This does not extend the B packet helper, its wait counter or LP1 proof.
static bool isLegacyV0IndexedAEncoding(Attribute attr) {
  return triton::hcugpu::isLegacyV0IndexedAPacket(attr);
}

static bool isLegacyV0IndexedADot(RankedTensorType t) {
  if (t.getRank() != 2 || t.getShape() != ArrayRef<int64_t>({128, 32}) ||
      !(t.getElementType().isF16() || t.getElementType().isBF16()))
    return false;
  auto dot = dyn_cast<DotOperandEncodingAttr>(t.getEncoding());
  auto mma = dot ? dyn_cast<AMDMfmaEncodingAttr>(dot.getParent())
                 : AMDMfmaEncodingAttr{};
  if (!mma || dot.getOpIdx() != 0 || dot.getKWidth() != 8 ||
      mma.getVersion() != 3 || mma.getElementBitWidth() != 32 ||
      mma.getMmacLayout() != MmacLayout::LEGACY || !mma.getIsTransposed() ||
      mma.getInstrShape() != ArrayRef<unsigned>({16, 16, 16}) ||
      mma.getTilesPerWarp() != ArrayRef<unsigned>({2, 2}) ||
      mma.getWarpsPerCTA() != ArrayRef<unsigned>({2, 2}) ||
      !llvm::equal(mma.getCTALayout().getCTAsPerCGA(), ArrayRef<unsigned>({1, 1})) ||
      !llvm::equal(mma.getCTALayout().getCTASplitNum(), ArrayRef<unsigned>({1, 1})))
    return false;
  auto ll = triton::gpu::toLinearLayout(t);
  auto *ctx = t.getContext();
  auto reg = StringAttr::get(ctx, "register"), lane = StringAttr::get(ctx, "lane");
  auto warp = StringAttr::get(ctx, "warp"), block = StringAttr::get(ctx, "block");
  if (!ll.hasInDim(reg) || !ll.hasInDim(lane) || !ll.hasInDim(warp) ||
      !ll.hasInDim(block) || ll.getInDimSize(reg) != 32 ||
      ll.getInDimSize(lane) != 64 || ll.getInDimSize(warp) != 4 ||
      ll.getInDimSize(block) != 1)
    return false;
  auto matches = [&](StringAttr dim,
                     ArrayRef<std::pair<int32_t, int32_t>> expected) {
    if (ll.getInDimSizeLog2(dim) != expected.size()) return false;
    for (unsigned bit = 0; bit < expected.size(); ++bit) {
      auto xy = ll.getBasis(dim, bit);
      if (xy.size() != 2 || xy[0] != expected[bit].first ||
          xy[1] != expected[bit].second) return false;
    }
    return true;
  };
  return matches(reg, {{0, 1}, {0, 2}, {0, 4}, {16, 0}, {64, 0}}) &&
      matches(lane, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 8}, {0, 16}}) &&
      matches(warp, {{0, 0}, {32, 0}});
}

static Value emitLegacyV0IndexedAPhysical(TritonLLVMOpBuilder &b, Value m, Value k) {
  // Canonical coordinate chart only; R below is the actual physical map.
  Value u = k;
  for (auto [mask, shift] : {std::pair<unsigned, unsigned>{1, 5}, {6, 8},
                            {56, 3}, {64, 5}})
    u = b.or_(u, b.shl(b.and_(m, b.i32_val(mask)), b.i32_val(shift)));
  Value rotation = b.shl(b.and_(b.lshr(u, b.i32_val(9)), b.i32_val(3)), b.i32_val(3));
  Value low = b.and_(b.add(b.and_(u, b.i32_val(511)), rotation), b.i32_val(511));
  return b.or_(b.and_(u, b.i32_val(0xfffffe00u)), low);
}

static LogicalResult lowerLegacyV0IndexedAProducer(
    triton::hcugpu::MatrixLoadToLocalOp op,
    triton::hcugpu::MatrixLoadToLocalOpAdaptor adaptor,
    const LLVMTypeConverter &converter, ConversionPatternRewriter &rewriter) {
  if (!isLegacyV0IndexedAEncoding(op.getDest().getType().getEncoding()))
    return op.emitError("legacy-v0 indexed A: invalid producer packet");
  // Standalone preflight proves its physical-base/prefix/extent ABI and
  // explicit fullwait0 publication. Paired primitives retain the normal
  // unmasked source/whole-view contract, token waits and generic membar;
  // a partial wait does not imply completion of all outstanding DMA.
  auto loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();
  auto elemTy = converter.convertType(op.getDest().getType().getElementType());
  auto smem = getSharedMemoryObjectFromStruct(loc, adaptor.getDest(), elemTy, rewriter);
  auto function = op->getParentOfType<LLVM::LLVMFuncOp>();
  if (!function) return op.emitError("legacy-v0 indexed A: needs LLVM function");
  auto saved = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(&function.getBody().front());
  auto [lane, wave] = getLaneAndWarpId(rewriter, loc);
  wave = rewriter.create<ROCDL::ReadfirstlaneOp>(loc, i32_ty, wave);
  rewriter.restoreInsertionPoint(saved);
  // Identical resource ABI to the already executed P2 struct.ptr probe.
  // 0x4800 is the measured structured stride/swizzle mode, not a byte extent.
  Value resource = rewriter.createOrFold<ROCDL::MakeBufferRsrcOp>(loc,
      ptr_ty(ctx, 8), adaptor.getBase(), b.int_val(16, 0x4800),
      adaptor.getShape()[0], b.i32_val(0x20000));
  Value row = b.add(b.shl(b.lshr(lane, b.i32_val(3)), b.i32_val(3)),
                   b.add(b.and_(b.lshr(lane, b.i32_val(2)), b.i32_val(1)),
                         b.shl(wave, b.i32_val(1))));
  // MatrixLoadToLocal indices are I32 by its existing op contract. The
  // closed preflight admits only the old tuple or two bounded full corners.
  row = b.add(row, adaptor.getIndices()[0]);
  Value offset = b.add(b.shl(adaptor.getIndices()[1], b.i32_val(1)),
                      b.shl(b.and_(lane, b.i32_val(3)), b.i32_val(4)));
  auto fn = getLegacyV0Function(op->getParentOfType<ModuleOp>(), loc, rewriter,
      "llvm.amdgcn.struct.ptr.buffer.load.lds", LLVM::LLVMVoidType::get(ctx),
      {ptr_ty(ctx, 8), ptr_ty(ctx, 3), i32_ty, i32_ty, i32_ty,
       i32_ty, i32_ty, i32_ty});
  for (unsigned call = 0; call < 2; ++call) {
    Value low = b.add(b.ptrtoint(i32_ty, smem.getBase()),
        b.add(b.mul(wave, b.i32_val(1024)), b.i32_val(4096 * call)));
    Value packed = b.or_(low, b.shl(b.and_(wave, b.i32_val(3)), b.i32_val(16)));
    // This packed control word is ONLY an argument to the vendor intrinsic;
    // it never becomes an inbounds GEP or an ordinary LDS load pointer.
    Value dst = b.inttoptr(ptr_ty(ctx, 3), packed);
    LLVM::createLLVMCallOp(rewriter, loc, fn,
        {resource, dst, b.i32_val(16), b.add(row, b.i32_val(64 * call)),
         offset, b.i32_val(0), b.i32_val(0), b.i32_val(0)});
  }
  // Like the existing B emitter, no new alias/noalias promise is introduced.
  rewriter.replaceOp(op, b.i32_val(0));
  return success();
}

static LogicalResult lowerLegacyV0IndexedAConsumer(LocalLoadOp op,
    triton::gpu::LocalLoadOpAdaptor adaptor, const LLVMTypeConverter &converter,
    ConversionPatternRewriter &rewriter) {
  if (!isLegacyV0IndexedAEncoding(op.getSrc().getType().getEncoding()) ||
      !isLegacyV0IndexedADot(op.getType()))
    return op.emitError("legacy-v0 indexed A: needs exact ordinary A-dot ownership");
  auto loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();
  auto elemTy = converter.convertType(op.getType().getElementType());
  auto smem = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(), elemTy, rewriter);
  auto reg = StringAttr::get(ctx, "register"), lane = StringAttr::get(ctx, "lane");
  auto warp = StringAttr::get(ctx, "warp");
  auto logical = triton::gpu::toLinearLayout(op.getType());
  logical = logical.sublayout({reg, lane, warp}, llvm::to_vector(logical.getOutDimNames()));
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  SmallVector<Value> values;
  for (unsigned base = 0; base < 32; base += 8) {
    auto mk = applyLinearLayout(loc, rewriter, logical,
        {{reg, b.i32_val(base)}, {lane, laneId}, {warp, warpId}});
    Value physical = emitLegacyV0IndexedAPhysical(b, mk[0].second, mk[1].second);
    Value ptr = b.gep(ptr_ty(ctx, 3), elemTy, smem.getBase(), physical);
    // The exact LL and R map prove each 8-half vector is contiguous,16B
    // aligned and contained in the selected view. These are ordinary LDS
    // loads, not matrix reads. Source writes must complete and be published:
    // standalone uses fullwait0+CTA; paired retains its token waits and
    // generic membar, without treating partial wait as draining all DMA.
    auto read = LLVM::LoadOp::create(rewriter, loc, vec_ty(elemTy, 8), ptr,
                                    16, false, false);
    for (unsigned j = 0; j < 8; ++j)
      values.push_back(b.extract_element(elemTy, read.getResult(), b.i32_val(j)));
  }
  rewriter.replaceOp(op, packLLElements(loc, &converter, values, rewriter, op.getType()));
  return success();
}

static LogicalResult lowerLegacyV0Producer(
    triton::hcugpu::MatrixLoadToLocalOp op,
    triton::hcugpu::MatrixLoadToLocalOpAdaptor adaptor,
    const LLVMTypeConverter &converter, ConversionPatternRewriter &rewriter) {
  auto packet = getLegacyV0Packet(op.getDest().getType().getEncoding());
  if (!packet)
    return op.emitError("invalid closed legacy-v0 producer packet");
  auto p = *packet;
  auto loopMode = op->getParentOfType<ModuleOp>()->getAttrOfType<IntegerAttr>(
      "ttg.hcu.legacy_v0_static_loop");
  auto pureLoopMode = op->getParentOfType<ModuleOp>()->getAttrOfType<IntegerAttr>(
      "ttg.hcu.v0_blockptr_loop");
  const bool staticLoop = (loopMode && loopMode.getInt() == 1) ||
      (pureLoopMode && pureLoopMode.getInt() == 1);
  auto source = getLegacyV0SourceTuple(op, p);
  if (!staticLoop && !source)
    return op.emitError("invalid closed legacy-v0 full-square source tuple");
  // Static-loop preflight proves the original CF/source SSA before conversion.
  // Only adaptors are consulted here; no proof attrs or mutable global cache.
  auto loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();
  auto elemTy = converter.convertType(op.getDest().getType().getElementType());
  auto smem = getSharedMemoryObjectFromStruct(loc, adaptor.getDest(), elemTy, rewriter);
  // Reuse the AMD direct-to-LDS entry placement: readfirstlane is convergent
  // and MachineLICM cannot hoist it from the loop. The closed v0 producer is
  // directly in the kernel CFG, without warp-specialized relative thread IDs.
  auto insertPt = rewriter.saveInsertionPoint();
  auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
  if (!funcOp)
    return op.emitError("legacy-v0 producer must lower inside an LLVM function");
  rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  auto [lane, wave] = getLaneAndWarpId(rewriter, loc);
  wave = rewriter.create<ROCDL::ReadfirstlaneOp>(loc, i32_ty, wave);
  rewriter.restoreInsertionPoint(insertPt);
  Value tid = b.add(b.mul(wave, b.i32_val(64)), lane);
  Value u = b.lshr(tid, b.i32_val(p.k == 32 ? 4 : 5));
  Value v = b.and_(u, b.i32_val(p.k == 32 ? 7 : 3));
  Value ka = b.shl(b.and_(v, b.i32_val(1)), b.i32_val(1));
  if (p.k == 32) {
    ka = b.add(ka, b.shl(b.and_(b.lshr(v, b.i32_val(1)), b.i32_val(1)), b.i32_val(3)));
    ka = b.add(ka, b.lshr(v, b.i32_val(2)));
    ka = b.add(ka, b.shl(b.lshr(u, b.i32_val(3)), b.i32_val(4)));
  } else {
    ka = b.add(ka, b.lshr(v, b.i32_val(1)));
    ka = b.add(ka, b.shl(b.lshr(u, b.i32_val(2)), b.i32_val(2)));
  }
  Value n = b.shl(b.and_(tid, b.i32_val(p.k == 32 ? 15 : 31)), b.i32_val(3));
  // Keep the measured raw descriptor format/flags; NUM_RECORDS now covers
  // the full source allocation. Base is the physical
  // allocation; the validated constant source row/column selects a full tile.
  Value base64 = b.ptrtoint(i64_ty, adaptor.getBase());
  // The closed preflight validated positive rectangular shape/stride constants
  // and2*K*N<=INT32_MAX. Use converted I32 shape operands here, not a lookup
  // for arith::ConstantOp after the conversion driver has replaced its producer.
  Value sourcePitch = staticLoop ? adaptor.getShape()[1] : b.i32_val(p.global);
  Value sourceRecords = staticLoop
      ? b.shl(b.mul(adaptor.getShape()[0], sourcePitch), b.i32_val(1))
      : b.i32_val(source->records);
  SmallVector<Value> words{
      b.trunc(i32_ty, base64),
      b.and_(b.trunc(i32_ty, b.lshr(base64, b.i64_val(32))), b.i32_val(0xffff)),
      sourceRecords, b.i32_val(0x20000)};
  Value resource = b.undef(vec_ty(i32_ty, 4));
  for (unsigned i = 0; i < 4; ++i)
    resource = b.insert_element(vec_ty(i32_ty, 4), resource, words[i], b.i32_val(i));
  auto module = op->getParentOfType<ModuleOp>();
  auto fn = getLegacyV0Function(module, loc, rewriter, "llvm.hcu.raw.buffer.load.lds",
      LLVM::LLVMVoidType::get(ctx), {vec_ty(i32_ty, 4), ptr_ty(ctx, 3),
                                   i32_ty, i32_ty, i32_ty, i32_ty, i32_ty});
  // Closed preflight proves the full 16-byte vector is within NUM_RECORDS;
  // keep the descriptor extent unchanged when moving the uniform origin.
  Value sourceRow = staticLoop ? adaptor.getIndices()[0] : b.i32_val(source->row);
  Value sourceColumn = staticLoop ? adaptor.getIndices()[1] : b.i32_val(source->column);
  Value sourceOffset = b.shl(b.add(b.mul(sourceRow, sourcePitch), sourceColumn),
                           b.i32_val(1));
  for (unsigned call = 0; call < (p.k == 32 ? 2u : 1u); ++call) {
    Value row = b.add(ka, b.i32_val(4 * call));
    Value offset = b.shl(b.add(b.mul(row, sourcePitch), n), b.i32_val(1));
    Value low = b.add(b.ptrtoint(i32_ty, smem.getBase()),
                     b.add(b.mul(wave, b.i32_val(1024)), b.i32_val(4096 * call)));
    Value packedM0 = b.or_(low, b.shl(b.and_(wave, b.i32_val(1)), b.i32_val(18)));
    // Packed m0 is a control operand ONLY to this exact vendor intrinsic.
    // It must never be used as an ordinary LDS pointer.
    Value dst = b.inttoptr(ptr_ty(ctx, 3), packedM0);
    LLVM::createLLVMCallOp(rewriter, loc, fn,
        {resource, dst, b.i32_val(16), offset, sourceOffset,
         b.i32_val(0), b.i32_val(0)});
  }
  rewriter.replaceOp(op, b.i32_val(0));
  return success();
}

static LogicalResult lowerLegacyV0Consumer(triton::gpu::LocalLoadOp op,
    triton::gpu::LocalLoadOpAdaptor adaptor, const LLVMTypeConverter &converter,
    ConversionPatternRewriter &rewriter) {
  auto packet = getLegacyV0Packet(op.getSrc().getType().getEncoding());
  bool s25Kw8 = packet && isLegacyV0S25Kw8Dot(op.getType(), *packet);
  if (!packet || (!isLegacyV0Dot(op.getType(), *packet) && !s25Kw8))
    return op.emitError("legacy-v0 local_load requires the proved standard B-dot ownership");
  auto p = *packet;
  auto loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();
  auto elemTy = converter.convertType(op.getType().getElementType());
  auto smem = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(), elemTy, rewriter);
  auto reg = StringAttr::get(ctx, "register"), lane = StringAttr::get(ctx, "lane");
  auto warp = StringAttr::get(ctx, "warp");
  auto logical = triton::gpu::toLinearLayout(op.getType());
  auto outDims = llvm::to_vector(logical.getOutDimNames());
  logical = logical.sublayout({reg, lane, warp}, outDims);
  std::optional<mlir::triton::ColumnAction> nativeRegOrder;
  if (s25Kw8) {
    // Standard [K1,K2,K4,N16,N64] -> measured normal-read packet order
    // [K1,K2,N16,N64,K4]. This only permutes a compiler-side LL/value list;
    // producerLane/Reg, canonical/physical address calculation and intrinsic
    // remain the existing normal-reader implementation.
    nativeRegOrder.emplace(ArrayRef<size_t>({0, 1, 3, 4, 2}), reg, 5);
    logical = nativeRegOrder->apply(logical);
  }
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value producerReg = b.or_(b.and_(b.lshr(laneId, b.i32_val(2)), b.i32_val(3)),
      b.shl(b.and_(b.lshr(laneId, b.i32_val(1)), b.i32_val(1)), b.i32_val(2)));
  Value producerLane = b.or_(b.and_(laneId, b.i32_val(48)),
                            b.shl(b.and_(laneId, b.i32_val(1)), b.i32_val(3)));
  auto fn = getLegacyV0Function(op->getParentOfType<ModuleOp>(), loc, rewriter,
      "llvm.hcu.ds.read.m32x16.b16.v4i32.p3", vec_ty(i32_ty, 4),
      {ptr_ty(ctx, 3), i32_ty});
  SmallVector<Value> values;
  for (unsigned base = 0; base < 32; base += 8) {
    auto kn = applyLinearLayout(loc, rewriter, logical,
        {{reg, b.or_(producerReg, b.i32_val(base))}, {lane, producerLane}, {warp, warpId}});
    Value canonical = emitLegacyV0Canonical(b, kn[0].second, kn[1].second, p.k);
    Value physical = emitLegacyV0Physical(b, canonical);
    Value ptr = b.gep(ptr_ty(ctx, 3), elemTy, smem.getBase(), physical);
    auto read = LLVM::createLLVMCallOp(rewriter, loc, fn, {ptr, b.i32_val(0)});
    // Full wait0 + CTA publication is mandatory. No new noalias promise.
    Value unpacked = b.bitcast(read.getResult(), vec_ty(elemTy, 8));
    for (unsigned j = 0; j < 8; ++j)
      values.push_back(b.extract_element(elemTy, unpacked, b.i32_val(j)));
  }
  if (nativeRegOrder)
    values = nativeRegOrder->inverse().apply(values);
  rewriter.replaceOp(op, packLLElements(loc, &converter, values, rewriter, op.getType()));
  return success();
}
// Source/layout ownership is closed around one B packet. No A LDS allocation,
// conversion scratch, extra global memory operations, epilogue, or loops.
static bool isLegacyV0GlobalTensorPointer(Value ptr, RankedTensorType valueTy,
                                         triton::FuncOp function) {
  auto ptrTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!ptrTy || ptrTy.getShape() != valueTy.getShape() ||
      ptrTy.getEncoding() != valueTy.getEncoding())
    return false;
  auto element = dyn_cast<triton::PointerType>(ptrTy.getElementType());
  if (!element || element.getAddressSpace() != 1 ||
      element.getPointeeType() != valueTy.getElementType())
    return false;
  auto add = ptr.getDefiningOp<triton::AddPtrOp>();
  auto splat = add ? add.getPtr().getDefiningOp<triton::SplatOp>()
                   : triton::SplatOp{};
  auto arg = splat ? dyn_cast<BlockArgument>(splat.getSrc()) : BlockArgument{};
  if (!arg || arg.getOwner() != &function.getBody().front())
    return false;
  auto scalar = dyn_cast<triton::PointerType>(arg.getType());
  auto alignment = function.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),
                                                          "tt.divisibility");
  auto offsetTy = dyn_cast<RankedTensorType>(add.getOffset().getType());
  return scalar && scalar.getAddressSpace() == 1 &&
         scalar.getPointeeType() == valueTy.getElementType() && alignment &&
         alignment.getInt() >= 16 && alignment.getInt() % 16 == 0 &&
         offsetTy && offsetTy.getElementType().isInteger(32) &&
         offsetTy.getShape() == valueTy.getShape() &&
         offsetTy.getEncoding() == valueTy.getEncoding();
}

// This checks typed pure pointer construction, not alias independence. It does
// not move instructions, change offsets, or assert any noalias property.

// Deliberately one packet, no dot/loop/extra effects. Existing ToLLVM root8
// allocation validation still checks recomputed64KiB size/base0, emitted
// allocation.offset0 and ttg.shared65536. MemDescIndex selects8192B packets.
static LogicalResult verifyLegacyV0IndexedAClosedModule(ModuleOp mod) {
  auto fail = [&](StringRef reason) -> LogicalResult {
    return mod.emitError("legacy-v0 indexed A: ") << reason;
  };
  auto integerEquals = [](IntegerAttr a, int64_t expected) {
    return a && a.getValue().isSignedIntN(64) && a.getInt() == expected;
  };
  auto hint = mod->getAttrOfType<IntegerAttr>("ttg.hcu.v0_indexed_a_packet");
  if (!hint || !hint.getType().isInteger(32) || !integerEquals(hint, 1) ||
      !integerEquals(mod->getAttrOfType<IntegerAttr>("ttg.num-warps"), 4))
    return fail("requires explicit private packet1/i32 and four waves");
  // Do not let B/loop/joint proof modes reinterpret the A-only root.
  for (StringRef name : {"ttg.hcu.v0_blockptr", "ttg.hcu.v0_blockptr_loop",
       "ttg.hcu.legacy_v0_static_loop", "ttg.hcu.v0_joint_phase",
       "ttg.hcu.v0_mls_local_prefetch_capture"})
    if (mod->hasAttr(name)) return fail("incompatible B/pipeline mode");
  SmallVector<LocalAllocOp> roots;
  SmallVector<LocalLoadOp> reads;
  SmallVector<triton::hcugpu::MatrixLoadToLocalOp> copies;
  SmallVector<triton::StoreOp> stores;
  SmallVector<triton::FuncOp> functions;
  unsigned indices = 0, commits = 0, waits = 0, barriers = 0;
  bool forbidden = false;
  mod.walk([&](Operation *op) {
    auto name = op->getName().getStringRef();
    if (!name.starts_with("arith.") && !llvm::is_contained(ArrayRef<StringRef>({
        "builtin.module", "tt.func", "tt.return", "tt.make_range",
        "tt.expand_dims", "tt.broadcast", "tt.splat", "tt.addptr", "tt.store",
        "ttg.local_alloc", "ttg.memdesc_index", "ttg.local_load",
        "ttg.async_commit_group", "hcug.async_wait", "ttg.local_barrier",
        "hcug.matrix_load_to_local"}), name)) forbidden = true;
    if (op->getNumRegions() && op != mod.getOperation() &&
        !isa<triton::FuncOp>(op)) forbidden = true;
    if (auto fn = dyn_cast<triton::FuncOp>(op)) functions.push_back(fn);
    if (auto a = dyn_cast<LocalAllocOp>(op)) roots.push_back(a);
    if (auto l = dyn_cast<LocalLoadOp>(op)) reads.push_back(l);
    if (auto c = dyn_cast<triton::hcugpu::MatrixLoadToLocalOp>(op)) copies.push_back(c);
    if (auto s = dyn_cast<triton::StoreOp>(op)) stores.push_back(s);
    indices += isa<MemDescIndexOp>(op); commits += isa<AsyncCommitGroupOp>(op);
    waits += isa<triton::hcugpu::AsyncWaitOp>(op); barriers += isa<LocalBarrierOp>(op);
  });
  if (forbidden || functions.size() != 1 || roots.size() != 1 || reads.size() != 1 ||
      copies.size() != 1 || stores.size() != 1 || indices != 1 || commits != 1 ||
      waits != 1 || barriers != 1) return fail("requires one closed packet and no extra effects");
  auto function = functions.front();
  if (!function.getBody().hasOneBlock() || function.getNumArguments() != 2)
    return fail("requires one block and physical-source/output pointer ABI");
  auto root = roots.front(); auto rt = root.getType();
  if (!isLegacyV0IndexedAEncoding(rt.getEncoding()) ||
      rt.getShape() != ArrayRef<int64_t>({8, 128, 32}) ||
      rt.getAllocShape() != rt.getShape() || !rt.getMutableMemory() ||
      !(rt.getElementType().isF16() || rt.getElementType().isBF16()) ||
      !root.isSharedMemoryAlloc() || root.getSrc() ||
      root.getAlignmentOrDefault() < 16 || root.getAlignmentOrDefault() % 16 ||
      !root.getResult().hasOneUse()) return fail("requires sole aligned mutable64KiB root");
  auto index = dyn_cast<MemDescIndexOp>(*root.getResult().getUsers().begin());
  auto slot = index ? legacyV0SafeInteger(index.getIndex()) : std::optional<int64_t>{};
  if (!slot || *slot < 0 || *slot > 3)
    return fail("requires finite whole packet view0..3");
  auto view = index.getType();
  if (view.getShape() != ArrayRef<int64_t>({128, 32}) ||
      view.getAllocShape() != view.getShape() || view.getEncoding() != rt.getEncoding() ||
      view.getElementType() != rt.getElementType() ||
      view.getMemorySpace() != rt.getMemorySpace() || !view.getMutableMemory() ||
      std::distance(index.getResult().use_begin(), index.getResult().use_end()) != 2)
    return fail("requires exact opaque packet view with two uses");
  auto copy = copies.front(); auto read = reads.front(); auto store = stores.front();
  if (copy.getDest() != index.getResult() || read.getSrc() != index.getResult() ||
      !isLegacyV0IndexedADot(read.getType()) ||
      read.getType().getElementType() != rt.getElementType())
    return fail("requires paired producer and exact ordinary A-dot consumer");
  auto source = function.getArgument(0);
  for (unsigned arg = 0; arg < 2; ++arg) {
    auto p = dyn_cast<triton::PointerType>(function.getArgument(arg).getType());
    auto a = function.getArgAttrOfType<IntegerAttr>(arg, "tt.divisibility");
    if (!p || p.getAddressSpace() != 1 || p.getPointeeType() != rt.getElementType() ||
        !a || !a.getValue().isSignedIntN(64) || a.getInt() < 16 || a.getInt() % 16)
      return fail("requires explicitly16B-aligned AS1 typed pointer arguments");
  }
  // Caller must actually allocate/check the declared bytes; attrs are an ABI contract,
  // not a runtime allocation proof. Argument0 is the PHYSICAL start. It is
  // never computed by subtracting16 from an ordinary logical tensor pointer.
  auto bytes = function.getArgAttrOfType<IntegerAttr>(0, "hcu.physical_source_bytes");
  auto prefix = function.getArgAttrOfType<IntegerAttr>(0, "hcu.logical_prefix_bytes");
  bool oldSourceABI = integerEquals(bytes, 33554496) && integerEquals(prefix, 16);
  bool finiteUnpaddedABI = integerEquals(prefix, 0) &&
      (integerEquals(bytes, 33554496) || integerEquals(bytes, 2097152));
  if (!bytes || !bytes.getType().isInteger(64) ||
      !prefix || !prefix.getType().isInteger(32) ||
      (!oldSourceABI && !finiteUnpaddedABI) || copy.getBase() != source)
    return fail("requires physical source33554496B/prefix16B or finite unpadded2MiB/(32MiB+64B) prefix0 ABI");
  if (oldSourceABI && *slot != 1)
    return fail("requires measured physical packet view1");
  auto ml = copy->getAttrOfType<MlsEncodingAttr>(MlsEncodingAttr::getMnemonic());
  if (!ml || ml.getVersion() != 0 || ml.getOpIdx() != 0 || ml.getElemBitWidth() != 16 ||
      ml.getAlt2Kind() != 0 || ml.getMlsTile() != ArrayRef<unsigned>({128, 32}) ||
      ml.getWarpsPerCTA() != ArrayRef<unsigned>({2, 2}) ||
      ml.getOrder() != ArrayRef<unsigned>({1, 0}) ||
      copy.getTensorShape() != ArrayRef<int32_t>({128, 32}) ||
      copy.getShape().size() != 2 || copy.getStrides().size() != 2 ||
      copy.getIndices().size() != 2 || !copy.getBoundaryCheck().empty() ||
      copy.getCache() != triton::CacheModifier::NONE ||
      copy.getEvict() != triton::EvictionPolicy::NORMAL || copy.getIsVolatile())
    return fail("requires exact indexed-A metadata/default memory semantics");
  for (unsigned d = 0; d < 2; ++d)
    if (legacyV0SafeInteger(copy.getShape()[d]) != 1024 ||
        legacyV0SafeInteger(copy.getStrides()[d]) != (d == 0 ? 1024 : 1))
      return fail("requires measured source shape/stride/index tuple");
  auto sourceRow = legacyV0SafeInteger(copy.getIndices()[0]);
  auto sourceColumn = legacyV0SafeInteger(copy.getIndices()[1]);
  if (oldSourceABI) {
    if (sourceRow != 0 || sourceColumn != 8)
      return fail("requires measured source shape/stride/index tuple");
  } else {
    if (!sourceRow || !sourceColumn ||
        !((*sourceRow == 0 && *sourceColumn == 0) ||
          (*sourceRow == 896 && *sourceColumn == 992)))
      return fail("requires physical source33554496B/prefix16B or finite unpadded full corners");
    // Descriptor stride[13:0]=2048B, bit14 is the cache/swizzle mode field (0x4800).
    // Full16B vectors stay within each record; no unchecked SOFFSET origin.
    uint64_t lastByte = 2ull * ((uint64_t(*sourceRow) + 127) * 1024 +
                               uint64_t(*sourceColumn) + 31) + 1;
    if (*sourceRow + 128 > 1024 || *sourceColumn + 32 > 1024 ||
        *sourceColumn % 8 || lastByte >= 2097152)
      return fail("finite indexed A source exceeds real2MiB tensor window");
  }
  if (!copy.getToken().hasOneUse()) return fail("producer token escapes");
  auto commit = dyn_cast<AsyncCommitGroupOp>(*copy.getToken().getUsers().begin());
  if (!commit || !commit.getResult().hasOneUse()) return fail("requires one commit");
  auto wait = dyn_cast<triton::hcugpu::AsyncWaitOp>(*commit.getResult().getUsers().begin());
  if (!wait || wait.getNumInst() != 0 || !wait.getResult().hasOneUse() ||
      read.getToken() != wait.getResult() ||
      copy->getNextNode() != commit.getOperation() || commit->getNextNode() != wait.getOperation() ||
      !isa_and_nonnull<LocalBarrierOp>(wait->getNextNode()) ||
      wait->getNextNode()->getNextNode() != read.getOperation())
    return fail("requires explicit fullwait0+CTA publication");
  if (store.getMask() || !store.getBoundaryCheck().empty() ||
      store.getCache() != triton::CacheModifier::NONE ||
      store.getEvict() != triton::EvictionPolicy::NORMAL)
    return fail("store requires full unmasked default memory semantics");
  if (!read.getResult().hasOneUse() || store.getValue() != read.getResult() ||
      !isLegacyV0GlobalTensorPointer(store.getPtr(), read.getType(), function))
    return fail("requires sole unmasked ordinary global roundtrip store");
  auto add = store.getPtr().getDefiningOp<triton::AddPtrOp>();
  auto splat = add.getPtr().getDefiningOp<triton::SplatOp>();
  if (splat.getSrc() != function.getArgument(1)) return fail("store must use output argument1");
  return success();
}

static bool isLegacyV0PureTLPointer(Value ptr, RankedTensorType valueTy,
                                    triton::FuncOp function, unsigned argNo) {
  if (!valueTy || valueTy.getRank() != 2 || function.getNumArguments() != 3 || argNo >= 3)
    return false;
  auto expectedArg = function.getArgument(argNo);
  auto scalar = dyn_cast<triton::PointerType>(expectedArg.getType());
  auto align = function.getArgAttrOfType<IntegerAttr>(argNo, "tt.divisibility");
  if (!scalar || scalar.getAddressSpace() != 1 ||
      scalar.getPointeeType() != valueTy.getElementType() || !align ||
      align.getInt() < 16 || align.getInt() % 16)
    return false;
  // The actual C3 pointer tree has AddPtr/Broadcast/AddPtr/Splat. In contrast,
  // old handwritten C fixtures have just AddPtr/Splat. Do not add IntToPtr,
  // pointer Select, function calls or arbitrary pointer-producing operations.
  std::function<bool(Value)> visit = [&](Value v) {
    auto ty = dyn_cast<RankedTensorType>(v.getType());
    if (!ty || ty.getRank() != 2 || ty.getEncoding() != valueTy.getEncoding() ||
        ty.getElementType() != expectedArg.getType())
      return false;
    for (unsigned d = 0; d < 2; ++d)
      if (ty.getShape()[d] != 1 && ty.getShape()[d] != valueTy.getShape()[d])
        return false;
    if (auto splat = v.getDefiningOp<triton::SplatOp>())
      return splat.getSrc() == expectedArg;
    if (auto broadcast = v.getDefiningOp<triton::BroadcastOp>())
      return visit(broadcast.getSrc());
    if (auto add = v.getDefiningOp<triton::AddPtrOp>()) {
      auto offTy = dyn_cast<RankedTensorType>(add.getOffset().getType());
      return offTy && offTy.getShape() == ty.getShape() &&
          offTy.getEncoding() == ty.getEncoding() &&
          (offTy.getElementType().isInteger(32) || offTy.getElementType().isInteger(64)) &&
          visit(add.getPtr());
    }
    return false;
  };
  auto finalTy = dyn_cast<RankedTensorType>(ptr.getType());
  return finalTy && finalTy.getShape() == valueTy.getShape() && visit(ptr);
}

// Existing FP32 output, or exactly one ordinary FP32->input-dtype truncf
// in the exit block. No general output expression or extra consumer is admitted.
static bool isLegacyV0LoopStore(triton::StoreOp store, Value accumulator,
                                 Type inputElement, Block *exit) {
  if (store.getValue() == accumulator)
    return true; // Preserve the existing FP32 store path verbatim.
  auto trunc = store.getValue().getDefiningOp<arith::TruncFOp>();
  auto accTy = dyn_cast<RankedTensorType>(accumulator.getType());
  auto outTy = dyn_cast<RankedTensorType>(store.getValue().getType());
  if (!trunc || trunc.getIn() != accumulator || !trunc.getOut().hasOneUse() ||
      trunc->getBlock() != exit || store->getBlock() != exit ||
      !trunc->isBeforeInBlock(store.getOperation()) || !accTy || !outTy ||
      !accTy.getElementType().isF32() ||
      !(inputElement.isF16() || inputElement.isBF16()) ||
      outTy.getElementType() != inputElement ||
      outTy.getShape() != accTy.getShape() || outTy.getEncoding() != accTy.getEncoding())
    return false;
  unsigned truncCount = 0;
  store->getParentOfType<ModuleOp>().walk([&](arith::TruncFOp) { ++truncCount; });
  return truncCount == 1;
}

// B packet/source/publication validation. Only the private pure-TL branch may
// additionally admit this exact ordinary A staging chain.
struct LegacyV0StagedAProtocol {
  triton::LoadOp globalA;
  LocalAllocOp allocA;
  LocalLoadOp localA;
  triton::DotOp dot;
  triton::StoreOp store;
};

static std::optional<LegacyV0StagedAProtocol> matchLegacyV0StagedA(
    ModuleOp module, triton::FuncOp function, LocalAllocOp rootB,
    LocalLoadOp localB, const LegacyV0Packet &packet,
    const LegacyV0CountedBody *loop = nullptr) {
  auto enabled = module->getAttrOfType<IntegerAttr>("ttg.hcu.v0_blockptr");
  if (!enabled || enabled.getInt() != 1 || (!loop && !function.getBody().hasOneBlock()) ||
      rootB.getType().getShape()[0] != 1)
    return std::nullopt;
  SmallVector<triton::LoadOp> loads;
  SmallVector<triton::StoreOp> stores;
  SmallVector<triton::DotOp> dots;
  SmallVector<LocalAllocOp> allocs;
  SmallVector<LocalLoadOp> locals;
  module.walk([&](Operation *op) {
    if (auto x = dyn_cast<triton::LoadOp>(op)) loads.push_back(x);
    if (auto x = dyn_cast<triton::StoreOp>(op)) stores.push_back(x);
    if (auto x = dyn_cast<triton::DotOp>(op)) dots.push_back(x);
    if (auto x = dyn_cast<LocalAllocOp>(op)) allocs.push_back(x);
    if (auto x = dyn_cast<LocalLoadOp>(op)) locals.push_back(x);
  });
  if (loads.size() != 1 || stores.size() != 1 || dots.size() != 1 ||
      allocs.size() != 2 || locals.size() != 2)
    return std::nullopt;
  auto load = loads.front();
  auto store = stores.front();
  auto dot = dots.front();
  auto localA = dot.getA().getDefiningOp<LocalLoadOp>();
  auto allocA = localA ? localA.getSrc().getDefiningOp<LocalAllocOp>() : LocalAllocOp{};
  if (!allocA || allocA == rootB || !localA || localA == localB ||
      allocA.getSrc() != load.getResult() ||
      !load.getResult().hasOneUse() || !allocA.getResult().hasOneUse() ||
      !localA.getResult().hasOneUse() || localA.getToken() ||
      !localB.getResult().hasOneUse() || dot.getB() != localB.getResult() ||
      !dot.getD().hasOneUse() || (!loop && store.getValue() != dot.getD()))
    return std::nullopt;
  unsigned m = 16 * (packet.warps / 2);
  auto aTy = localA.getType();
  if (loop && aTy.getRank() == 2 && aTy.getShape()[0] == 4 * m) {
    auto joint = module->getAttrOfType<IntegerAttr>("ttg.hcu.v0_joint_phase");
    if (!joint || joint.getInt() != 1)
      return std::nullopt;
    // Only M128/K32/N128/W4 or M256/K16/N256/W8, inherited from packet.
    // Keep every actual A shared-layout, B ownership, SSA and effect guard.
    m *= 4;
  }
  auto gTy = dyn_cast<RankedTensorType>(load.getType());
  auto bEnc = cast<DotOperandEncodingAttr>(localB.getType().getEncoding());
  auto aEnc = dyn_cast<DotOperandEncodingAttr>(aTy.getEncoding());
  auto mma = bEnc.getParent();
  auto mem = allocA.getType();
  auto swizzle = dyn_cast<SwizzledSharedEncodingAttr>(mem.getEncoding());
  auto cTy = dot.getType();
  if (!gTy || aTy.getShape() != ArrayRef<int64_t>({m, packet.k}) ||
      aTy.getElementType() != localB.getType().getElementType() ||
      gTy.getShape() != aTy.getShape() || gTy.getElementType() != aTy.getElementType() ||
      !isa<BlockedEncodingAttr>(gTy.getEncoding()) || !aEnc ||
      aEnc.getOpIdx() != 0 || aEnc.getParent() != mma || aEnc.getKWidth() != 4 ||
      mem.getRank() != 2 || mem.getShape() != aTy.getShape() ||
      mem.getAllocShape() != mem.getShape() || mem.getElementType() != aTy.getElementType() ||
      mem.getMutableMemory() || allocA.getAlignmentOrDefault() < 16 ||
      allocA.getAlignmentOrDefault() % 16 || !isa<SharedMemorySpaceAttr>(mem.getMemorySpace()) ||
      !swizzle || swizzle.getVec() != 4 ||
      swizzle.getPerPhase() != (packet.k == 32 ? 2 : 4) ||
      swizzle.getMaxPhase() != (packet.k == 32 ? 8 : 4) ||
      swizzle.getOrder() != ArrayRef<unsigned>({1, 0}) ||
      !llvm::all_of(getCTAsPerCGA(swizzle), [](unsigned n) { return n == 1; }) ||
      !llvm::all_of(getCTASplitNum(swizzle), [](unsigned n) { return n == 1; }) ||
      cTy.getShape() != ArrayRef<int64_t>({m, packet.n}) ||
      !cTy.getElementType().isF32() || cTy.getEncoding() != mma ||
      dot.getInputPrecision() != triton::InputPrecision::IEEE ||
      dot.getMaxNumImpreciseAcc() != 0)
    return std::nullopt;
  if (load.getMask() || load.getOther() || !load.getBoundaryCheck().empty() ||
      load.getIsVolatile() || load.getCache() != triton::CacheModifier::NONE ||
      load.getEvict() != triton::EvictionPolicy::NORMAL ||
      store.getMask() || !store.getBoundaryCheck().empty() ||
      store.getCache() != triton::CacheModifier::NONE ||
      store.getEvict() != triton::EvictionPolicy::NORMAL ||
      !isLegacyV0PureTLPointer(load.getPtr(), gTy, function, 0) ||
      !isLegacyV0PureTLPointer(store.getPtr(),
          loop ? dyn_cast<RankedTensorType>(store.getValue().getType()) : cTy, function, 2))
    return std::nullopt;
  Block *body = loop ? loop->body : &function.getBody().front();
  Value initial = dot.getC();
  if (loop) {
    auto back = cast<cf::BranchOp>(body->getTerminator());
    auto *header = back.getDest();
    auto condition = cast<cf::CondBranchOp>(header->getTerminator());
    auto *exit = condition.getFalseDest();
    auto enter = cast<cf::BranchOp>(function.getBody().front().getTerminator());
    initial = enter.getDestOperands()[1];
    if (rootB->getBlock() != body || store->getBlock() != exit ||
        dot.getC() != header->getArgument(1) || back.getDestOperands()[1] != dot.getD() ||
        !isLegacyV0LoopStore(store, header->getArgument(1),
                             localB.getType().getElementType(), exit) ||
        std::distance(header->getArgument(1).use_begin(), header->getArgument(1).use_end()) != 2)
      return std::nullopt;
  } else if (store->getBlock() != body) {
    return std::nullopt;
  }
  auto zero = initial.getDefiningOp<arith::ConstantOp>();
  auto dense = zero ? dyn_cast<DenseFPElementsAttr>(zero.getValue()) : DenseFPElementsAttr{};
  if (!dense || !dense.isSplat() ||
      !dense.getSplatValue<llvm::APFloat>().isZero() || initial.getType() != cTy ||
      dot.getC().getType() != cTy)
    return std::nullopt;
  for (Operation *op : {load.getOperation(), allocA.getOperation(), localA.getOperation(),
                         localB.getOperation(), dot.getOperation()})
    if (op->getBlock() != body)
      return std::nullopt;
  // Original one-shot and counted-loop arithmetic retain ordinary semantics.
  // All intervening ops still pass the closed whitelist and exact memory counts.
  if (!load->isBeforeInBlock(allocA.getOperation()) || !allocA->isBeforeInBlock(localA.getOperation()) ||
      !localA->isBeforeInBlock(dot.getOperation()) || !localB->isBeforeInBlock(dot.getOperation()) ||
      (!loop && !dot->isBeforeInBlock(store.getOperation())))
    return std::nullopt;
  return LegacyV0StagedAProtocol{load, allocA, localA, dot, store};
}

static LogicalResult verifyLegacyV0ValueUses(ModuleOp module, LocalLoadOp bLoad,
    const LegacyV0Packet &packet, triton::FuncOp function) {
  auto fail = [&](StringRef message) { return module.emitError(message); };
  SmallVector<triton::LoadOp> loads;
  SmallVector<triton::DotOp> dots;
  SmallVector<triton::StoreOp> stores;
  module.walk([&](Operation *op) {
    if (auto load = dyn_cast<triton::LoadOp>(op)) loads.push_back(load);
    if (auto dot = dyn_cast<triton::DotOp>(op)) dots.push_back(dot);
    if (auto store = dyn_cast<triton::StoreOp>(op)) stores.push_back(store);
  });
  if (stores.size() != 1)
    return fail("legacy-v0 value protocol needs exactly one global store");
  auto store = stores.front();
  if (store.getMask() || !store.getBoundaryCheck().empty() ||
      store.getCache() != triton::CacheModifier::NONE ||
      store.getEvict() != triton::EvictionPolicy::NORMAL)
    return fail("legacy-v0 store requires full unmasked default memory semantics");
  auto output = dyn_cast<RankedTensorType>(store.getValue().getType());
  if (!output)
    return fail("legacy-v0 value protocol requires tensor output");
  if (!isLegacyV0GlobalTensorPointer(store.getPtr(), output, function))
    return fail("legacy-v0 output pointer must be direct aligned AS1 splat+addptr in value layout");
  if (dots.empty()) {
    if (!loads.empty() || !bLoad.getResult().hasOneUse() ||
        store.getValue() != bLoad.getResult())
      return fail("legacy-v0 roundtrip requires direct sole B-to-store value use");
    return success();
  }
  if (dots.size() != 1 || loads.size() != 1)
    return fail("legacy-v0 dot protocol requires exactly one global A load and one dot");
  auto aLoad = loads.front();
  auto dot = dots.front();
  auto bTy = bLoad.getType();
  auto bEnc = cast<DotOperandEncodingAttr>(bTy.getEncoding());
  auto mma = cast<AMDMfmaEncodingAttr>(bEnc.getParent());
  auto aTy = dyn_cast<RankedTensorType>(aLoad.getType());
  if (!aTy)
    return fail("legacy-v0 A load requires a ranked tensor");
  auto aEnc = dyn_cast<DotOperandEncodingAttr>(aTy.getEncoding());
  auto cTy = dot.getType();
  unsigned m = 16 * (packet.warps / 2);
  if (aTy.getRank() != 2 ||
      aTy.getShape() != ArrayRef<int64_t>({m, packet.k}) ||
      aTy.getElementType() != bTy.getElementType() || !aEnc ||
      aEnc.getOpIdx() != 0 || aEnc.getParent() != mma || aEnc.getKWidth() != 4 ||
      cTy.getRank() != 2 || cTy.getShape() != ArrayRef<int64_t>({m, packet.n}) ||
      !cTy.getElementType().isF32() || cTy.getEncoding() != mma ||
      dot.getInputPrecision() != triton::InputPrecision::IEEE ||
      dot.getMaxNumImpreciseAcc() != 0)
    return fail("legacy-v0 dot requires exact compatible A-dot and MMA-C shapes/types");
  if (aLoad.getMask() || aLoad.getOther() || !aLoad.getBoundaryCheck().empty() ||
      aLoad.getCache() != triton::CacheModifier::NONE ||
      aLoad.getEvict() != triton::EvictionPolicy::NORMAL || aLoad.getIsVolatile() ||
      !isLegacyV0GlobalTensorPointer(aLoad.getPtr(), aTy, function))
    return fail("legacy-v0 A load must be direct aligned full AS1/default load");
  auto zero = dot.getC().getDefiningOp<arith::ConstantOp>();
  auto dense = zero ? dyn_cast<DenseFPElementsAttr>(zero.getValue())
                    : DenseFPElementsAttr{};
  if (!dense || !dense.isSplat() || !dense.getSplatValue<llvm::APFloat>().isZero() ||
      dot.getC().getType() != cTy)
    return fail("legacy-v0 one-shot dot requires exact f32 zero accumulator");
  if (dot.getA() != aLoad.getResult() || dot.getB() != bLoad.getResult() ||
      !aLoad.getResult().hasOneUse() || !bLoad.getResult().hasOneUse() ||
      !dot.getD().hasOneUse() || store.getValue() != dot.getD() ||
      bLoad->getNextNode() != aLoad.getOperation() ||
      aLoad->getNextNode() != dot.getOperation() ||
      dot->getNextNode() != store.getOperation())
    return fail("legacy-v0 requires the exact B-read/A-load/dot/C-store SSA chain");
  return success();
}

// Complete closed static-loop verifier. This is separate from
// pure-TL staged A: direct global A in ordinary DotOperand layout only.
static LogicalResult verifyLegacyV0StaticLoop(ModuleOp mod) {
  auto fail = [&](StringRef message) { return mod.emitError(message); };
  auto pure = mod->getAttrOfType<IntegerAttr>("ttg.hcu.v0_blockptr");
  auto pureLoopAttr = mod->getAttrOfType<IntegerAttr>("ttg.hcu.v0_blockptr_loop");
  const bool pureTL = pureLoopAttr && pureLoopAttr.getInt() == 1;
  if (pureTL && (!pure || pure.getInt() != 1))
    return fail("legacy-v0 pure loop requires the existing blockptr experiment");
  if (!pureTL && pure && pure.getInt() != 0)
    return fail("legacy-v0 static loop does not admit staged-A pure-TL mode");
  SmallVector<triton::FuncOp> functions;
  SmallVector<LocalAllocOp> allocs;
  SmallVector<MemDescIndexOp> views;
  SmallVector<triton::hcugpu::MatrixLoadToLocalOp> copies;
  SmallVector<LocalLoadOp> localLoads;
  SmallVector<triton::LoadOp> globalLoads;
  SmallVector<triton::DotOp> dots;
  SmallVector<triton::StoreOp> stores;
  SmallVector<AsyncCommitGroupOp> commits;
  SmallVector<triton::hcugpu::AsyncWaitOp> waits;
  SmallVector<LocalBarrierOp> barriers;
  bool forbidden = false;
  unsigned branches = 0, conditions = 0, returns = 0;
  mod.walk([&](Operation *op) {
    auto name = op->getName().getStringRef();
    if (!name.starts_with("arith.") && !llvm::is_contained(
        ArrayRef<StringRef>({"builtin.module", "tt.func", "tt.return",
          "tt.get_program_id", "tt.make_range", "tt.expand_dims", "tt.broadcast", "tt.splat",
          "tt.addptr", "tt.load", "tt.store", "tt.dot", "ttg.local_alloc",
          "ttg.memdesc_index", "ttg.local_load", "ttg.async_commit_group",
          "hcug.async_wait", "ttg.local_barrier", "hcug.matrix_load_to_local",
          "cf.br", "cf.cond_br"}), name))
      forbidden = true;
    if (op->getNumRegions() && op != mod.getOperation() && !isa<triton::FuncOp>(op))
      forbidden = true;
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          forbidden |= isa<MemDescType>(arg.getType());
    if (auto x = dyn_cast<triton::FuncOp>(op)) functions.push_back(x);
    if (auto x = dyn_cast<LocalAllocOp>(op)) allocs.push_back(x);
    if (auto x = dyn_cast<MemDescIndexOp>(op)) views.push_back(x);
    if (auto x = dyn_cast<triton::hcugpu::MatrixLoadToLocalOp>(op)) copies.push_back(x);
    if (auto x = dyn_cast<LocalLoadOp>(op)) localLoads.push_back(x);
    if (auto x = dyn_cast<triton::LoadOp>(op)) globalLoads.push_back(x);
    if (auto x = dyn_cast<triton::DotOp>(op)) dots.push_back(x);
    if (auto x = dyn_cast<triton::StoreOp>(op)) stores.push_back(x);
    if (auto x = dyn_cast<AsyncCommitGroupOp>(op)) commits.push_back(x);
    if (auto x = dyn_cast<triton::hcugpu::AsyncWaitOp>(op)) waits.push_back(x);
    if (auto x = dyn_cast<LocalBarrierOp>(op)) barriers.push_back(x);
    branches += isa<cf::BranchOp>(op);
    conditions += isa<cf::CondBranchOp>(op);
    returns += isa<triton::ReturnOp>(op);
  });
  if (forbidden || functions.size() != 1 || allocs.size() != (pureTL ? 2 : 1) ||
      views.size() != 1 || copies.size() != 1 || localLoads.size() != (pureTL ? 2 : 1) ||
      globalLoads.size() != 1 || dots.size() != 1 || stores.size() != 1 ||
      commits.size() != 1 || waits.size() != 1 || barriers.size() != 2 ||
      branches != 2 || conditions != 1 || returns != 1)
    return fail("legacy-v0 static loop requires its exact closed operation set");
  auto fn = functions.front();
  auto view = views.front(); auto copy = copies.front();
  auto root = view.getSrc().getDefiningOp<LocalAllocOp>();
  LocalLoadOp localB;
  for (auto candidate : localLoads)
    if (candidate.getSrc() == view.getResult()) {
      if (localB)
        return fail("legacy-v0 loop requires unique paired B local read");
      localB = candidate;
    }
  if (!root || !localB)
    return fail("legacy-v0 loop requires direct B root/view/read ownership");
  auto aLoad = globalLoads.front(); auto dot = dots.front();
  auto store = stores.front(); auto commit = commits.front(); auto wait = waits.front();
  auto rt = root.getType();
  auto packet = getLegacyV0Packet(rt.getEncoding());
  auto nw = mod->getAttrOfType<IntegerAttr>("ttg.num-warps");
  if (!packet || !nw || nw.getInt() != packet->warps ||
      rt.getRank() != 3 || rt.getShape() != ArrayRef<int64_t>({1, packet->k, packet->n}) ||
      rt.getAllocShape() != rt.getShape() ||
      !(rt.getElementType().isF16() || rt.getElementType().isBF16()) ||
      !rt.getMutableMemory() || !root.isSharedMemoryAlloc() || root.getSrc() ||
      root.getAlignmentOrDefault() < 16 || root.getAlignmentOrDefault() % 16 ||
      !root.getResult().hasOneUse() || view.getSrc() != root.getResult() ||
      legacyV0SafeInteger(view.getIndex()) != 0)
    return fail("legacy-v0 static loop requires the exact8KiB root and slot0");
  auto vt = view.getType();
  if (vt.getRank() != 2 || vt.getShape() != ArrayRef<int64_t>({packet->k, packet->n}) ||
      vt.getAllocShape() != vt.getShape() || vt.getEncoding() != rt.getEncoding() ||
      vt.getElementType() != rt.getElementType() || vt.getMemorySpace() != rt.getMemorySpace() ||
      !vt.getMutableMemory() || !isa<SharedMemorySpaceAttr>(rt.getMemorySpace()) ||
      std::distance(view.getResult().use_begin(), view.getResult().use_end()) != 2 ||
      copy.getDest() != view.getResult() || localB.getSrc() != view.getResult() ||
      !isLegacyV0Dot(localB.getType(), *packet))
    return fail("legacy-v0 static loop requires exact paired opaque B ownership");
  auto source = getLegacyV0SourceWindow(copy, *packet);
  auto loop = source ? getLegacyV0CountedBody(copy, source->totalK / packet->k)
                     : std::nullopt;
  if (!loop || !source)
    return fail("legacy-v0 static loop requires proved IV*BK rectangular bounds");
  auto row = copy.getIndices()[0].getDefiningOp<arith::MulIOp>();
  if (!row || row.getLhs() != loop->iv || legacyV0SafeInteger(row.getRhs()) != packet->k)
    return fail("legacy-v0 static loop requires exact IV*BK source row");
  Block *body = loop->body;
  auto back = cast<cf::BranchOp>(body->getTerminator());
  Block *header = back.getDest();
  auto condition = cast<cf::CondBranchOp>(header->getTerminator());
  Block *exit = condition.getFalseDest();
  Block *entry = &fn.getBody().front();
  auto enter = cast<cf::BranchOp>(entry->getTerminator());
  Block *frameBlock = pureTL ? body : entry;
  if (root->getBlock() != frameBlock || view->getBlock() != frameBlock ||
      !root->isBeforeInBlock(view.getOperation()) ||
      (pureTL && !view->isBeforeInBlock(copy.getOperation())) || fn.getNumArguments() != 3 ||
      fn.getFunctionType().getNumResults() != 0 ||
      store->getBlock() != exit || !isa<triton::ReturnOp>(exit->getTerminator()))
    return fail("legacy-v0 static loop requires exact entry/frame and exit/store roles");
  for (Operation *op : {copy.getOperation(), localB.getOperation(), aLoad.getOperation(),
                        dot.getOperation(), commit.getOperation(), wait.getOperation(),
                        barriers[0].getOperation(), barriers[1].getOperation()})
    if (op->getBlock() != body)
      return fail("legacy-v0 static loop memory and compute must be in its single body");
  for (unsigned i = 0; i < 3; ++i) {
    auto pointer = dyn_cast<triton::PointerType>(fn.getArgument(i).getType());
    auto alignment = fn.getArgAttrOfType<IntegerAttr>(i, "tt.divisibility");
    if (!pointer || pointer.getAddressSpace() != 1 || !alignment ||
        alignment.getInt() < 16 || alignment.getInt() % 16 ||
        (i < 2 ? pointer.getPointeeType() != rt.getElementType()
               : (!pointer.getPointeeType().isF32() &&
                  !(pureTL && pointer.getPointeeType() == rt.getElementType()))))
      return fail("legacy-v0 static loop requires aligned typed A/B/C AS1 arguments");
  }
  auto ml = copy->getAttrOfType<triton::hcugpu::MlsEncodingAttr>(
      triton::hcugpu::MlsEncodingAttr::getMnemonic());
  if (!ml || ml.getVersion() != 0 || ml.getOpIdx() != 1 ||
      ml.getElemBitWidth() != 16 || ml.getAlt2Kind() != 0 ||
      ml.getMlsTile() != ArrayRef<unsigned>({packet->k, packet->n}) ||
      ml.getOrder() != ArrayRef<unsigned>({1, 0}) ||
      ml.getWarpsPerCTA() != ArrayRef<unsigned>({packet->warps / 2, 2}) ||
      copy.getTensorShape() != ArrayRef<int32_t>({int32_t(packet->k), int32_t(packet->n)}) ||
      copy.getBase() != fn.getArgument(1) || !copy.getBoundaryCheck().empty() ||
      copy.getCache() != triton::CacheModifier::NONE ||
      copy.getEvict() != triton::EvictionPolicy::NORMAL || copy.getIsVolatile())
    return fail("legacy-v0 static loop requires unchanged proved B producer metadata");
  if (pureTL) {
    if (!matchLegacyV0StagedA(mod, fn, root, localB, *packet, &*loop))
      return fail("legacy-v0 pure loop requires exact ordinary staged A and counted accumulator/C-store protocol");
  } else {
    auto aTy = dyn_cast<RankedTensorType>(aLoad.getType());
    auto bTy = localB.getType();
    auto mma = cast<DotOperandEncodingAttr>(bTy.getEncoding()).getParent();
    auto aEnc = aTy ? dyn_cast<DotOperandEncodingAttr>(aTy.getEncoding()) : DotOperandEncodingAttr{};
    auto cTy = dot.getType();
    unsigned m = 16 * (packet->warps / 2);
    if (!aTy || aTy.getRank() != 2 ||
        aTy.getShape() != ArrayRef<int64_t>({m, packet->k}) ||
        aTy.getElementType() != bTy.getElementType() || !aEnc ||
        aEnc.getOpIdx() != 0 || aEnc.getParent() != mma || aEnc.getKWidth() != 4 ||
        cTy.getShape() != ArrayRef<int64_t>({m, packet->n}) ||
        !cTy.getElementType().isF32() || cTy.getEncoding() != mma ||
        dot.getInputPrecision() != triton::InputPrecision::IEEE || dot.getMaxNumImpreciseAcc() != 0 ||
        aLoad.getMask() || aLoad.getOther() || !aLoad.getBoundaryCheck().empty() ||
        aLoad.getCache() != triton::CacheModifier::NONE ||
        aLoad.getEvict() != triton::EvictionPolicy::NORMAL || aLoad.getIsVolatile() ||
        store.getMask() || !store.getBoundaryCheck().empty() ||
        store.getCache() != triton::CacheModifier::NONE ||
        store.getEvict() != triton::EvictionPolicy::NORMAL ||
        !isLegacyV0GlobalTensorPointer(aLoad.getPtr(), aTy, fn) ||
        !isLegacyV0GlobalTensorPointer(store.getPtr(), cTy, fn))
      return fail("legacy-v0 static loop requires ordinary direct A-dot and C-store types");
    auto pointerBase = [](Value ptr) {
      auto add = ptr.getDefiningOp<triton::AddPtrOp>();
      auto splat = add ? add.getPtr().getDefiningOp<triton::SplatOp>() : triton::SplatOp{};
      return splat ? splat.getSrc() : Value{};
    };
    if (pointerBase(aLoad.getPtr()) != fn.getArgument(0) ||
        pointerBase(store.getPtr()) != fn.getArgument(2))
      return fail("legacy-v0 static loop requires exact A/C global pointer roots");
    auto zero = enter.getDestOperands()[1].getDefiningOp<arith::ConstantOp>();
    auto dense = zero ? dyn_cast<DenseFPElementsAttr>(zero.getValue()) : DenseFPElementsAttr{};
    if (!dense || !dense.isSplat() || !dense.getSplatValue<llvm::APFloat>().isZero() ||
        enter.getDestOperands()[1].getType() != cTy || header->getArgument(1).getType() != cTy ||
        dot.getC() != header->getArgument(1) || back.getDestOperands()[1] != dot.getD() ||
        dot.getA() != aLoad.getResult() || dot.getB() != localB.getResult() ||
        !aLoad.getResult().hasOneUse() || !localB.getResult().hasOneUse() ||
        !dot.getD().hasOneUse() || store.getValue() != header->getArgument(1) ||
        std::distance(header->getArgument(1).use_begin(),
                      header->getArgument(1).use_end()) != 2)
      return fail("legacy-v0 static loop requires exact zero/accumulator/backedge/output SSA chain");
  }
  if (!copy.getToken().hasOneUse() || commit.getOperands().size() != 1 ||
      commit.getOperands()[0] != copy.getToken() || !commit.getResult().hasOneUse() ||
      wait.getOperands().size() != 1 || wait.getOperands()[0] != commit.getResult() ||
      wait.getNumInst() != 0 || !wait.getResult().hasOneUse() ||
      localB.getToken() != wait.getResult() ||
      copy->getNextNode() != commit.getOperation() || commit->getNextNode() != wait.getOperation() ||
      !isa_and_nonnull<LocalBarrierOp>(wait->getNextNode()) ||
      wait->getNextNode()->getNextNode() != localB.getOperation() ||
      (!pureTL && (localB->getNextNode() != aLoad.getOperation() ||
                   aLoad->getNextNode() != dot.getOperation())))
    return fail("legacy-v0 static loop requires fullwait0/publication/read/A/dot ordering");
  auto publication = cast<LocalBarrierOp>(wait->getNextNode());
  auto war = barriers[0] == publication ? barriers[1] : barriers[0];
  if (!war->isBeforeInBlock(copy.getOperation()))
    return fail("legacy-v0 static loop requires explicit head WAR completion before slot reuse");
  // All operations/effects are enumerated. Entry has the sole frame/view; body
  // has all loads/dot and both barriers; exit has the sole global store. Thus
  // the known view cannot escape and no unknown op can allocate hidden LDS.
  // Ordinary A/C offsets retain ordinary Triton host bounds obligations.
  return success();
}

} // namespace

#include "WholePacketLowering.inc"
#include "StructuredRegisterLoad.inc"
#include "RawWindowLowering.inc"

namespace mlir::triton::HCU {
// Closed one-shot IR only. Reject invalid manual v0 before allocation/membar;
// never silently fall through to v1 or generic shared-layout conversions.
// Local primitive contract for the existing two-stage MLS ring. Ordinary A,
// accumulation and output operations retain their normal verifier/lowering.
// No membar filter or asynchronous alias annotation is introduced here.
static LogicalResult verifyLegacyV0Prefix2(
    ModuleOp mod, LegacyV0Prefix2Queue *witness = nullptr, bool diagnose = true) {
  auto fail = [&](StringRef why) -> LogicalResult {
    if (diagnose)
      mod.emitError("legacy-v0 prefix2: ") << why;
    return failure();
  };
  SmallVector<LocalAllocOp> roots;
  SmallVector<triton::hcugpu::MatrixLoadToLocalOp> copies;
  mod.walk([&](Operation *op) {
    if (auto a = dyn_cast<LocalAllocOp>(op))
      if (auto e = dyn_cast<HCUMlsSharedEncodingAttr>(a.getType().getEncoding()))
        if (e.getVersion() == 0) roots.push_back(a);
    if (auto c = dyn_cast<triton::hcugpu::MatrixLoadToLocalOp>(op)) copies.push_back(c);
  });
  if (roots.size() != 1 || copies.size() != 2)
    return fail("one opaque root and prologue/body copies required");
  auto pure = mod->getAttrOfType<IntegerAttr>("ttg.hcu.v0_blockptr_loop");
  auto joint = mod->getAttrOfType<IntegerAttr>("ttg.hcu.v0_joint_phase");
  if (!pure || pure.getInt() != 1 || (joint && joint.getInt() != 0))
    return fail("existing block-pointer loop mode, no single-slot joint relocation");
  auto root = roots.front();
  auto rt = root.getType();
  auto packet = getLegacyV0Packet(rt.getEncoding());
  auto fn = root->getParentOfType<triton::FuncOp>();
  auto warps = mod->getAttrOfType<IntegerAttr>("ttg.num-warps");
  if (!packet || !fn || !warps || root->getBlock() != &fn.getBody().front() ||
      warps.getInt() != packet->warps ||
      rt.getShape() != ArrayRef<int64_t>({2, packet->k, packet->n}) ||
      rt.getAllocShape() != rt.getShape() || !rt.getMutableMemory() ||
      !(rt.getElementType().isF16() || rt.getElementType().isBF16()) ||
      !root.isSharedMemoryAlloc() || root.getSrc() ||
      root.getAlignmentOrDefault() < 16 || root.getAlignmentOrDefault() % 16)
    return fail("exact two-slot mutable16-bit packet root required");
  auto entry = &fn.getBody().front();
  triton::hcugpu::MatrixLoadToLocalOp initial, future;
  for (auto c : copies) (c->getBlock() == entry ? initial : future) = c;
  if (!initial || !future) return fail("copy block roles");
  auto source = getLegacyV0SourceWindow(future, *packet, true);
  auto loop = source ? getLegacyV0CountedBody(future, source->totalK / packet->k - 1, 0)
                     : std::nullopt;
  if (!loop) return fail("positive static source bounds and loop queues");
  auto row = future.getIndices()[0].getDefiningOp<arith::MulIOp>();
  auto next = row ? row.getLhs().getDefiningOp<arith::AddIOp>() : arith::AddIOp{};
  if (!row || !next || next.getLhs() != loop->iv ||
      legacyV0SafeInteger(next.getRhs()) != 1 || legacyV0SafeInteger(row.getRhs()) != packet->k)
    return fail("positive static source bounds and loop queues");
  Block *body = loop->body;
  auto back = cast<cf::BranchOp>(body->getTerminator());
  Block *header = back.getDest();
  auto enter = cast<cf::BranchOp>(entry->getTerminator());
  Block *exit = cast<cf::CondBranchOp>(header->getTerminator()).getFalseDest();
  auto initialView = initial.getDest().getDefiningOp<MemDescIndexOp>();
  auto futureView = future.getDest().getDefiningOp<MemDescIndexOp>();
  if (!initialView || !futureView || initialView->getBlock() != entry ||
      futureView->getBlock() != body || initialView.getSrc() != root.getResult() ||
      futureView.getSrc() != root.getResult() ||
      legacyV0SafeInteger(initialView.getIndex()) != 0)
    return fail("anchored prefix views and counter queue");
  auto select = futureView.getIndex().getDefiningOp<arith::SelectOp>();
  auto inc = select ? select.getTrueValue().getDefiningOp<arith::AddIOp>() : arith::AddIOp{};
  auto cmp = select ? select.getCondition().getDefiningOp<arith::CmpIOp>() : arith::CmpIOp{};
  auto counter = inc ? dyn_cast<BlockArgument>(inc.getLhs()) : BlockArgument{};
  if (!select || !inc || !cmp || !counter || counter.getOwner() != header ||
      !counter.getType().isInteger(32) || select->getBlock() != body ||
      legacyV0SafeInteger(enter.getDestOperands()[counter.getArgNumber()]) != 0 ||
      back.getDestOperands()[counter.getArgNumber()] != futureView.getIndex() ||
      legacyV0SafeInteger(inc.getRhs()) != 1 ||
      cmp.getLhs() != inc.getResult() || legacyV0SafeInteger(cmp.getRhs()) != 2 ||
      cmp.getPredicate() != arith::CmpIPredicate::slt ||
      legacyV0SafeInteger(select.getFalseValue()) != 0)
    return fail("uniform modulo2 prefix index required");
  auto vt = initialView.getType();
  if (vt.getShape() != ArrayRef<int64_t>({packet->k, packet->n}) ||
      vt.getAllocShape() != vt.getShape() || vt.getEncoding() != rt.getEncoding() ||
      vt.getElementType() != rt.getElementType() || vt.getMemorySpace() != rt.getMemorySpace() ||
      !vt.getMutableMemory() || !isa<SharedMemorySpaceAttr>(rt.getMemorySpace()) ||
      futureView.getType() != vt)
    return fail("whole identical physical packet views required");
  if (initial.getIndices().size() != 2 || initial.getShape().size() != 2 ||
      initial.getStrides().size() != 2 ||
      legacyV0SafeInteger(initial.getIndices()[0]) != 0 ||
      initial.getBase() != future.getBase() || !llvm::equal(initial.getShape(), future.getShape()) ||
      !llvm::equal(initial.getStrides(), future.getStrides()) ||
      initial.getIndices()[1] != future.getIndices()[1])
    return fail("same bounded source with row0 prologue required");
  auto base = dyn_cast<BlockArgument>(future.getBase());
  auto ptr = dyn_cast<triton::PointerType>(future.getBase().getType());
  auto alignment = base && base.getOwner() == entry
      ? fn.getArgAttrOfType<IntegerAttr>(base.getArgNumber(), "tt.divisibility") : IntegerAttr{};
  if (!base || !ptr || ptr.getAddressSpace() != 1 || ptr.getPointeeType() != rt.getElementType() ||
      !alignment || alignment.getInt() < 16 || alignment.getInt() % 16)
    return fail("aligned typed AS1 source argument required");
  SmallVector<Operation *> commits;
  for (auto copy : {initial, future}) {
    auto ml = copy->getAttrOfType<triton::hcugpu::MlsEncodingAttr>(
        triton::hcugpu::MlsEncodingAttr::getMnemonic());
    auto commit = copy.getToken().hasOneUse()
        ? dyn_cast<AsyncCommitGroupOp>(*copy.getToken().getUsers().begin()) : AsyncCommitGroupOp{};
    if (!ml || ml.getVersion() != 0 || ml.getOpIdx() != 1 || ml.getAlt2Kind() != 0 ||
        ml.getElemBitWidth() != 16 || ml.getOrder() != ArrayRef<unsigned>({1, 0}) ||
        ml.getMlsTile() != ArrayRef<unsigned>({packet->k, packet->n}) ||
        ml.getWarpsPerCTA() != ArrayRef<unsigned>({packet->warps / 2, 2}) ||
        copy.getTensorShape() != ArrayRef<int32_t>({int32_t(packet->k), int32_t(packet->n)}) ||
        !copy.getBoundaryCheck().empty() || copy.getCache() != triton::CacheModifier::NONE ||
        copy.getEvict() != triton::EvictionPolicy::NORMAL || copy.getIsVolatile() ||
        !commit || commit.getOperands().size() != 1 ||
        copy->getNextNode() != commit.getOperation() || !commit.getResult().hasOneUse())
      return fail("raw producer metadata and sole commit required");
    commits.push_back(commit.getOperation());
  }
  // Discover B's queue by its actual initial/future commits, not A/B argument
  // ordering. Ordinary A may introduce another token and a combined fullwait.
  BlockArgument bToken;
  for (unsigned i = 0; i < header->getNumArguments(); ++i)
    if (enter.getDestOperands()[i] == commits[0]->getResult(0) &&
        back.getDestOperands()[i] == commits[1]->getResult(0)) {
      if (bToken) return fail("ambiguous current B token queue");
      bToken = header->getArgument(i);
    }
  if (!bToken) return fail("current token queue required");
  SmallVector<LocalLoadOp> reads;
  SmallVector<triton::hcugpu::AsyncWaitOp> waits;
  mod.walk([&](LocalLoadOp load) { if (load.getSrc().getType() == vt) reads.push_back(load); });
  mod.walk([&](triton::hcugpu::AsyncWaitOp wait) {
    if (llvm::is_contained(wait.getOperands(), bToken)) waits.push_back(wait);
  });
  if (reads.size() != 2 || waits.size() != 2) return fail("main and epilogue read/fullwait required");
  auto bDesc = dyn_cast<BlockArgument>(reads.front().getSrc());
  if (!bDesc || bDesc.getOwner() != header ||
      enter.getDestOperands()[bDesc.getArgNumber()] != initialView.getResult() ||
      back.getDestOperands()[bDesc.getArgNumber()] != futureView.getResult())
    return fail("anchored current B descriptor queue required");
  // Retain the old reader branch verbatim. The new alternative changes only
  // B ownership; ordinary A/C keep generic verification/lowering and membar.
  bool oldReaders = llvm::all_of(reads, [&](LocalLoadOp read) {
    return isLegacyV0Dot(read.getType(), *packet);
  });
  auto kw8Hint = [&](StringRef key, int32_t value, bool optional = false) {
    auto a = mod->getAttrOfType<IntegerAttr>(key);
    return a ? a.getType().isInteger(32) && a.getInt() == value
             : optional && value == 0;
  };
  bool s25Kw8Readers = !oldReaders && source->totalK == 1024 && loop->trips == 31 &&
      legacyV0SafeInteger(future.getShape()[1]) == 1024 &&
      kw8Hint("ttg.hcu.v0_blockptr", 1) && kw8Hint("ttg.hcu.mmac_tpw_n", 2) &&
      kw8Hint("ttg.hcu.mmac_transposed", 1) && kw8Hint("ttg.hcu.canonical_matrix", 0) &&
      kw8Hint("ttg.hcu.async_slot_war", 0, true) &&
      kw8Hint("ttg.hcu.v0_mls_local_prefetch_capture", 0, true) &&
      kw8Hint("ttg.hcu.local_prefetch", 0, true) &&
      kw8Hint("ttg.hcu.alt_matrix", 0, true) &&
      kw8Hint("ttg.hcu.v0_swizzled_a_vec8", 0, true) &&
      llvm::all_of(reads, [&](LocalLoadOp read) {
        return isLegacyV0S25Kw8Dot(read.getType(), *packet);
      });
  for (Block *where : {body, exit}) {
    LocalLoadOp read;
    triton::hcugpu::AsyncWaitOp wait;
    for (auto x : reads) if (x->getBlock() == where) read = x;
    for (auto x : waits) if (x->getBlock() == where) wait = x;
    if (!read || !wait || read.getSrc() != bDesc || read.getToken() ||
        !(isLegacyV0Dot(read.getType(), *packet) ||
          (s25Kw8Readers && isLegacyV0S25Kw8Dot(read.getType(), *packet))) ||
        wait.getNumInst() != 0 ||

        !wait->isBeforeInBlock(read.getOperation()) ||
        (where == body && !read->isBeforeInBlock(future.getOperation())))
      return fail("tokenless native reader after current fullwait and before future copy");
  }
  // Local opaque use closure only. Generic values, A memory and accumulation
  // use their normal dialect verification and conservative membar analysis.
  bool escape = false;
  llvm::SmallPtrSet<Operation *, 16> allowed{root, initialView, futureView,
      initial, future, enter, back, reads[0], reads[1]};
  mod.walk([&](Operation *op) {
    if (auto free = dyn_cast<LocalDeallocOp>(op)) {
      if (free.getSrc() == root.getResult() && free->getBlock() == exit &&
          reads[0]->getBlock() == exit && reads[0]->isBeforeInBlock(free)) allowed.insert(op);
      else if (free.getSrc() == root.getResult() && free->getBlock() == exit &&
          reads[1]->getBlock() == exit && reads[1]->isBeforeInBlock(free)) allowed.insert(op);
    }
    auto check = [&](Type type) {
      auto mem = dyn_cast<MemDescType>(type);
      auto e = mem ? dyn_cast<HCUMlsSharedEncodingAttr>(mem.getEncoding()) : HCUMlsSharedEncodingAttr{};
      if (e && e.getVersion() == 0 && !allowed.contains(op)) escape = true;
    };
    for (Type t : op->getOperandTypes()) check(t);
    for (Type t : op->getResultTypes()) check(t);
  });
  if (escape) return fail("opaque descriptor escaped primitive view/copy/read/queue closure");
  if (witness) {
    witness->root = root.getResult();
    witness->initialView = initialView.getResult();
    witness->futureView = futureView.getResult();
    witness->currentDescriptor = bDesc;
    witness->currentToken = bToken;
    witness->counter = counter;
    witness->nextIndex = futureView.getIndex();
    witness->entry = entry; witness->header = header;
    witness->body = body; witness->exit = exit;
    witness->initialCopy = initial; witness->futureCopy = future;
    witness->initialCommit = commits[0]; witness->futureCommit = commits[1];
    for (auto read : reads)
      (read->getBlock() == body ? witness->mainRead : witness->exitRead) = read;
    for (auto wait : waits)
      (wait->getBlock() == body ? witness->mainWait : witness->exitWait) = wait;
  }
  return success();
}

// Private LP1 reuses the captured two/four-slot packet queues.
// This is a separate queue proof, never an expansion of the old prefix2 WAR
// witness. No barriers/alias filters are removed or certificate attrs created.
// Pure SSA queue matching shared by independent B and A witnesses.
static std::optional<SmallVector<BlockArgument>>
matchLegacyV0HeaderQueue(Block *header, cf::BranchOp enter, cf::BranchOp back,
                        ArrayRef<Value> seeds, Value tail) {
    SmallVector<BlockArgument> queue(seeds.size());
    for (int j = int(seeds.size()) - 1; j >= 0; --j) {
      BlockArgument found;
      for (unsigned i = 0; i < header->getNumArguments(); ++i)
        if (enter.getDestOperands()[i] == seeds[j] &&
            back.getDestOperands()[i] == tail) {
          if (found || header->getArgument(i).getType() != seeds[j].getType())
            return std::nullopt;
          found = header->getArgument(i);
        }
      if (!found || llvm::is_contained(queue, found)) return std::nullopt;
      queue[j] = found;
      tail = found;
    }
    return queue;
}

static LogicalResult verifyLegacyV0LocalPrefetch(
    ModuleOp mod, std::pair<Operation *, Operation *> *rawPair = nullptr,
    bool diagnose = true) {
  auto fail = [&](StringRef why) -> LogicalResult {
    if (diagnose)
      mod.emitError("legacy-v0 two-slot local-prefetch: ") << why;
    return failure();
  };
  auto attr = [&](StringRef name, int64_t value, bool optional = false) {
    auto a = mod->getAttrOfType<IntegerAttr>(name);
    return a ? a.getType().isInteger(32) && a.getInt() == value
             : optional && value == 0;
  };
  if (!attr("ttg.hcu.v0_mls_local_prefetch_capture", 1) ||
      !attr("ttg.hcu.v0_blockptr", 1) || !attr("ttg.hcu.v0_blockptr_loop", 1) ||
      !attr("ttg.hcu.v0_default_async", 1) || !attr("ttg.num-warps", 4) ||
      !attr("ttg.hcu.canonical_matrix", 0) || !attr("ttg.hcu.alt_matrix", 0) ||
      !attr("ttg.hcu.async_slot_war", 0) || !attr("ttg.hcu.local_prefetch", 0) ||
      !attr("ttg.hcu.mmac_tpw_n", 0) || !attr("ttg.hcu.v0_swizzled_a_vec8", 0) ||
      !attr("ttg.hcu.v0_joint_phase", 0, true))
    return fail("private captured W4/canonical0/no-WAR/no-joint scope required");
  SmallVector<LocalAllocOp> allRoots, roots;
  SmallVector<triton::hcugpu::MatrixLoadToLocalOp> copies;
  unsigned ordinaryLoads = 0;
  bool wrongAPolicy = false;
  mod.walk([&](Operation *op) {
    if (auto root = dyn_cast<LocalAllocOp>(op)) {
      allRoots.push_back(root);
      auto enc = dyn_cast<HCUMlsSharedEncodingAttr>(root.getType().getEncoding());
      if (enc && enc.getVersion() == 0) roots.push_back(root);
    }
    if (auto copy = dyn_cast<triton::hcugpu::MatrixLoadToLocalOp>(op)) copies.push_back(copy);
    if (auto load = dyn_cast<triton::LoadOp>(op)) {
      ++ordinaryLoads;
      auto policy = load->getAttrOfType<BoolAttr>("tt.load.async");
      wrongAPolicy |= !policy || policy.getValue();
    }
    wrongAPolicy |= isa<AsyncCopyGlobalToLocalOp>(op) ||
                   op->getName().getStringRef().starts_with("hcug.buffer");
  });
  if (roots.size() != 1 || allRoots.size() != 2 || wrongAPolicy)
    return fail("one opaque/two total roots and explicit synchronous A required");
  auto root = roots.front();
  auto rt = root.getType();
  if (rt.getRank() != 3 || (rt.getShape()[0] != 2 && rt.getShape()[0] != 4))
    return fail("two or four physical packet slots required");
  unsigned slots = unsigned(rt.getShape()[0]);
  if (copies.size() != slots + 1 || ordinaryLoads != slots + 1)
    return fail("one prefix copy/load per slot and one body issue required");
  auto packet = getLegacyV0Packet(rt.getEncoding());
  auto fn = root->getParentOfType<triton::FuncOp>();
  if (!packet || packet->k != 32 || packet->n != 128 || packet->warps != 4 ||
      !fn || root->getBlock() != &fn.getBody().front() ||
      rt.getShape() != ArrayRef<int64_t>({int64_t(slots), 32, 128}) ||
      rt.getAllocShape() != rt.getShape() || !rt.getMutableMemory() ||
      !(rt.getElementType().isF16() || rt.getElementType().isBF16()) ||
      !root.isSharedMemoryAlloc() || root.getSrc() ||
      root.getAlignmentOrDefault() < 16 || root.getAlignmentOrDefault() % 16)
    return fail("exact aligned mutable s25 packet root required");
  // A is not part of the B witness. Keep the captured policy/storage scope;
  // its local-store/accumulator/C semantics and all membar remain generic.
  auto aRoot = allRoots[0] == root ? allRoots[1] : allRoots[0];
  auto at = aRoot.getType();
  if (aRoot->getParentOfType<triton::FuncOp>() != fn ||
      aRoot->getBlock() != root->getBlock() || aRoot.getSrc() ||
      at.getShape() != ArrayRef<int64_t>({int64_t(slots), 128, 32}) ||
      at.getAllocShape() != at.getShape() || !at.getMutableMemory() ||
      at.getElementType() != rt.getElementType() ||
      !isa<SwizzledSharedEncodingAttr>(at.getEncoding()) ||
      at.getMemorySpace() != rt.getMemorySpace() || !aRoot.isSharedMemoryAlloc())
    return fail("captured synchronous Swizzled A storage required");
  Block *entry = &fn.getBody().front();
  SmallVector<triton::hcugpu::MatrixLoadToLocalOp> prologue;
  triton::hcugpu::MatrixLoadToLocalOp future;
  for (auto copy : copies) {
    if (copy->getParentOfType<triton::FuncOp>() != fn)
      return fail("copy function mismatch");
    if (copy->getBlock() == entry) prologue.push_back(copy);
    else if (future) return fail("copy block roles");
    else future = copy;
  }
  if (prologue.size() != slots || !future)
    return fail("one prologue copy per slot and one body copy required");
  llvm::sort(prologue, [](auto a, auto b) {
    return a->isBeforeInBlock(b.getOperation());
  });
  auto first = prologue.front();
  auto source = getLegacyV0LocalPrefetchSourceWindow(prologue, future, *packet);
  auto loop = source ? getLegacyV0CountedBody(future, source->totalK / 32 - slots, 0)
                     : std::nullopt;
  if (!loop) return fail(slots == 2 ? "bounded full static T>=4 lookahead2 source required"
                            : "bounded full static T>=slots+2 lookahead source required");
  Block *body = loop->body;
  auto back = cast<cf::BranchOp>(body->getTerminator());
  Block *header = back.getDest();
  auto enter = cast<cf::BranchOp>(entry->getTerminator());
  Block *exit = cast<cf::CondBranchOp>(header->getTerminator()).getFalseDest();
  SmallVector<MemDescIndexOp> views;
  SmallVector<AsyncCommitGroupOp> commits;
  SmallVector<triton::hcugpu::MatrixLoadToLocalOp> orderedCopies = prologue;
  orderedCopies.push_back(future);
  for (auto copy : orderedCopies) {
    auto view = copy.getDest().getDefiningOp<MemDescIndexOp>();
    auto ml = copy->getAttrOfType<triton::hcugpu::MlsEncodingAttr>(
        triton::hcugpu::MlsEncodingAttr::getMnemonic());
    auto commit = copy.getToken().hasOneUse()
        ? dyn_cast<AsyncCommitGroupOp>(*copy.getToken().getUsers().begin()) : AsyncCommitGroupOp{};
    if (!view || view.getSrc() != root.getResult() || view->getBlock() != copy->getBlock() ||
        !ml || ml.getVersion() != 0 || ml.getOpIdx() != 1 || ml.getAlt2Kind() != 0 ||
        ml.getElemBitWidth() != 16 || ml.getOrder() != ArrayRef<unsigned>({1, 0}) ||
        ml.getMlsTile() != ArrayRef<unsigned>({32, 128}) ||
        ml.getWarpsPerCTA() != ArrayRef<unsigned>({2, 2}) ||
        copy.getTensorShape() != ArrayRef<int32_t>({32, 128}) ||
        !copy.getBoundaryCheck().empty() || copy.getCache() != triton::CacheModifier::NONE ||
        copy.getEvict() != triton::EvictionPolicy::NORMAL || copy.getIsVolatile() ||
        !commit || commit.getOperands().size() != 1 ||
        copy->getNextNode() != commit.getOperation() || !commit.getResult().hasOneUse())
      return fail("whole packet views, raw metadata and sole commits required");
    views.push_back(view); commits.push_back(commit);
  }
  auto vt = views[0].getType();
  if (vt.getShape() != ArrayRef<int64_t>({32, 128}) || vt.getAllocShape() != vt.getShape() ||
      vt.getEncoding() != rt.getEncoding() || vt.getElementType() != rt.getElementType() ||
      vt.getMemorySpace() != rt.getMemorySpace() || !vt.getMutableMemory() ||
      !isa<SharedMemorySpaceAttr>(rt.getMemorySpace()))
    return fail("identical full packet view type required");
  for (unsigned j = 0; j <= slots; ++j)
    if (views[j].getType() != vt ||
        (j < slots && legacyV0SafeInteger(views[j].getIndex()) != j))
      return fail("identical full packet views with ordered prefix slots required");
  auto select = views.back().getIndex().getDefiningOp<arith::SelectOp>();
  auto inc = select ? select.getTrueValue().getDefiningOp<arith::AddIOp>() : arith::AddIOp{};
  auto cmp = select ? select.getCondition().getDefiningOp<arith::CmpIOp>() : arith::CmpIOp{};
  auto counter = inc ? dyn_cast<BlockArgument>(inc.getLhs()) : BlockArgument{};
  if (!select || !inc || !cmp || !counter || counter.getOwner() != header ||
      !counter.getType().isInteger(32) || select->getBlock() != body ||
      inc->getBlock() != body || cmp->getBlock() != body ||
      legacyV0SafeInteger(enter.getDestOperands()[counter.getArgNumber()]) != slots - 1 ||
      back.getDestOperands()[counter.getArgNumber()] != select.getResult() ||
      legacyV0SafeInteger(inc.getRhs()) != 1 || cmp.getLhs() != inc.getResult() ||
      legacyV0SafeInteger(cmp.getRhs()) != slots ||
      cmp.getPredicate() != arith::CmpIPredicate::slt ||
      legacyV0SafeInteger(select.getFalseValue()) != 0)
    return fail(slots == 2 ? "current slot counter must be anchored at1 and advance modulo2"
                          : "slot counter must start at slots-1 and advance modulo slots");
  // Match only exact entry seeds and backedge shifts, independently of
  // unrelated A/accumulator arguments and without a fixed header arity.
  auto matchQueue = [&](ArrayRef<Value> seeds, Value tail) {
    return matchLegacyV0HeaderQueue(header, enter, back, seeds, tail);
  };
  SmallVector<Value> tokenSeeds, descSeeds;
  for (unsigned j = 1; j < slots; ++j) {
    tokenSeeds.push_back(commits[j].getResult());
    descSeeds.push_back(views[j].getResult());
  }
  auto tokens = matchQueue(tokenSeeds, commits.back().getResult());
  auto descs = matchQueue(descSeeds, views.back().getResult());
  if (!tokens || !descs)
    return fail(slots == 2 ? "current descriptor/token must be anchored at prefix1/future"
                          : "exact prefix-seeded token/descriptor queue shifts required");
  auto bToken = tokens->front();
  auto base = dyn_cast<BlockArgument>(future.getBase());
  auto ptr = dyn_cast<triton::PointerType>(future.getBase().getType());
  auto alignment = base && base.getOwner() == entry
      ? fn.getArgAttrOfType<IntegerAttr>(base.getArgNumber(), "tt.divisibility") : IntegerAttr{};
  if (!base || !ptr || ptr.getAddressSpace() != 1 || ptr.getPointeeType() != rt.getElementType() ||
      !alignment || !alignment.getType().isInteger(32) ||
      alignment.getInt() < 16 || alignment.getInt() % 16)
    return fail("aligned typed AS1 B source argument required");
  SmallVector<LocalLoadOp> reads;
  SmallVector<triton::hcugpu::AsyncWaitOp> waits;
  mod.walk([&](LocalLoadOp read) { if (read.getSrc().getType() == vt) reads.push_back(read); });
  mod.walk([&](triton::hcugpu::AsyncWaitOp wait) {
    bool relevant = llvm::is_contained(wait.getOperands(), commits[0].getResult());
    for (auto token : *tokens)
      relevant |= llvm::is_contained(wait.getOperands(), token);
    if (relevant) waits.push_back(wait);
  });
  if (reads.size() != slots + 1 || waits.size() != 3)
    return fail("prefix/main/drain native reads and three packet waits required");
  LocalLoadOp firstRead, mainRead;
  SmallVector<LocalLoadOp> exitReads;
  triton::hcugpu::AsyncWaitOp firstWait, mainWait, exitWait;
  for (auto read : reads) {
    if (!isLegacyV0Dot(read.getType(), *packet) || read.getToken() ||
        read.getType().getElementType() != rt.getElementType())
      return fail("tokenless normal B native ownership required");
    if (read->getBlock() == entry && !firstRead) firstRead = read;
    else if (read->getBlock() == body && !mainRead) mainRead = read;
    else if (read->getBlock() == exit) exitReads.push_back(read);
    else return fail("native read block roles");
  }
  llvm::sort(exitReads, [](auto a, auto b) {
    return a->isBeforeInBlock(b.getOperation());
  });
  // The known s25 packet emits two native raw instructions per issue.
  // N-2 newer packets precede the front-token wait; ordinary A is uncounted.
  unsigned partial = (slots - 2) * 2;
  for (auto wait : waits) {
    if (wait->getBlock() == entry && !firstWait && wait.getNumInst() == partial &&
        wait.getOperands().size() == 1 && wait.getOperand(0) == commits[0].getResult())
      firstWait = wait;
    else if (wait->getBlock() == body && !mainWait && wait.getNumInst() == partial &&
             wait.getOperands().size() == 1 && wait.getOperand(0) == bToken)
      mainWait = wait;
    else if (wait->getBlock() == exit && !exitWait && wait.getNumInst() == 0 &&
             llvm::equal(wait.getOperands(), *tokens))
      exitWait = wait;
    else return fail(slots == 2 ? "sole B token fullwait0 required"
                               : "exact front-token partial waits and merged drain fullwait0 required");
  }
  if (!firstRead || !mainRead || exitReads.size() != slots - 1 ||
      !firstWait || !mainWait || !exitWait ||
      firstRead.getSrc() != views[0].getResult() || mainRead.getSrc() != descs->front() ||
      !commits[slots - 2]->isBeforeInBlock(firstWait) ||
      !firstWait->isBeforeInBlock(prologue.back()) ||
      !commits[slots - 1]->isBeforeInBlock(firstRead) ||
      &body->front() != mainWait.getOperation() ||
      !mainWait->isBeforeInBlock(future) || !commits.back()->isBeforeInBlock(mainRead))
    return fail("captured prefix/head/future/current-read order required");
  for (unsigned j = 0; j < exitReads.size(); ++j)
    if (exitReads[j].getSrc() != (*descs)[j] || !exitWait->isBeforeInBlock(exitReads[j]))
      return fail("ordered complete descriptor drain after merged fullwait0 required");
  // At iteration i the front descriptor is slot(i+1)%N while counter is
  // (i+N-1)%N; future writes slot(i+N)%N. These slots differ for N=2/4.
  // The head wait protects the front packet, not the new future write. Keep
  // wait->LocalBarrier DS0+CTA for prior-iteration local-read completion.
  bool escape = false;
  llvm::SmallPtrSet<Operation *, 16> allowed{root, enter, back, firstRead, mainRead};
  for (auto view : views) allowed.insert(view.getOperation());
  for (auto copy : orderedCopies) allowed.insert(copy.getOperation());
  for (auto read : exitReads) allowed.insert(read.getOperation());
  mod.walk([&](Operation *op) {
    if (auto free = dyn_cast<LocalDeallocOp>(op))
      if (free.getSrc() == root.getResult() && free->getBlock() == exit &&
          exitReads.back()->isBeforeInBlock(free)) allowed.insert(op);
    auto isOpaque = [](Type t) {
      auto m = dyn_cast<MemDescType>(t);
      auto e = m ? dyn_cast<HCUMlsSharedEncodingAttr>(m.getEncoding()) : HCUMlsSharedEncodingAttr{};
      return e && e.getVersion() == 0;
    };
    for (Type t : op->getOperandTypes()) if (isOpaque(t) && !allowed.contains(op)) escape = true;
    for (Type t : op->getResultTypes()) if (isOpaque(t) && !allowed.contains(op)) escape = true;
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          if (isOpaque(arg.getType()) && !llvm::is_contained(*descs, arg)) escape = true;
  });
  if (escape) return fail("opaque descriptor escaped the local-prefetch packet queue");
  // Only publish an ephemeral pair after EVERY existing proof check succeeded.
  if (rawPair)
    *rawPair = {future.getOperation(), mainRead.getOperation()};
  return success();
}

llvm::SmallVector<std::pair<Operation *, Operation *>>
prepareLegacyV0LocalPrefetchRAW(ModuleOp mod) {
  auto hasI32 = [&](StringRef name, int32_t value) {
    auto a = mod->getAttrOfType<IntegerAttr>(name);
    return a && a.getType().isInteger(32) && a.getInt() == value;
  };
  auto target = mod->getAttrOfType<StringAttr>("ttg.target");
  if (!hasI32("ttg.hcu.v0_lp1_slot_raw", 1) ||
      !hasI32("ttg.num-ctas", 1) || !hasI32("ttg.threads-per-warp", 64) ||
      !target || target.getValue() != "hip:gfx936")
    return {};
  std::pair<Operation *, Operation *> pair{nullptr, nullptr};
  // Re-run the SAME complete queue/physical/source proof immediately before
  // membar. No diagnostics or IR mutation if an unsupported module is seen.
  if (failed(verifyLegacyV0LocalPrefetch(mod, &pair, /*diagnose=*/false)))
    return {};
  if (!pair.first || !pair.second ||
      pair.first->getBlock() != pair.second->getBlock())
    return {};
  auto wait = dyn_cast<triton::hcugpu::AsyncWaitOp>(
      &pair.first->getBlock()->front());
  auto publication = wait ? dyn_cast_or_null<LocalBarrierOp>(wait->getNextNode())
                          : LocalBarrierOp{};
  // The existing front-packet wait + DS0/CTA must remain on every iteration.
  // It resets membar's dangling effects and protects cross-iteration slot reuse.
  // This is not a WAR exemption and never creates/removes a barrier itself.
  if (!publication || !publication->isBeforeInBlock(pair.first))
    return {};
  return {pair};
}

// Schedule only the exact, already verified B pair after membar has finished.
// Keep every inserted publication/WAR barrier. No alias promise or wait edit.
LogicalResult scheduleLegacyV0LocalPrefetchReadFirst(ModuleOp mod,
    ArrayRef<std::pair<Operation *, Operation *>> pairs) {
  auto requested = mod->getAttr("ttg.hcu.v0_lp1_read_first");
  if (!requested) return success();
  auto hint = dyn_cast<IntegerAttr>(requested);
  auto fail = [&]() -> LogicalResult {
    return mod.emitError("legacy-v0 read-first requires the exact N4 RAW1 pair and unchanged copy/commit/read adjacency");
  };
  if (!hint || !hint.getType().isInteger(32)) return fail();
  if (hint.getInt() == 0) return success();
  if (hint.getInt() != 1) return fail();
  if (pairs.size() != 1) return fail();
  auto copy = dyn_cast_or_null<triton::hcugpu::MatrixLoadToLocalOp>(pairs[0].first);
  auto read = dyn_cast_or_null<LocalLoadOp>(pairs[0].second);
  auto view = copy ? copy.getDest().getDefiningOp<MemDescIndexOp>() : MemDescIndexOp{};
  auto root = view ? view.getSrc().getDefiningOp<LocalAllocOp>() : LocalAllocOp{};
  auto shape = root ? root.getType().getShape() : ArrayRef<int64_t>{};
  auto commit = copy ? dyn_cast_or_null<AsyncCommitGroupOp>(copy->getNextNode()) : AsyncCommitGroupOp{};
  if (!root || shape != ArrayRef<int64_t>({4, 32, 128}) || !copy || !read ||
      read.getToken() || copy->getBlock() != read->getBlock() || !commit ||
      commit->getNextNode() != read.getOperation() ||
      commit.getOperands().size() != 1 || commit.getOperand(0) != copy.getToken())
    return fail();
  // The full queue witness requires the current descriptor to be a header
  // argument, which dominates the body's future copy. Recheck the SSA fact
  // here; the move cannot cross another barrier, region or side effect.
  auto source = dyn_cast<BlockArgument>(read.getSrc());
  auto wait = dyn_cast<triton::hcugpu::AsyncWaitOp>(&copy->getBlock()->front());
  auto publication = wait ? dyn_cast_or_null<LocalBarrierOp>(wait->getNextNode()) : LocalBarrierOp{};
  if (!source || source.getOwner() == read->getBlock() || !publication ||
      !publication->isBeforeInBlock(copy.getOperation()))
    return fail();
  read->moveBefore(copy.getOperation());
  return success();
}

// Private N4-only same-iteration A WAR witness. This is independent of the
// opaque B alias proof and never changes an existing barrier or IR operand.
llvm::SmallVector<std::pair<Operation *, Operation *>>
prepareLegacyV0LocalPrefetchAWAR(ModuleOp mod, ModuleAllocation &allocation,
    ArrayRef<std::pair<Operation *, Operation *>> bPairs) {
  auto hasI32 = [&](StringRef name, int32_t value) {
    auto a = mod->getAttrOfType<IntegerAttr>(name);
    return a && a.getType().isInteger(32) && a.getInt() == value;
  };
  if (!hasI32("ttg.hcu.v0_lp1_a_slot_war", 1) ||
      !hasI32("ttg.hcu.v0_lp1_read_first", 1) ||
      !hasI32("ttg.hcu.v0_lp1_slot_raw", 1) || bPairs.size() != 1)
    return {};
  // Same CF, before membar/movement. Reuse the complete quiet B queue/source
  // proof, not a cached attribute or an untrusted caller-supplied pair.
  auto verified = prepareLegacyV0LocalPrefetchRAW(mod);
  if (verified.size() != 1 || verified.front() != bPairs.front()) return {};
  auto futureB = cast<triton::hcugpu::MatrixLoadToLocalOp>(bPairs[0].first);
  auto bView = futureB.getDest().getDefiningOp<MemDescIndexOp>();
  auto bRoot = bView.getSrc().getDefiningOp<LocalAllocOp>();
  if (bRoot.getType().getShape() != ArrayRef<int64_t>({4, 32, 128})) return {};
  auto fn = futureB->getParentOfType<triton::FuncOp>();
  if (fn.getNumArguments() != 3 || llvm::any_of(fn.getArguments(), [](Value arg) {
        auto p = dyn_cast<triton::PointerType>(arg.getType());
        return !p || p.getAddressSpace() != 1;
      })) return {};
  Block *entry = &fn.getBody().front(), *body = futureB->getBlock();
  auto back = cast<cf::BranchOp>(body->getTerminator());
  Block *header = back.getDest();
  auto enter = cast<cf::BranchOp>(entry->getTerminator());
  Block *exit = cast<cf::CondBranchOp>(header->getTerminator()).getFalseDest();
  auto headWait = cast<triton::hcugpu::AsyncWaitOp>(&body->front());
  auto publication = dyn_cast_or_null<LocalBarrierOp>(headWait->getNextNode());
  if (!publication) return {};
  LocalAllocOp root;
  for (auto r : entry->getOps<LocalAllocOp>()) if (r != bRoot) {
    if (root) return {};
    root = r;
  }
  if (!root || root.getSrc() || !root.isSharedMemoryAlloc()) return {};
  auto rt = root.getType();
  auto enc = dyn_cast<SwizzledSharedEncodingAttr>(rt.getEncoding());
  if (rt.getShape() != ArrayRef<int64_t>({4, 128, 32}) ||
      rt.getAllocShape() != rt.getShape() || !rt.getMutableMemory() || !enc ||
      rt.getElementType() != bRoot.getType().getElementType() ||
      rt.getMemorySpace() != bRoot.getType().getMemorySpace() ||
      llvm::any_of(enc.getCTALayout().getCTAsPerCGA(), [](unsigned x) { return x != 1; }) ||
      llvm::any_of(enc.getCTALayout().getCTASplitNum(), [](unsigned x) { return x != 1; }))
    return {};
  // Recomputed physical frames, not shape-only or manually supplied offsets.
  auto *data = allocation.getFuncData(cast<FunctionOpInterface>(fn.getOperation()));
  if (!data) return {};
  auto aId = data->getBufferId(root.getResult()), bId = data->getBufferId(bRoot.getResult());
  if (aId == Allocation::InvalidBufferId || bId == Allocation::InvalidBufferId ||
      data->getAllocatedSize(aId) != 32768 || data->getAllocatedSize(bId) != 32768 ||
      allocation.getSharedMemorySize() != 65536) return {};
  uint64_t aBase = data->getOffset(aId), bBase = data->getOffset(bId);
  auto emitted = [&](Operation *op, StringRef name, uint64_t n) {
    auto a = op->getAttrOfType<IntegerAttr>(name);
    return a && a.getType().isInteger(32) && a.getInt() >= 0 && uint64_t(a.getInt()) == n;
  };
  if (aBase % 8192 || bBase % 8192 || aBase > 32768 || bBase > 32768 ||
      !(aBase + 32768 <= bBase || bBase + 32768 <= aBase) ||
      !emitted(root, "allocation.offset", aBase) ||
      !emitted(bRoot, "allocation.offset", bBase) || !emitted(mod.getOperation(), "ttg.shared", 65536)) return {};
  SmallVector<MemDescIndexOp> prefix(4);
  MemDescIndexOp futureView;
  LocalDeallocOp dealloc;
  for (Operation *user : root.getResult().getUsers()) {
    if (auto view = dyn_cast<MemDescIndexOp>(user)) {
      auto t = view.getType();
      if (t.getShape() != rt.getShape().drop_front() || t.getAllocShape() != t.getShape() ||
          t.getEncoding() != rt.getEncoding() || t.getElementType() != rt.getElementType() ||
          t.getMemorySpace() != rt.getMemorySpace() || t.getMutableMemory() != rt.getMutableMemory()) return {};
      if (user->getBlock() == entry) {
        auto index = legacyV0SafeInteger(view.getIndex());
        if (!index || *index < 0 || *index > 3 || prefix[*index]) return {};
        prefix[*index] = view;
      } else if (user->getBlock() == body && !futureView && view.getIndex() == bView.getIndex())
        futureView = view;
      else return {};
    } else if (auto free = dyn_cast<LocalDeallocOp>(user)) {
      if (dealloc || user->getBlock() != exit) return {};
      dealloc = free;
    } else return {};
  }
  if (!futureView || !dealloc || llvm::any_of(prefix, [](auto v) { return !v; })) return {};
  SmallVector<Value> seeds{prefix[1].getResult(), prefix[2].getResult(), prefix[3].getResult()};
  auto queue = matchLegacyV0HeaderQueue(header, enter, back, seeds, futureView.getResult());
  if (!queue) return {};
  auto vt = prefix[0].getType();
  SmallVector<Value> descs = seeds;
  descs.push_back(prefix[0].getResult()); descs.push_back(futureView.getResult());
  for (auto a : *queue) descs.push_back(a);
  SmallVector<LocalLoadOp> reads;
  SmallVector<LocalStoreOp> stores;
  for (Block &block : fn.getBody()) for (Operation &op : block) {
    if (auto load = dyn_cast<LocalLoadOp>(op); load && load.getSrc().getType() == vt) reads.push_back(load);
    if (auto store = dyn_cast<LocalStoreOp>(op); store && store.getDst().getType() == vt) stores.push_back(store);
  }
  if (reads.size() != 5 || stores.size() != 5) return {};
  SmallVector<LocalStoreOp> prefixStores(4);
  LocalStoreOp futureStore;
  for (auto store : stores) {
    auto value = store.getSrc().getDefiningOp<triton::LoadOp>();
    auto ptrTy = value ? dyn_cast<RankedTensorType>(value.getPtr().getType()) : RankedTensorType{};
    auto p = ptrTy ? dyn_cast<triton::PointerType>(ptrTy.getElementType()) : triton::PointerType{};
    if (!value || !value->hasOneUse() || value->getBlock() != store->getBlock() ||
        !value->isBeforeInBlock(store) || !p || p.getAddressSpace() != 1 ||
        value.getMask() || value.getOther() || value.getIsVolatile() ||
        cast<RankedTensorType>(value.getResult().getType()).getShape() != ArrayRef<int64_t>({128, 32}) ||
        cast<RankedTensorType>(value.getResult().getType()).getElementType() != rt.getElementType()) return {};
    if (store->getBlock() == body && !futureStore && store.getDst() == futureView.getResult()) futureStore = store;
    else if (store->getBlock() == entry) {
      int found = -1;
      for (unsigned i = 0; i < 4; ++i) if (store.getDst() == prefix[i].getResult()) found = i;
      if (found < 0 || prefixStores[found]) return {};
      prefixStores[found] = store;
    } else return {};
  }
  LocalLoadOp initialRead, mainRead;
  SmallVector<LocalLoadOp> drain;
  for (auto read : reads) {
    auto dot = dyn_cast<DotOperandEncodingAttr>(read.getType().getEncoding());
    if (!dot || dot.getOpIdx() != 0 || dot.getKWidth() != 4 || read.getToken() ||
        read.getType().getShape() != ArrayRef<int64_t>({128, 32}) ||
        read.getType().getElementType() != rt.getElementType()) return {};
    if (read->getBlock() == entry && !initialRead && read.getSrc() == prefix[0].getResult()) initialRead = read;
    else if (read->getBlock() == body && !mainRead && read.getSrc() == queue->front()) mainRead = read;
    else if (read->getBlock() == exit) drain.push_back(read);
    else return {};
  }
  llvm::sort(drain, [](auto a, auto b) { return a->isBeforeInBlock(b.getOperation()); });
  if (!initialRead || !mainRead || !futureStore || drain.size() != 3 ||
      llvm::any_of(prefixStores, [](auto x) { return !x; }) ||
      !prefixStores[0]->isBeforeInBlock(initialRead) ||
      !publication->isBeforeInBlock(mainRead) || !publication->isBeforeInBlock(futureStore) ||
      !mainRead->isBeforeInBlock(futureStore)) return {};
  for (unsigned j = 0; j < 3; ++j)
    if (drain[j].getSrc() != (*queue)[j] || !drain[j]->isBeforeInBlock(dealloc)) return {};
  auto carry = matchLegacyV0HeaderQueue(header, enter, back,
      ArrayRef<Value>{initialRead.getResult()}, mainRead.getResult());
  if (!carry || initialRead.getType() != mainRead.getType() ||
      llvm::any_of(drain, [&](auto r) { return r.getType() != mainRead.getType(); })) return {};
  auto hasOnlyUser = [](Value v, Operation *op) { return v.hasOneUse() && *v.getUsers().begin() == op; };
  if (!hasOnlyUser(initialRead.getResult(), enter.getOperation()) || !hasOnlyUser(mainRead.getResult(), back.getOperation())) return {};
  SmallVector<triton::DotOp> bodyDots, exitDots;
  for (auto dot : body->getOps<triton::DotOp>()) bodyDots.push_back(dot);
  for (auto dot : exit->getOps<triton::DotOp>()) exitDots.push_back(dot);
  if (bodyDots.size() != 1 || exitDots.size() != 4 ||
      bodyDots[0].getA() != carry->front() || exitDots[0].getA() != carry->front() ||
      std::distance(carry->front().use_begin(), carry->front().use_end()) != 2) return {};
  for (unsigned j = 0; j < 3; ++j)
    if (exitDots[j + 1].getA() != drain[j].getResult() ||
        !hasOnlyUser(drain[j].getResult(), exitDots[j + 1].getOperation())) return {};
  // Exact root/view/phi alias closure. Unknown desc users/SSA aliases reject.
  for (Value desc : descs) for (OpOperand &use : desc.getUses()) {
    Operation *user = use.getOwner();
    if (auto read = dyn_cast<LocalLoadOp>(user)) {
      if (read.getSrc() != desc || !llvm::is_contained(reads, read)) return {};
    } else if (auto store = dyn_cast<LocalStoreOp>(user)) {
      if (store.getDst() != desc || !llvm::is_contained(stores, store)) return {};
    } else if (auto branch = dyn_cast<cf::BranchOp>(user)) {
      unsigned i = use.getOperandNumber();
      if ((branch != enter && branch != back) || branch.getDest() != header ||
          i >= header->getNumArguments() || !llvm::is_contained(*queue, header->getArgument(i))) return {};
    } else return {};
  }
  for (Block &block : fn.getBody()) {
    for (BlockArgument arg : block.getArguments()) if (arg.getType() == vt && !llvm::is_contained(descs, Value(arg))) return {};
    for (Operation &op : block) {
      if (op.getNumRegions()) return {};
      if (auto load = dyn_cast<triton::LoadOp>(op)) {
        auto t = dyn_cast<RankedTensorType>(load.getPtr().getType());
        auto p = t ? dyn_cast<triton::PointerType>(t.getElementType()) : triton::PointerType{};
        auto policy = load->getAttrOfType<BoolAttr>("tt.load.async");
        if (!p || p.getAddressSpace() != 1 || !policy || policy.getValue()) return {};
      }
      for (Value v : op.getResults()) if (v.getType() == vt && !llvm::is_contained(descs, v)) return {};
      bool arithmetic = op.getName().getDialectNamespace() == "arith" && isMemoryEffectFree(&op);
      if (!arithmetic && !isa<triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp,
          triton::MakeRangeOp, triton::AddPtrOp, triton::GetProgramIdOp, triton::LoadOp,
          triton::StoreOp, triton::DotOp, triton::ReturnOp, cf::BranchOp, cf::CondBranchOp,
          LocalAllocOp, LocalDeallocOp, MemDescIndexOp, LocalLoadOp, LocalStoreOp,
          AsyncCommitGroupOp, LocalBarrierOp, triton::hcugpu::AsyncWaitOp,
          triton::hcugpu::MatrixLoadToLocalOp, ROCDL::SchedBarrier>(&op)) return {};
      if (auto store = dyn_cast<triton::StoreOp>(op)) {
        auto t = dyn_cast<RankedTensorType>(store.getPtr().getType());
        auto p = t ? dyn_cast<triton::PointerType>(t.getElementType()) : triton::PointerType{};
        if (!p || p.getAddressSpace() != 1) return {};
      }
    }
  }
  // Independently: Aqueue seeds1,2,3/backshift/tail root[Cnext] establish
  // read=(i+1)%4, write=i%4. Prior iteration read=i%4 DOES alias the next write;
  // retain mandatory head DS0+CTA on every backedge, every RAW/WAW/other WAR.
  return {{mainRead.getOperation(), futureStore.getOperation()}};
}

std::optional<LegacyV0Prefix2Queue>
getVerifiedLegacyV0Prefix2Queue(ModuleOp mod) {
  LegacyV0Prefix2Queue witness;
  if (failed(verifyLegacyV0Prefix2(mod, &witness, /*diagnose=*/false)))
    return std::nullopt;
  return witness;
}

// Paired capture uses the existing A/B packet emitters as local primitives.
// This does not prove a queue or remove synchronization: valid whole-view
// MemDescIndex and unmasked source-tile bounds retain their normal IR/TL
// contracts. Generic membar and the real allocator remain mandatory.
static LogicalResult verifyLegacyV0PairedPrimitives(ModuleOp mod) {
  auto fail = [&](StringRef why) -> LogicalResult {
    return mod.emitError("legacy-v0 paired primitive: ") << why;
  };
  auto is = [&](StringRef key, int32_t value) {
    auto a = mod->getAttrOfType<IntegerAttr>(key);
    return a && a.getType().isInteger(32) && a.getInt() == value;
  };
  if (!is("ttg.hcu.v0_indexed_a_packet", 1) || !is("ttg.hcu.v0_blockptr", 1) ||
      !is("ttg.hcu.v0_mls_local_prefetch_capture", 1) ||
      !is("ttg.hcu.v0_blockptr_loop", 1) || !is("ttg.num-warps", 4))
    return fail("requires explicit paired capture and four waves");
  for (StringRef key : {"ttg.hcu.async_slot_war", "ttg.hcu.v0_lp1_slot_raw",
                       "ttg.hcu.v0_lp1_a_slot_war", "ttg.hcu.v0_lp1_read_first",
                       "ttg.hcu.v0_joint_phase"})
    if (mod->hasAttr(key) && !is(key, 0))
      return fail("paired primitive does not enable alias filters or scheduling");
  auto kind = [](Attribute encoding) -> int {
    if (isLegacyV0IndexedAEncoding(encoding)) return 0;
    auto packet = getLegacyV0Packet(encoding);
    return packet && packet->k == 32 && packet->n == 128 && packet->warps == 4 ? 1 : -1;
  };
  LocalAllocOp roots[2];
  bool invalid = false;
  mod.walk([&](LocalAllocOp root) {
    auto enc = dyn_cast<HCUMlsSharedEncodingAttr>(root.getType().getEncoding());
    if (!enc || enc.getVersion() != 0) return;
    int i = kind(enc);
    auto t = root.getType();
    if (i < 0 || (i >= 0 && roots[i]) || t.getRank() != 3 ||
        t.getShape() != (i == 0 ? ArrayRef<int64_t>({4, 128, 32})
                               : ArrayRef<int64_t>({4, 32, 128})) ||
        t.getAllocShape() != t.getShape() || !t.getMutableMemory() ||
        !(t.getElementType().isF16() || t.getElementType().isBF16()) ||
        root.getSrc() || !root.isSharedMemoryAlloc() ||
        root.getAlignmentOrDefault() < 16 || root.getAlignmentOrDefault() % 16 ||
        !isa<triton::FuncOp>(root->getParentOp())) {
      invalid = true; return;
    }
    roots[i] = root;
  });
  if (invalid || !roots[0] || !roots[1])
    return fail("requires exact aligned mutable four-slot A/B packet roots");
  auto viewKind = [&](MemDescType t) -> int {
    int i = kind(t.getEncoding());
    if (i < 0 || t.getRank() != 2 ||
        t.getShape() != (i == 0 ? ArrayRef<int64_t>({128, 32})
                               : ArrayRef<int64_t>({32, 128})) ||
        t.getAllocShape() != t.getShape() || !t.getMutableMemory() ||
        t.getElementType() != roots[i].getType().getElementType() ||
        t.getMemorySpace() != roots[i].getType().getMemorySpace()) return -1;
    return i;
  };
  auto isV0 = [](Type t) {
    auto m = dyn_cast<MemDescType>(t);
    auto e = m ? dyn_cast<HCUMlsSharedEncodingAttr>(m.getEncoding()) : HCUMlsSharedEncodingAttr{};
    return e && e.getVersion() == 0;
  };
  // Address provenance only: opaque views originate in aligned local roots,
  // whole MemDescIndex views and ordinary CF transport. No unknown descriptor
  // constructor or external memdesc ABI may bypass the alignment contract.
  mod.walk([&](Operation *op) {
    for (Type t : op->getResultTypes())
      if (isV0(t) && !isa<LocalAllocOp, MemDescIndexOp>(op)) invalid = true;
    // Only packet-aware memory operations may consume the opaque address
    // recipe. Generic shared stores/conversions/calls cannot reinterpret it.
    for (Value operand : op->getOperands()) {
      if (!isV0(operand.getType())) continue;
      bool supported = isa<cf::BranchOp, cf::CondBranchOp, LocalDeallocOp>(op);
      if (auto copy = dyn_cast<triton::hcugpu::MatrixLoadToLocalOp>(op))
        supported = operand == copy.getDest();
      if (auto read = dyn_cast<LocalLoadOp>(op)) supported = operand == read.getSrc();
      if (auto index = dyn_cast<MemDescIndexOp>(op)) supported = operand == index.getSrc();
      if (!supported) invalid = true;
    }
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          if (isV0(arg.getType()) && (!isa<triton::FuncOp>(op) ||
              &block == &region.front() || viewKind(cast<MemDescType>(arg.getType())) < 0))
            invalid = true;
    if (auto index = dyn_cast<MemDescIndexOp>(op)) {
      if (!isV0(index.getType())) return;
      int i = viewKind(index.getType());
      if (i < 0 || index.getSrc() != roots[i].getResult()) invalid = true;
    }
  });
  if (invalid) return fail("requires whole packet views and only supported opaque producers/consumers");
  triton::ModuleAxisInfoAnalysis axis(mod);
  unsigned producers[2] = {0, 0}, consumers[2] = {0, 0};
  LogicalResult result = success();
  mod.walk([&](Operation *op) {
    if (failed(result)) return;
    if (auto copy = dyn_cast<triton::hcugpu::MatrixLoadToLocalOp>(op)) {
      if (!isV0(copy.getDest().getType())) return;
      int i = viewKind(copy.getDest().getType());
      auto fn = copy->getParentOfType<triton::FuncOp>();
      auto base = dyn_cast<BlockArgument>(copy.getBase());
      auto ptr = dyn_cast<triton::PointerType>(copy.getBase().getType());
      auto ml = copy->getAttrOfType<MlsEncodingAttr>(MlsEncodingAttr::getMnemonic());
      if (i < 0 || !fn || copy->getBlock()->getParentOp() != fn.getOperation() ||
          !base || base.getOwner() != &fn.getBody().front() || base.getArgNumber() != i ||
          !ptr || ptr.getAddressSpace() != 1 ||
          ptr.getPointeeType() != copy.getDest().getType().getElementType() ||
          !ml || ml.getVersion() != 0 || ml.getOpIdx() != unsigned(i) ||
          ml.getElemBitWidth() != 16 || ml.getAlt2Kind() != 0 ||
          ml.getMlsTile() != (i == 0 ? ArrayRef<unsigned>({128, 32})
                                    : ArrayRef<unsigned>({32, 128})) ||
          ml.getOrder() != ArrayRef<unsigned>({1, 0}) ||
          ml.getWarpsPerCTA() != ArrayRef<unsigned>({2, 2}) ||
          copy.getTensorShape() != (i == 0 ? ArrayRef<int32_t>({128, 32})
                                          : ArrayRef<int32_t>({32, 128})) ||
          copy.getShape().size() != 2 || copy.getStrides().size() != 2 ||
          copy.getIndices().size() != 2 || !copy.getBoundaryCheck().empty() ||
          copy.getCache() != triton::CacheModifier::NONE ||
          copy.getEvict() != triton::EvictionPolicy::NORMAL || copy.getIsVolatile()) {
        result = fail("invalid packet producer/source/default-memory contract"); return;
      }
      auto align = fn.getArgAttrOfType<IntegerAttr>(i, "tt.divisibility");
      if (!align || !align.getValue().isSignedIntN(64) ||
          align.getInt() < 16 || align.getInt() % 16) {
        result = fail("source argument needs explicit 16B alignment"); return;
      }
      auto extent = legacyV0SafeInteger(copy.getShape()[i]);
      if (!extent || *extent < 128 || *extent % 128 ||
          *extent >= (int64_t(1) << 20) ||
          legacyV0SafeInteger(copy.getShape()[1 - i]) != 1024 ||
          legacyV0SafeInteger(copy.getStrides()[0]) != (i == 0 ? 1024 : *extent) ||
          legacyV0SafeInteger(copy.getStrides()[1]) != 1) {
        result = fail("source requires contiguous fixed-K1024 full-128 tiles and I32 byte extent"); return;
      }
      for (unsigned d = 0; d < 2; ++d)
        if (!copy.getIndices()[d].getType().isInteger(32)) {
          result = fail("source indices require I32 operands"); return;
        }
      auto *column = axis.getAxisInfo(copy.getIndices()[1]);
      if (!column || column->getDivisibility().size() != 1 ||
          column->getDivisibility(0) < 8) {
        result = fail("source column needs proven eight-half alignment"); return;
      }
      ++producers[i];
    }
    if (auto read = dyn_cast<LocalLoadOp>(op)) {
      if (!isV0(read.getSrc().getType())) return;
      int i = viewKind(read.getSrc().getType());
      auto fn = read->getParentOfType<triton::FuncOp>();
      auto packet = getLegacyV0Packet(read.getSrc().getType().getEncoding());
      if (i < 0 || !fn || read->getBlock()->getParentOp() != fn.getOperation() ||
          read.getType().getElementType() != read.getSrc().getType().getElementType() ||
          (i == 0 ? !isLegacyV0IndexedADot(read.getType())
                  : !packet || !isLegacyV0S25Kw8Dot(read.getType(), *packet))) {
        result = fail("consumer needs exact A/B KW8 TPW22 dot ownership"); return;
      }
      ++consumers[i];
    }
  });
  if (failed(result)) return failure();
  if (!producers[0] || !producers[1] || !consumers[0] || !consumers[1])
    return fail("both packet primitives need producers and consumers");
  return success();
}

// Exact same-iteration WARs of the already-validated paired indexed packets.
// Called after addLocalBarrierAfterAmdGpuAsyncWait, before generic membar.
llvm::SmallVector<std::pair<Operation *, Operation *>>
prepareLegacyV0IndexedABWAR(ModuleOp mod, ModuleAllocation &allocation) {
  auto is = [&](StringRef key, int32_t value) {
    auto a = mod->getAttrOfType<IntegerAttr>(key);
    return a && a.getType().isInteger(32) && a.getInt() == value;
  };
  auto target = mod->getAttrOfType<StringAttr>("ttg.target");
  if (!is("ttg.hcu.v0_indexed_a_packet", 1) || !is("ttg.hcu.v0_blockptr", 1) ||
      !is("ttg.hcu.v0_mls_local_prefetch_capture", 1) ||
      !is("ttg.hcu.v0_blockptr_loop", 1) || !is("ttg.num-warps", 4) ||
      !is("ttg.num-ctas", 1) || !is("ttg.threads-per-warp", 64) ||
      !target || target.getValue() != "hip:gfx936") return {};
  // The existing primitive preflight has checked types/ownership and forbids
  // unknown opaque users. Do not duplicate source, token or GEMM semantics.
  LocalAllocOp roots[2];
  mod.walk([&](LocalAllocOp root) {
    auto e = dyn_cast<HCUMlsSharedEncodingAttr>(root.getType().getEncoding());
    if (e && e.getVersion() == 0 && e.getOpIdx() < 2 && !roots[e.getOpIdx()])
      roots[e.getOpIdx()] = root;
  });
  if (!roots[0] || !roots[1]) return {};
  auto fn = roots[0]->getParentOfType<triton::FuncOp>();
  if (!fn || roots[1]->getParentOfType<triton::FuncOp>() != fn) return {};
  Block *entry = &fn.getBody().front();
  SmallVector<MemDescIndexOp> prefix[2];
  MemDescIndexOp futureViews[2];
  triton::hcugpu::MatrixLoadToLocalOp future[2];
  for (unsigned i = 0; i < 2; ++i) {
    auto rt = roots[i].getType();
    if (rt.getRank() != 3 || rt.getShape()[0] != 4 ||
        roots[i]->getBlock() != entry) return {};
    prefix[i].resize(4);
    for (Operation *user : roots[i].getResult().getUsers()) {
      if (isa<LocalDeallocOp>(user)) continue;
      auto view = dyn_cast<MemDescIndexOp>(user);
      if (!view || view.getType().getShape() != rt.getShape().drop_front() ||
          view.getType().getAllocShape() != view.getType().getShape() ||
          view.getType().getEncoding() != rt.getEncoding()) return {};
      if (view->getBlock() == entry) {
        auto slot = legacyV0SafeInteger(view.getIndex());
        if (!slot || *slot < 0 || *slot > 3 || prefix[i][*slot]) return {};
        prefix[i][*slot] = view;
      } else {
        if (futureViews[i]) return {};
        futureViews[i] = view;
        for (Operation *vuser : view.getResult().getUsers())
          if (auto copy = dyn_cast<triton::hcugpu::MatrixLoadToLocalOp>(vuser)) {
            if (future[i] || copy.getDest() != view.getResult()) return {};
            future[i] = copy;
          }
      }
    }
    if (!future[i] || llvm::any_of(prefix[i], [](auto v) { return !v; })) return {};
  }
  auto loop = getLegacyV0CountedBody(future[0], 28, 0);
  if (!loop || future[1]->getBlock() != loop->body) return {};
  Block *body = loop->body;
  auto back = cast<cf::BranchOp>(body->getTerminator());
  Block *header = back.getDest();
  auto enter = cast<cf::BranchOp>(entry->getTerminator());
  Block *exit = cast<cf::CondBranchOp>(header->getTerminator()).getFalseDest();
  auto head = dyn_cast<triton::hcugpu::AsyncWaitOp>(&body->front());
  auto publication = head ? dyn_cast_or_null<LocalBarrierOp>(head->getNextNode()) : LocalBarrierOp{};
  if (!head || head.getNumInst() != 8 || !publication) return {};
  Value index = futureViews[0].getIndex();
  if (futureViews[1].getIndex() != index) return {};
  auto select = index.getDefiningOp<arith::SelectOp>();
  auto inc = select ? select.getTrueValue().getDefiningOp<arith::AddIOp>() : arith::AddIOp{};
  auto cmp = select ? select.getCondition().getDefiningOp<arith::CmpIOp>() : arith::CmpIOp{};
  auto counter = inc ? dyn_cast<BlockArgument>(inc.getLhs()) : BlockArgument{};
  if (!select || !inc || !cmp || !counter || counter.getOwner() != header ||
      !counter.getType().isInteger(32) || select->getBlock() != body ||
      inc->getBlock() != body || cmp->getBlock() != body ||
      legacyV0SafeInteger(enter.getDestOperands()[counter.getArgNumber()]) != 3 ||
      back.getDestOperands()[counter.getArgNumber()] != index ||
      legacyV0SafeInteger(inc.getRhs()) != 1 || cmp.getLhs() != inc.getResult() ||
      legacyV0SafeInteger(cmp.getRhs()) != 4 ||
      cmp.getPredicate() != arith::CmpIPredicate::slt ||
      legacyV0SafeInteger(select.getFalseValue()) != 0) return {};
  auto *data = allocation.getFuncData(cast<FunctionOpInterface>(fn.getOperation()));
  if (!data || allocation.getSharedMemorySize() != 65536 || !is("ttg.shared", 65536)) return {};
  uint64_t bases[2];
  LocalLoadOp reads[2];
  for (unsigned i = 0; i < 2; ++i) {
    auto id = data->getBufferId(roots[i].getResult());
    if (id == Allocation::InvalidBufferId || data->getAllocatedSize(id) != 32768) return {};
    bases[i] = data->getOffset(id);
    auto offset = roots[i]->getAttrOfType<IntegerAttr>("allocation.offset");
    if (!offset || !offset.getType().isInteger(32) || offset.getInt() < 0 ||
        uint64_t(offset.getInt()) != bases[i] || bases[i] % 8192 || bases[i] > 32768) return {};
    SmallVector<Value> seeds{prefix[i][1].getResult(), prefix[i][2].getResult(), prefix[i][3].getResult()};
    auto queue = matchLegacyV0HeaderQueue(header, enter, back, seeds, futureViews[i].getResult());
    if (!queue) return {};
    for (auto read : body->getOps<LocalLoadOp>()) if (read.getSrc() == queue->front()) {
      if (reads[i] || read.getToken() != head.getResult()) return {};
      reads[i] = read;
    }
    if (!reads[i] || !publication->isBeforeInBlock(reads[i]) ||
        !reads[i]->isBeforeInBlock(future[i])) return {};
    for (Operation *user : roots[i].getResult().getUsers())
      if (auto free = dyn_cast<LocalDeallocOp>(user)) {
        if (free->getBlock() != exit) return {};
        for (auto read : exit->getOps<LocalLoadOp>())
          if (!read->isBeforeInBlock(free)) return {};
      }
  }
  if (!(bases[0] + 32768 <= bases[1] || bases[1] + 32768 <= bases[0])) return {};
  // Prefix1/2/3 + backshift establish R=(i+1)%4; counter3/+1%4
  // establishes W=i%4. The head DS0+CTA protects R(i-1)=W(i),
  // so only same-iteration R->W differs from whole-allocation aliasing.
  return {{reads[0].getOperation(), future[0].getOperation()},
          {reads[1].getOperation(), future[1].getOperation()}};
}

LogicalResult verifyLegacyMlsV0ClosedModule(ModuleOp mod, StringRef arch) {
  if (failed(verifyRawWindows(mod, arch))) return failure();
  if (failed(verifyWholePackets(mod, arch))) return failure();
  if (failed(verifyStructuredRegisterLoads(mod, arch))) return failure();
  bool hasV0 = false;
  mod.walk([&](Operation *op) {
    SmallVector<Type> types = llvm::to_vector(op->getResultTypes());
    llvm::append_range(types, op->getOperandTypes());
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          types.push_back(arg.getType());
    for (Type type : types)
      if (auto mem = dyn_cast<MemDescType>(type))
        if (auto enc = dyn_cast<HCUMlsSharedEncodingAttr>(mem.getEncoding()))
          hasV0 |= enc.getVersion() == 0;
  });
  if (!hasV0)
    return success();
  auto fail = [&](StringRef message) { return mod.emitError(message); };
  if (arch != "gfx936" || TritonGPUDialect::getThreadsPerWarp(mod) != 64 ||
      TritonGPUDialect::getNumCTAs(mod) != 1)
    return fail("legacy-v0 requires gfx936, wave64, one CTA");
  bool hasIndexedA = false;
  mod.walk([&](LocalAllocOp a) {
    auto e = dyn_cast<HCUMlsSharedEncodingAttr>(a.getType().getEncoding());
    hasIndexedA |= e && e.getVersion() == 0 && e.getOpIdx() == 0;
  });
  if (hasIndexedA) {
    auto isOne = [&](StringRef key) {
      auto a = mod->getAttrOfType<IntegerAttr>(key);
      return a && a.getType().isInteger(32) && a.getInt() == 1;
    };
    if (isOne("ttg.hcu.v0_indexed_a_packet") && isOne("ttg.hcu.v0_blockptr") &&
        isOne("ttg.hcu.v0_mls_local_prefetch_capture"))
      return verifyLegacyV0PairedPrimitives(mod);
    return verifyLegacyV0IndexedAClosedModule(mod);
  }
  bool hasPrefix2 = false;
  mod.walk([&](LocalAllocOp a) {
    auto e = dyn_cast<HCUMlsSharedEncodingAttr>(a.getType().getEncoding());
    hasPrefix2 |= e && e.getVersion() == 0 && a.getType().getRank() == 3 &&
                  a.getType().getShape()[0] == 2;
  });
  auto localPrefetch = mod->getAttrOfType<IntegerAttr>(
      "ttg.hcu.v0_mls_local_prefetch_capture");
  if (localPrefetch && !localPrefetch.getType().isInteger(32))
    return fail("legacy-v0 local-prefetch hint must be i32");
  if (localPrefetch && localPrefetch.getInt() == 1)
    return verifyLegacyV0LocalPrefetch(mod);
  if (hasPrefix2)
    return verifyLegacyV0Prefix2(mod);
  auto loopMode = mod->getAttrOfType<IntegerAttr>("ttg.hcu.legacy_v0_static_loop");
  auto pureLoopMode = mod->getAttrOfType<IntegerAttr>("ttg.hcu.v0_blockptr_loop");
  if ((loopMode && loopMode.getInt() == 1) ||
      (pureLoopMode && pureLoopMode.getInt() == 1))
    return verifyLegacyV0StaticLoop(mod);
  SmallVector<LocalAllocOp> allocs;
  bool forbidden = false;
  unsigned functionCount = 0;
  mod.walk([&](Operation *op) {
    if (auto alloc = dyn_cast<LocalAllocOp>(op))
      allocs.push_back(alloc);
    auto name = op->getName().getStringRef();
    if (!name.starts_with("arith.") &&
        !llvm::is_contained(ArrayRef<StringRef>({
          "builtin.module", "tt.func", "tt.return", "tt.make_range",
          "tt.expand_dims", "tt.broadcast", "tt.splat", "tt.addptr",
          "tt.store", "tt.load", "tt.dot", "ttg.local_alloc", "ttg.memdesc_index",
          "ttg.local_load", "ttg.async_commit_group", "hcug.async_wait",
          "ttg.local_barrier", "hcug.matrix_load_to_local"}), name))
      forbidden = true;
    // No generic layout exchange/scratch, calls, control flow, loops or
    // escaping memdesc ABI. Slice+expand_dim on distributed indices is fine.
    if (isa<ConvertLayoutOp, triton::CallOp, LocalStoreOp>(op) ||
        (op->getNumRegions() && op != mod.getOperation() && !isa<triton::FuncOp>(op)))
      forbidden = true;
    if (auto fn = dyn_cast<triton::FuncOp>(op)) {
      ++functionCount;
      if (!fn.getBody().hasOneBlock())
        forbidden = true;
      // The closed frame cannot alias a pointer-valued LDS function ABI.
      for (Type type : fn.getArgumentTypes())
        if (auto pointer = dyn_cast<triton::PointerType>(type))
          forbidden |= pointer.getAddressSpace() != 1;
    }
    if (auto store = dyn_cast<triton::StoreOp>(op)) {
      Type type = store.getPtr().getType();
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        type = tensor.getElementType();
      auto pointer = dyn_cast<triton::PointerType>(type);
      forbidden |= !pointer || pointer.getAddressSpace() != 1;
    }
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          forbidden |= isa<MemDescType>(arg.getType());
  });
  auto pureTLAttr = mod->getAttrOfType<IntegerAttr>("ttg.hcu.v0_blockptr");
  bool pureTL = pureTLAttr && pureTLAttr.getInt() == 1;
  if (forbidden || allocs.size() != (pureTL ? 2 : 1) || functionCount != 1)
    return fail("legacy-v0 requires its exact closed allocation protocol and no control/layout exchange");
  // Ordinary A staging has its own initialized root. Identify B by the opaque
  // version0 encoding, not traversal order; the value matcher below validates A.
  LocalAllocOp alloc;
  for (auto candidate : allocs) {
    auto enc = dyn_cast<HCUMlsSharedEncodingAttr>(candidate.getType().getEncoding());
    if (!enc || enc.getVersion() != 0)
      continue;
    if (alloc)
      return fail("legacy-v0 requires exactly one opaque B root");
    alloc = candidate;
  }
  if (!alloc)
    return fail("legacy-v0 requires an opaque B root");
  auto rootTy = alloc.getType();
  auto packet = getLegacyV0Packet(rootTy.getEncoding());
  if (!packet || rootTy.getRank() != 3 ||
      (rootTy.getShape()[0] != 1 && rootTy.getShape()[0] != 8) ||
      rootTy.getShape()[1] != packet->k || rootTy.getShape()[2] != packet->n ||
      !(rootTy.getElementType().isF16() || rootTy.getElementType().isBF16()) ||
      rootTy.getAllocShape() != rootTy.getShape() ||
      alloc.getSrc() || !alloc.isSharedMemoryAlloc() ||
      alloc.getAlignmentOrDefault() < 16 || alloc.getAlignmentOrDefault() % 16 ||
      !alloc.getResult().hasOneUse())
    return fail("legacy-v0 requires uninitialized aligned8KiB/64KiB frame with1/8 packet regions");
  auto index = dyn_cast<MemDescIndexOp>(*alloc.getResult().getUsers().begin());
  auto slot = index ? legacyV0Constant(index.getIndex()) : std::optional<int64_t>{};
  // Keep the64KiB origin whitelist; a one-packet8KiB root admits view0 only.
  if (!slot || !llvm::is_contained(ArrayRef<int64_t>({0, 1, 2, 4}), *slot) ||
      *slot >= rootTy.getShape()[0])
    return fail("legacy-v0 prototype permits only in-frame whole packet views0/1/2/4");
  auto slice = index.getType();
  if (slice.getRank() != 2 || slice.getShape()[0] != packet->k ||
      slice.getShape()[1] != packet->n || slice.getEncoding() != rootTy.getEncoding() ||
      slice.getElementType() != rootTy.getElementType() ||
      slice.getMemorySpace() != rootTy.getMemorySpace() || !slice.getMutableMemory() ||
      slice.getAllocShape() != slice.getShape())
    return fail("legacy-v0 index must preserve exact packet type/encoding");
  triton::hcugpu::MatrixLoadToLocalOp copy;
  LocalLoadOp load;
  unsigned users = 0;
  for (Operation *user : index.getResult().getUsers()) {
    ++users;
    if (auto c = dyn_cast<triton::hcugpu::MatrixLoadToLocalOp>(user))
      copy = c;
    else if (auto l = dyn_cast<LocalLoadOp>(user))
      load = l;
    else
      return fail("legacy-v0 memdesc escapes paired producer/consumer");
  }
  bool s25Kw8 = load && isLegacyV0S25Kw8Dot(load.getType(), *packet);
  if (users != 2 || !copy || !load || copy.getDest() != index.getResult() ||
      load.getSrc() != index.getResult() ||
      (!isLegacyV0Dot(load.getType(), *packet) && !s25Kw8) ||
      copy->getBlock() != load->getBlock() || !copy->isBeforeInBlock(load))
    return fail("legacy-v0 needs exactly one producer and proved normal B-dot consumer");
  auto ml = copy->getAttrOfType<triton::hcugpu::MlsEncodingAttr>(
      triton::hcugpu::MlsEncodingAttr::getMnemonic());
  auto nw = mod->getAttrOfType<IntegerAttr>("ttg.num-warps");
  if (!ml || !nw || nw.getInt() != packet->warps || ml.getVersion() != 0 ||
      ml.getOpIdx() != 1 || ml.getElemBitWidth() != 16 || ml.getAlt2Kind() != 0 ||
      ml.getMlsTile() != ArrayRef<unsigned>({packet->k, packet->n}) ||
      ml.getOrder() != ArrayRef<unsigned>({1, 0}) ||
      ml.getWarpsPerCTA() != ArrayRef<unsigned>({packet->warps / 2, 2}) ||
      copy.getTensorShape() != ArrayRef<int32_t>({int32_t(packet->k), int32_t(packet->n)}) ||
      copy.getShape().size() != 2 || copy.getStrides().size() != 2 ||
      copy.getIndices().size() != 2 || !copy.getBoundaryCheck().empty() ||
      copy.getCache() != triton::CacheModifier::NONE ||
      copy.getEvict() != triton::EvictionPolicy::NORMAL || copy.getIsVolatile())
    return fail("legacy-v0 producer requires exact measured metadata and default memory semantics");
  auto source = getLegacyV0SourceTuple(copy, *packet);
  if (!source)
    return fail("legacy-v0 requires exact full-square source tuple and proved full-tile corner/control bounds");
  auto sourceArg = dyn_cast<BlockArgument>(copy.getBase());
  auto function = alloc->getParentOfType<triton::FuncOp>();
  auto pointer = dyn_cast<triton::PointerType>(copy.getBase().getType());
  auto alignment = sourceArg && function
      ? function.getArgAttrOfType<IntegerAttr>(sourceArg.getArgNumber(), "tt.divisibility")
      : IntegerAttr{};
  if (!sourceArg || !function || sourceArg.getOwner() != &function.getBody().front() ||
      !pointer || pointer.getAddressSpace() != 1 ||
      pointer.getPointeeType() != rootTy.getElementType() || !alignment ||
      alignment.getInt() < 16 || alignment.getInt() % 16)
    return fail("legacy-v0 requires an explicitly16B-aligned global source argument");
  // Source position, last vector, descriptor and signed-i32 bounds were
  // checked by the same helper used by emission. A full2*G*G-byte backing
  // allocation remains an explicit kernel ABI/host obligation.
  if (!copy.getToken().hasOneUse())
    return fail("legacy-v0 producer token must have one commit/wait chain");
  auto commit = dyn_cast<AsyncCommitGroupOp>(*copy.getToken().getUsers().begin());
  if (!commit || !commit.getResult().hasOneUse())
    return fail("legacy-v0 requires one async commit");
  auto wait = dyn_cast<triton::hcugpu::AsyncWaitOp>(*commit.getResult().getUsers().begin());
  if (!wait || wait.getNumInst() != 0 || !wait.getResult().hasOneUse() ||
      load.getToken() != wait.getResult() ||
      copy->getNextNode() != commit.getOperation() || commit->getNextNode() != wait.getOperation() ||
      !isa_and_nonnull<LocalBarrierOp>(wait->getNextNode()) ||
      wait->getNextNode()->getNextNode() != load.getOperation())
    return fail("legacy-v0 requires explicit fullwait0+CTA publication before read");
  if (s25Kw8) {
    // Only the existing opaque-v0, one-shot B-copy protocol opts in. Keep
    // old A staging, loop, prefix2 and local-prefetch witnesses unchanged.
    bool hasAOrDot = false;
    mod.walk([&](Operation *op) {
      hasAOrDot |= isa<triton::LoadOp, triton::DotOp>(op);
    });
    if (pureTL || rootTy.getShape()[0] != 1 || *slot != 0 || hasAOrDot)
      return fail("legacy-v0 KW8/TPW22 requires the single-packet B-copy-only protocol");
  }
  if (pureTL) {
    if (!matchLegacyV0StagedA(mod, function, alloc, load, *packet))
      return fail("legacy-v0 pure-TL requires the exact ordinary A staging and direct MMA C-store protocol");
    return success();
  }
  return verifyLegacyV0ValueUses(mod, load, *packet, function);
}
// Private final-position diagnostic. Default path is a no-op. This is called
// only by the existing HCU allocation pass, after all ordinary reordering.
LogicalResult prepareLegacyMlsV0JointPhase(ModuleOp module) {
    auto fail = [&](StringRef reason) -> LogicalResult {
      return module.emitError("legacy-v0 joint phase: ") << reason;
    };
    auto requested = module->getAttrOfType<IntegerAttr>("ttg.hcu.v0_joint_phase");
    if (!requested || requested.getInt() == 0)
      return success();
    if (requested.getInt() != 1)
      return fail("only private mode0/1 is supported");
    auto isOne = [&](StringRef name) {
      auto attr = module->getAttrOfType<IntegerAttr>(name);
      return attr && attr.getInt() == 1;
    };
    auto target = module->getAttrOfType<StringAttr>("ttg.target");
    if (!target || target.getValue() != "hip:gfx936" ||
        !isOne("ttg.hcu.v0_blockptr") || !isOne("ttg.hcu.v0_blockptr_loop") ||
        module->hasAttr("ttg.shared"))
      return fail("needs unallocated existing pure-TL loop input, no derived allocation metadata");
    // Full current closed CF/source/token/type/effect proof precedes movement.
    if (failed(triton::HCU::verifyLegacyMlsV0ClosedModule(module, "gfx936")))
      return failure();
    SmallVector<triton::gpu::LocalAllocOp> roots;
    SmallVector<triton::gpu::LocalLoadOp> reads;
    SmallVector<triton::hcugpu::MatrixLoadToLocalOp> copies;
    SmallVector<triton::LoadOp> globalLoads;
    module.walk([&](Operation *op) {
      if (auto x = dyn_cast<triton::gpu::LocalAllocOp>(op)) roots.push_back(x);
      if (auto x = dyn_cast<triton::gpu::LocalLoadOp>(op)) reads.push_back(x);
      if (auto x = dyn_cast<triton::hcugpu::MatrixLoadToLocalOp>(op)) copies.push_back(x);
      if (auto x = dyn_cast<triton::LoadOp>(op)) globalLoads.push_back(x);
    });
    if (roots.size() != 2 || reads.size() != 2 || copies.size() != 1 ||
        globalLoads.size() != 1)
      return fail("exact two-root/two-read/one-copy/one-global-load counts");
    auto copy = copies.front();
    triton::gpu::LocalAllocOp aRoot, bRoot;
    for (auto root : roots) {
      if (root->hasAttr("allocation.offset"))
        return fail("allocation attributes must be absent; rerun allocator afterwards");
      if (root.getSrc()) aRoot = root;
      else bRoot = root;
    }
    if (!aRoot || !bRoot || aRoot.getSrc() != globalLoads.front().getResult())
      return fail("unique initialized A and uninitialized B roots");
    auto view = copy.getDest().getDefiningOp<triton::gpu::MemDescIndexOp>();
    if (!view || view.getSrc() != bRoot.getResult())
      return fail("unchanged direct B root/view/copy");
    triton::gpu::LocalLoadOp aRead, bRead;
    for (auto read : reads) {
      if (read.getSrc() == aRoot.getResult()) aRead = read;
      if (read.getSrc() == view.getResult()) bRead = read;
    }
    // The artifact-only test pass can already have performed this exact move.
    // Prove actual positions again; the module flag alone is never a certificate.
    const bool alreadyJoint = aRoot->getNextNode() == copy.getOperation() &&
        isa_and_nonnull<triton::gpu::LocalBarrierOp>(aRoot->getPrevNode());
    auto *head = alreadyJoint ? aRoot->getPrevNode() : copy->getPrevNode();
    Block *body = copy->getBlock();
    if (!aRead || !bRead || !isa_and_nonnull<triton::gpu::LocalBarrierOp>(head) ||
        aRoot->getBlock() != body || aRead->getBlock() != body ||
        bRead->getBlock() != body || globalLoads.front()->getBlock() != body ||
        !globalLoads.front()->isBeforeInBlock(head) ||
        !head->isBeforeInBlock(bRead.getOperation()) ||
        (!alreadyJoint && (!bRead->isBeforeInBlock(aRoot.getOperation()) ||
                           aRoot->getNextNode() != aRead.getOperation())))
      return fail("exact globalA/head/Bcopy/Bread/Ainit/Aread source order");
    // The existing head barrier remains before BOTH shared writes. All local
    // reads remain after the existing wait0+publication barrier. SSA source
    // of A initialization already dominates the insertion point in this block.
    if (alreadyJoint)
      return success(); // Full closed proof and head/Ainit/copy positions checked.
    Operation *oldNext = aRoot->getNextNode();
    aRoot->moveBefore(copy.getOperation());
    if (failed(triton::HCU::verifyLegacyMlsV0ClosedModule(module, "gfx936"))) {
      aRoot->moveBefore(oldNext);
      return failure();
    }
    // No allocator metadata is installed here. The immediately following REAL
    // allocator must independently produce B0/8192 and the proved A payload
    // at8192 (2KiB/8KiB), total10240/16384; final allocation checks both.
    return success();
}
} // namespace mlir::triton::HCU

namespace {

// ===----------------------------------------------------------------------===//
// Utility functions
// ===----------------------------------------------------------------------===//

SmallVector<unsigned> getShapePerCTA(ArrayRef<unsigned> mlsTile,
                                     ArrayRef<unsigned> warpsPerCTA) {
  assert(mlsTile.size() == warpsPerCTA.size() && mlsTile.size() == 2);

  return {warpsPerCTA[0] * mlsTile[0], warpsPerCTA[1] * mlsTile[1]};
}

SmallVector<unsigned> getNumReps(const MlsEncodingAttr &blockLayout,
                                 ArrayRef<int64_t> shape) {
  auto rank = shape.size();
  assert(rank == 2);

  SmallVector<unsigned> numReps(rank);
  auto shapePerCTA =
      getShapePerCTA(blockLayout.getMlsTile(), blockLayout.getWarpsPerCTA());
  for (unsigned d = 0; d < rank; ++d) {
    numReps[d] = std::max<unsigned>(1, shape[d] / shapePerCTA[d]);
  }
  return numReps;
}

bool isKMajor(llvm::ArrayRef<unsigned> order, int opIdx) {
  auto rank = order.size();
  int kdim = opIdx == 0 ? rank - 1 : rank - 2;
  return order[0] == kdim;
}

} // namespace

//--------------------------------------------------------------------------------------------------

namespace {

struct MLSMatrixLoadToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::hcugpu::MatrixLoadToLocalOp> {
  using ConvertOpToLLVMPattern<
      triton::hcugpu::MatrixLoadToLocalOp>::ConvertOpToLLVMPattern;

  MLSMatrixLoadToLocalOpConversion(LLVMTypeConverter &converter,
                                   const HCU::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::hcugpu::MatrixLoadToLocalOp>(converter,
                                                                    benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::hcugpu::MatrixLoadToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    bool mABIsNeed =
        targetInfo.getGPUKind() == llvm::AMDGPU::GPUKind::GK_GFX938;

    auto ctx = rewriter.getContext();

    auto dstTy = cast<MemDescType>(op.getDest().getType());
    auto sharedLayout = dyn_cast<HCUMlsSharedEncodingAttr>(dstTy.getEncoding());
    if (sharedLayout && isPacketVersion(sharedLayout.getVersion()))
      return lowerWholePacketProducer(op, adaptor, *getTypeConverter(), rewriter);
    if (sharedLayout && sharedLayout.getVersion() == 0) {
      if (sharedLayout.getOpIdx() == 0)
        return lowerLegacyV0IndexedAProducer(op, adaptor, *getTypeConverter(), rewriter);
      return lowerLegacyV0Producer(op, adaptor, *getTypeConverter(), rewriter);
    }
    auto blockLayout =
        op->getAttrOfType<MlsEncodingAttr>(MlsEncodingAttr::getMnemonic());
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getDest(),
                                                         llvmElemTy, rewriter);

    auto opIdx = blockLayout.getOpIdx();
    auto kDimIdx = opIdx == 0 ? 1 : 0;
    auto nonKDimIdx = opIdx == 0 ? 0 : 1;
    auto mlsInsn = MlsInsn::selectOrGetMlsInsn(
        blockLayout.getMlsTile()[nonKDimIdx], blockLayout.getMlsTile()[kDimIdx],
        blockLayout.getElemBitWidth(), opIdx,
        blockLayout.getOrder()[0] == kDimIdx,
        static_cast<MlsInterleaveKind>(blockLayout.getAlt2Kind()),
        blockLayout.getVersion());
    auto mlInsnAttr = mlsInsn->getMatrixLoadInsnAttr();

    auto allocOp = op.getResult().getDefiningOp<triton::gpu::LocalAllocOp>();
    auto elemByteWidth = blockLayout.getElemBitWidth() / 8;

    auto llBasePtr = adaptor.getBase();
    auto llShape = adaptor.getShape();
    auto llStrides = adaptor.getStrides();
    auto llOffsets = adaptor.getIndices();
    auto llBlockPos = adaptor.getIndices();

    auto ldBlkOff = dot64(loc, rewriter, llOffsets, llStrides);

    auto boundaryCheck = adaptor.getBoundaryCheck();
    bool needBoundaryCheck = boundaryCheck.size() > 0;
    SmallVector<bool> boundaryCheckInfo = {
        needBoundaryCheck && boundaryCheck[0] == 0,
        needBoundaryCheck && boundaryCheck[boundaryCheck.size() - 1] == 1};
    SmallVector<unsigned> numReps = getNumReps(blockLayout, dstTy.getShape());
    unsigned numRepsX = numReps[0] * mlInsnAttr.instrsPerWarp[0];
    unsigned numRepsY = numReps[1] * mlInsnAttr.instrsPerWarp[1];

    bool useMatrixLoadStoreOffsetsWithMlOffs = true;
    if (boundaryCheckInfo[0] && boundaryCheckInfo[1]) {
      // Per mls inst need a rsrc desc for boundary padding.
      useMatrixLoadStoreOffsetsWithMlOffs = false;
    }

    if (!useMatrixLoadStoreOffsetsWithMlOffs) {
      SmallVector<std::pair<Value, Value>> ldInBlkOffCoordMappings;
      auto ldstInBlkMappings = computeMatrixLoadStoreOffsets(
          loc, rewriter, blockLayout, mlInsnAttr, sharedLayout,
          dstTy.getShape(), llStrides, mlsInsn->isRowMajor(), boundaryCheckInfo,
          ldInBlkOffCoordMappings);

      unsigned iterIdx = 0;
      for (auto &ldstInBlkGroup : ldstInBlkMappings) {
        Value ldInBlkOff64 = ldstInBlkGroup.first;

        // global mem address calc
        Value ldOffset = b.add(ldInBlkOff64, ldBlkOff);
        Value ldPtr = b.gep(ptr_ty(ctx, 1), elemByteWidth == 1 ? i8_ty : i16_ty,
                            llBasePtr, ldOffset);

        // create matrix load lds inst
        Value llStride = llStrides[sharedLayout.getOrder()[1]];
        llStride = b.trunc(i32_ty, llStride);

        Value paddingLenX = b.i32_val(0);
        Value paddingLenY = b.i32_val(0);
        if (boundaryCheckInfo[0]) {
          Value ldInBlockOffCoordX = ldInBlkOffCoordMappings[iterIdx].first;
          Value posEnd = b.add(b.add(llBlockPos[0], ldInBlockOffCoordX),
                               b.i32_val(mlInsnAttr.instrShape[0]));
          Value posGt = b.icmp_sgt(posEnd, llShape[0]);
          paddingLenX =
              b.select(posGt, b.sub(posEnd, llShape[0]), b.i32_val(0));
        }
        if (boundaryCheckInfo[1]) {
          Value ldInBlockOffCoordY = ldInBlkOffCoordMappings[iterIdx].second;
          Value posEnd = b.add(b.add(llBlockPos[1], ldInBlockOffCoordY),
                               b.i32_val(mlInsnAttr.instrShape[1]));
          Value posGt = b.icmp_sgt(posEnd, llShape[1]);
          paddingLenY =
              b.select(posGt, b.sub(posEnd, llShape[1]), b.i32_val(0));
        }
        Value mPadding =
            mlsInsn->getMajorDimIndex() == 0 ? paddingLenX : paddingLenY;
        Value nmPadding =
            mlsInsn->getMajorDimIndex() == 0 ? paddingLenY : paddingLenX;

        Value rsrcDesc =
            createRsrcDesc(loc, rewriter, ldPtr, llStride,
                           blockLayout.getAlt2Kind(), mPadding, nmPadding);

        // share mem address calc
        Value stInBlkOff32 = ldstInBlkGroup.second;
        Value smemOff = stInBlkOff32;
        Value smemPtr =
            b.gep(ptr_ty(ctx, 3), elemByteWidth == 1 ? i8_ty : i16_ty,
                  smemObj.getBase(), smemOff);
        if (mlsInsn->isTranspose() && mABIsNeed) {
          Value stPtrInt = b.ptrtoint(i32_ty, smemPtr);
          stPtrInt = b.or_(stPtrInt, b.i32_val(0x80000000));
          smemPtr = b.inttoptr(ptr_ty(ctx, 3), stPtrInt);
        }
        bool t = mlsInsn->isRowMajor();
        bool r = false;
        generateMatrixLoadLDSOp(loc, rewriter, mlInsnAttr.insn, rsrcDesc,
                                smemPtr, t, 0, r, false, false, false);
        iterIdx++;
      }
    } else {
      // compute matrix load/store offsets with desc group and diff ml offset.
      bool mlOffsInY;
      DenseMap<Value, Value> ldInBlkOffCoordMappings;
      auto ldstInBlkMappings = computeMatrixLoadStoreOffsetsWithMlOffs(
          loc, rewriter, blockLayout, mlInsnAttr, sharedLayout,
          dstTy.getShape(), llStrides, mlsInsn->isRowMajor(), boundaryCheckInfo,
          mlOffsInY, ldInBlkOffCoordMappings);
      bool groupPaddingInY = !mlOffsInY;

      for (auto &ldstInBlkGroup : ldstInBlkMappings) {
        Value ldInBlkOff64 = ldstInBlkGroup.first;

        // global mem address calc
        Value ldOffset = b.add(ldInBlkOff64, ldBlkOff);
        Value ldPtr = b.gep(ptr_ty(ctx, 1), elemByteWidth == 1 ? i8_ty : i16_ty,
                            llBasePtr, ldOffset);

        // create matrix load lds inst
        Value llStride = llStrides[sharedLayout.getOrder()[1]];
        llStride = b.trunc(i32_ty, llStride);

        Value rsrcDesc;
        for (auto &ldstInBlkOff : ldstInBlkGroup.second) {
          unsigned ldInBlkOffC = ldstInBlkOff.first;
          Value stInBlkOff32 = ldstInBlkOff.second;

          if (!rsrcDesc) {
            Value paddingLen = b.i32_val(0);
            if (needBoundaryCheck) {
              int dimIdx = groupPaddingInY ? 1 : 0;
              Value ldInBlockOffCoordV = ldInBlkOffCoordMappings[ldInBlkOff64];
              Value posEnd =
                  b.add(b.add(llBlockPos[dimIdx], ldInBlockOffCoordV),
                        b.i32_val(mlInsnAttr.instrShape[dimIdx]));
              Value posGt = b.icmp_sgt(posEnd, llShape[dimIdx]);
              paddingLen =
                  b.select(posGt, b.sub(posEnd, llShape[dimIdx]), b.i32_val(0));
            }

            Value mPadding = ((groupPaddingInY && mlsInsn->isRowMajor()) ||
                              (!groupPaddingInY && !mlsInsn->isRowMajor()))
                                 ? paddingLen
                                 : b.i32_val(0);
            Value nmPadding =
                mPadding == paddingLen ? b.i32_val(0) : paddingLen;
            rsrcDesc =
                createRsrcDesc(loc, rewriter, ldPtr, llStride,
                               blockLayout.getAlt2Kind(), mPadding, nmPadding);
          }

          // share mem address calc
          Value smemOff = stInBlkOff32;
          Value smemPtr =
              b.gep(ptr_ty(ctx, 3), elemByteWidth == 1 ? i8_ty : i16_ty,
                    smemObj.getBase(), smemOff);
          if (mlsInsn->isTranspose() && mABIsNeed) {
            Value stPtrInt = b.ptrtoint(i32_ty, smemPtr);
            stPtrInt = b.or_(stPtrInt, b.i32_val(0x80000000));
            smemPtr = b.inttoptr(ptr_ty(ctx, 3), stPtrInt);
          }

          bool t = mlsInsn->isRowMajor();
          bool r = mlOffsInY;
          assert(ldInBlkOffC < 1024 && "ldInBlkOffC out of range"); // 2^10
          generateMatrixLoadLDSOp(loc, rewriter, mlInsnAttr.insn, rsrcDesc,
                                  smemPtr, t, ldInBlkOffC, r, false, false,
                                  false);
        }
      }
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const HCU::TargetInfo &targetInfo;

  /*
   * Calculate matrix load store offsets in blk: ldOffVal, stOffVal
   **/
  SmallVector<std::pair<Value, Value>> computeMatrixLoadStoreOffsets(
      Location loc, RewriterBase &rewriter, const MlsEncodingAttr &blockLayout,
      const MatrixLoadInsnAttr &mlInsnAttr,
      const HCUMlsSharedEncodingAttr &sharedLayout,
      ArrayRef<int64_t> tensorShape, ValueRange llStrides, bool mlsIsRowMajor,
      SmallVector<bool> boundaryCheckInfo,
      SmallVector<std::pair<Value, Value>> &ldInBlkOffCoordMappings) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto warpsPerCta = blockLayout.getWarpsPerCTA();
    auto shapePerCta = getShapePerCTA(blockLayout.getMlsTile(), warpsPerCta);
    SmallVector<Value> multiDimWarpIds;
    auto dummyMapping = computeMatrixLoadStoreOffsetsMappingInCTA(
        loc, rewriter, blockLayout, mlInsnAttr, tensorShape, multiDimWarpIds);

    auto dsByteOffsets = mlInsnAttr.dsByteOffsets;
    auto elemByteWidth = blockLayout.getElemBitWidth() / 8;
    SmallVector<unsigned> flattenedMlsShape{tensorShape};
    flattenedMlsShape[0] = tensorShape[0] / blockLayout.getMlsTile()[0];
    flattenedMlsShape[1] = tensorShape[1] / blockLayout.getMlsTile()[1];

    auto rank = tensorShape.size();
    Value offWarpX = b.mul(multiDimWarpIds[rank - 2],
                           b.i32_val(blockLayout.getMlsTile()[rank - 2]));
    Value offWarpY = b.mul(multiDimWarpIds[rank - 1],
                           b.i32_val(blockLayout.getMlsTile()[rank - 1]));

    SmallVector<unsigned> numReps = getNumReps(blockLayout, tensorShape);
    unsigned numRepsX = numReps[0] * mlInsnAttr.instrsPerWarp[0];
    unsigned numRepsY = numReps[1] * mlInsnAttr.instrsPerWarp[1];
    bool needBoundaryCheck = boundaryCheckInfo[0] || boundaryCheckInfo[1];

    SmallVector<std::pair<Value, Value>> loadStoreOffsets;

    for (unsigned ctaX = 0; ctaX < numReps[0]; ++ctaX) {
      for (unsigned ctaY = 0; ctaY < numReps[1]; ++ctaY) {
        for (unsigned instIdxX = 0; instIdxX < mlInsnAttr.instrsPerWarp[0];
             ++instIdxX) {
          for (unsigned instIdxY = 0; instIdxY < mlInsnAttr.instrsPerWarp[1];
               ++instIdxY) {
            unsigned ctaRow = ctaX * shapePerCta[0];
            unsigned ctaCol = ctaY * shapePerCta[1];
            Value ldRow =
                b.add(offWarpX,
                      b.i32_val(ctaRow + instIdxX * mlInsnAttr.instrShape[0]));
            Value ldCol =
                b.add(offWarpY,
                      b.i32_val(ctaCol + instIdxY * mlInsnAttr.instrShape[1]));

            Value ldOffV = dot64(loc, rewriter, {ldRow, ldCol}, llStrides);
            Value stMlsRow =
                b.add(b.i32_val(ctaX * warpsPerCta[0]), multiDimWarpIds[0]);
            Value stMlsCol =
                b.add(b.i32_val(ctaY * warpsPerCta[1]), multiDimWarpIds[1]);

            Value stMlsOff =
                b.mul(linearize(rewriter, loc, {stMlsRow, stMlsCol},
                                flattenedMlsShape, sharedLayout.getOrder()),
                      b.i32_val(product(blockLayout.getMlsTile())));
            unsigned instIdx =
                instIdxX * mlInsnAttr.instrsPerWarp[1] + instIdxY;
            Value stInMlsOff =
                b.i32_val(dsByteOffsets[instIdx] /
                          elemByteWidth); // element offset in mls
            Value stOffV = b.add(stMlsOff, stInMlsOff);

            loadStoreOffsets.push_back(std::make_pair(ldOffV, stOffV));
            ldInBlkOffCoordMappings.push_back(std::make_pair(ldRow, ldCol));
          }
        }
      }
    }

    return loadStoreOffsets;
  }

  /*
   * Calculate matrix load with mls offset field.
   * For each vector, share same rsrc desc with ldOffValue, but has different
   *ldOffConst offset & stOffValue. key: ldOffValue, value:
   *std::pair<ldOffConst, stOffValue>
   **/
  DenseMap<Value, SmallVector<std::pair<unsigned, Value>>>
  computeMatrixLoadStoreOffsetsWithMlOffs(
      Location loc, RewriterBase &rewriter, const MlsEncodingAttr &blockLayout,
      const MatrixLoadInsnAttr &mlInsnAttr,
      const HCUMlsSharedEncodingAttr &sharedLayout,
      ArrayRef<int64_t> tensorShape, ValueRange llStrides, bool mlsIsRowMajor,
      SmallVector<bool> boundaryCheckInfo,
      bool &mlOffsInY, /* key: ldOffValue, value: ldCoordValue */
      DenseMap<Value, Value> &ldInBlkOffCoordMappings) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto warpsPerCta = blockLayout.getWarpsPerCTA();
    auto shapePerCta = getShapePerCTA(blockLayout.getMlsTile(), warpsPerCta);
    SmallVector<Value> multiDimWarpIds;
    auto dummyMapping = computeMatrixLoadStoreOffsetsMappingInCTA(
        loc, rewriter, blockLayout, mlInsnAttr, tensorShape, multiDimWarpIds);

    auto dsByteOffsets = mlInsnAttr.dsByteOffsets;
    auto elemByteWidth = blockLayout.getElemBitWidth() / 8;
    SmallVector<unsigned> flattenedMlsShape{tensorShape};
    flattenedMlsShape[0] = tensorShape[0] / blockLayout.getMlsTile()[0];
    flattenedMlsShape[1] = tensorShape[1] / blockLayout.getMlsTile()[1];

    auto rank = tensorShape.size();
    Value offWarpX = b.mul(multiDimWarpIds[rank - 2],
                           b.i32_val(blockLayout.getMlsTile()[rank - 2]));
    Value offWarpY = b.mul(multiDimWarpIds[rank - 1],
                           b.i32_val(blockLayout.getMlsTile()[rank - 1]));

    SmallVector<unsigned> numReps = getNumReps(blockLayout, tensorShape);
    unsigned numRepsX = numReps[0] * mlInsnAttr.instrsPerWarp[0];
    unsigned numRepsY = numReps[1] * mlInsnAttr.instrsPerWarp[1];
    bool membersPerGroupInY =
        (numRepsX == numRepsY)
            ? mlsIsRowMajor
            : numRepsX < numRepsY; // per numRepsY mls insts share same coordX ?
    bool needBoundaryPadding = boundaryCheckInfo[0] || boundaryCheckInfo[1];
    if (needBoundaryPadding) {
      assert(!(boundaryCheckInfo[0] && boundaryCheckInfo[1]) &&
             "both padding condition should not here!");
      if (boundaryCheckInfo[1] && membersPerGroupInY)
        membersPerGroupInY = false;
    }
    mlOffsInY = membersPerGroupInY;
    bool groupPaddingInY = !mlOffsInY;

    SmallVector<SmallVector<std::tuple<Value, unsigned, Value, Value>>>
        loadStoreOffsets;
    unsigned numGroups = membersPerGroupInY
                             ? numRepsX
                             : numRepsY; /* need create numGroups rsrc desc */
    unsigned membersPerGroup =
        membersPerGroupInY
            ? numRepsY
            : numRepsX; /* per group has membersPerGroup mls insts */
    loadStoreOffsets.resize(numGroups);
    for (auto &innerVec : loadStoreOffsets) {
      innerVec.resize(membersPerGroup);
    }

    for (unsigned ctaX = 0; ctaX < numReps[0]; ++ctaX) {
      for (unsigned ctaY = 0; ctaY < numReps[1]; ++ctaY) {
        for (unsigned instIdxX = 0; instIdxX < mlInsnAttr.instrsPerWarp[0];
             ++instIdxX) {
          for (unsigned instIdxY = 0; instIdxY < mlInsnAttr.instrsPerWarp[1];
               ++instIdxY) {
            unsigned groupIdx =
                membersPerGroupInY
                    ? ctaX * mlInsnAttr.instrsPerWarp[0] + instIdxX
                    : ctaY * mlInsnAttr.instrsPerWarp[1] + instIdxY;
            unsigned memberIdx =
                membersPerGroupInY
                    ? ctaY * mlInsnAttr.instrsPerWarp[1] + instIdxY
                    : ctaX * mlInsnAttr.instrsPerWarp[0] + instIdxX;

            unsigned ctaRow = ctaX * shapePerCta[0];
            unsigned ctaCol = ctaY * shapePerCta[1];
            Value ldRow = membersPerGroupInY
                              ? b.add(offWarpX, b.i32_val(ctaRow))
                              : offWarpX;
            Value ldCol = membersPerGroupInY
                              ? offWarpY
                              : b.add(offWarpY, b.i32_val(ctaCol));

            Value ldOffV = dot64(loc, rewriter, {ldRow, ldCol}, llStrides);
            unsigned ldOffC =
                membersPerGroupInY
                    ? ctaCol + instIdxY * mlInsnAttr.instrShape[1]
                    : ctaRow + instIdxX * mlInsnAttr.instrShape[0];

            Value stMlsRow =
                b.add(b.i32_val(ctaX * warpsPerCta[0]), multiDimWarpIds[0]);
            Value stMlsCol =
                b.add(b.i32_val(ctaY * warpsPerCta[1]), multiDimWarpIds[1]);

            Value stMlsOff =
                b.mul(linearize(rewriter, loc, {stMlsRow, stMlsCol},
                                flattenedMlsShape, sharedLayout.getOrder()),
                      b.i32_val(product(blockLayout.getMlsTile())));
            unsigned instIdx =
                instIdxX * mlInsnAttr.instrsPerWarp[1] + instIdxY;
            Value stInMlsOff =
                b.i32_val(dsByteOffsets[instIdx] /
                          elemByteWidth); // element offset in mls
            Value stOffset = b.add(stMlsOff, stInMlsOff);

            Value ldInBlockOffVal = groupPaddingInY ? ldCol : ldRow;
            loadStoreOffsets[groupIdx][memberIdx] =
                std::make_tuple(ldOffV, ldOffC, stOffset, ldInBlockOffVal);
          }
        }
      }
    }

    DenseMap<Value, SmallVector<std::pair<unsigned, Value>>>
        loadStoreOffsetsMap;
    for (auto &group : loadStoreOffsets) {
      Value ldOffV = std::get<0>(group[0]);
      auto &curMap = loadStoreOffsetsMap[ldOffV];
      curMap.reserve(group.size());
      for (auto &member : group) {
        unsigned ldOffC = std::get<1>(member);
        Value stOffset = std::get<2>(member);
        curMap.emplace_back(ldOffC, stOffset);
      }

      Value ldInBlockOffVal = std::get<3>(group[0]);
      ldInBlkOffCoordMappings[ldOffV] = ldInBlockOffVal;
    }

    return loadStoreOffsetsMap;
  }

  // reference: emitBaseIndexForMfmaLayout, emitMfmaOffsetForCTA
  SmallVector<SmallVector<Value>> computeMatrixLoadStoreOffsetsMappingInCTA(
      Location loc, RewriterBase &rewriter, const MlsEncodingAttr &blockLayout,
      const MatrixLoadInsnAttr &mlInsnAttr, ArrayRef<int64_t> shape,
      SmallVector<Value> &multiDimWarpIds) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto rank = shape.size();
    assert(rank == 2);

    auto warpOrder = blockLayout.getOrder();
    auto _warpsPerCTA = blockLayout.getWarpsPerCTA();
    SmallVector<Value> warpsPerCTA;
    for (unsigned i = 0; i < rank; ++i)
      warpsPerCTA.push_back(b.i32_val(_warpsPerCTA[i]));

    auto mlsTile = blockLayout.getMlsTile();

    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = b.i32_val(64);
    Value laneId = b.urem(threadId, warpSize);
    // Note: To make the warpId as a uniform value for Matrix Load Rsrc Desc.
    Value warpId = rewriter.create<ROCDL::ReadfirstlaneOp>(
        loc, i32_ty, b.udiv(threadId, warpSize));
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, _warpsPerCTA, warpOrder);

    if (shape[rank - 2] >= mlsTile[rank - 2]) {
      assert(shape[rank - 2] % mlsTile[rank - 2] == 0);
      multiDimWarpId[rank - 2] =
          b.urem(multiDimWarpId[rank - 2],
                 b.i32_val(ceil<unsigned>(shape[rank - 2], mlsTile[rank - 2])));
    }
    if (shape[rank - 1] >= mlsTile[rank - 1]) {
      assert(shape[rank - 1] % mlsTile[rank - 1] == 0);
      multiDimWarpId[rank - 1] =
          b.urem(multiDimWarpId[rank - 1],
                 b.i32_val(ceil<unsigned>(shape[rank - 1], mlsTile[rank - 1])));
    }

    multiDimWarpIds.push_back(multiDimWarpId[rank - 2]);
    multiDimWarpIds.push_back(multiDimWarpId[rank - 1]);

    Value offWarpX =
        b.mul(multiDimWarpId[rank - 2], b.i32_val(mlsTile[rank - 2]));
    Value offWarpY =
        b.mul(multiDimWarpId[rank - 1], b.i32_val(mlsTile[rank - 1]));

    SmallVector<SmallVector<Value>> multiDimOffsets;
    for (unsigned i = 0; i < mlInsnAttr.instrsPerWarp[0]; i++) {
      for (unsigned j = 0; j < mlInsnAttr.instrsPerWarp[1]; j++) {
        assert((mlInsnAttr.instrsPerWarp[0] == 1 ||
                mlInsnAttr.instrsPerWarp[1] == 1) &&
               "add more code support!");
        Value offElemX = b.add(
            offWarpX, b.mul(b.i32_val(i), b.i32_val(mlInsnAttr.instrShape[0])));
        Value offElemY = b.add(
            offWarpY, b.mul(b.i32_val(j), b.i32_val(mlInsnAttr.instrShape[1])));

        multiDimOffsets.push_back({offElemX, offElemY});
      }
    }

    return multiDimOffsets;
  }

  Value createRsrcDesc(Location loc, RewriterBase &rewriter, Value llBasePtr,
                       Value llStride, unsigned alt2Kind, Value mPadding,
                       Value nmPadding) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // 1. Create the resource descriptor
    // dword 0: base_addr_lo, bit pos: [31:0], Byte base address 32 LSBs for
    // global adressing dwrod 1: base_addr_hi, bit pos: [15:0], Byte base
    // address 16 MSBs for global adressing dword 2: stride,       bit pos:
    // [31:0], The unit is matrix element. dword 3:
    //   mfilter,      bit pos: [7:0]
    //                 The number of row/column needed to be zero-padded in
    //                 matrix major direction.
    //   nfilter,      bit pos: [15:8]
    //                 The number of row/column needed to be zero-padded ...
    //   cache_swizzle_enable, bit pos: [16]
    //                 1: enable cache swizzle operation in L1 cache.
    //   mfmt,         bit pos: [18:17], The matrix layout format in memory.
    //                 0: non-interleaved
    //                 1: 2-interleaved
    //                 2: 4-interleaved
    //                 3: 8-interleaved
    // Note: from Spec, it requires the base ptr to be 16-byte aligned !!!

    // dword 0: base_addr_lo [31:0]
    Value baseLow = b.ptrtoint(i32_ty, llBasePtr);
    // dword 1: base_addr_hi [15:0]
    Value baseHigh = b.and_(
        b.trunc(i32_ty, b.lshr(b.ptrtoint(i64_ty, llBasePtr), b.i64_val(32))),
        b.i32_val(0xFFFF));

    // dword 2: stride [31:0]
    Value stride = llStride;

    // dword 3:
    Value mfilter = b.and_(mPadding, b.i32_val(0xFF));
    Value nfilter = b.and_(nmPadding, b.i32_val(0xFF));
    Value cacheSwizzle = b.i32_val(0);
    Value mfmt = b.i32_val(alt2Kind);

    Value controlField = mfilter;                                     // [7:0]
    controlField = b.or_(controlField, b.shl(nfilter, b.i32_val(8))); // [15:8]
    controlField =
        b.or_(controlField, b.shl(cacheSwizzle, b.i32_val(16)));    // [16]
    controlField = b.or_(controlField, b.shl(mfmt, b.i32_val(17))); // [18:17]

    Value resource = b.undef(vec_ty(i32_ty, 4));
    resource =
        b.insert_element(vec_ty(i32_ty, 4), resource, baseLow, b.i32_val(0));
    resource =
        b.insert_element(vec_ty(i32_ty, 4), resource, baseHigh, b.i32_val(1));
    resource =
        b.insert_element(vec_ty(i32_ty, 4), resource, stride, b.i32_val(2));
    resource = b.insert_element(vec_ty(i32_ty, 4), resource, controlField,
                                b.i32_val(3));

    return resource;
  }

  void generateMatrixLoadLDSOp(
      Location loc, RewriterBase &rewriter, StringRef mlInsnOpName,
      Value rsrcDesc, Value soffset,
      bool t,         // Transposition. 0: column major, 1: row major
      int32_t offset, // Global address offset [9:0]
      bool r,         // Golbal address offset is in row direction ?
      bool glc = false, bool slc = false, bool bps = false) const {
    OperationState loweredOp(loc, mlInsnOpName);
    loweredOp.addOperands(rsrcDesc);
    loweredOp.addOperands(soffset);

    loweredOp.addAttribute("offset", rewriter.getI32IntegerAttr(offset));
    loweredOp.addAttribute("t", rewriter.getBoolAttr(t));
    loweredOp.addAttribute("r", rewriter.getBoolAttr(r));
    loweredOp.addAttribute("glc", rewriter.getBoolAttr(glc));
    loweredOp.addAttribute("slc", rewriter.getBoolAttr(slc));
    loweredOp.addAttribute("bps", rewriter.getBoolAttr(bps));

    rewriter.create(loweredOp);
  }

  Value dot64(Location loc, RewriterBase &rewriter, ValueRange offsets32,
              ValueRange strides64) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    assert(offsets32.size() == strides64.size());
    Value ret = b.i64_val(0);
    for (auto [offset, stride] : llvm::zip(offsets32, strides64)) {
      ret = b.add(ret, b.mul(b.zext(i64_ty, offset), stride));
    }
    return ret;
  }
};

//--------------------------------------------------------------------------------------------------

struct MLSLocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  MLSLocalAllocOpConversion(LLVMTypeConverter &converter,
                            const HCU::TargetInfo &targetInfo,
                            ModuleAxisInfoAnalysis &axisAnalysisPass,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isSharedMemoryAlloc())
      return failure();
    Location loc = op->getLoc();
    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto resultTy = cast<MemDescType>(op.getType());
    auto typeConverter = getTypeConverter();
    auto sharedLayout =
        dyn_cast<HCUMlsSharedEncodingAttr>(resultTy.getEncoding());
    if (!sharedLayout)
      return failure();

    auto llvmElemTy = typeConverter->convertType(resultTy.getElementType());
    auto smemObj = LLVM::SharedMemoryObject(smemBase, llvmElemTy,
                                            resultTy.getRank(), loc, rewriter);
    assert(!op.getSrc());
    auto retVal = LLVM::getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

//--------------------------------------------------------------------------------------------------

// Get warpId inside block of warps.
Value getWarpIdInBlock(ConversionPatternRewriter &rewriter, Location loc,
                       Value warpId, const ArrayRef<unsigned int> &wpt,
                       int elemPerInstrNonK, int tensorSizeNonK, int nonKIdx,
                       const ArrayRef<unsigned int> &order) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, wpt, order);

  return b.urem(multiDimWarpId[nonKIdx],
                b.i32_val(tensorSizeNonK / elemPerInstrNonK));
}

struct MLSLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalLoadOp>::ConvertOpToLLVMPattern;

  MLSLocalLoadOpConversion(LLVMTypeConverter &converter,
                           const HCU::TargetInfo &targetInfo,
                           ModuleAxisInfoAnalysis &axisAnalysisPass,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<triton::gpu::HCUMlsSharedEncodingAttr>(srcLayout)) {
      if (isPacketVersion(cast<HCUMlsSharedEncodingAttr>(srcLayout).getVersion()))
        return lowerWholePacketConsumer(op, adaptor, *getTypeConverter(), rewriter);
      if (cast<HCUMlsSharedEncodingAttr>(srcLayout).getVersion() == 0) {
        if (cast<HCUMlsSharedEncodingAttr>(srcLayout).getOpIdx() == 0)
          return lowerLegacyV0IndexedAConsumer(op, adaptor, *getTypeConverter(), rewriter);
        return lowerLegacyV0Consumer(op, adaptor, *getTypeConverter(), rewriter);
      }
      if (isa<DotOperandEncodingAttr>(dstLayout) &&
          isa<AMDMfmaEncodingAttr>(
              cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
        return lowerMLSSharedToDotOperand(op, adaptor, getTypeConverter(),
                                          rewriter);
      } else {
        assert(false && "should been canonicalized and not reach here!");
        return failure();
      }
    }

    return failure();
  }

private:
  const HCU::TargetInfo &targetInfo;

  // shared -> matrix_core_dot_operand
  // reference: SharedToDotOperandMFMA::convertLayout in
  // SharedToDotOperandMFMA.cpp
  LogicalResult
  lowerMLSSharedToDotOperand(triton::gpu::LocalLoadOp op,
                             triton::gpu::LocalLoadOpAdaptor adaptor,
                             const LLVMTypeConverter *typeConverter,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto dstTensorTy = cast<RankedTensorType>(dst.getType());
    auto srcTensorTy = cast<MemDescType>(src.getType());
    auto dotOperandLayout =
        cast<DotOperandEncodingAttr>(dstTensorTy.getEncoding());

    auto llvmElemTy = typeConverter->convertType(
        cast<MemDescType>(src.getType()).getElementType());

    auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                   llvmElemTy, rewriter);

    Value res;
    auto opIdx = dotOperandLayout.getOpIdx();
    auto mfmaLayout = cast<AMDMfmaEncodingAttr>(dotOperandLayout.getParent());
    Value threadId = getThreadId(rewriter, loc);
    if ((opIdx == 0 &&
         mfmaLayout.getMfmaTile()[0] == mfmaLayout.getInstrShape()[0]) ||
        (opIdx == 1 &&
         mfmaLayout.getMfmaTile()[1] == mfmaLayout.getInstrShape()[1])) {
      res = convertLayoutUnitTilesPerWarp(dotOperandLayout.getOpIdx(), rewriter,
                                          loc, src, dotOperandLayout, smemObj,
                                          typeConverter, threadId);
    } else {
      res = convertLayoutMultiTilesPerWarp(dotOperandLayout.getOpIdx(),
                                           rewriter, loc, src, dotOperandLayout,
                                           smemObj, typeConverter, threadId);
    }

    if (!res)
      return failure();
    rewriter.replaceOp(op, res);
    return success();
  }

  Value convertLayoutUnitTilesPerWarp(int opIdx,
                                      ConversionPatternRewriter &rewriter,
                                      Location loc, Value tensor,
                                      DotOperandEncodingAttr encoding,
                                      const SharedMemoryObject &smemObj,
                                      const LLVMTypeConverter *typeConverter,
                                      Value thread) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    assert((opIdx == 0 || opIdx == 1) && "unexpected operand idx");
    auto tensorTy = cast<MemDescType>(tensor.getType());
    ArrayRef<int64_t> shape = tensorTy.getShape();
    auto rank = shape.size();
    int kDimIdx = opIdx == 0 ? rank - 1 : rank - 2;
    int nonKDimIdx = opIdx == 0 ? rank - 2 : rank - 1;
    int kDimIdx2D = opIdx == 0 ? 1 : 0;
    int nonKDimIdx2D = opIdx == 0 ? 0 : 1;

    auto mfmaLayout = cast<AMDMfmaEncodingAttr>(encoding.getParent());
    assert(((opIdx == 0 && mfmaLayout.getInstrsPerWarp()[0] == 1) ||
            (opIdx == 1 && mfmaLayout.getInstrsPerWarp()[1] == 1)) &&
           "only support unit tiles per warp mfma layout!");
    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

    auto elemTy = tensorTy.getElementType();
    auto kWidth = encoding.getKWidth();
    auto elemsPerInstr = mfmaLayout.getInstrShapeForOperand(kWidth, opIdx);

    int64_t mfmaInstrNonK;
    int64_t mfmaInstrK;
    // TODO(Lixun): make it simpler
    // getInstrShapeForOperand always returns a 2D vector
    if (rank == 3) {
      mfmaInstrNonK = elemsPerInstr[nonKDimIdx - 1];
      mfmaInstrK = elemsPerInstr[kDimIdx - 1];
    } else {
      mfmaInstrNonK = elemsPerInstr[nonKDimIdx];
      mfmaInstrK = elemsPerInstr[kDimIdx];
    }

    if (mfmaInstrNonK > shape[nonKDimIdx] || mfmaInstrK > shape[kDimIdx]) {
      // This pattern does not support cases tensor shape is smaller than
      // one instruction size, it will be processed by LinearLayout converter
      return Value();
    }

    auto numReps = mfmaLayout.getRepForOperand(shape, kWidth, opIdx);
    auto numRepNonK = numReps[nonKDimIdx];
    auto numRepK = numReps[kDimIdx];
    auto repB = numReps[0];
    // TODO(Lixun): make it simpler
    // getRepForOperand always returns a 3D vector
    if (rank == 2) {
      numRepNonK = numReps[nonKDimIdx + 1];
      numRepK = numReps[kDimIdx + 1];
    }

    unsigned iWarpSize = targetInfo.getWarpSize();
    assert(iWarpSize == 64);
    Value warpSize = b.i32_val(iWarpSize);
    Value linearWarpId = b.udiv(thread, warpSize);
    Value lane = b.urem(thread, warpSize);

    auto warpOrder = triton::gpu::getMatrixOrder(rank, /*rowMajor*/ true);
    Value spatialWarpId = getWarpIdInBlock(
        rewriter, loc, linearWarpId, warpsPerCTA, mfmaInstrNonK,
        shape[nonKDimIdx], nonKDimIdx, warpOrder);

    int numSubBlocks = 1;
    // numOfElemsPerThreadPerMfmaInstr
    int numOfElems = mfmaInstrNonK * mfmaInstrK * numSubBlocks / iWarpSize;
    assert(numOfElems >= 1);

    unsigned int maxNumWarps = shape[nonKDimIdx] / mfmaInstrNonK;
    int warpsPerBlockNonK = std::min(warpsPerCTA[nonKDimIdx], maxNumWarps);
    int warpsPerBatch =
        rank == 3 ? std::min<unsigned>(shape[0], warpsPerCTA[0]) : 1;
    Value warpIdInBatch = b.urem(linearWarpId, b.i32_val(warpsPerBatch));
    elemTy = typeConverter->convertType(elemTy);

    // 1. get ds_read_matrix inst info
    auto sharedLayout = cast<HCUMlsSharedEncodingAttr>(tensorTy.getEncoding());
    auto mlsTile = sharedLayout.getMlsTile();
    bool kMajor = sharedLayout.getOrder()[0] == kDimIdx;
    assert(kMajor && "m16n16 mfma & mls only support kMajor layout!");

    auto mlsInsn = MlsInsn::selectOrGetMlsInsn(
        mlsTile[nonKDimIdx2D], mlsTile[kDimIdx2D],
        sharedLayout.getElemBitWidth(), opIdx, kMajor,
        static_cast<MlsInterleaveKind>(sharedLayout.getAlt2Kind()),
        sharedLayout.getVersion());

    auto dsInsnAttr = mlsInsn->getDsReadMatrixInsnAttr();

    assert(mlsTile[kDimIdx2D] % mfmaInstrK == 0 &&
           shape[kDimIdx] % mlsTile[kDimIdx2D] == 0 &&
           "mls tile kdim and shape kdim should be divisible!");

    unsigned mlsNumRepK = shape[kDimIdx] / mlsTile[kDimIdx];
    unsigned mlsNumRepNonK = numRepNonK;
    SmallVector<unsigned> mlsNumReps{mlsNumRepNonK, mlsNumRepK};
    if (opIdx == 1) {
      std::swap(mlsNumReps[0], mlsNumReps[1]);
    }

    // 2. compute ds_read_matrix offsets
    auto offsets = computeDsReadMatrixOffsets(
        rewriter, loc, sharedLayout, dsInsnAttr, spatialWarpId,
        warpsPerBlockNonK, mlsNumReps, smemObj, shape);
    Value smemBase = smemObj.getBase();
    Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);

    assert(mlsInsn->getElemBitWidth() == 16 || mlsInsn->getElemBitWidth() == 8);
    auto dsLoadsPerK = product(dsInsnAttr.instrsPerWarp);

    // 3. create ds_read_matrix ops
    SmallVector<Value> loadedValues;
    for (int batchIdx = 0; batchIdx < repB; ++batchIdx) {
      int operandSize = shape[rank - 1] * shape[rank - 2];
      Value batchOffset =
          b.mul(b.i32_val(operandSize),
                b.add(warpIdInBatch, b.i32_val(batchIdx * warpsPerBatch)));
      for (int mlsNonK = 0; mlsNonK < mlsNumRepNonK; ++mlsNonK) {
        for (int mlsKIdx = 0; mlsKIdx < mlsNumRepK; ++mlsKIdx) {
          auto loadOffset = offsets[mlsNonK * mlsNumRepK + mlsKIdx];
          loadOffset = b.add(loadOffset, batchOffset);
          for (int loadIdx = 0; loadIdx < dsLoadsPerK; ++loadIdx) {
            Value loadAddress = b.gep(smemPtrTy, elemTy, smemBase, loadOffset);

            auto dsInsnResTy =
                getDsReadMatrixInsnResType(loc, rewriter, tensorTy, dsInsnAttr,
                                           mlsInsn->getElemBitWidth());
            Value loadedValue = generateDsReadMatrixOp(
                loc, rewriter, dsInsnAttr.insn, dsInsnResTy, loadAddress,
                dsInsnAttr.dsByteOffsets[loadIdx], dsInsnAttr.flags);
            auto unpackedValues = unpackDsReadMatrixInsnRes(
                loc, rewriter, tensorTy, dsInsnAttr, mlsInsn->getElemBitWidth(),
                loadedValue);
            loadedValues.append(unpackedValues);
          }
        }
      }
    }

    assert(loadedValues.size() ==
           repB * mlsNumRepNonK * mlsNumRepK * dsLoadsPerK *
               (dsInsnAttr.instrShape[kDimIdx2D] *
                dsInsnAttr.instrShape[nonKDimIdx2D] / iWarpSize));

    MLIRContext *ctx = mfmaLayout.getContext();
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(loadedValues.size(), loadedValues[0].getType()));
    auto result =
        packLLElements(loc, typeConverter, loadedValues, rewriter, structTy);
    return result;
  }

  Value convertLayoutMultiTilesPerWarp(int opIdx,
                                       ConversionPatternRewriter &rewriter,
                                       Location loc, Value tensor,
                                       DotOperandEncodingAttr encoding,
                                       const SharedMemoryObject &smemObj,
                                       const LLVMTypeConverter *typeConverter,
                                       Value thread) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    assert((opIdx == 0 || opIdx == 1) && "unexpected operand idx");
    auto tensorTy = cast<MemDescType>(tensor.getType());
    ArrayRef<int64_t> shape = tensorTy.getShape();
    auto rank = shape.size();
    int kDimIdx = opIdx == 0 ? rank - 1 : rank - 2;
    int nonKDimIdx = opIdx == 0 ? rank - 2 : rank - 1;
    int kDimIdx2D = opIdx == 0 ? 1 : 0;
    int nonKDimIdx2D = opIdx == 0 ? 0 : 1;

    auto mfmaLayout = cast<AMDMfmaEncodingAttr>(encoding.getParent());
    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

    auto elemTy = tensorTy.getElementType();
    auto kWidth = encoding.getKWidth();
    auto elemsPerInstr = mfmaLayout.getInstrShapeForOperand(kWidth, opIdx);

    int64_t mfmaInstrNonK;
    int64_t mfmaInstrK;
    // TODO(Lixun): make it simpler
    // getInstrShapeForOperand always returns a 2D vector
    if (rank == 3) {
      mfmaInstrNonK = elemsPerInstr[nonKDimIdx - 1];
      mfmaInstrK = elemsPerInstr[kDimIdx - 1];
    } else {
      mfmaInstrNonK = elemsPerInstr[nonKDimIdx];
      mfmaInstrK = elemsPerInstr[kDimIdx];
    }

    if (mfmaInstrNonK > shape[nonKDimIdx] || mfmaInstrK > shape[kDimIdx]) {
      // This pattern does not support cases tensor shape is smaller than
      // one instruction size, it will be processed by LinearLayout converter
      return Value();
    }

    auto numReps = mfmaLayout.getRepForOperand(shape, kWidth, opIdx);
    auto numRepNonK = numReps[nonKDimIdx];
    auto numRepK = numReps[kDimIdx];
    auto repB = numReps[0];
    // TODO(Lixun): make it simpler
    // getRepForOperand always returns a 3D vector
    if (rank == 2) {
      numRepNonK = numReps[nonKDimIdx + 1];
      numRepK = numReps[kDimIdx + 1];
    }

    auto mfmaTileNonK =
        opIdx == 0 ? mfmaLayout.getMfmaTile()[0] : mfmaLayout.getMfmaTile()[1];
    auto mfmaTileK = mfmaInstrK;
    auto mfmaInstrsPerWarpNonK = mfmaLayout.getInstrsPerWarp()[nonKDimIdx2D];
    auto mfmaTileNumReps =
        mfmaLayout.getMfmaTileRepForOperand(shape, kWidth, opIdx);
    auto mfmaTileNumRepNonK = mfmaTileNumReps[nonKDimIdx];
    if (rank == 2) {
      mfmaTileNumRepNonK = mfmaTileNumReps[nonKDimIdx + 1];
    }

    unsigned iWarpSize = targetInfo.getWarpSize();
    assert(iWarpSize == 64);
    Value warpSize = b.i32_val(iWarpSize);
    Value linearWarpId = b.udiv(thread, warpSize);
    Value lane = b.urem(thread, warpSize);

    auto warpOrder = triton::gpu::getMatrixOrder(rank, /*rowMajor*/ true);
    Value spatialWarpId =
        getWarpIdInBlock(rewriter, loc, linearWarpId, warpsPerCTA, mfmaTileNonK,
                         shape[nonKDimIdx], nonKDimIdx, warpOrder);

    int numSubBlocks = 1;
    // numOfElemsPerThreadPerMfmaInstr
    int numOfElems = mfmaInstrNonK * mfmaInstrK * numSubBlocks / iWarpSize;
    assert(numOfElems >= 1);

    unsigned int maxNumWarps = shape[nonKDimIdx] / mfmaTileNonK;
    int warpsPerBlockNonK = std::min(warpsPerCTA[nonKDimIdx], maxNumWarps);
    int warpsPerBatch =
        rank == 3 ? std::min<unsigned>(shape[0], warpsPerCTA[0]) : 1;
    Value warpIdInBatch = b.urem(linearWarpId, b.i32_val(warpsPerBatch));
    elemTy = typeConverter->convertType(elemTy);

    // 1. get ds_read_matrix inst info
    auto sharedLayout = cast<HCUMlsSharedEncodingAttr>(tensorTy.getEncoding());
    auto mlsTile = sharedLayout.getMlsTile();
    bool kMajor = sharedLayout.getOrder()[0] == kDimIdx;

    auto mlsInsn = MlsInsn::selectOrGetMlsInsn(
        mlsTile[nonKDimIdx2D], mlsTile[kDimIdx2D],
        sharedLayout.getElemBitWidth(), opIdx, kMajor,
        static_cast<MlsInterleaveKind>(sharedLayout.getAlt2Kind()),
        sharedLayout.getVersion());

    auto dsInsnAttr = mlsInsn->getDsReadMatrixInsnAttr();

    assert(mlsTile[kDimIdx2D] % mfmaInstrK == 0 &&
           shape[kDimIdx] % mlsTile[kDimIdx2D] == 0 &&
           "mls tile kdim and shape kdim should be divisible!");

    unsigned mlsNumRepK = shape[kDimIdx] / mlsTile[kDimIdx2D];
    unsigned mlsNumRepNonK = mfmaTileNumRepNonK;
    SmallVector<unsigned> mlsNumReps{mlsNumRepNonK, mlsNumRepK};
    if (opIdx == 1) {
      std::swap(mlsNumReps[0], mlsNumReps[1]);
    }

    // 2. compute ds_read_matrix offsets
    auto offsets = computeDsReadMatrixOffsets(
        rewriter, loc, sharedLayout, dsInsnAttr, spatialWarpId,
        warpsPerBlockNonK, mlsNumReps, smemObj, shape);
    Value smemBase = smemObj.getBase();
    Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);

    assert(mlsInsn->getElemBitWidth() == 16 || mlsInsn->getElemBitWidth() == 8);
    auto dsLoadsPerK = product(dsInsnAttr.instrsPerWarp);

    // 3. create ds_read_matrix ops
    assert((dsInsnAttr.instrsPerWarp[0] == 1 ||
            dsInsnAttr.instrsPerWarp[1] == 1) &&
           "add more code support!");
    bool isDsInsnRepInKDirection = dsInsnAttr.instrsPerWarp[kDimIdx2D] != 1;
    unsigned mfmaGroupPerDsInsn =
        dsInsnAttr.instrShape[nonKDimIdx2D] / mfmaInstrNonK;
    assert(mfmaGroupPerDsInsn != 1);

    unsigned numOfElemsPerDsInsn = dsInsnAttr.instrShape[kDimIdx2D] *
                                   dsInsnAttr.instrShape[nonKDimIdx2D] /
                                   iWarpSize;
    unsigned totalElemsMfma = repB *
                              (mfmaTileNumRepNonK * mfmaInstrsPerWarpNonK) *
                              numRepK * numOfElems;
    unsigned totalElemsMls =
        repB * mlsNumRepNonK * mlsNumRepK * dsLoadsPerK * numOfElemsPerDsInsn;
    assert(totalElemsMfma == totalElemsMls);

    SmallVector<Value> loadedValues(totalElemsMfma);
    for (int batchIdx = 0; batchIdx < repB; ++batchIdx) {
      int operandSize = shape[rank - 1] * shape[rank - 2];
      Value batchOffset =
          b.mul(b.i32_val(operandSize),
                b.add(warpIdInBatch, b.i32_val(batchIdx * warpsPerBatch)));
      for (int mlsNonK = 0; mlsNonK < mlsNumRepNonK; ++mlsNonK) {
        int mfmaTileNonKIdx = mlsNonK;
        for (int mlsKIdx = 0; mlsKIdx < mlsNumRepK; ++mlsKIdx) {
          int mfmaTileKIdx = mlsKIdx * (mlsTile[kDimIdx2D] / mfmaTileK);

          auto loadOffset = offsets[mlsNonK * mlsNumRepK + mlsKIdx];
          loadOffset = b.add(loadOffset, batchOffset);
          for (int loadIdx = 0; loadIdx < dsLoadsPerK; ++loadIdx) {
            Value loadAddress = b.gep(smemPtrTy, elemTy, smemBase, loadOffset);

            auto dsInsnResTy =
                getDsReadMatrixInsnResType(loc, rewriter, tensorTy, dsInsnAttr,
                                           mlsInsn->getElemBitWidth());
            Value loadedValue = generateDsReadMatrixOp(
                loc, rewriter, dsInsnAttr.insn, dsInsnResTy, loadAddress,
                dsInsnAttr.dsByteOffsets[loadIdx], dsInsnAttr.flags);
            auto unpackedValues = unpackDsReadMatrixInsnRes(
                loc, rewriter, tensorTy, dsInsnAttr, mlsInsn->getElemBitWidth(),
                loadedValue);

            assert(unpackedValues.size() % mfmaGroupPerDsInsn == 0 &&
                   "unpackedValues should be divisible by dsGroupSize");
            unsigned elemsPerGroup = unpackedValues.size() / mfmaGroupPerDsInsn;

            for (int i = 0; i < mfmaGroupPerDsInsn; ++i) {
              int mfmaInsnNonKIdxInTile =
                  isDsInsnRepInKDirection ? i
                                          : loadIdx * mfmaGroupPerDsInsn + i;
              int mfmaInsnKIdxInTile =
                  isDsInsnRepInKDirection
                      ? loadIdx * (dsInsnAttr.instrShape[kDimIdx2D] / mfmaTileK)
                      : 0;
              unsigned groupOffset =
                  batchIdx * (mfmaTileNumRepNonK * mfmaInstrsPerWarpNonK) *
                      numRepK * numOfElems /* batch idx */
                  + mfmaTileNonKIdx * mfmaInstrsPerWarpNonK * numRepK *
                        numOfElems /* block idx*/
                  + mfmaInsnNonKIdxInTile * numRepK *
                        numOfElems + /* inblock idx */
                  (mfmaTileKIdx + mfmaInsnKIdxInTile) * numOfElems;

              for (int j = 0; j < elemsPerGroup; ++j) {
                loadedValues[groupOffset + j] =
                    unpackedValues[i * elemsPerGroup + j];
              }
            }
          }
        }
      }
    }

    assert(loadedValues.size() == totalElemsMfma);

    MLIRContext *ctx = mfmaLayout.getContext();
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(loadedValues.size(), loadedValues[0].getType()));
    auto result =
        packLLElements(loc, typeConverter, loadedValues, rewriter, structTy);
    return result;
  }

  llvm::SmallVector<Value> computeDsReadMatrixOffsets(
      ConversionPatternRewriter &rewriter, Location loc,
      const HCUMlsSharedEncodingAttr &sharedLayout,
      const DsReadMatrixInsnAttr &dsInsnAttr, Value warpNonKId,
      int warpsPerBlockNonK, ArrayRef<unsigned> mlsReps,
      SharedMemoryObject smemObj, ArrayRef<int64_t> shape) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto opIdx = sharedLayout.getOpIdx();
    auto mlsTile = sharedLayout.getMlsTile();
    auto kDimIdx = opIdx == 0 ? 1 : 0;
    auto nonKDimIdx = opIdx == 0 ? 0 : 1;

    assert(mlsReps.size() == 2);
    auto numRepK = mlsReps[kDimIdx];
    auto numRepNonK = mlsReps[nonKDimIdx];

    const auto numBlocks = numRepNonK;
    const auto numMlsTilesPerBlock = numRepK;
    const auto blockSize = numMlsTilesPerBlock;

    SmallVector<unsigned> flattenedMlsShape{shape};
    flattenedMlsShape[0] = shape[0] / sharedLayout.getMlsTile()[0];
    flattenedMlsShape[1] = shape[1] / sharedLayout.getMlsTile()[1];

    SmallVector<unsigned> tensorShape{shape};
    llvm::SmallVector<Value> offsets(numBlocks * numMlsTilesPerBlock);
    for (int block = 0; block < numBlocks; ++block) {
      // assert(isKMajor(sharedLayout.getOrder(), opIdx));
      int blockNonKOffset = block * warpsPerBlockNonK;

      for (int mlsTile = 0; mlsTile < numMlsTilesPerBlock; ++mlsTile) {
        Value mlsNonKOff = b.add(b.i32_val(blockNonKOffset), warpNonKId);
        Value mlsKOff = b.i32_val(mlsTile);

        std::array<Value, 2> mlsCoords =
            opIdx == 0 ? std::array<Value, 2>{mlsNonKOff, mlsKOff}
                       : std::array<Value, 2>{mlsKOff, mlsNonKOff};

        Value mlsOff =
            b.mul(linearize(rewriter, loc, mlsCoords, flattenedMlsShape,
                            sharedLayout.getOrder()),
                  b.i32_val(product(sharedLayout.getMlsTile())));
        offsets[block * blockSize + mlsTile] = mlsOff;
      }
    }

    return offsets;
  }

  Type getDsReadMatrixInsnResType(Location loc, RewriterBase &rewriter,
                                  const MemDescType &tensorTy,
                                  const DsReadMatrixInsnAttr &dsInsnAttr,
                                  unsigned elemBitWidth,
                                  unsigned iWarpSize = 64) const {
    auto dsInsnElemTy = tensorTy.getElementType();
    auto dsInsnNumOfElems =
        dsInsnAttr.instrShape[0] * dsInsnAttr.instrShape[1] / iWarpSize;

    if (elemBitWidth == 8) {
      dsInsnElemTy = i32_ty;
      dsInsnNumOfElems = 4;
    }
    return vec_ty(dsInsnElemTy, dsInsnNumOfElems);
  }

  Value generateDsReadMatrixOp(Location loc, RewriterBase &rewriter,
                               StringRef dsOpName, Type resType, Value soffset,
                               unsigned offset, unsigned flags) const {
    OperationState loweredOp(loc, dsOpName);
    loweredOp.addTypes(resType);
    loweredOp.addOperands(soffset);
    loweredOp.addAttribute("offset", rewriter.getI32IntegerAttr(offset));

    int8_t elem3 = MLS_DS_FLAGS_GET_ELEM3(flags);
    int8_t row3 = MLS_DS_FLAGS_GET_ROW3(flags);
    int8_t col3 = MLS_DS_FLAGS_GET_COL3(flags);
    int8_t alt2 = MLS_DS_FLAGS_GET_ALT2(flags);

    loweredOp.addAttribute("element", rewriter.getI8IntegerAttr(elem3));
    loweredOp.addAttribute("row", rewriter.getI8IntegerAttr(row3));
    loweredOp.addAttribute("col", rewriter.getI8IntegerAttr(col3));
    loweredOp.addAttribute("alt", rewriter.getI8IntegerAttr(alt2));

    return rewriter.create(loweredOp)->getResult(0);
  }

  SmallVector<Value>
  unpackDsReadMatrixInsnRes(Location loc, RewriterBase &rewriter,
                            const MemDescType &tensorTy,
                            const DsReadMatrixInsnAttr &dsInsnAttr,
                            unsigned elemBitWidth, Value dsLoadedValue) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto dsInsnRetTy = cast<VectorType>(dsLoadedValue.getType());
    auto dsInsnElemTy = dsInsnRetTy.getElementType();
    auto dsInsnNumOfElems = dsInsnRetTy.getNumElements();
    bool needCvt = false;
    unsigned numExtracts = 1;
    if (elemBitWidth == 8) {
      numExtracts = dsInsnElemTy.getIntOrFloatBitWidth() / elemBitWidth;
    }

    SmallVector<Value> retValues;
    auto elemTy = typeConverter->convertType(tensorTy.getElementType());
    for (int elemId = 0; elemId < dsInsnNumOfElems; ++elemId) {
      Value elemVal =
          b.extract_element(dsInsnElemTy, dsLoadedValue, b.i32_val(elemId));

      if (numExtracts != 1) {
        auto vecVal = b.bitcast(elemVal, vec_ty(elemTy, numExtracts));
        for (int extractId = 0; extractId < numExtracts; ++extractId) {
          Value subElemVal =
              b.extract_element(elemTy, vecVal, b.i32_val(extractId));
          retValues.push_back(subElemVal);
        }
      } else {
        retValues.push_back(elemVal);
      }
    }
    return retValues;
  }
};

} // namespace

namespace mlir::triton::HCU {
void populateMLSOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 const TargetInfo &targetInfo,
                                 RewritePatternSet &patterns,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 PatternBenefit benefit) {
  patterns.add<StructuredRegisterLoadConversion>(typeConverter, targetInfo,
      axisInfoAnalysis, benefit);
  patterns.add<RawWindowCopyConversion, RawWindowReadConversion, RawWindowWaitConversion, RawWarpGroupIdConversion>(typeConverter, benefit);
  patterns.add<MLSLocalAllocOpConversion, MLSMatrixLoadToLocalOpConversion,
               MLSLocalLoadOpConversion>(typeConverter, targetInfo,
                                         axisInfoAnalysis, benefit);
}
} // namespace mlir::triton::HCU
