/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// PPU has no TMA unit. Lower descriptor copies to descriptor loads/stores
// paired with accesses through tle.local_pointers. The generic descriptor-to-
// pointer pass then expands them to bounds-checked accesses; global-to-shared
// copies become cp.async in the local-pointer store optimization. Copies that
// signal an mbarrier remain unsupported.
//
// The second pass prepares the expanded loads for vectorized cp.async:
// - Replace padding with zero and assert that the descriptor does not request
//   NaN padding, which cp.async cannot provide.
// - Fold the unit innermost stride guaranteed by tensor descriptors.
// - Restore alignment hints from ppu.tensordesc_divisibility in the cache key:
//   bit 0 records a 16-byte aligned base; bit 1 + d records an extent divisible
//   by 16. Host validation guarantees 16-byte aligned outer strides. The first
//   pass copies these bits from the descriptor argument to its load, and the
//   second applies them to the expanded pointer and mask operands.

#include "TritonPPUGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#ifdef __TLE__
#include "tle/dialect/include/IR/Dialect.h"
#endif

namespace mlir {
#define GEN_PASS_DEF_TRITONPPUTLELOWERTMACOPY
#define GEN_PASS_DEF_TRITONPPUTLEZEROPADDESCRIPTORCOPIES
#include "TritonPPUGPUTransforms/Passes.h.inc"

namespace {
namespace tt = triton;
namespace ttg = triton::gpu;

#ifdef __TLE__
// Marks the descriptor load of a global-to-shared copy; the descriptor-to-
// pointer rewrite carries it over to the pointer load it creates.
constexpr llvm::StringLiteral kDescriptorCopyLoadAttr("ppu.tle_copy_load");
// The alignment bits of a host descriptor argument (see the file comment),
// copied from the argument to the descriptor load.
constexpr llvm::StringLiteral
    kDescriptorDivisibilityAttr("ppu.tensordesc_divisibility");
constexpr int64_t kVectorBytes = 16;

// Index tensors that address every element of a buffer of `shape`, in the
// form tle.local_pointers and the local-pointer store optimization expect:
// per axis, a make_range expanded to the rank and broadcast to the shape.
static SmallVector<Value> createFullIndices(OpBuilder &builder, Location loc,
                                            ArrayRef<int64_t> shape) {
  Type i32 = builder.getI32Type();
  SmallVector<Value> indices;
  for (auto [axis, dim] : llvm::enumerate(shape)) {
    Value index = tt::MakeRangeOp::create(
        builder, loc, RankedTensorType::get({dim}, i32),
        builder.getI32IntegerAttr(0), builder.getI32IntegerAttr(dim));
    SmallVector<int64_t> expanded{dim};
    for (size_t d = 0; d < shape.size(); ++d) {
      if (d == axis)
        continue;
      expanded.insert(expanded.begin() + d, 1);
      index = tt::ExpandDimsOp::create(builder, loc,
                                       RankedTensorType::get(expanded, i32),
                                       index, builder.getI32IntegerAttr(d));
    }
    indices.push_back(tt::BroadcastOp::create(
        builder, loc, RankedTensorType::get(shape, i32), index));
  }
  return indices;
}

static Value createLocalPointers(OpBuilder &builder, Location loc,
                                 Value memDesc) {
  auto memDescTy = cast<ttg::MemDescType>(memDesc.getType());
  auto ptrTy = tt::PointerType::get(memDescTy.getElementType(), 3);
  return triton::tle::LocalPointersOp::create(
      builder, loc, RankedTensorType::get(memDescTy.getShape(), ptrTy), memDesc,
      createFullIndices(builder, loc, memDescTy.getShape()));
}

// The descriptor block may have unit leading dimensions beyond the buffer's
// rank; those are the only shape difference a copy may have.
static LogicalResult checkShapes(ttg::TMACopyOp op, RankedTensorType blockTy,
                                 ttg::MemDescType memDescTy) {
  ArrayRef<int64_t> blockShape = blockTy.getShape();
  ArrayRef<int64_t> memShape = memDescTy.getShape();
  bool ok = blockShape.size() >= memShape.size() &&
            blockShape.take_back(memShape.size()) == memShape &&
            llvm::all_of(blockShape.drop_back(memShape.size()),
                         [](int64_t dim) { return dim == 1; });
  if (!ok)
    return op.emitOpError("descriptor block shape ")
           << blockShape << " must match the buffer shape " << memShape
           << " up to unit leading dimensions";
  if (blockTy.getElementType() != memDescTy.getElementType())
    return op.emitOpError("descriptor element type ")
           << blockTy.getElementType() << " must match the buffer element type "
           << memDescTy.getElementType();
  return success();
}

static LogicalResult lowerCopy(ttg::TMACopyOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value src = op.getSrc();
  Value dst = op.getDst();
  if (op.getBarrier())
    return op.emitOpError(
        "PPU has no mbarrier; descriptor copies with a completion barrier "
        "are not supported");

  bool toShared = isa<tt::TensorDescType>(src.getType());
  Value desc = toShared ? src : dst;
  Value memDesc = toShared ? dst : src;
  auto descTy = cast<tt::TensorDescType>(desc.getType());
  auto memDescTy = dyn_cast<ttg::MemDescType>(memDesc.getType());
  if (!memDescTy || isa<tt::TensorDescType>(memDesc.getType()))
    return op.emitOpError("expects one tensor descriptor and one shared "
                          "buffer operand");
  RankedTensorType blockTy = descTy.getSignlessBlockType();
  if (failed(checkShapes(op, blockTy, memDescTy)))
    return failure();
  if (op.getIndices().size() != blockTy.getRank())
    return op.emitOpError("expects ")
           << blockTy.getRank() << " coordinates, got "
           << op.getIndices().size();

  auto bufferTy =
      RankedTensorType::get(memDescTy.getShape(), memDescTy.getElementType());
  Value ptrs = createLocalPointers(builder, loc, memDesc);
  if (toShared) {
    auto load = tt::DescriptorLoadOp::create(
        builder, loc, blockTy, desc, op.getIndices(),
        tt::CacheModifierAttr::get(op.getContext(), tt::CacheModifier::NONE));
    load->setAttr(kDescriptorCopyLoadAttr, builder.getUnitAttr());
    if (auto arg = dyn_cast<BlockArgument>(desc))
      if (auto func = dyn_cast<tt::FuncOp>(arg.getOwner()->getParentOp()))
        if (auto bits = func.getArgAttrOfType<IntegerAttr>(
                arg.getArgNumber(), kDescriptorDivisibilityAttr))
          load->setAttr(kDescriptorDivisibilityAttr, bits);
    Value value = load.getResult();
    if (blockTy.getRank() != bufferTy.getRank())
      value = tt::ReshapeOp::create(builder, loc, bufferTy.getShape(), value);
    tt::StoreOp::create(builder, loc, ptrs, value, tt::CacheModifier::NONE,
                        tt::EvictionPolicy::NORMAL);
  } else {
    Value value =
        tt::LoadOp::create(builder, loc, ptrs, tt::CacheModifier::NONE,
                           tt::EvictionPolicy::NORMAL, /*isVolatile=*/false);
    if (blockTy.getRank() != bufferTy.getRank())
      value = tt::ReshapeOp::create(builder, loc, blockTy.getShape(), value);
    tt::DescriptorStoreOp::create(builder, loc, desc, value, op.getIndices());
  }
  op.erase();
  return success();
}
#endif // __TLE__

#ifdef __TLE__
// `other` of a pointer load made from a descriptor load: select(padding, nan,
// zero) for a float element type; the zero constant alone for other types.
struct PaddingSelect {
  Value padding;
  Value zero;
};

static std::optional<PaddingSelect> matchPaddingSelect(Value other) {
  auto select = other.getDefiningOp<arith::SelectOp>();
  if (!select || !select.getCondition().getType().isInteger(1))
    return std::nullopt;
  auto zero = select.getFalseValue().getDefiningOp<arith::ConstantOp>();
  if (!zero)
    return std::nullopt;
  auto splat = dyn_cast<SplatElementsAttr>(zero.getValue());
  if (!splat)
    return std::nullopt;
  auto value = dyn_cast<FloatAttr>(splat.getSplatValue<Attribute>());
  if (!value || !value.getValue().isZero())
    return std::nullopt;
  return PaddingSelect{select.getCondition(), zero.getResult()};
}

// The pointer of a rewritten descriptor load is
//   addptr(..., broadcast(muli(offsets, splat(stride))))
// per dimension, innermost last. Replaces the innermost product by its
// offsets when the stride is not already a constant.
static void dropInnermostStride(tt::LoadOp load) {
  auto addptr = load.getPtr().getDefiningOp<tt::AddPtrOp>();
  if (!addptr)
    return;
  auto broadcast = addptr.getOffset().getDefiningOp<tt::BroadcastOp>();
  if (!broadcast)
    return;
  auto mul = broadcast.getSrc().getDefiningOp<arith::MulIOp>();
  if (!mul)
    return;
  auto splat = mul.getRhs().getDefiningOp<tt::SplatOp>();
  if (!splat || splat.getSrc().getDefiningOp<arith::ConstantOp>())
    return;
  broadcast.getSrcMutable().assign(mul.getLhs());
}

// A per-dimension hint: the axis analysis ignores a scalar hint on a tensor
// of rank > 1.
static void setDivisibility(Operation *op, int64_t value) {
  auto type = cast<RankedTensorType>(op->getResult(0).getType());
  OpBuilder builder(op);
  SmallVector<int64_t> values(type.getRank(), value);
  op->setAttr("tt.divisibility",
              DenseIntElementsAttr::get(
                  RankedTensorType::get({type.getRank()}, builder.getI64Type()),
                  values));
}

// The position of the dimension a per-dimension offset or bound tensor
// spans: the rewrite expands each dimension's range to the block rank with
// unit extents elsewhere.
static std::optional<unsigned> spannedDim(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return std::nullopt;
  std::optional<unsigned> dim;
  for (auto [d, extent] : llvm::enumerate(type.getShape())) {
    if (extent == 1)
      continue;
    if (dim)
      return std::nullopt;
    dim = d;
  }
  return dim;
}

// Divisibility hints for the pointer and mask of a rewritten host descriptor
// load, from the descriptor argument's alignment bits: the base splat, the
// extent splats of the aligned dimensions, and the stride splats of the outer
// dimensions (16-byte multiples, checked on the host).
static void hintDescriptorAlignment(tt::LoadOp load, int64_t bits) {
  auto ptrType = cast<RankedTensorType>(load.getPtr().getType());
  int64_t elemBytes =
      std::max<int64_t>(cast<tt::PointerType>(ptrType.getElementType())
                                .getPointeeType()
                                .getIntOrFloatBitWidth() /
                            8,
                        1);
  Value ptr = load.getPtr();
  while (auto addptr = ptr.getDefiningOp<tt::AddPtrOp>()) {
    if (auto broadcast = addptr.getOffset().getDefiningOp<tt::BroadcastOp>())
      if (auto mul = broadcast.getSrc().getDefiningOp<arith::MulIOp>())
        if (auto stride = mul.getRhs().getDefiningOp<tt::SplatOp>())
          setDivisibility(stride,
                          std::max<int64_t>(kVectorBytes / elemBytes, 1));
    ptr = addptr.getPtr();
  }
  if (auto base = ptr.getDefiningOp<tt::SplatOp>(); base && (bits & 1))
    setDivisibility(base, kVectorBytes);

  if (!load.getMask())
    return;
  SmallVector<Value> worklist{load.getMask()};
  while (!worklist.empty()) {
    Operation *def = worklist.pop_back_val().getDefiningOp();
    if (!def)
      continue;
    if (isa<arith::AndIOp, tt::BroadcastOp>(def)) {
      llvm::append_range(worklist, def->getOperands());
      continue;
    }
    auto cmp = dyn_cast<arith::CmpIOp>(def);
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
      continue;
    auto extent = cmp.getRhs().getDefiningOp<tt::SplatOp>();
    std::optional<unsigned> dim = spannedDim(cmp.getLhs());
    if (extent && dim && (bits & (2 << *dim)))
      setDivisibility(extent, kVectorBytes);
  }
}

static LogicalResult zeroPadCopyLoad(tt::LoadOp load,
                                     llvm::DenseSet<Value> &asserted) {
  if (!load.getOther())
    return success();
  std::optional<PaddingSelect> padding = matchPaddingSelect(load.getOther());
  if (!padding)
    return success();
  load.getOtherMutable().assign(padding->zero);
  dropInnermostStride(load);
  if (auto bits = load->getAttrOfType<IntegerAttr>(kDescriptorDivisibilityAttr))
    hintDescriptorAlignment(load, bits.getInt());
  if (!asserted.insert(padding->padding).second)
    return success();
  // The flag is a kernel argument or a constant: check it once, where it is
  // defined, so that every copy of the descriptor is covered.
  OpBuilder builder(load.getContext());
  if (Operation *def = padding->padding.getDefiningOp())
    builder.setInsertionPointAfter(def);
  else
    builder.setInsertionPointToStart(
        cast<BlockArgument>(padding->padding).getOwner());
  Location loc = load.getLoc();
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 1);
  Value zeroPadding =
      arith::XOrIOp::create(builder, loc, padding->padding, one);
  tt::AssertOp::create(
      builder, loc, zeroPadding,
      builder.getStringAttr("PPU descriptor copies to shared memory pad with "
                            "zero; the descriptor asks for NaN padding"));
  return success();
}
#endif // __TLE__

class TritonPPUTleZeroPadDescriptorCopies
    : public impl::TritonPPUTleZeroPadDescriptorCopiesBase<
          TritonPPUTleZeroPadDescriptorCopies> {
public:
  using impl::TritonPPUTleZeroPadDescriptorCopiesBase<
      TritonPPUTleZeroPadDescriptorCopies>::
      TritonPPUTleZeroPadDescriptorCopiesBase;

  void runOnOperation() override {
#ifdef __TLE__
    SmallVector<tt::LoadOp> loads;
    getOperation().walk([&](tt::LoadOp load) {
      if (load->hasAttr(kDescriptorCopyLoadAttr))
        loads.push_back(load);
    });
    llvm::DenseSet<Value> asserted;
    for (tt::LoadOp load : loads) {
      if (failed(zeroPadCopyLoad(load, asserted))) {
        signalPassFailure();
        return;
      }
      load->removeAttr(kDescriptorCopyLoadAttr);
      load->removeAttr(kDescriptorDivisibilityAttr);
    }
#endif
  }
};

class TritonPPUTleLowerTmaCopy
    : public impl::TritonPPUTleLowerTmaCopyBase<TritonPPUTleLowerTmaCopy> {
public:
  using impl::TritonPPUTleLowerTmaCopyBase<
      TritonPPUTleLowerTmaCopy>::TritonPPUTleLowerTmaCopyBase;

  void runOnOperation() override {
#ifdef __TLE__
    SmallVector<ttg::TMACopyOp> copies;
    getOperation().walk([&](ttg::TMACopyOp op) { copies.push_back(op); });
    for (ttg::TMACopyOp op : copies)
      if (failed(lowerCopy(op))) {
        signalPassFailure();
        return;
      }
#endif
  }
};

} // namespace
} // namespace mlir
