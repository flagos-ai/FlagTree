#include "CombineEmit.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Conversion/TritonXPUToLLVM/LegacyLLVMHelpers.h" // dragon-style macros, XPU only
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/LLVMXPU/IR/Dialect.h"
#include "triton/Target/LLVMIR/IntrinsicAttrTable.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::xpu;

namespace mlir::triton::xpu {

bool isSupportedCombineOp(Operation *op) {
  return isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::MaxNumFOp, arith::MinNumFOp, arith::OrIOp, arith::XOrIOp,
             arith::AndIOp, arith::CmpFOp, arith::CmpIOp, arith::AddIOp,
             arith::SubIOp, arith::MulIOp, arith::MaxSIOp,
             arith::MinSIOp, arith::MaxUIOp, arith::MinUIOp,
             arith::MaximumFOp, arith::MinimumFOp, arith::DivSIOp,
             arith::DivUIOp, arith::SelectOp, arith::ConstantOp>(op);
}

bool isUniformConstant(Value v) {
  auto cstOp = v.getDefiningOp<arith::ConstantOp>();
  if (!cstOp)
    return false;
  if (!isa<VectorType>(cstOp.getType()))
    return true;
  auto dense = dyn_cast<DenseElementsAttr>(cstOp.getValue());
  return dense && dense.isSplat();
}

bool canInterpretCombine(Region &combine) {
  if (!llvm::hasSingleElement(combine))
    return false;
  Block &block = combine.front();
  if (!isa<triton::xpu::ReduceReturnOp, triton::xpu::ScanReturnOp>(
          block.getTerminator()))
    return false;
  llvm::SmallPtrSet<Value, 8> defined;
  for (Value arg : block.getArguments())
    defined.insert(arg);
  for (Operation &op : block.without_terminator()) {
    if (!isSupportedCombineOp(&op) || op.getNumResults() != 1)
      return false;
    for (Value operand : op.getOperands())
      // Anything but a uniform constant is captured from outside and has no
      // lowered counterpart here. Constants get rematerialized instead: the
      // canonicalizer hoists them out of the region, so rejecting them would
      // reject welford's combine outright.
      if (!defined.contains(operand) && !isUniformConstant(operand))
        return false;
    defined.insert(op.getResult(0));
  }
  for (Value v : block.getTerminator()->getOperands())
    if (!defined.contains(v))
      return false;
  return true;
}

Value splatToWidth(ConversionPatternRewriter &rewriter, Location loc, Value v,
                   int64_t width) {
  if (isa<VectorType>(v.getType()))
    return v;
  auto vecTy = VectorType::get({width}, v.getType());
  Value splat = undef(vecTy);
  for (int64_t i = 0; i < width; ++i)
    splat = insert_element(vecTy, splat, v, i32_val(i));
  return splat;
}

Value materializeConstant(ConversionPatternRewriter &rewriter, Location loc,
                          arith::ConstantOp cstOp, int64_t width) {
  Attribute attr = cstOp.getValue();
  if (auto dense = dyn_cast<DenseElementsAttr>(attr))
    attr = dense.getSplatValue<Attribute>(); // isUniformConstant checked this
  Value cst = rewriter.create<LLVM::ConstantOp>(
      loc, getElementTypeOrSelf(cstOp.getType()), attr);
  return width > 0 ? splatToWidth(rewriter, loc, cst, width) : cst;
}

namespace {
// Predicate converters, verbatim from the reduce lowering (ReduceOpToLLVM's
// members of the same name). Local rather than shared in the header: they are
// pure switch tables with nothing to drift.
LLVM::FCmpPredicate ArithCmpFPredicateToLLVM(arith::CmpFPredicate predicate) {
  switch (predicate) {
#define __PRED_ENUM(item__, item1__)                                           \
  case arith::CmpFPredicate::item__:                                           \
    return LLVM::FCmpPredicate::item1__

    __PRED_ENUM(OEQ, oeq);
    __PRED_ENUM(ONE, one);
    __PRED_ENUM(OGT, ogt);
    __PRED_ENUM(OGE, oge);
    __PRED_ENUM(OLT, olt);
    __PRED_ENUM(OLE, ole);
    __PRED_ENUM(ORD, ord);
    __PRED_ENUM(UEQ, ueq);
    __PRED_ENUM(UGT, ugt);
    __PRED_ENUM(UGE, uge);
    __PRED_ENUM(ULT, ult);
    __PRED_ENUM(ULE, ule);
    __PRED_ENUM(UNE, une);
    __PRED_ENUM(UNO, uno);
    __PRED_ENUM(AlwaysTrue, _true);
    __PRED_ENUM(AlwaysFalse, _false);

#undef __PRED_ENUM
  }
  llvm_unreachable("Unknown arith::CmpFPredicate");
}

LLVM::ICmpPredicate ArithCmpIPredicateToLLVM(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    return LLVM::ICmpPredicate::eq;
  case arith::CmpIPredicate::ne:
    return LLVM::ICmpPredicate::ne;
  case arith::CmpIPredicate::sgt:
    return LLVM::ICmpPredicate::sgt;
  case arith::CmpIPredicate::sge:
    return LLVM::ICmpPredicate::sge;
  case arith::CmpIPredicate::slt:
    return LLVM::ICmpPredicate::slt;
  case arith::CmpIPredicate::sle:
    return LLVM::ICmpPredicate::sle;
  case arith::CmpIPredicate::ugt:
    return LLVM::ICmpPredicate::ugt;
  case arith::CmpIPredicate::uge:
    return LLVM::ICmpPredicate::uge;
  case arith::CmpIPredicate::ult:
    return LLVM::ICmpPredicate::ult;
  case arith::CmpIPredicate::ule:
    return LLVM::ICmpPredicate::ule;
  }
  llvm_unreachable("Unknown arith::CmpIPredicate");
}
} // namespace

Value emitCombineOp(ConversionPatternRewriter &rewriter, Location loc,
                    Operation *op, SmallVector<Value> &args, int64_t width) {
  if (width > 0)
    for (Value &a : args)
      a = splatToWidth(rewriter, loc, a, width);

  return TypeSwitch<Operation *, Value>(op)
      .Case<arith::AddFOp>(
          [&](auto) -> Value { return fadd(args[0], args[1]); })
      .Case<arith::SubFOp>([&](auto) -> Value {
        return rewriter.create<LLVM::FSubOp>(loc, args[0], args[1]);
      })
      .Case<arith::MulFOp>([&](auto) -> Value { return fmul(args[0], args[1]); })
      .Case<arith::DivFOp>([&](auto) -> Value {
        return rewriter.create<LLVM::FDivOp>(loc, args[0], args[1]);
      })
      .Case<arith::MaxNumFOp>(
          [&](auto) -> Value { return fmax(args[0], args[1]); })
      .Case<arith::MinNumFOp>(
          [&](auto) -> Value { return fmin(args[0], args[1]); })
      .Case<arith::OrIOp>([&](auto) -> Value { return or_(args[0], args[1]); })
      .Case<arith::XOrIOp>(
          [&](auto) -> Value { return xor_(args[0], args[1]); })
      .Case<arith::AndIOp>(
          [&](auto) -> Value { return and_(args[0], args[1]); })
      .Case<arith::CmpFOp>([&](arith::CmpFOp cmpfOp) -> Value {
        Type resTy = rewriter.getI1Type();
        if (auto ty = dyn_cast<VectorType>(args[0].getType()))
          resTy = VectorType::get(ty.getShape(), resTy);
        return rewriter.create<LLVM::FCmpOp>(
            loc, resTy, ArithCmpFPredicateToLLVM(cmpfOp.getPredicate()),
            args[0], args[1]);
      })
      .Case<arith::CmpIOp>([&](arith::CmpIOp cmpiOp) -> Value {
        Type resTy = rewriter.getI1Type();
        if (auto ty = dyn_cast<VectorType>(args[0].getType()))
          resTy = VectorType::get(ty.getShape(), resTy);
        return rewriter.create<LLVM::ICmpOp>(
            loc, resTy, ArithCmpIPredicateToLLVM(cmpiOp.getPredicate()),
            args[0], args[1]);
      })
      .Case<arith::AddIOp>([&](auto) -> Value {
        return rewriter.create<LLVM::AddOp>(loc, args[0], args[1]);
      })
      .Case<arith::SubIOp>([&](auto) -> Value {
        return rewriter.create<LLVM::SubOp>(loc, args[0], args[1]);
      })
      .Case<arith::MulIOp>([&](auto) -> Value {
        return rewriter.create<LLVM::MulOp>(loc, args[0], args[1]);
      })
      .Case<arith::MaxSIOp>([&](auto) -> Value {
        return rewriter.create<LLVM::SMaxOp>(loc, args[0], args[1]);
      })
      .Case<arith::MinSIOp>([&](auto) -> Value {
        return rewriter.create<LLVM::SMinOp>(loc, args[0], args[1]);
      })
      .Case<arith::MaxUIOp>([&](auto) -> Value {
        return rewriter.create<LLVM::UMaxOp>(loc, args[0], args[1]);
      })
      .Case<arith::MinUIOp>([&](auto) -> Value {
        return rewriter.create<LLVM::UMinOp>(loc, args[0], args[1]);
      })
      .Case<arith::MaximumFOp>([&](auto) -> Value {
        return rewriter.create<LLVM::MaximumOp>(loc, args[0], args[1]);
      })
      .Case<arith::MinimumFOp>([&](auto) -> Value {
        return rewriter.create<LLVM::MinimumOp>(loc, args[0], args[1]);
      })
      .Case<arith::DivSIOp>([&](auto) -> Value {
        return rewriter.create<LLVM::SDivOp>(loc, args[0], args[1]);
      })
      .Case<arith::DivUIOp>([&](auto) -> Value {
        return rewriter.create<LLVM::UDivOp>(loc, args[0], args[1]);
      })
      .Case<arith::SelectOp>([&](auto) -> Value {
        return select(args[0], args[1], args[2]);
      })
      .Case<arith::ConstantOp>([&](arith::ConstantOp cstOp) -> Value {
        return materializeConstant(rewriter, loc, cstOp, width);
      })
      .Default([](auto) -> Value { return Value(); });
}

void interpretCombine(ConversionPatternRewriter &rewriter, Location loc,
                      Region &combine, SmallVector<Value> &acc, ValueRange cur) {
  Block &block = combine.front();
  assert(block.getNumArguments() == acc.size() + cur.size() &&
         "combine arity does not match the accumulator");

  llvm::DenseMap<Value, Value> map;
  for (unsigned i = 0; i < acc.size(); ++i) {
    map[block.getArgument(i)] = acc[i];
    map[block.getArgument(acc.size() + i)] = cur[i];
  }

  // The accumulators decide the stage: vector within a core, scalar across
  // cores. Everything else in the region is brought to that shape. A scalar
  // accumulator combined into vector stage values (the scan's carry
  // add-back) still runs at the vector stage, hence the fallback.
  int64_t width = 0;
  for (Value v : acc)
    if (auto vecTy = dyn_cast<VectorType>(v.getType()))
      width = vecTy.getNumElements();
  if (width == 0)
    for (Value v : cur)
      if (auto vecTy = dyn_cast<VectorType>(v.getType()))
        width = vecTy.getNumElements();

  for (Operation &op : block.without_terminator()) {
    SmallVector<Value> args;
    for (Value operand : op.getOperands()) {
      Value mapped = map.lookup(operand);
      if (!mapped) // hoisted out of the region; rematerialize it here
        mapped = materializeConstant(
            rewriter, loc, operand.getDefiningOp<arith::ConstantOp>(), width);
      args.push_back(mapped);
    }
    Value res = emitCombineOp(rewriter, loc, &op, args, width);
    assert(res && "isSupportedCombineOp is out of sync with emitCombineOp");
    map[op.getResult(0)] = res;
  }

  // A combine that only forwards a block argument (e.g. get_first_element's
  // `return a`) has no op for emitCombineOp to widen, so at the vector stage a
  // scalar operand -- the cross-core carry folded into a vector chunk -- would
  // leak a scalar into a vector-typed result and trip packLLElements. Broadcast
  // forwarded values to the active width (no-op when already a vector).
  SmallVector<Value> results;
  for (Value v : block.getTerminator()->getOperands()) {
    Value mapped = map.lookup(v);
    if (width > 0)
      mapped = splatToWidth(rewriter, loc, mapped, width);
    results.push_back(mapped);
  }
  acc = results;
}

void emitMaskedLanePrefix(ConversionPatternRewriter &rewriter, Location loc,
                          Region &combine, SmallVector<Value> &vecs,
                          bool needsMask) {
  if (vecs.empty())
    return;
  auto netTy = cast<VectorType>(vecs[0].getType());
  Type elemTy = netTy.getElementType();
  unsigned lanes = netTy.getNumElements();
  unsigned bitw = elemTy.getIntOrFloatBitWidth();
  bool isF16 = elemTy.isF16();
  bool isFloat = isa<FloatType>(elemTy);
  auto mod = vecs[0].getParentRegion()->getParentOfType<mlir::ModuleOp>();
  // one lane-shift left by k lanes = packed shift by k*bitwidth(T) bits, the
  // (i32, <W x T>) arg order XTDK clang uses for __builtin_xpu*_svsllp_*.
  auto shiftLanes = [&](Value v, unsigned k) -> Value {
    const char *name =
        isF16 ? "llvm.xpu.svsllp_hf"
              : (isFloat ? "llvm.xpu.svsllp_f" : "llvm.xpu.svsllp_s");
    LLVM::LLVMFuncOp callee = mod.lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (!callee) {
      OpBuilder b(mod.getContext());
      b.setInsertionPointToStart(mod.getBody());
      callee = mlir::intrinsic_attr_table::getOrCreateDeclaration(
          b, mod, loc, name,
          LLVM::LLVMFunctionType::get(netTy, {rewriter.getI32Type(), netTy}));
    }
    return rewriter
        .create<LLVM::CallOp>(loc, callee, ValueRange{i32_val(k * bitw), v})
        .getResult();
  };
  Value laneId;
  if (needsMask) {
    auto liTy = VectorType::get(lanes, rewriter.getI32Type());
    laneId = undef(liTy);
    for (unsigned l = 0; l < lanes; ++l)
      laneId = insert_element(liTy, laneId, i32_val(l), i32_val(l));
  }
  for (unsigned k = 1; k < lanes; k <<= 1) {
    SmallVector<Value> prev(vecs.begin(), vecs.end());
    SmallVector<Value> shifted(prev.size());
    for (size_t j = 0; j < prev.size(); ++j)
      shifted[j] = shiftLanes(prev[j], k);
    // folded = combine(shifted, prev): the earlier lane (i-k) is the
    // accumulated prefix (lhs), the current lane the new element (rhs) -- the
    // same operand order as ScanOpToLLVM's cross-chunk carry fold and the
    // serial replay, so a non-commutative associative combine scans correctly
    // (commutative add/max/min/mul are unaffected).
    SmallVector<Value> folded(shifted.begin(), shifted.end());
    interpretCombine(rewriter, loc, combine, folded, prev);
    if (!needsMask) {
      for (size_t j = 0; j < prev.size(); ++j)
        vecs[j] = folded[j];
      continue;
    }
    // keep-lane<k mask via the vvor masked select: <32 x i1>, f16's 32-lane
    // keep is the mask directly, f32's 16-lane keep packs into the low 16 bits.
    auto laneVecTy = VectorType::get(lanes, rewriter.getI32Type());
    Value kSpl = rewriter.create<LLVM::ConstantOp>(
        loc, laneVecTy,
        SplatElementsAttr::get(laneVecTy, rewriter.getI32IntegerAttr(k)));
    Value keep = rewriter.create<LLVM::ICmpOp>(
        loc, VectorType::get(lanes, rewriter.getI1Type()),
        LLVM::ICmpPredicate::ult, laneId, kSpl);
    Value maskV;
    if (isF16) {
      maskV = keep;
    } else {
      Value keepBits =
          bitcast(keep, IntegerType::get(rewriter.getContext(), lanes));
      Value keepI32 =
          rewriter.create<LLVM::ZExtOp>(loc, rewriter.getI32Type(), keepBits);
      maskV = bitcast(keepI32, VectorType::get(32, rewriter.getI1Type()));
    }
    // all-zero vector of the chunk type (keep path does 0 | prev = prev),
    // built at the element's own width so the bitcast is size-preserving.
    auto zEltTy = IntegerType::get(rewriter.getContext(), bitw);
    auto zIntTy = VectorType::get(lanes, zEltTy);
    Value zerosI = rewriter.create<LLVM::ConstantOp>(
        loc, zIntTy,
        SplatElementsAttr::get(zIntTy, rewriter.getIntegerAttr(zEltTy, 0)));
    for (size_t j = 0; j < prev.size(); ++j) {
      Value zeros = bitcast(zerosI, prev[j].getType());
      // dst = keep ? (zeros | prev) : folded -- lane<k keeps the original.
      // vvor is a bitwise masked-or, split by element kind/width: f16 is the
      // 32-lane form, f32 and i32 share the 16-lane form (vvor_f / vvor_s).
      if (isF16)
        vecs[j] = rewriter.create<mlir::LLVM::XPU::VVOR_HF_MHOp>(
            loc, prev[j].getType(), maskV, zeros, prev[j], folded[j]);
      else if (isFloat)
        vecs[j] = rewriter.create<mlir::LLVM::XPU::VVOR_F_MHOp>(
            loc, prev[j].getType(), maskV, zeros, prev[j], folded[j]);
      else
        vecs[j] = rewriter.create<mlir::LLVM::XPU::VVOR_S_MHOp>(
            loc, prev[j].getType(), maskV, zeros, prev[j], folded[j]);
    }
  }
}

} // namespace mlir::triton::xpu
