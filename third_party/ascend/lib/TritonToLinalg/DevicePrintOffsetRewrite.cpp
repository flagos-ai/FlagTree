/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/TritonToLinalg/DevicePrintOffsetRewrite.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <functional>

namespace mlir {
namespace triton {
namespace {

static bool hasOnlyPrintUsers(Value v) {
  if (v.use_empty())
    return false;
  for (Operation *u : v.getUsers()) {
    auto callOp = dyn_cast<func::CallOp>(u);
    if (!callOp)
      return false;
    if (!callOp.getCallee().starts_with("triton_print"))
      return false;
  }
  return true;
}

static Value materializeIndex(OpBuilder &bd, Location loc, OpFoldResult ofr) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    int64_t v = cast<IntegerAttr>(attr).getInt();
    return bd.create<arith::ConstantIndexOp>(loc, v);
  }
  auto val = llvm::cast<Value>(ofr);
  Type ty = val.getType();
  if (ty.isIndex())
    return val;
  if (isa<IntegerType>(ty))
    return bd.create<arith::IndexCastOp>(loc, bd.getIndexType(), val);
  return nullptr;
}

static bool isStaticEqual(OpFoldResult ofr, int64_t target) {
  if (auto attr = ofr.dyn_cast<Attribute>())
    return cast<IntegerAttr>(attr).getInt() == target;
  return false;
}

class ScalarChainWalker {
public:
  ScalarChainWalker(OpBuilder &bd, Location loc) : bd(bd), loc(loc) {}

  Value emit(Value v, ArrayRef<Value> indices) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return nullptr;

    if (auto cstOp = dyn_cast<arith::ConstantOp>(def))
      return scalarFromConstant(cstOp);
    if (auto fillOp = dyn_cast<linalg::FillOp>(def))
      return scalarFromFill(fillOp);
    if (auto generic = dyn_cast<linalg::GenericOp>(def))
      return scalarFromMakeRange(generic, indices);

    if (auto bc = dyn_cast<linalg::BroadcastOp>(def))
      return emitBroadcast(bc, indices);
    if (auto exp = dyn_cast<tensor::ExpandShapeOp>(def))
      return emitExpandShape(exp, indices);
    if (auto col = dyn_cast<tensor::CollapseShapeOp>(def))
      return emitCollapseShape(col, indices);

    if (def->getNumOperands() != 2)
      return nullptr;
    Value lhs = emit(def->getOperand(0), indices);
    Value rhs = emit(def->getOperand(1), indices);
    if (!lhs || !rhs)
      return nullptr;
    return emitBinaryArith(def, lhs, rhs);
  }

private:
  Value scalarFromConstant(arith::ConstantOp op) {
    auto i32Ty = bd.getI32Type();
    if (auto dense = dyn_cast<DenseIntElementsAttr>(op.getValue()))
      if (dense.isSplat())
        return bd.create<arith::ConstantIntOp>(
            loc, dense.getSplatValue<APInt>().getSExtValue(), i32Ty.getWidth());
    if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue()))
      return bd.create<arith::ConstantIntOp>(loc, intAttr.getInt(),
                                             i32Ty.getWidth());
    return nullptr;
  }

  Value scalarFromFill(linalg::FillOp op) {
    if (op.getInputs().empty())
      return nullptr;
    Value sc = op.getInputs()[0];
    auto i32Ty = bd.getI32Type();
    auto scTy = sc.getType();

    if (auto cstOp = sc.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(cstOp.getValue()))
        return bd.create<arith::ConstantIntOp>(loc, intAttr.getInt(),
                                               i32Ty.getWidth());
      return nullptr;
    }
    if (scTy.isInteger(32))
      return sc;
    if (scTy.isIndex())
      return bd.create<arith::IndexCastOp>(loc, i32Ty, sc);
    if (auto intTy = dyn_cast<IntegerType>(scTy)) {
      if (intTy.getWidth() < 32)
        return bd.create<arith::ExtSIOp>(loc, i32Ty, sc);
      if (intTy.getWidth() > 32)
        return bd.create<arith::TruncIOp>(loc, i32Ty, sc);
    }
    return nullptr;
  }

  Value scalarFromMakeRange(linalg::GenericOp op, ArrayRef<Value> indices) {
    auto i32Ty = bd.getI32Type();
    if (!op->hasAttr("tt.from_make_range"))
      return nullptr;
    if (indices.size() != 1)
      return nullptr;
    int64_t off = 0;
    if (auto a = op->getAttrOfType<IntegerAttr>("tt.make_range_offset"))
      off = a.getInt();
    Value v32 = bd.create<arith::IndexCastOp>(loc, i32Ty, indices[0]);
    if (off != 0) {
      Value oc = bd.create<arith::ConstantIntOp>(loc, off, i32Ty.getWidth());
      v32 = bd.create<arith::AddIOp>(loc, v32, oc);
    }
    return v32;
  }

  Value emitBroadcast(linalg::BroadcastOp op, ArrayRef<Value> indices) {
    auto bcDims = op.getDimensions();
    llvm::DenseSet<int64_t> dropSet(bcDims.begin(), bcDims.end());
    SmallVector<Value> inputIdx;
    for (size_t k = 0; k < indices.size(); ++k)
      if (!dropSet.count(k))
        inputIdx.push_back(indices[k]);
    return emit(op.getInput(), inputIdx);
  }

  Value emitExpandShape(tensor::ExpandShapeOp op, ArrayRef<Value> indices) {
    auto resTy = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<Value> inputIdx;
    for (auto group : op.getReassociationIndices()) {
      Value combined;
      for (size_t i = 0; i < group.size(); ++i) {
        int64_t d = group[i];
        int64_t weight = 1;
        for (size_t j = i + 1; j < group.size(); ++j)
          weight *= resTy.getDimSize(group[j]);
        Value term = indices[d];
        if (weight != 1) {
          Value w = bd.create<arith::ConstantIndexOp>(loc, weight);
          term = bd.create<arith::MulIOp>(loc, term, w);
        }
        combined =
            !combined ? term : bd.create<arith::AddIOp>(loc, combined, term);
      }
      if (!combined)
        combined = bd.create<arith::ConstantIndexOp>(loc, 0);
      inputIdx.push_back(combined);
    }
    return emit(op.getSrc(), inputIdx);
  }

  Value emitCollapseShape(tensor::CollapseShapeOp op, ArrayRef<Value> indices) {
    auto inTy = cast<RankedTensorType>(op.getSrc().getType());
    SmallVector<Value> inputIdx(inTy.getRank());
    Value c0 = bd.create<arith::ConstantIndexOp>(loc, 0);

    auto reassoc = op.getReassociationIndices();
    for (size_t outDim = 0; outDim < indices.size(); ++outDim) {
      auto &group = reassoc[outDim];
      Value out = indices[outDim];

      int64_t leftmostNonOne = -1;
      for (size_t i = 0; i < group.size(); ++i)
        if (inTy.getDimSize(group[i]) != 1) {
          leftmostNonOne = (int64_t)i;
          break;
        }

      for (size_t i = 0; i < group.size(); ++i) {
        int64_t d = group[i];
        int64_t size = inTy.getDimSize(d);
        if (size == 1) {
          inputIdx[d] = c0;
          continue;
        }
        int64_t weight = 1;
        for (size_t j = i + 1; j < group.size(); ++j)
          weight *= inTy.getDimSize(group[j]);
        Value v = out;
        if (weight != 1) {
          Value w = bd.create<arith::ConstantIndexOp>(loc, weight);
          v = bd.create<arith::DivUIOp>(loc, v, w);
        }
        if ((int64_t)i != leftmostNonOne) {
          Value s = bd.create<arith::ConstantIndexOp>(loc, size);
          v = bd.create<arith::RemUIOp>(loc, v, s);
        }
        inputIdx[d] = v;
      }
    }
    return emit(op.getSrc(), inputIdx);
  }

  Value emitBinaryArith(Operation *def, Value lhs, Value rhs) {
    if (isa<arith::MulIOp>(def))
      return bd.create<arith::MulIOp>(loc, lhs, rhs);
    if (isa<arith::AddIOp>(def))
      return bd.create<arith::AddIOp>(loc, lhs, rhs);
    if (isa<arith::SubIOp>(def))
      return bd.create<arith::SubIOp>(loc, lhs, rhs);
    if (isa<arith::RemSIOp>(def))
      return bd.create<arith::RemSIOp>(loc, lhs, rhs);
    if (isa<arith::DivSIOp>(def))
      return bd.create<arith::DivSIOp>(loc, lhs, rhs);
    if (isa<arith::RemUIOp>(def))
      return bd.create<arith::RemUIOp>(loc, lhs, rhs);
    if (isa<arith::DivUIOp>(def))
      return bd.create<arith::DivUIOp>(loc, lhs, rhs);
    return nullptr;
  }

  OpBuilder &bd;
  Location loc;
};

static void emitLoopsAndPrint(
    OpBuilder &builder, Location loc, func::FuncOp scalarPrintFn,
    ArrayRef<int64_t> shape,
    llvm::function_ref<Value(OpBuilder &, Location, ArrayRef<Value>)>
        scalarGen) {
  auto i32Ty = builder.getI32Type();
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value> ivs(shape.size(), c0);
  OpBuilder bd = builder;
  for (size_t k = 0; k < shape.size(); ++k) {
    if (shape[k] <= 1)
      continue;
    Value upper = bd.create<arith::ConstantIndexOp>(loc, shape[k]);
    auto forOp = bd.create<scf::ForOp>(loc, c0, upper, c1);
    ivs[k] = forOp.getInductionVar();
    bd.setInsertionPoint(forOp.getBody()->getTerminator());
  }

  Value scalar = scalarGen(bd, loc, ivs);
  if (!scalar)
    return;
  if (scalar.getType().isIndex())
    scalar = bd.create<arith::IndexCastOp>(loc, i32Ty, scalar);
  else if (!scalar.getType().isInteger(32))
    return;
  bd.create<func::CallOp>(loc, scalarPrintFn, ValueRange{scalar});
}

enum class ChainKind {
  Unsupported,
  Linear,
  Linearized,
};

struct OffsetChainInfo {
  ChainKind kind = ChainKind::Unsupported;
  StringRef rejectReason;
  SmallVector<int64_t> shape;
  Value chainRoot;
  SmallVector<OpFoldResult> strides;
  OpFoldResult baseOffset;
};

static void classifyChain(Value root, bool &canWalk, bool &hasDivMod) {
  canWalk = true;
  hasDivMod = false;

  llvm::DenseSet<Operation *> visited;
  std::function<void(Value)> walk = [&](Value v) {
    if (!canWalk)
      return;
    auto ty = dyn_cast<RankedTensorType>(v.getType());
    if (!ty)
      return;
    if (!ty.hasStaticShape()) {
      canWalk = false;
      return;
    }

    Operation *def = v.getDefiningOp();
    if (!def) {
      canWalk = false;
      return;
    }
    if (!visited.insert(def).second)
      return;

    if (auto fillOp = dyn_cast<linalg::FillOp>(def)) {
      if (fillOp.getInputs().empty() ||
          !fillOp.getInputs()[0].getType().isIntOrIndex())
        canWalk = false;
      return;
    }
    if (isa<arith::ConstantOp, tensor::EmptyOp>(def))
      return;
    if (auto g = dyn_cast<linalg::GenericOp>(def)) {
      if (!g->hasAttr("tt.from_make_range"))
        canWalk = false;
      return;
    }
    if (auto bc = dyn_cast<linalg::BroadcastOp>(def)) {
      walk(bc.getInput());
      return;
    }
    if (auto exp = dyn_cast<tensor::ExpandShapeOp>(def)) {
      auto resTy = cast<RankedTensorType>(exp.getResult().getType());
      for (auto group : exp.getReassociationIndices())
        for (int64_t d : group)
          if (ShapedType::isDynamic(resTy.getDimSize(d))) {
            canWalk = false;
            return;
          }
      walk(exp.getSrc());
      return;
    }
    if (auto col = dyn_cast<tensor::CollapseShapeOp>(def)) {
      auto inTy = cast<RankedTensorType>(col.getSrc().getType());
      for (auto group : col.getReassociationIndices())
        for (int64_t d : group)
          if (ShapedType::isDynamic(inTy.getDimSize(d))) {
            canWalk = false;
            return;
          }
      walk(col.getSrc());
      return;
    }
    if (isa<arith::RemSIOp, arith::DivSIOp, arith::RemUIOp, arith::DivUIOp>(
            def)) {
      hasDivMod = true;
      for (Value operand : def->getOperands())
        walk(operand);
      return;
    }
    if (isa<arith::MulIOp, arith::AddIOp, arith::SubIOp>(def)) {
      for (Value operand : def->getOperands())
        walk(operand);
      return;
    }
    canWalk = false;
  };

  walk(root);
}

static memref::ReinterpretCastOp
findMatchingCast(func::FuncOp funcOp, ArrayRef<int64_t> targetShape) {
  memref::ReinterpretCastOp best;
  unsigned bestScore = 0;
  funcOp.walk([&](memref::ReinterpretCastOp c) {
    auto resTy = dyn_cast<MemRefType>(c.getResult().getType());
    if (!resTy)
      return;
    if (resTy.getShape() != targetShape)
      return;
    unsigned score = 0;
    if (auto blockArg = dyn_cast<BlockArgument>(c.getSource())) {
      unsigned idx = blockArg.getArgNumber();
      if (auto argDict = funcOp.getArgAttrDict(idx))
        if (argDict.contains("tt.tensor_kind"))
          score = 1;
    }
    if (score >= bestScore) {
      bestScore = score;
      best = c;
    }
  });
  return best;
}

static OffsetChainInfo analyzeChain(Value arg, func::FuncOp funcOp,
                                    Operation *anchorOp,
                                    DominanceInfo &domInfo) {
  OffsetChainInfo info;

  auto argTy = dyn_cast<RankedTensorType>(arg.getType());
  if (!argTy) {
    info.rejectReason = "arg not RankedTensorType";
    return info;
  }
  if (!argTy.hasStaticShape()) {
    info.rejectReason = "arg has dynamic shape";
    return info;
  }
  if (!hasOnlyPrintUsers(arg)) {
    info.rejectReason = "arg has non-print users";
    return info;
  }

  info.shape.assign(argTy.getShape().begin(), argTy.getShape().end());
  info.chainRoot = arg;

  bool canWalk = false;
  bool hasDivMod = false;
  classifyChain(arg, canWalk, hasDivMod);
  if (!canWalk) {
    info.rejectReason = "chain contains unsupported op";
    return info;
  }

  if (argTy.getRank() > 1 && !hasDivMod) {
    auto castOp = findMatchingCast(funcOp, argTy.getShape());
    if (castOp) {
      SmallVector<OpFoldResult> mixedStrides = castOp.getMixedStrides();
      SmallVector<OpFoldResult> mixedOffsets = castOp.getMixedOffsets();
      if (mixedOffsets.size() == 1 &&
          (int64_t)mixedStrides.size() == argTy.getRank()) {
        auto domOk = [&](OpFoldResult ofr) -> bool {
          if (auto val = ofr.dyn_cast<Value>())
            return domInfo.dominates(val, anchorOp);
          return true;
        };
        bool domAllOk = domOk(mixedOffsets[0]);
        if (domAllOk)
          for (auto s : mixedStrides)
            if (!domOk(s)) {
              domAllOk = false;
              break;
            }
        if (domAllOk) {
          info.kind = ChainKind::Linear;
          info.strides = std::move(mixedStrides);
          info.baseOffset = mixedOffsets[0];
          return info;
        }
      }
    }
  }

  info.kind = ChainKind::Linearized;
  return info;
}

struct RewriteCandidate {
  func::CallOp callOp;
  OffsetChainInfo info;
};

static void rewriteCandidate(RewriteCandidate &cand, ModuleOp moduleOp) {
  auto i32Ty = IntegerType::get(moduleOp.getContext(), 32);
  auto oldFn = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(moduleOp, cand.callOp.getCalleeAttr()));
  if (!oldFn)
    return;

  auto newFnTy = FunctionType::get(moduleOp.getContext(), {i32Ty}, {});
  oldFn.setFunctionType(newFnTy);

  OpBuilder b(cand.callOp);
  Location loc = cand.callOp.getLoc();
  OffsetChainInfo &info = cand.info;

  switch (info.kind) {
  case ChainKind::Linear: {
    auto gen = [&info](OpBuilder &bd, Location loc,
                       ArrayRef<Value> ivs) -> Value {
      Value acc;
      if (!isStaticEqual(info.baseOffset, 0)) {
        acc = materializeIndex(bd, loc, info.baseOffset);
        if (!acc)
          return nullptr;
      }
      for (size_t k = 0; k < info.shape.size(); ++k) {
        if (info.shape[k] <= 1)
          continue;
        Value idx = ivs[k];
        if (!isStaticEqual(info.strides[k], 1)) {
          Value s = materializeIndex(bd, loc, info.strides[k]);
          if (!s)
            return nullptr;
          idx = bd.create<arith::MulIOp>(loc, idx, s);
        }
        if (!acc)
          acc = idx;
        else
          acc = bd.create<arith::AddIOp>(loc, acc, idx);
      }
      if (!acc)
        acc = bd.create<arith::ConstantIndexOp>(loc, 0);
      return acc;
    };
    emitLoopsAndPrint(b, loc, oldFn, info.shape, gen);
    break;
  }

  case ChainKind::Linearized: {
    Value root = info.chainRoot;
    auto gen = [root](OpBuilder &bd, Location loc,
                      ArrayRef<Value> ivs) -> Value {
      ScalarChainWalker walker(bd, loc);
      return walker.emit(root, ivs);
    };
    emitLoopsAndPrint(b, loc, oldFn, info.shape, gen);
    break;
  }

  case ChainKind::Unsupported:
    return;
  }

  cand.callOp.erase();
}

static void rewriteOneFunc(func::FuncOp funcOp) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return;

  DominanceInfo domInfo(funcOp);
  SmallVector<RewriteCandidate> candidates;

  funcOp.walk([&](func::CallOp callOp) {
    if (!callOp.getCallee().starts_with("triton_print"))
      return;
    if (callOp.getNumOperands() != 1)
      return;

    OffsetChainInfo info = analyzeChain(callOp.getOperand(0), funcOp,
                                        callOp.getOperation(), domInfo);
    if (info.kind == ChainKind::Unsupported)
      return;

    candidates.push_back({callOp, std::move(info)});
  });

  for (auto &cand : candidates)
    rewriteCandidate(cand, moduleOp);
}

} // namespace

void rewriteDevicePrintOffsets(ModuleOp moduleOp) {
  moduleOp.walk([&](func::FuncOp funcOp) {
    if (funcOp.isPrivate())
      return;
    rewriteOneFunc(funcOp);
  });
}

} // namespace triton
} // namespace mlir
