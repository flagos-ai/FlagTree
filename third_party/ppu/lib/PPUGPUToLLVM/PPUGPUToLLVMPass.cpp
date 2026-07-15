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

#include "PPUGPUToLLVM/PPUGPUToLLVMPass.h"
#include "PPUGPUToLLVM/Passes.h"

#include "Dialect/PPUGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "ppu/lib/TritonPPUGPUToLLVM/Utility.h"
#include "llvm/Support/ErrorHandling.h"

namespace ttn = mlir::triton::ppugpu;
using ttn::Constraints;
using ttn::OperandsAndConstraints;

using namespace mlir::triton::ppu;

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_CONVERTPPUGPUTOLLVM
#include "PPUGPUToLLVM/Passes.h.inc"

namespace {

bool isNumber(const std::string &s) {
  return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) {
                         return !std::isdigit(c);
                       }) == s.end();
}

Type getTypeFromConstraint(char constraint, PatternRewriter &rewriter) {
  Type ty;
  if (constraint == 'b')
    ty = IntegerType::get(rewriter.getContext(), 1);
  else if (constraint == 'h')
    ty = IntegerType::get(rewriter.getContext(), 16);
  else if (constraint == 'r')
    ty = IntegerType::get(rewriter.getContext(), 32);
  else if (constraint == 'l')
    ty = IntegerType::get(rewriter.getContext(), 64);
  else if (constraint == 'f')
    ty = Float32Type::get(rewriter.getContext());
  else if (constraint == 'd')
    ty = Float64Type::get(rewriter.getContext());
  else {
    assert(false && "Unsupported constraint");
  }
  return ty;
}

// Converts the given value to the type represented by the constraint
// E.g. if val is of type llvmptr and constraint is 'r', then we convert
// val to i32 using ptrtoint(i32_ty, val)
Value convertToType(Value val, std::string constraint, Location loc,
                    PatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto isConstraintNumber = isNumber(constraint);
  if (!isConstraintNumber) {
    auto ty = getTypeFromConstraint(constraint[0], rewriter);
    if (isa<LLVM::LLVMPointerType>(val.getType())) {
      return b.ptrtoint(ty, val);
    } else {
      assert(val.getType().getIntOrFloatBitWidth() <=
                 ty.getIntOrFloatBitWidth() &&
             "Cannot convert to a smaller type");
      if (val.getType().getIntOrFloatBitWidth() < ty.getIntOrFloatBitWidth())
        return b.zext(ty, val);
    }
  }
  return val;
}

SmallVector<TIXBuilder::Operand *>
getTixOutputs(const ppugpu::Constraints &outputConstraints,
              TIXBuilder &tixBuilder) {
  SmallVector<TIXBuilder::Operand *> tixOutputs;
  for (unsigned i = 0; i < outputConstraints.size(); i++) {
    auto *tixOutput = tixBuilder.newOperand(outputConstraints[i]);
    tixOutputs.push_back(tixOutput);
  }
  return tixOutputs;
}

OperandsAndConstraints
unpackOperands(const OperandsAndConstraints &operandsAndConstraints,
               TIXBuilder &tixBuilder, Location loc,
               PatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  OperandsAndConstraints unpackedOperands;
  for (const auto &[operand, constraint] : operandsAndConstraints) {
    auto llvmStruct = llvm::dyn_cast<LLVM::LLVMStructType>(operand.getType());
    // if a constraint is a number, then we are doing input/output tying
    // if the operand is a struct, then we need to unpack it, and
    // add the constraint to each of the unpacked operands uses the constraint
    // as an offset
    auto isConstraintNumber = isNumber(constraint);
    if (llvmStruct) {
      for (unsigned i = 0; i < llvmStruct.getBody().size(); i++) {
        if (isConstraintNumber) {
          auto constraintInt = std::stoi(constraint) + i;
          unpackedOperands.push_back(
              {b.extract_val(llvmStruct.getBody()[i], operand, i),
               std::to_string(constraintInt)});
        } else {
          unpackedOperands.push_back(
              {b.extract_val(llvmStruct.getBody()[i], operand, i), constraint});
        }
      }
    } else {
      unpackedOperands.push_back({operand, constraint});
    }
  }
  return unpackedOperands;
}

SmallVector<TIXBuilder::Operand *>
getTixOperands(const OperandsAndConstraints &operandsAndConstraints,
               TIXBuilder &tixBuilder, Location loc,
               PatternRewriter &rewriter) {
  SmallVector<TIXBuilder::Operand *> tixOperands;
  auto unpackedOperandsAndConstraints =
      unpackOperands(operandsAndConstraints, tixBuilder, loc, rewriter);
  for (auto &[operand, constraint] : unpackedOperandsAndConstraints) {
    auto convertedOperand = convertToType(operand, constraint, loc, rewriter);
    auto *tixOperand = tixBuilder.newOperand(convertedOperand, constraint);
    tixOperands.push_back(tixOperand);
  }
  return tixOperands;
}

std::string patchTixAsm(Operation *op, std::string tixAsm) {
  std::vector<std::pair<int, int>> patchLocations;
  std::vector<std::string> patchValues;
  auto start = tixAsm.find("#", 0);
  while (start != std::string::npos) {
    auto endIterator =
        std::find_if(tixAsm.begin() + start + 1, tixAsm.end(),
                     [](unsigned char c) { return !std::isalnum(c); });

    assert(endIterator != tixAsm.end() && "unexpected asm format");

    auto end = std::distance(tixAsm.begin(), endIterator);
    auto patchLocation = std::make_pair(start, end);
    patchLocations.push_back(patchLocation);
    auto patchValue = tixAsm.substr(start + 1, end - start - 1);
    patchValues.push_back(patchValue);
    start = tixAsm.find("#", end);
  }
  assert(patchLocations.size() == patchValues.size() &&
         "patchLocations and patchValues should have the same size");
  if (patchLocations.size() == 0) {
    return tixAsm;
  }
  std::string res = "";
  size_t prevStart = 0;
  unsigned i = 0;
  for (auto &[start, end] : patchLocations) {
    res += tixAsm.substr(prevStart, start - prevStart);
    auto integerAttr = op->getAttrOfType<IntegerAttr>(patchValues[i]);
    auto attr = integerAttr.getInt();
    res += std::to_string(attr);
    prevStart = end;
    i++;
  }
  if (prevStart < tixAsm.size())
    res += tixAsm.substr(prevStart, tixAsm.size() - prevStart);
  return res;
}

template <typename SourceOp>
class PPUGPUOpGenericPattern : public OpRewritePattern<SourceOp> {
public:
  explicit PPUGPUOpGenericPattern(MLIRContext *context, std::string tixAsm,
                                 Constraints outputConstraints,
                                 Constraints inputConstraints)
      : OpRewritePattern<SourceOp>(context), tixAsm(std::move(tixAsm)),
        outputConstraints(outputConstraints),
        inputConstraints(inputConstraints) {}

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    OperandsAndConstraints operandsAndConstraints;
    for (unsigned i = 0; i < inputConstraints.size(); i++) {
      operandsAndConstraints.push_back(
          {op->getOperand(i), inputConstraints[i]});
    }
    return rewriteAsTixAsm(op, rewriter, tixAsm, operandsAndConstraints,
                           outputConstraints);
  }

private:
  std::string tixAsm;
  Constraints outputConstraints;
  Constraints inputConstraints;
};

class WarpIdOpPattern : public OpRewritePattern<ttn::WarpIdOp> {
public:
  using OpRewritePattern<ttn::WarpIdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::WarpIdOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    if (triton::gpu::lookupNumWarps(op) == 1) {
      // If there is only one warp, the warp ID is always 0.
      rewriter.replaceOp(op, b.i32_val(0));
      return success();
    }

    // If this is inside a warp specialize op, compute the relative thread ID
    // within the warp group.
    Value tid = NVVM::ThreadIdXOp::create(rewriter, loc, i32_ty);
    if (std::optional<int> startId =
            getWarpGroupStartThreadId(rewriter.getInsertionBlock()))
      tid = LLVM::SubOp::create(rewriter, loc, tid, b.i32_val(*startId));

    Value warpId = b.udiv(tid, b.i32_val(32));
    // This indicates to ppu-llc that the result and its derived values are
    // uniform across the warp. For example, if a branch condition derives from
    // this value, it can be proven to be non-divergent.
    warpId = LLVM::PPU::shuffleIdx(loc, rewriter, warpId, 0);
    rewriter.replaceOp(op, warpId);
    return success();
  }
};

class ClusterCTAIdOpPattern : public OpRewritePattern<ttn::ClusterCTAIdOp> {
  using OpRewritePattern<ttn::ClusterCTAIdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::ClusterCTAIdOp op,
                                PatternRewriter &rewriter) const override {
    // TODO Should we pass in the range of the cluster ID?
    // We should benchmark as when doing so for thread_id it regressed lol
    // auto numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(
    //     op->getParentOfType<ModuleOp>());
    auto res = NVVM::ClusterId::create(rewriter, op.getLoc(), i32_ty);
    rewriter.replaceOp(op, res);
    return success();
  }
};

// PPULoadMatrixOp Pattern
class PPULoadMatrixOpPattern : public OpRewritePattern<ttn::PPULoadMatrixOp> {
public:
  using OpRewritePattern<ttn::PPULoadMatrixOp>::OpRewritePattern;
  static constexpr const char *kOpCode = "ppu.ldmatrix";

  LogicalResult matchAndRewrite(ttn::PPULoadMatrixOp op,
                                PatternRewriter &rewriter) const override {
    unsigned vecSize = getVectorSize(op);
    bool trans = op.getTrans();
    std::string tixAsm = (llvm::Twine(PPULoadMatrixOpPattern::kOpCode) +
                          getTixModifiers(vecSize, trans, op) + " " +
                          getOperands(op, vecSize) + ";")
                             .str();

    OperandsAndConstraints operandAndConstraints =
        getOperandsAndConstraints(op, vecSize);
    Constraints outputConstraints = getOutputConstraints(op, vecSize);

    return rewriteAsTixAsm(op, rewriter, tixAsm, operandAndConstraints,
                           outputConstraints);
  }

protected:
  // Shared helper methods
  std::string getTixModifiers(unsigned vecSize, bool trans,
                              ttn::PPULoadMatrixOp op) const {
    if (op.getOpb8bLdmatrix())
      return ".sync.aligned.m16n16.x1.trans.b8.shared";
    auto tixAsmBase = llvm::Twine(".sync.aligned.m8n8");
    auto tixAsmBaseTrans = llvm::Twine(".sync.aligned.m16n16");
    const std::string suffix = trans ? ".trans.shared.b16" : ".shared.b16";
    switch (vecSize) {
    case 1:
      return (tixAsmBase + ".x1" + suffix).str();
    case 2:
      return (tixAsmBase + ".x2" + suffix).str();
    case 4:
      if (trans)
        return (tixAsmBaseTrans + ".x1" + suffix).str();
      else
        return (tixAsmBase + ".x4" + suffix).str();
    default:
      llvm_unreachable("Invalid vector size");
    }
  }

  std::string getTixRegOperands(unsigned startIdx, unsigned count) const {
    llvm::SmallString<20> regOperands;
    llvm::raw_svector_ostream stream(regOperands);
    stream << "{";
    for (unsigned i = 0; i < count; i++) {
      stream << "$" + llvm::utostr(startIdx + i);
      if (i != count - 1)
        stream << ", ";
    }
    stream << "}";
    return std::string(regOperands.str());
  }

  std::string getTixAddrOperand(unsigned idx) const {
    return (llvm::Twine("[$") + llvm::utostr(idx) + "]").str();
  }

  unsigned getVectorSize(ttn::PPULoadMatrixOp op) const {
    auto resultType = cast<LLVM::LLVMStructType>(op.getType());
    return resultType.getBody().size();
  }

  std::string getOperands(ttn::PPULoadMatrixOp op, unsigned vecSize) const {
    return (llvm::Twine(getTixRegOperands(0, vecSize)) + ", " +
            getTixAddrOperand(vecSize))
        .str();
  }

  OperandsAndConstraints getOperandsAndConstraints(ttn::PPULoadMatrixOp op,
                                                   unsigned vecSize) const {
    return {{op.getAddr(), "r"}};
  }

  Constraints getOutputConstraints(ttn::PPULoadMatrixOp op,
                                   unsigned vecSize) const {
    return Constraints(vecSize, "=r");
  }
};

class LoadAcquireOpPattern : public OpRewritePattern<ttn::LoadAcquireOp> {
public:
  using OpRewritePattern<ttn::LoadAcquireOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::LoadAcquireOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Type valueTy = op.getType();
    const unsigned valueNBits =
        std::max(8u, (unsigned)getIntOrFloatOrPtrBitWidth(valueTy));
    const size_t maxWordWidth = std::max<size_t>(32, valueNBits);
    const size_t width = std::min((size_t)valueNBits, maxWordWidth);

    const std::string writeConstraint =
        (width == 64) ? "=l" : ((width == 32) ? "=r" : "=c");
    TIXBuilder tixBuilder;
    bool init = true;
    auto *dstOpr = tixBuilder.newOperand(writeConstraint, init); // =r operation
    auto *addrOpr =
        tixBuilder.newAddrOperand(op.getAddr(), "l", 0 /* in_off */);
    auto &ld =
        tixBuilder.create("ppu.ld")
            ->global()
            .o("cta", op.getScope() == triton::ppugpu::MemSyncScope::CTA)
            .o("gpu", op.getScope() == triton::ppugpu::MemSyncScope::GPU)
            .o("sys", op.getScope() == triton::ppugpu::MemSyncScope::SYSTEM)
            .o("acquire", op.getSem() == triton::ppugpu::MemSemantic::ACQUIRE)
            .o("relaxed", op.getSem() == triton::ppugpu::MemSemantic::RELAXED)
            .b(width);
    ld(dstOpr, addrOpr).maybePredicate(op.getMask(), "b");

    // Create inline ASM signature
    Type retTy = IntegerType::get(getContext(), width);
    Value ret = tixBuilder.launch(rewriter, loc, retTy);
    ret = b.bitcast(ret, op.getType());

    rewriter.replaceOp(op, {ret});
    return success();
  }
};
} // anonymous namespace

class ConvertPPUGPUToLLVM
    : public impl::ConvertPPUGPUToLLVMBase<ConvertPPUGPUToLLVM> {
public:
  using impl::ConvertPPUGPUToLLVMBase<
      ConvertPPUGPUToLLVM>::ConvertPPUGPUToLLVMBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    RewritePatternSet patterns(context);

    patterns.add<ClusterCTAIdOpPattern, PPULoadMatrixOpPattern,
                 LoadAcquireOpPattern, WarpIdOpPattern>(context);

    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();

    makeAllWarpGroupsIsolatedFromAbove(mod);
  }
};

LogicalResult
ppugpu::rewriteAsTixAsm(Operation *op, PatternRewriter &rewriter,
                       std::string tixAsm,
                       const OperandsAndConstraints &operandsAndConstraints,
                       const Constraints &outputConstraints) {
  auto ctx = rewriter.getContext();
  auto loc = op->getLoc();
  tixAsm = patchTixAsm(op, std::move(tixAsm));
  auto hasSideEffects = !isMemoryEffectFree(op);

  TIXBuilder tixBuilder;
  auto tixOutputs = getTixOutputs(outputConstraints, tixBuilder);
  auto tixOperands =
      getTixOperands(operandsAndConstraints, tixBuilder, loc, rewriter);
  SmallVector<TIXBuilder::Operand *> outputsAndOperands = tixOutputs;
  outputsAndOperands.append(tixOperands.begin(), tixOperands.end());
  auto &tixInstr = *tixBuilder.create(tixAsm);
  tixInstr(outputsAndOperands, /*onlyAttachMLIRArgs=*/true);
  auto retTy =
      op->getNumResults() == 0 ? void_ty(ctx) : op->getResult(0).getType();
  auto res = tixBuilder.launch(rewriter, loc, retTy,
                               /*hasSideEffects*/ hasSideEffects);
  if (op->getNumResults() == 0) {
    rewriter.eraseOp(op);
  } else {
    rewriter.replaceOp(op, res);
  }

  return success();
}

} // namespace triton
} // namespace mlir
