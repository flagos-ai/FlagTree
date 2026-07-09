#include "PatternTritonXPUOpToLLVM.h"
#include "mlir/IR/Value.h"
#include "triton/Conversion/TritonXPUToLLVM/LegacyLLVMHelpers.h" // LLVM22 dragon-style macros for XPU only
#include "triton/Dialect/LLVMXPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstddef>

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::xpu;

struct XPUPrintOpConversion : public ConvertOpToLLVMPattern<XPUPrintOp> {
  explicit XPUPrintOpConversion(LLVMTypeConverter &typeConverter,
                                const TargetInfoBase &targetInfo,
                                PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<XPUPrintOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(XPUPrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    Value logicClusterId = rewriter.create<LLVM::XPU::LoadParamOp>(
        loc, type::i32Ty(rewriter.getContext()), i32_val(0));
    Value physicalClusterId = rewriter.create<LLVM::XPU::ClusterIdOp>(
        loc, type::i32Ty(rewriter.getContext()));
    Value coreId = rewriter.create<LLVM::XPU::CoreIdOp>(
        loc, type::i32Ty(rewriter.getContext()));

    assert(op.getNumOperands() >= 8);
    auto pidX = op.getOperand(0);
    auto pidY = op.getOperand(1);
    auto pidZ = op.getOperand(2);
    auto outerIdx = op.getOperand(3);
    auto innerIdx = op.getOperand(4);
    auto ucIdx = op.getOperand(5);
    auto innerBound = op->getOperand(6);
    auto ucBound = op->getOperand(7);

    std::string idStr;
    llvm::raw_string_ostream os(idStr);
    os << "pid (" << getFormatSubstr(pidX) << ", " << getFormatSubstr(pidY)
       << ", " << getFormatSubstr(pidZ) << ")";
    SmallVector<Value, 9> idOperands = {pidX, pidY, pidZ};

    std::string printVerboseStr =
        mlir::triton::tools::getStrEnvXPU("TRITON_PRINT_VERBOSE");
    if (printVerboseStr != "0" && printVerboseStr != "false") {
      os << ", hw_id (" << getFormatSubstr(physicalClusterId) << ", "
         << getFormatSubstr(logicClusterId) << ", " << getFormatSubstr(coreId)
         << ")"
         << ", loop_index (" << getFormatSubstr(outerIdx) << ", "
         << getFormatSubstr(innerIdx) << ", " << getFormatSubstr(ucIdx) << ")";
      idOperands.append({physicalClusterId, logicClusterId, coreId, outerIdx,
                         innerIdx, ucIdx});
    }

    if (op.getNumOperands() == 8) {
      // pid (<x>, <y>, <z>), hw_id (<physical cluster id>, <logic cluster
      // id>, <core id>), loop_index (<outer_index>, <inner_index>,
      // <unroll_control_index>)
      std::string formatStr;
      llvm::raw_string_ostream os(formatStr);
      os << idStr << op.getPrefix();
      llPrintf(formatStr, idOperands, rewriter);
      rewriter.eraseOp(op);
      return success();
    }

    for (size_t i = 8; i < op.getNumOperands(); i++) {
      auto elems = unpackLLElements(loc, adaptor.getOperands()[i], rewriter);

      SmallVector<int, 8> dimWidths;
      SmallVector<SmallVector<Value>> indices;
      if (auto rankedTy =
              dyn_cast<RankedTensorType>(op.getOperand(i).getType())) {
        indices = getIndicesAndDimWidths(
            loc, rewriter, targetInfo, rankedTy.getEncoding(), rankedTy,
            dimWidths, innerIdx, ucIdx, innerBound, ucBound);
      } else {
        assert(elems.size() == 1);
        indices.push_back({});
      }

      if (!elems.empty()) {
        printTensor(idStr, op.getPrefix(), i - 8, op.getNumOperands() - 8,
                    elems, idOperands, indices, dimWidths, op.getHex(),
                    rewriter);
      }
    }

    rewriter.eraseOp(op);

    return success();
  }

  void printTensor(StringRef idStr, StringRef prefixStr, size_t operand,
                   size_t numOperands, ArrayRef<Value> elems,
                   SmallVector<Value, 9> idOperands,
                   ArrayRef<SmallVector<Value>> indices,
                   ArrayRef<int> dimWidths, bool hex,
                   ConversionPatternRewriter &rewriter) const {
    assert(!elems.empty());
    assert(elems.size() == indices.size());
    assert(dimWidths.size() == indices.front().size());

    size_t rank = dimWidths.size();

    // pid (<x>, <y>, <z>), hw_id (<physical cluster id>, <logic cluster
    // id>, <core id>), loop_index (<outer_index>, <inner_index>,
    // <unroll_control_index>)
    Value formatStrValue;
    int formatStrByteCount = 0;
    for (int i = 0; i < elems.size(); i++) {
      std::string formatStr;
      llvm::raw_string_ostream os(formatStr);

      constexpr int kMaxPrintfOperands = 16;
      SmallVector<Value, kMaxPrintfOperands> printfOperands;

      os << idStr << " ";
      printfOperands.append(idOperands);

      int maxAllowedRank = kMaxPrintfOperands - printfOperands.size() - 2;

      // idx (<i1>, <i2>, ...)<prefix> (operand <n>) <elem>
      os << "idx (";
      const auto &index = indices[i];
      for (size_t dim = 0; dim < index.size(); dim++) {
        if (dim != 0) {
          os << ", ";
        }
        if (dim == maxAllowedRank) {
          os << "... (truncated)";
          break;
        }
        os << getFormatSubstr(index[dim], false, dimWidths[dim]);
        printfOperands.push_back(index[dim]);
      }
      os << ")" << prefixStr;

      if (numOperands > 1) {
        os << "(operand " << operand << ") ";
      }

      auto elem = elems[i];
      os << getFormatSubstr(elem, hex);
      printfOperands.push_back(elem);

      if (i == 0) {
        formatStrValue =
            llPrintf(formatStr, printfOperands, rewriter, &formatStrByteCount);
      } else {
        targetInfo.printf(rewriter, formatStrValue, formatStrByteCount,
                          printfOperands);
      }
    }
  }

  Value llPrintf(StringRef msg, ValueRange args,
                 ConversionPatternRewriter &rewriter,
                 int *formatStrByteCount = nullptr) const {
    assert(!msg.empty() && "printf with empty string not supported");
    llvm::SmallString<64> msgNewline(msg);
    msgNewline.push_back('\n');
    msgNewline.push_back('\0');
    Value msgValue =
        LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()),
                                rewriter, "printfFormat_", msgNewline);
    targetInfo.printf(rewriter, msgValue, msgNewline.size_in_bytes(), args);
    if (formatStrByteCount)
      *formatStrByteCount = msgNewline.size_in_bytes();
    return msgValue;
  }

  std::string getFormatSubstr(Value value, bool hex = false,
                              std::optional<int> width = std::nullopt) const {
    Type type = value.getType();
    if (isa<LLVM::LLVMPointerType>(type)) {
      return "%p";
    }
    if (hex) {
      std::string ret =
          "0x%0" + std::to_string(type.getIntOrFloatBitWidth() / 4);
      if (type.getIntOrFloatBitWidth() > 32) {
        ret += "ll";
      }
      ret += "x";
      return ret;
    }

    std::string prefix = "%";
    if (width.has_value()) {
      prefix += std::to_string(*width);
    } else if (hex) {
      prefix += "0";
      prefix += std::to_string(value.getType().getIntOrFloatBitWidth() / 4);
    }

    if (type.isBF16() || type.isF16() || type.isF32()) {
      return prefix + "f";
    } else if (type.isF64()) {
      return prefix + "F";
    } else if (type.isSignedInteger()) {
      if (type.getIntOrFloatBitWidth() == 64)
        return prefix + "lli";
      else
        return prefix + "i";
    } else if (type.isUnsignedInteger() || type.isSignlessInteger()) {
      if (type.getIntOrFloatBitWidth() == 64)
        return type.isUnsignedInteger() ? prefix + "llu" : prefix + "lld";
      else
        return type.isUnsignedInteger() ? prefix + "u" : prefix + "d";
    }
    assert(false && "not supported type");
    return "";
  }

  inline SmallVector<SmallVector<Value>> getIndicesAndDimWidths(
      Location loc, RewriterBase &rewriter, const TargetInfoBase &target,
      Attribute layout, RankedTensorType type, SmallVector<int, 8> &dimWidths,
      Value innerIndex, Value ucIndex, Value innerBound, Value ucBound) const {

    auto clusterLayout = mlir::cast<triton::xpu::ClusterLayoutAttr>(layout);
    auto shape = type.getShape();
    unsigned rank = shape.size();
    unsigned totalElemsPerThread =
        clusterLayout.getTotalElemsPerThread(shape, type);
    SmallVector<SmallVector<Value>> indices(totalElemsPerThread,
                                            SmallVector<Value>(rank));
    Value coreId = rewriter.create<LLVM::SExtOp>(
        loc, i64_ty, mlir::LLVM::XPU::getThreadId(rewriter, loc));

    if (rank == 1) {
      unsigned elemsPerThread = clusterLayout.getElemsPerThread(shape, type)[0];
      assert(elemsPerThread == totalElemsPerThread);
      Value threadsPerCore = mul(innerBound, ucBound);
      Value elemsPerCore = mul(threadsPerCore, i64_val(elemsPerThread));
      Value coreBase = mul(coreId, elemsPerCore);
      Value threadBase =
          add(mul(innerIndex, mul(ucBound, i64_val(elemsPerThread))),
              mul(ucIndex, i64_val(elemsPerThread)));
      Value base = add(coreBase, threadBase);

      for (unsigned n = 0; n < elemsPerThread; ++n) {
        indices[n][0] = add(base, i64_val(n));
      }

      if (auto innerBoundConst =
              innerBound.getDefiningOp<arith::ConstantIntOp>()) {
        if (auto ucBoundConst = ucBound.getDefiningOp<arith::ConstantIntOp>()) {
          int64_t ucRange = ucBoundConst.value();
          int64_t innerRange = innerBoundConst.value();
          int64_t maxRange = ucRange * innerRange * elemsPerThread * 63 +
                             (innerRange - 1) * ucRange * elemsPerThread +
                             (ucRange - 1) * elemsPerThread + elemsPerThread -
                             1;
          dimWidths.push_back(
              static_cast<int>(std::ceil(std::log10(maxRange))));
        } else {
          dimWidths.push_back(0);
        }
      }
    } else if (rank == 2) {
      unsigned coresPerGroup = product(clusterLayout.getCoresPerGroup());
      Value groupId = sdiv(coreId, i64_val(coresPerGroup));
      unsigned groupNum = product(clusterLayout.getGroupsPerCluster());
      Value baseX = add(mul(innerIndex, i64_val(groupNum)), groupId);

      Value idInsideGroup = srem(coreId, i64_val(coresPerGroup));
      unsigned sizePerCore = product(clusterLayout.getSizePerCore());
      // unsigned sizePerCore = totalElemsPerThread;
      assert(sizePerCore == totalElemsPerThread);
      Value elemsPerCore = mul(ucBound, i64_val(sizePerCore));
      Value baseInGroup = mul(idInsideGroup, elemsPerCore);
      unsigned elemsPerThread = clusterLayout.getElemsPerThread(shape, type)[1];
      Value baseY = add(baseInGroup, mul(ucIndex, i64_val(elemsPerThread)));

      for (unsigned n = 0; n < sizePerCore; ++n) {
        indices[n][0] = baseX;
        indices[n][1] = add(baseY, i64_val(n));
      }

      if (auto innerBoundConst =
              innerBound.getDefiningOp<arith::ConstantIntOp>()) {
        int64_t innerRange = innerBoundConst.value();
        int64_t maxRange = (innerRange - 1) * groupNum + 63 / coresPerGroup;
        dimWidths.push_back(static_cast<int>(std::ceil(std::log10(maxRange))));
      } else {
        dimWidths.push_back(0);
      }
      if (auto ucBoundConst = ucBound.getDefiningOp<arith::ConstantIntOp>()) {
        int64_t ucRange = ucBoundConst.value();
        int64_t maxRange = (ucRange - 1) * elemsPerThread +
                           (coresPerGroup - 1) * ucRange * sizePerCore +
                           sizePerCore - 1;
        dimWidths.push_back(static_cast<int>(std::ceil(std::log10(maxRange))));
      } else {
        dimWidths.push_back(0);
      }
    } else {
      llvm_unreachable("unsupported rank of XPUPrintOp");
    }

    return indices;
  }

protected:
  const TargetInfoBase &targetInfo;
};
} // namespace

void mlir::triton::xpu::populateXPUPrintOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<XPUPrintOpConversion>(typeConverter, targetInfo, benefit);
}
