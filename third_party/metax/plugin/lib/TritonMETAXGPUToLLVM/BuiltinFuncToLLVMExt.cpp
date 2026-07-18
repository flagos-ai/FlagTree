#include "TritonMETAXGPUToLLVM/Passes.h"

#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTBUILTINFUNCTOLLVM
#include "TritonMETAXGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

class CallOpConversion : public mlir::RewritePattern {
public:
  CallOpConversion(mlir::MLIRContext *context)
      : mlir::RewritePattern(LLVM::CallOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto callOp = cast<LLVM::CallOp>(op);
    if (isWrappedLLVMIntrinsic(callOp)) {
      return convertToLLVMIntrinsic(callOp, rewriter);
    } else {
      return failure();
    }
  }

private:
  bool isWrappedLLVMIntrinsic(LLVM::CallOp callOp) const {
    if (std::optional<StringRef> callee = callOp.getCallee()) {
      if (callee.value().starts_with("__triton_maca_")) {
        return true;
      }
    }
    return false;
  }

  std::optional<SmallVector<StringRef>>
  matchPrefixAndSplitRemainder(StringRef name, StringRef prefix) const {
    if (name.starts_with(prefix)) {
      StringRef remainder = name.substr(prefix.size());
      SmallVector<StringRef> parts;

      static const SmallVector<StringRef> protectedPatterns = {"acq_rel",
                                                               "seq_cst"};

      // Type suffix patterns to be excluded from parts
      static const SmallVector<StringRef> typeSuffixes = {
          "i32", "u32", "i64", "u64", "f32", "f16", "bf16"};

      while (!remainder.empty()) {
        // Check if remainder starts with a type suffix
        bool isTypeSuffix = false;
        for (StringRef typeSuffix : typeSuffixes) {
          if (remainder.starts_with(typeSuffix)) {
            // If the type suffix is at the end or followed by '_', skip it
            if (remainder.size() == typeSuffix.size() ||
                remainder[typeSuffix.size()] == '_') {
              // Skip the type suffix and the trailing '_'
              remainder = remainder.substr(typeSuffix.size());
              if (remainder.starts_with('_')) {
                remainder = remainder.substr(1);
              }
              isTypeSuffix = true;
              break;
            }
          }
        }
        if (isTypeSuffix)
          continue;

        bool foundProtected = false;

        for (StringRef pattern : protectedPatterns) {
          if (remainder.starts_with(pattern)) {
            if (remainder.size() == pattern.size() ||
                remainder[pattern.size()] == '_') {
              parts.push_back(remainder.substr(0, pattern.size()));
              remainder = remainder.substr(pattern.size());
              if (remainder.starts_with('_')) {
                remainder = remainder.substr(1);
              }
              foundProtected = true;
              break;
            }
          }
        }

        if (foundProtected)
          continue;

        size_t pos = remainder.find('_');
        if (pos == StringRef::npos) {
          parts.push_back(remainder);
          break;
        }
        parts.push_back(remainder.substr(0, pos));
        remainder = remainder.substr(pos + 1);
      }

      return parts;
    }
    return std::nullopt;
  }

  LogicalResult convertToLLVMIntrinsic(LLVM::CallOp callOp,
                                       mlir::PatternRewriter &rewriter) const {
    StringRef calleeName = callOp.getCallee().value();
    auto strToMemoryOrder = [](StringRef str) {
      if (str == "monotonic")
        return LLVM::AtomicOrdering::monotonic;
      else if (str == "acquire")
        return LLVM::AtomicOrdering::acquire;
      else if (str == "release")
        return LLVM::AtomicOrdering::release;
      else if (str == "acq_rel")
        return LLVM::AtomicOrdering::acq_rel;
      else
        llvm_unreachable("unknown memory order string");
    };

    auto strToScope = [](StringRef str) {
      if (str == "device")
        return "device";
      else if (str == "system")
        return "system";
      else if (str == "block")
        return "block";
      else if (str == "warp")
        return "warp";
      else if (str == "one-as")
        return "one-as";
      else if (str == "device-one-as")
        return "device-one-as";
      else if (str == "block-one-as")
        return "block-one-as";
      else if (str == "warp-one-as")
        return "warp-one-as";
      else if (str == "singlethread-one-as")
        return "singlethread-one-as";
      else
        llvm_unreachable("unknown scope string");
    };

    auto operands = callOp.getOperands();
    auto result = callOp.getResult();

    LLVM::LLVMFunctionType calleeType = callOp.getCalleeFunctionType();
    Type returnType = calleeType.getReturnType();

    auto loc = callOp.getLoc();
    auto buildAtomicLoad =
        [&rewriter, &loc,
         returnType](Value inputPtr, LLVM::AtomicOrdering ordering,
                     std::optional<StringRef> scopeStr = std::nullopt) {
          assert(llvm::isa<LLVM::LLVMPointerType>(inputPtr.getType()) &&
                 "expected pointer type for atomic load");
          int alignment = returnType.getIntOrFloatBitWidth() / 8;
          return rewriter.create<LLVM::LoadOp>(
              loc, returnType, inputPtr, /*alignment=*/alignment,
              /*isVolatile=*/false, /*isNonTemporal=*/false,
              /*isInvariant=*/false, /*isInvariantGroup=*/false, ordering,
              scopeStr.value_or(StringRef()));
        };

    auto buildAtomicStore =
        [&rewriter, &loc](Value value, Value inputPtr,
                          LLVM::AtomicOrdering ordering,
                          std::optional<StringRef> scopeStr = std::nullopt) {
          int32_t alignment = value.getType().getIntOrFloatBitWidth() / 8;
          return rewriter.create<LLVM::StoreOp>(
              loc, value, inputPtr, /*alignment=*/alignment,
              /*isVolatile=*/false, /*isNonTemporal=*/false,
              /*isInvariantGroup=*/false, ordering,
              scopeStr.value_or(StringRef()));
        };

    auto buildAtomicFetchAdd =
        [&rewriter, &loc](Value atomicAddr, Value value,
                          LLVM::AtomicOrdering ordering,
                          std::optional<StringRef> scopeStr = std::nullopt) {
          int32_t alignment = value.getType().getIntOrFloatBitWidth() / 8;
          return rewriter.create<LLVM::AtomicRMWOp>(
              loc, LLVM::AtomicBinOp::add, atomicAddr, value, ordering,
              scopeStr.value_or(StringRef()), /*alignment=*/alignment);
        };

    auto buildAtomicCompareExchangeStrong =
        [&rewriter, &loc](Value atomicAddr, Value cmpVal, Value value,
                          LLVM::AtomicOrdering successOrdering,
                          LLVM::AtomicOrdering failureOrdering,
                          std::optional<StringRef> scopeStr = std::nullopt) {
          int32_t alignment = cmpVal.getType().getIntOrFloatBitWidth() / 8;
          auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
              loc, atomicAddr, cmpVal, value, successOrdering, failureOrdering,
              scopeStr.value_or(StringRef()), /*alignment=*/alignment);
          auto atomPtrVal = rewriter.create<LLVM::ExtractValueOp>(
              loc, cmpxchg, SmallVector<int64_t>{0});
          return atomPtrVal;
        };

    auto buildUnpackBf16x2x4ToF32x8 = [&rewriter, &loc](ValueRange inVals,
                                                        Type retTy) -> Value {
      assert(inVals.size() == 4 && "expected 4 i32 packed bf16x2 inputs");
      auto b = TritonLLVMOpBuilder(loc, rewriter);
      Type bf16x2VecTy = vec_ty(bf16_ty, 2);
      SmallVector<Value> outVals;
      outVals.reserve(8);

      auto structTy = dyn_cast<LLVM::LLVMStructType>(retTy);
      assert(structTy &&
             "expected unpack_bf16x2_f32 call to return LLVM struct type");
      assert(structTy.getBody().size() == 8 &&
             "expected 8 return values for unpack_bf16x2_f32");

      for (Value inVal : inVals) {
        Value vecBf16x2 =
            rewriter.create<LLVM::BitcastOp>(loc, bf16x2VecTy, inVal);
        Value loBf16 = rewriter.create<LLVM::ExtractElementOp>(
            loc, bf16_ty, vecBf16x2, b.i32_val(0));
        Value hiBf16 = rewriter.create<LLVM::ExtractElementOp>(
            loc, bf16_ty, vecBf16x2, b.i32_val(1));
        outVals.push_back(
            mlir::LLVM::METAX::convertBf16ToFp32(loc, rewriter, loBf16));
        outVals.push_back(
            mlir::LLVM::METAX::convertBf16ToFp32(loc, rewriter, hiBf16));
      }

      Value packedRet = rewriter.create<LLVM::UndefOp>(loc, retTy);
      for (unsigned i = 0; i < outVals.size(); ++i) {
        packedRet = rewriter.create<LLVM::InsertValueOp>(loc, retTy, packedRet,
                                                         outVals[i], i);
      }
      return packedRet;
    };

    auto buildPackF32x8ToBf16x2x4 =
        [&rewriter, &loc, &callOp](ValueRange inVals, Type retTy) -> Value {
      assert(inVals.size() == 8 && "expected 8 f32 inputs");
      auto b = TritonLLVMOpBuilder(loc, rewriter);
      Type i16x2VecTy = vec_ty(i16_ty, 2);

      auto structTy = dyn_cast<LLVM::LLVMStructType>(retTy);
      assert(structTy &&
             "expected pack_f32_bf16x2 call to return LLVM struct type");
      assert(structTy.getBody().size() == 4 &&
             "expected 4 return values from pack_f32_bf16x2");

      Value packedRet = rewriter.create<LLVM::UndefOp>(loc, retTy);
      for (int i = 0; i < 4; ++i) {
        Value loBf16 = mlir::LLVM::METAX::convertFp32ToBf16(
            loc, callOp.getOperation(), rewriter, inVals[2 * i + 0]);
        Value hiBf16 = mlir::LLVM::METAX::convertFp32ToBf16(
            loc, callOp.getOperation(), rewriter, inVals[2 * i + 1]);

        Value pairVec = rewriter.create<LLVM::UndefOp>(loc, i16x2VecTy);
        pairVec = rewriter.create<LLVM::InsertElementOp>(
            loc, i16x2VecTy, pairVec, loBf16, b.i32_val(0));
        pairVec = rewriter.create<LLVM::InsertElementOp>(
            loc, i16x2VecTy, pairVec, hiBf16, b.i32_val(1));

        Value packedI32 =
            rewriter.create<LLVM::BitcastOp>(loc, i32_ty, pairVec);
        Type outElemTy = structTy.getBody()[i];
        Value outVal = packedI32;
        if (outElemTy != i32_ty) {
          if (outElemTy == f32_ty) {
            outVal = rewriter.create<LLVM::BitcastOp>(loc, f32_ty, packedI32);
          } else {
            llvm_unreachable("unsupported output element type for "
                             "__triton_maca_pack_f32_bf16x2");
          }
        }

        packedRet = rewriter.create<LLVM::InsertValueOp>(
            loc, retTy, packedRet, outVal, ArrayRef<int64_t>{(int64_t)i});
      }
      return packedRet;
    };

    Operation *replacementOp = nullptr;
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // load
    if (auto maybeParts =
            matchPrefixAndSplitRemainder(calleeName, "__triton_maca_load_")) {
      auto parts = maybeParts.value();
      assert(parts.size() == 2 &&
             "expected load function to have 2 parts after prefix");
      LLVM::AtomicOrdering memOrder = strToMemoryOrder(parts[0]);
      auto scopeStr = strToScope(parts[1]);
      assert(operands.size() == 1 && "expected load to have 1 operand");

      replacementOp = buildAtomicLoad(operands[0], memOrder, scopeStr);
    }

    // store
    else if (auto maybeParts = matchPrefixAndSplitRemainder(
                 calleeName, "__triton_maca_store_")) {
      auto parts = maybeParts.value();
      assert(parts.size() == 2 &&
             "expected store function to have 2 parts after prefix");
      LLVM::AtomicOrdering memOrder = strToMemoryOrder(parts[0]);
      auto scopeStr = strToScope(parts[1]);
      assert(operands.size() == 2 && "expected store to have 2 operands");
      buildAtomicStore(operands[1], operands[0], memOrder, scopeStr);
      rewriter.eraseOp(callOp);
      return mlir::success();
    }

    // atomic add
    else if (auto maybeParts = matchPrefixAndSplitRemainder(
                 calleeName, "__triton_maca_atomic_add_")) {
      auto parts = maybeParts.value();
      assert(parts.size() == 2 &&
             "expected atomic add function to have 2 parts after prefix");
      assert(operands.size() == 2 && "expected atomic add to have 2 operands");
      LLVM::AtomicOrdering memOrder = strToMemoryOrder(parts[0]);
      auto scopeStr = strToScope(parts[1]);
      replacementOp =
          buildAtomicFetchAdd(operands[0], operands[1], memOrder, scopeStr);
    }

    // atomic cas
    else if (auto maybeParts = matchPrefixAndSplitRemainder(
                 calleeName, "__triton_maca_atomic_cas_")) {
      auto parts = maybeParts.value();
      assert(parts.size() == 3 &&
             "expected atomic cas function to have 3 parts after prefix");
      assert(operands.size() == 3 && "expected atomic cas to have 3 operands");
      LLVM::AtomicOrdering successOrdering = strToMemoryOrder(parts[0]);
      LLVM::AtomicOrdering failureOrdering = strToMemoryOrder(parts[1]);
      auto scopeStr = strToScope(parts[2]);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2], successOrdering,
          failureOrdering, scopeStr);
    }

    // load_b128
    else if (calleeName == "__triton_maca_loadv4_b128") {
      auto inputPtr = operands[0];
      assert(llvm::isa<LLVM::LLVMPointerType>(inputPtr.getType()) &&
             "expected pointer type for maca_loadv4");
      auto structTy = dyn_cast<LLVM::LLVMStructType>(returnType);
      assert(structTy &&
             "expected triton_maca_loadv4 call to return LLVM struct type");
      assert(structTy.getBody().size() == 4 &&
             "expected 4 return values from triton_maca_loadv4");
      auto retElemTy =
          structTy.getBody()[0]; // should be all same in one struct
      Type retTy = vec_ty(retElemTy, 4);
      Value loadRes = rewriter.create<LLVM::LoadOp>(loc, retTy, inputPtr);
      Value packedRet = rewriter.create<LLVM::UndefOp>(loc, structTy);
      for (int i = 0; i < 4; ++i) {
        auto val = b.extract_element(retElemTy, loadRes, b.i32_val(i));
        packedRet = b.insert_val(retElemTy, packedRet, val, i);
      }
      replacementOp = packedRet.getDefiningOp();
    }

    else if (calleeName == "__triton_maca_storev4_b128") {
      assert(operands.size() == 5 &&
             "expected triton_maca_storev4_b128 to have 5 operands");
      auto outPtr = operands[0];
      assert(llvm::isa<LLVM::LLVMPointerType>(outPtr.getType()) &&
             "expected pointer type for maca_storev4");
      auto elemTy = operands[1].getType();
      assert(elemTy.getIntOrFloatBitWidth() == 32 &&
             "triton_maca_storev4_b128 store value width should be 32bits");

      Type vecTy = vec_ty(elemTy, 4);
      Type vecPtrTy = ptr_ty(rewriter.getContext(), 1);
      Value packedVals = rewriter.create<LLVM::UndefOp>(loc, vecTy);
      for (int i = 0; i < 4; ++i) {
        packedVals =
            b.insert_element(vecTy, packedVals, operands[i + 1], b.i32_val(i));
      }
      replacementOp = rewriter.create<LLVM::StoreOp>(
          loc, packedVals, b.bitcast(outPtr, vecPtrTy));
    }

    // shuffle_idx
    else if (calleeName == "__triton_maca_shfl_idx") {
      assert(operands.size() == 2 && "expected shfl to have 2 operands");
      auto val = operands[0];
      auto laneid = operands[1];
      auto ret = mlir::LLVM::METAX::shuffleIdxIntrinsic(
          loc, rewriter, val, laneid, callOp.getOperation());
      replacementOp = ret.getDefiningOp();
    }

    // unpack four bf16x2 to eight f32
    else if (calleeName == "__triton_maca_unpack_bf16x2_f32") {
      Value unpacked = buildUnpackBf16x2x4ToF32x8(operands, returnType);
      replacementOp = unpacked.getDefiningOp();
    }

    // pack eight f32 to four bf16x2
    else if (calleeName == "__triton_maca_pack_f32_bf16x2") {
      Value packed = buildPackF32x8ToBf16x2x4(operands, returnType);
      replacementOp = packed.getDefiningOp();
    }

    if (replacementOp) {
      rewriter.replaceOp(callOp, replacementOp);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct ConvertBuiltinFuncToLLVM
    : public triton::impl::ConvertBuiltinFuncToLLVMBase<
          ConvertBuiltinFuncToLLVM> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<CallOpConversion>(context);

    if (mlir::applyPatternsGreedily(mod, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace METAX {

std::unique_ptr<OperationPass<ModuleOp>> createConvertBuiltinFuncToLLVMPass() {
  return std::make_unique<ConvertBuiltinFuncToLLVM>();
}

} // namespace METAX
} // namespace triton
} // namespace mlir
