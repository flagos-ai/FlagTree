#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

using ::mlir::LLVM::linearize;
using ::mlir::LLVM::METAX::createBuiltinFunc;
namespace {
LLVM::LLVMFuncOp getVprintfDeclaration(RewriterBase &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();

  StringRef funcName("llvm.mxc.vprintf");
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *context = rewriter.getContext();

  Type newType;
  Value newArg;
  SmallVector<Type> argsType{ptr_ty(context)};

  auto funcType = LLVM::LLVMFunctionType::get(i32_ty, argsType, true);

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto *ctx = rewriter.getContext();
  return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(context), funcName,
                                           funcType);
}

// extend integer to int32, extend float to float64
// this comes from vprintf alignment requirements.
std::pair<Type, Value> printfPromoteValue(RewriterBase &rewriter, Value value,
                                          bool isSigned) {
  auto *context = rewriter.getContext();
  auto type = value.getType();
  Value newOp = value;
  Type newType = type;
  auto loc = UnknownLoc::get(context);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
    newType = i32_ty;
    if (isSigned) {
      newOp = b.sext(newType, value);
    } else {
      newOp = b.zext(newType, value);
    }
  } else if (type.isBF16() || type.isF16() || type.isF32()) {
    newType = f64_ty;
    newOp = b.fpext(newType, value);
  }

  return {newType, newOp};
}
LLVM::LLVMFuncOp getAssertfailDeclaration(RewriterBase &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("llvm.mxc.assertfail");
  {
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);
  }
  auto *ctx = rewriter.getContext();
  SmallVector<Type> argsType{ptr_ty(ctx), ptr_ty(ctx), i64_ty, ptr_ty(ctx)};
  auto funcType = LLVM::LLVMFunctionType::get(void_ty(ctx), argsType);
  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx),
                                                  funcName, funcType);

  funcOp.setPassthroughAttr(
      ArrayAttr::get(ctx, StringAttr::get(ctx, "noreturn")));
  return funcOp;
}
} // namespace

namespace mlir::triton::METAX {

bool TargetInfo::supportMaximumMinimum() const {
  return computeCapability >= 80;
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  return rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  llvm_unreachable("unsupported code path");
}

Value TargetInfo::ballotIntrinsic(RewriterBase &rewriter, Location loc,
                                  Type type, Value cmp, Operation *op) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value threadMask = b.int_val(type.getIntOrFloatBitWidth(), -1);
  StringRef funcName("llvm.mxc.icmp.i64.i32");
  Value icmpNeVal = b.i32_val(33);
  Value zeroVal = b.i32_val(0);
  Value cmpValue = b.zext(i32_ty, cmp);
  ValueRange valueRange({cmpValue, zeroVal, icmpNeVal});
  auto ret = createBuiltinFunc(rewriter, loc, op, funcName, type, valueRange);
  return ret;
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         bool isWarpSync) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (!isWarpSync) {
    b.barrier();
  }
}

static std::string getConstraintForBitwidth(unsigned bitwidth) {
  switch (bitwidth) {
  case 8:
  case 16:
    return "h";
  case 32:
    return "r";
  case 64:
    return "l";
  default:
    llvm_unreachable("unsupported bitwidth");
  }
}

static bool isConstantTruePred(Value pred) {
  if (auto constOp = pred.getDefiningOp<LLVM::ConstantOp>()) {
    return cast<IntegerAttr>(constOp.getValue()).getInt() == -1;
  }
  return false;
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = rewriter.getContext();
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");

  if (!isa<VectorType>(val.getType())) {
    storeDShared(rewriter, loc, ptr, ctaId, packLLVector(loc, {val}, rewriter),
                 pred);
    return;
  }

  auto vecTy = cast<VectorType>(val.getType());
  Type elemTy = vecTy.getElementType();
  unsigned vec = vecTy.getNumElements();
  unsigned elemBitwidth = elemTy.getIntOrFloatBitWidth();
  assert(llvm::isPowerOf2_32(vec));

  if (elemBitwidth < 8) {
    assert(vec == 1 &&
           "don't know how to load/store vectors of sub-byte elems");
    SmallVector<Value> vals = unpackLLVector(loc, val, rewriter);
    for (Value &v : vals) {
      v = b.zext(int_ty(8), b.bitcast(v, int_ty(elemBitwidth)));
    }
    storeDShared(rewriter, loc, ptr, ctaId, packLLVector(loc, vals, rewriter),
                 pred);
    return;
  }

  if (!elemTy.isInteger()) {
    SmallVector<Value> vals = unpackLLVector(loc, val, rewriter);
    for (Value &v : vals) {
      v = b.bitcast(v, int_ty(elemBitwidth));
    }
    storeDShared(rewriter, loc, ptr, ctaId, packLLVector(loc, vals, rewriter),
                 pred);
    return;
  }

  // load/store ops only support v2 and v4.  If the vector width is larger than
  // 4, we have two strategies for dealing with it.
  //  1. If the element type is smaller than b32, store b32's instead.
  //  2. Otherwise, split the store into multiple stores.
  if (vec > 4 && elemBitwidth < 32) {
    assert(llvm::isPowerOf2_32(vec));
    int elemsPerPack = 32 / elemBitwidth;
    SmallVector<Value> oldVals = unpackLLVector(loc, val, rewriter);

    SmallVector<Value> newVals;
    for (int i = 0; i < vec / elemsPerPack; i++) {
      Value v = packLLVector(
          loc, ArrayRef(oldVals).slice(i * elemsPerPack, elemsPerPack),
          rewriter);
      newVals.push_back(b.bitcast(v, i32_ty));
    }
    storeDShared(rewriter, loc, ptr, ctaId,
                 packLLVector(loc, newVals, rewriter), pred);
    return;
  }

  if (vec * elemBitwidth > 128) {
    assert(llvm::isPowerOf2_32(vec));
    assert(elemBitwidth == 32 || elemBitwidth == 64);
    int maxVec = 128 / elemBitwidth;

    auto newVecTy = vec_ty(elemTy, maxVec);
    SmallVector<Value> vals = unpackLLVector(loc, val, rewriter);
    for (int i = 0; i < vec / maxVec; i++) {
      auto newPtr = b.gep(ptr.getType(), elemTy, ptr, b.i32_val(i * maxVec),
                          LLVM::GEPNoWrapFlags::inbounds);
      storeDShared(
          rewriter, loc, newPtr, ctaId,
          packLLVector(loc, ArrayRef(vals).slice(i * maxVec, maxVec), rewriter),
          pred);
    }
    return;
  }

  // At this point we're committed to doing the store!
  assert(elemBitwidth >= 8);
  assert(elemTy.isInteger());
  assert(1 <= vec && vec <= 4);
  assert(vec * elemBitwidth <= 128);

  // Get pointer to remote shared memory if needed.
  if (ctaId.has_value()) {
    llvm_unreachable("unsupported loadDShared, ctaId has value");
  }

  if (isConstantTruePred(pred)) {
    b.store(val, ptr, /*align=*/vec * elemBitwidth / 8);
  } else {
    rewriter.create<scf::IfOp>(
        loc, pred,
        [&](OpBuilder &builder, Location loc) {
          b.store(val, ptr, /*align=*/vec * elemBitwidth / 8);
          builder.create<mlir::scf::YieldOp>(loc);
        },
        nullptr);
  }
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type loadTy,
                              Value pred, Operation *localLoadOp) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = rewriter.getContext();
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");

  if (!isa<VectorType>(loadTy)) {
    SmallVector<Value> values = unpackLLVector(
        loc, loadDShared(rewriter, loc, ptr, ctaId, vec_ty(loadTy, 1), pred),
        rewriter);
    assert(values.size() == 1);
    return values[0];
  }

  auto vecTy = cast<VectorType>(loadTy);
  Type elemTy = vecTy.getElementType();
  unsigned vec = vecTy.getNumElements();
  unsigned elemBitwidth = elemTy.getIntOrFloatBitWidth();
  assert(llvm::isPowerOf2_32(vec));

  if (elemBitwidth < 8) {
    assert(vec == 1 &&
           "don't know how to load/store vectors of sub-byte elems");
    SmallVector<Value> vals = unpackLLVector(
        loc, loadDShared(rewriter, loc, ptr, ctaId, int_ty(8), pred), rewriter);
    assert(vals.size() == 1);
    return b.bitcast(b.trunc(int_ty(elemBitwidth), vals[0]), elemTy);
  }

  // We only know how to load integers.
  if (!elemTy.isInteger()) {
    Type newLoadTy = vec_ty(int_ty(elemBitwidth), vec);
    SmallVector<Value> vals = unpackLLVector(
        loc, loadDShared(rewriter, loc, ptr, ctaId, newLoadTy, pred), rewriter);
    for (Value &v : vals) {
      v = b.bitcast(v, elemTy);
    }
    return packLLVector(loc, vals, rewriter);
  }

  // load/store ops only support v2 and v4.  If the vector width is larger than
  // 4, we have two strategies for dealing with it.
  //  1. If the element type is smaller than b32, load b32's instead.
  //  2. Otherwise, split the load into multiple loads.
  if (vec > 4 && elemBitwidth < 32) {
    int newVec = vec / (32 / elemBitwidth);
    auto newVecTy = vec_ty(i32_ty, newVec);
    auto res = loadDShared(rewriter, loc, ptr, ctaId, newVecTy, pred);

    // Unpack the b32's into the original vector type.
    SmallVector<Value> vals;
    for (Value v : unpackLLVector(loc, res, rewriter)) {
      Value vv = b.bitcast(v, vec_ty(elemTy, 32 / elemBitwidth));
      for (Value vvv : unpackLLVector(loc, vv, rewriter)) {
        vals.push_back(vvv);
      }
    }
    return packLLVector(loc, vals, rewriter);
  }

  if (vec * elemBitwidth > 128) {
    assert(elemBitwidth == 32 || elemBitwidth == 64);
    assert(llvm::isPowerOf2_32(vec));
    int maxVec = 128 / elemBitwidth;

    SmallVector<Value> vals;
    for (int i = 0; i < vec / maxVec; i++) {
      auto newPtr = b.gep(ptr.getType(), elemTy, ptr, b.i32_val(i * maxVec),
                          LLVM::GEPNoWrapFlags::inbounds);
      auto newVal = loadDShared(rewriter, loc, newPtr, ctaId,
                                vec_ty(elemTy, maxVec), pred);
      for (Value v : unpackLLVector(loc, newVal, rewriter)) {
        vals.push_back(v);
      }
    }
    return packLLVector(loc, vals, rewriter);
  }

  // At this point we're committed to actually do the load!
  assert(elemBitwidth >= 8);
  assert(elemTy.isInteger());
  assert(1 <= vec && vec <= 4);
  assert(vec * elemBitwidth <= 128);

  // Get pointer to remote shared memory if needed.
  if (ctaId.has_value()) {
    llvm_unreachable("unsupported loadDShared, ctaId has value");
  }

  Value load;
  if (isConstantTruePred(pred)) {
    Type resultTy = vec == 1 ? Type(int_ty(elemBitwidth))
                             : Type(vec_ty(int_ty(elemBitwidth), vec));
    load = b.load(resultTy, ptr, /*align=*/vec * elemBitwidth / 8);
    if (vec > 1) {
      Type structTy = struct_ty(SmallVector<Type>(vec, int_ty(elemBitwidth)));
      Value structValue = b.undef(structTy);
      for (int i = 0; i < vec; i++) {
        structValue = b.insert_val(structTy, structValue,
                                   b.extract_element(load, b.i32_val(i)), i);
      }
      load = structValue;
    }
  } else {
    Type resultTy = vec == 1 ? Type(int_ty(elemBitwidth))
                             : Type(vec_ty(int_ty(elemBitwidth), vec));
    Value zeroVal = b.bitcast(b.int_val(elemBitwidth * vec, 0), resultTy);
    auto loaded = rewriter.create<scf::IfOp>(
        loc, pred,
        [&](OpBuilder &builder, Location loc) {
          auto loadVal =
              b.load(resultTy, ptr, /*align=*/vec * elemBitwidth / 8);
          builder.create<mlir::scf::YieldOp>(loc, ValueRange({loadVal}));
        },
        [&](OpBuilder &builder, Location loc) {
          Value otherVal = zeroVal;
          builder.create<mlir::scf::YieldOp>(loc, ValueRange({otherVal}));
        });
    load = loaded->getResult(0);
    if (vec > 1) {
      Type structTy = struct_ty(SmallVector<Type>(vec, int_ty(elemBitwidth)));
      Value structValue = b.undef(structTy);
      for (int i = 0; i < vec; i++) {
        structValue = b.insert_val(structTy, structValue,
                                   b.extract_element(load, b.i32_val(i)), i);
      }
      load = structValue;
    }
  }
  SmallVector<Value> resultVals = unpackLLElements(loc, load, rewriter);
  return packLLVector(loc, resultVals, rewriter);
}

Value TargetInfo::loadDSharedIntrinsic(RewriterBase &rewriter, Location loc,
                                       Value ptr, std::optional<Value> ctaId,
                                       Type loadTy, Value pred,
                                       Operation *localLoadOp) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = rewriter.getContext();
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");

  auto vecTy = cast<VectorType>(loadTy);
  Type elemTy = vecTy.getElementType();                   // i8
  unsigned vec = vecTy.getNumElements();                  // 16
  unsigned elemBitwidth = elemTy.getIntOrFloatBitWidth(); // 8
  assert(llvm::isPowerOf2_32(vec));

  triton::gpu::LocalLoadOp localloadOp =
      dyn_cast_or_null<triton::gpu::LocalLoadOp>(localLoadOp);
  assert(localloadOp);

  Type i8xnTy = vec_ty(elemTy, vec); // A:K, B:N
  Type i8xnPtrTy = ptr_ty(rewriter.getContext(), 3);

  // tmp value, need to change
  std::string lds_intrinsic = "llvm.mxc.lds";
  mlir::LLVM::METAX::appendIntrinsicModifer(lds_intrinsic, vec, elemTy);
  auto elemXNPtr = b.bitcast(ptr, i8xnPtrTy);

  Value elemXN = b.undef(i8xnTy);
  ValueRange valueRange({elemXNPtr});
  elemXN = createBuiltinFunc(rewriter, loc, localloadOp, lds_intrinsic, i8xnTy,
                             valueRange);
  return elemXN;
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::METAX::shuffleXor(loc, rewriter, val, i);
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  return LLVM::METAX::shuffleUp(loc, rewriter, val, i);
}

Value TargetInfo::shuffleXorIntrinsic(RewriterBase &rewriter, Location loc,
                                      Value val, int i, Operation *op) const {
  return LLVM::METAX::shuffleXorIntrinsic(loc, rewriter, val, i, op);
}

Value TargetInfo::shuffleUpIntrinsic(RewriterBase &rewriter, Location loc,
                                     Value val, int i, Operation *op) const {
  return LLVM::METAX::shuffleUpIntrinsic(loc, rewriter, val, i, op);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::METAX::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdxIntrinsic(RewriterBase &rewriter, Location loc,
                                      Value val, int i, Operation *op) const {
  return LLVM::METAX::shuffleIdxIntrinsic(loc, rewriter, val, i, op);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  return LLVM::METAX::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdxIntrinsic(RewriterBase &rewriter, Location loc,
                                      Value val, Value i, Operation *op) const {
  return LLVM::METAX::shuffleIdxIntrinsic(loc, rewriter, val, i, op);
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  return LLVM::METAX::permute(loc, rewriter, a, b, selector);
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  return LLVM::METAX::llGetPid(loc, rewriter, moduleOp, axis);
}
bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName = resultElementTy.isInteger(32)
                             ? "mc_math_func_umulhi"
                             : "mc_math_func_umul64hi";
  return funcName;
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int /*formatStrByteCount*/, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  auto *ctx = rewriter.getContext();
  auto loc = UnknownLoc::get(ctx);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type ptr = ptr_ty(ctx);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto funcOp = getVprintfDeclaration(rewriter);
  SmallVector<Value> operands{formatStrStart};
  for (int i = 0; i < args.size(); i++) {
    Type newType;
    Value newArg;
    std::tie(newType, newArg) = printfPromoteValue(
        rewriter, args[i], isSigned.empty() ? true : isSigned[i]);
    operands.push_back(newArg);
  }
  b.call(funcOp, operands);
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue =
      LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()), rewriter,
                              "printfFormat_", msgNewline);
  printf(rewriter, msgValue, msgNewline.size_in_bytes(), args, isSigned);
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto funcOp = getAssertfailDeclaration(rewriter);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  llvm::SmallString<64> messageString(message), fileString(file),
      funcString(func);
  messageString.push_back('\0');
  fileString.push_back('\0');
  funcString.push_back('\0');
  Value messageStringVal =
      LLVM::addStringToModule(loc, rewriter, "assertMessage_", messageString);
  Value fileStringVal =
      LLVM::addStringToModule(loc, rewriter, "assertFile_", fileString);
  Value funcStringVal =
      LLVM::addStringToModule(loc, rewriter, "assertFunc_", funcString);
  Value lineNumber = b.i64_val(line);
  SmallVector<Value> operands = {messageStringVal, fileStringVal, lineNumber,
                                 funcStringVal};
  b.call(funcOp, operands);
}

int TargetInfo::getSharedAddressSpace() const { return 3; }

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  int spaceId = 0;
  if (isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace)) {
    spaceId = 3;
  } else {
    llvm::report_fatal_error("Only support SharedMemorySpace");
  }
  return spaceId;
}

bool TargetInfo::supportVectorizedAtomics() const {
  return computeCapability >= 90 && ptxVersion >= 81;
}

} // namespace mlir::triton::METAX
