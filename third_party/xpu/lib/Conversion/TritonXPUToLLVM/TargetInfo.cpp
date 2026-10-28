// clang-format off
#include "TargetInfo.h"  // TargetInfo

#include "triton/Analysis/NewAnalysis/Utility.h"
#include "triton/Dialect/LLVMXPU/IR/Dialect.h"
#include "triton/Target/LLVMIR/IntrinsicAttrTable.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
// clang-format on

#include <algorithm>
#include <cstring>

using namespace mlir;

namespace mlir {
namespace triton {
namespace xpu {

namespace {

// declare __assert_fail(i8*, i8*, i32, i8*) as an external function.
// Same shape as glibc's, which is what the XTDK device runtime provides.
LLVM::LLVMFuncOp getAssertFailDeclaration(RewriterBase &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("__assert_fail");
  if (Operation *funcOp = moduleOp.lookupSymbol(funcName))
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *ctx = rewriter.getContext();
  auto funcType = LLVM::LLVMFunctionType::get(
      void_ty(ctx), {ptr_ty(ctx), ptr_ty(ctx), i32_ty, ptr_ty(ctx)},
      /*isVarArg=*/true);
  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  // The one gate (see IntrinsicAttrTable.h): not a table name, created plain.
  auto fn = mlir::intrinsic_attr_table::getOrCreateDeclaration(
      rewriter, moduleOp, UnknownLoc::get(ctx), funcName, funcType);
  fn.setPassthroughAttr(ArrayAttr::get(ctx, StringAttr::get(ctx, "noreturn")));
  return fn;
}

// ---------------------------------------------------------------------------
// Device printf lowering (public LLVM 22 only).
//
// The XTDK frontend ships a private LLVM pass (XPULowerPrintfAssert) that
// expands `call @printf` into the `__xpuprintf_*` runtime sequence at the O3
// pipeline.  Public LLVM 22 does not ship that pass, so the bare `printf`
// declaration survives to link time and ld.lld fails with
// `undefined symbol: printf` (the runtime library only exports
// `__xpuprintf_*`).  When built against the public frontend we therefore
// lower printf in the MLIR layer, emitting exactly the call sequence the XTDK
// pass would have produced.
//
// The sequence was recovered byte-for-byte from the t36 (XTDK) .llir ground
// truth for a representative kernel and validated across the full
// test_subprocess::test_print matrix:
//
//   %buf = alloca { i64, i64, i64, i64 }            ; BufferInfo
//   call void @__xpuprintf_init__(i32 packetLen, ptr %buf)
//   call void @__xpuprintf_upload_chunk_header__(i64 0, i64 0, ptr %buf)
//   call void @__xpuprintf_upload_header(i32 packetLen, i32 argc,
//                                        i64 outstream, i64 strOffset, ptr
//                                        %buf)
//   call void @__xpuprintf_upload_string_dw__(i32 nBytes, i64 dw, i64 off, ptr
//   %buf)  × ceil(len/8) call void @__xpuprintf_upload_arg__(i64 arg, i64 off,
//   ptr %buf)                   × argc call void @__xpuprintf_update_tail__(i32
//   argc, i64 packetLen, ptr %buf)
//
// Layout: chunk_header (8B) + header (32B) + argc × 8B args, then the format
// string.  packetLen = alignTo(8 + 32 + argc*8 + strlen, 256).
//
// Unlike a SIMT-style lowering there is no SIMT lane loop here: XPU runs
// one core sequentially, and the runtime helpers own the ring-buffer/lock
// state (BufferInfo is the 4-field XPU struct, not the 11-field SIMT one).
// ---------------------------------------------------------------------------

constexpr unsigned xpuPrintfChunkAlign = 256;
constexpr size_t xpuPrintfChunkHeaderSize = sizeof(uint64_t);
constexpr size_t xpuPrintfHeaderSize =
    4 * sizeof(uint32_t) + 2 * sizeof(uint64_t);

size_t alignTo(size_t value, size_t align) {
  return (value + align - 1) / align * align;
}

// Format strings are materialized by addStringToModule() as a GEP into a
// constant LLVM global.  The XPU printf ABI uploads the raw format bytes into
// the packet rather than passing a pointer, so recover them from the global's
// initializer here.  Same helper as the alternate cluster backend.
StringRef getFormatStringFromGlobal(Value formatStrStart,
                                    int formatStrByteCount, ModuleOp moduleOp) {
  if (formatStrByteCount < 0)
    llvm::report_fatal_error("XPU printf format string has a negative size");

  auto formatGEP = formatStrStart.getDefiningOp<LLVM::GEPOp>();
  if (!formatGEP)
    llvm::report_fatal_error(
        "XPU printf requires a format string backed by an LLVM global");

  auto addressOf = formatGEP->getOperand(0).getDefiningOp<LLVM::AddressOfOp>();
  if (!addressOf)
    llvm::report_fatal_error(
        "XPU printf requires a format string backed by an LLVM global");

  auto globalName = addressOf->getAttrOfType<FlatSymbolRefAttr>("global_name");
  auto global =
      globalName ? moduleOp.lookupSymbol<LLVM::GlobalOp>(globalName.getValue())
                 : LLVM::GlobalOp();
  auto globalValue =
      global ? global->getAttrOfType<StringAttr>("value") : StringAttr();
  if (!globalValue ||
      static_cast<size_t>(formatStrByteCount) > globalValue.getValue().size())
    llvm::report_fatal_error(
        "XPU printf format string does not match its LLVM global initializer");

  return globalValue.getValue().take_front(formatStrByteCount);
}

// For the XTDK frontend: declare printf(i8*, ...) as an external variadic
// function, leaving the call for the LLVM-layer XPULowerPrintfAssert pass.
LLVM::LLVMFuncOp getPrintfDeclaration(RewriterBase &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("printf");
  if (Operation *funcOp = moduleOp.lookupSymbol(funcName))
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *ctx = rewriter.getContext();
  auto funcType = LLVM::LLVMFunctionType::get(i32_ty, {ptr_ty(ctx)},
                                              /*isVarArg=*/true);
  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  // The one gate (see IntrinsicAttrTable.h): `printf` is not a table name.
  return mlir::intrinsic_attr_table::getOrCreateDeclaration(
      rewriter, moduleOp, UnknownLoc::get(ctx), funcName, funcType);
}

// Apply C variadic default argument promotion so the values match the
// conversion specifiers emitted by XPUPrintOpConversion::getFormatSubstr
// (%d/%i read a 32-bit int, %f reads a double).  Only used on the XTDK path,
// where the call stays as `call @printf` for the LLVM pass to lower.
Value printfPromoteValue(RewriterBase &rewriter, Value value, bool isSigned) {
  auto *ctx = rewriter.getContext();
  auto loc = UnknownLoc::get(ctx);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type type = value.getType();

  if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32)
    return isSigned ? Value(b.sext(i32_ty, value))
                    : Value(b.zext(i32_ty, value));

  if (type.isBF16() || type.isF16() || type.isF32())
    return b.fpext(f64_ty, value);

  return value;
}

LLVM::LLVMFuncOp getOrInsertPrintfFunc(RewriterBase &rewriter,
                                       ModuleOp moduleOp, StringRef name,
                                       Type retType, ArrayRef<Type> argTypes) {
  auto loc = UnknownLoc::get(rewriter.getContext());
  // The one gate (see IntrinsicAttrTable.h): the `__xpuprintf_*` runtime
  // helpers are not table names, so they are created plain.
  return mlir::intrinsic_attr_table::getOrCreateDeclaration(
      rewriter, moduleOp, loc, name,
      LLVM::LLVMFunctionType::get(retType, argTypes));
}

// Extend values to the 64-bit representation the upload_arg runtime expects:
//   int  -> zext/sext to i64
//   fp   -> fpext to f64, then bitcast to i64 (the ABI uploads the raw bits)
//   ptr  -> ptrtoint to i64
Value printfPromoteToI64(RewriterBase &rewriter, Value value, bool isSigned) {
  auto *ctx = rewriter.getContext();
  auto loc = UnknownLoc::get(ctx);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type type = value.getType();

  if (isa<LLVM::LLVMPointerType>(type))
    return b.ptrtoint(i64_ty, value);

  if (type.isIntOrIndex()) {
    unsigned width = type.getIntOrFloatBitWidth();
    if (width == 64)
      return value;
    return isSigned ? Value(b.sext(i64_ty, value))
                    : Value(b.zext(i64_ty, value));
  }

  if (type.isF64())
    return b.bitcast(value, i64_ty);

  if (type.isBF16() || type.isF16() || type.isF32())
    return b.bitcast(b.fpext(f64_ty, value), i64_ty);

  llvm_unreachable("unsupported printf argument type");
  return Value();
}

// Emit the full __xpuprintf_* packet upload for one printf call site.
void emitXPUPrintf(RewriterBase &rewriter, StringRef fmtStr, ValueRange args,
                   ArrayRef<bool> isSigned) {
  auto *ctx = rewriter.getContext();
  auto loc = UnknownLoc::get(ctx);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto funcOp =
      rewriter.getBlock()->getParent()->getParentOfType<LLVM::LLVMFuncOp>();

  size_t argc = args.size();
  size_t fmtLen = fmtStr.size();

  // Packet layout: chunk header + header + argc×8 + format string, aligned.
  size_t strOffset = xpuPrintfChunkHeaderSize + xpuPrintfHeaderSize + argc * 8;
  size_t packetLen = alignTo(strOffset + fmtLen, xpuPrintfChunkAlign);

  // BufferInfo: { phg_buf, ring_buf, ring_size, head } all i64.
  SmallVector<Type, 4> bufferMembers(4, i64_ty);
  auto bufferInfoTy = LLVM::LLVMStructType::getLiteral(ctx, bufferMembers);

  Value bufferInfo;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Block &entryBlock = funcOp.getBody().front();
    rewriter.setInsertionPointToStart(&entryBlock);
    bufferInfo = rewriter.create<LLVM::AllocaOp>(loc, ptr_ty(ctx), bufferInfoTy,
                                                 b.i32_val(1), 0);
  }

  Type voidTy = LLVM::LLVMVoidType::get(ctx);
  auto initFn = getOrInsertPrintfFunc(rewriter, moduleOp, "__xpuprintf_init__",
                                      voidTy, {i32_ty, ptr_ty(ctx)});
  auto uploadChunkHeaderFn = getOrInsertPrintfFunc(
      rewriter, moduleOp, "__xpuprintf_upload_chunk_header__", voidTy,
      {i64_ty, i64_ty, ptr_ty(ctx)});
  // NOTE: no trailing underscore, matching the runtime symbol.
  auto uploadHeaderFn = getOrInsertPrintfFunc(
      rewriter, moduleOp, "__xpuprintf_upload_header", voidTy,
      {i32_ty, i32_ty, i64_ty, i64_ty, ptr_ty(ctx)});
  auto uploadStrDwFn = getOrInsertPrintfFunc(
      rewriter, moduleOp, "__xpuprintf_upload_string_dw__", voidTy,
      {i32_ty, i64_ty, i64_ty, ptr_ty(ctx)});
  auto uploadArgFn =
      getOrInsertPrintfFunc(rewriter, moduleOp, "__xpuprintf_upload_arg__",
                            voidTy, {i64_ty, i64_ty, ptr_ty(ctx)});
  auto updateTailFn =
      getOrInsertPrintfFunc(rewriter, moduleOp, "__xpuprintf_update_tail__",
                            i32_ty, {i32_ty, i64_ty, ptr_ty(ctx)});

  rewriter.create<LLVM::CallOp>(loc, initFn,
                                ValueRange{b.i32_val(packetLen), bufferInfo});
  rewriter.create<LLVM::CallOp>(
      loc, uploadChunkHeaderFn,
      ValueRange{b.i64_val(0), b.i64_val(0), bufferInfo});
  rewriter.create<LLVM::CallOp>(
      loc, uploadHeaderFn,
      ValueRange{b.i32_val(packetLen), b.i32_val(argc), b.i64_val(0),
                 b.i64_val(xpuPrintfChunkHeaderSize), bufferInfo});

  // Upload the format string as little-endian 8-byte words.  The XTDK
  // XPULowerPrintfAssert lowering appends one trailing NUL beyond the NUL-
  // terminated string it reads from the LLVM global (so a 36-byte string
  // uploads as 8+8+8+8+5 = 37 bytes; see the t36 .llir ground truth).
  // Match it exactly: the word is zero-initialized, so bytes past fmtLen are
  // NUL without reading out of bounds.
  size_t uploadLen = fmtLen + 1;
  for (size_t cursor = 0; cursor < uploadLen; cursor += 8) {
    size_t bytesToUpload = std::min<size_t>(8, uploadLen - cursor);
    uint64_t word = 0;
    size_t bytesToCopy = std::min(bytesToUpload, fmtLen - cursor);
    memcpy(&word, fmtStr.data() + cursor, bytesToCopy);
    rewriter.create<LLVM::CallOp>(
        loc, uploadStrDwFn,
        ValueRange{b.i32_val(bytesToUpload), b.i64_val(word),
                   b.i64_val(strOffset + cursor), bufferInfo});
  }

  // Upload arguments at offsets 8 + 32 + i*8 (after chunk header + header).
  for (size_t i = 0; i < argc; ++i) {
    bool argIsSigned = isSigned.empty() ? true : isSigned[i];
    Value arg = printfPromoteToI64(rewriter, args[i], argIsSigned);
    rewriter.create<LLVM::CallOp>(
        loc, uploadArgFn,
        ValueRange{
            arg,
            b.i64_val(xpuPrintfChunkHeaderSize + xpuPrintfHeaderSize + i * 8),
            bufferInfo});
  }

  rewriter.create<LLVM::CallOp>(
      loc, updateTailFn,
      ValueRange{b.i32_val(argc), b.i64_val(packetLen), bufferInfo});
}

} // namespace

bool TargetInfo::supportMaximumMinimum() const {
  llvm_unreachable("not impl");
  return false;
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  llvm_unreachable("not impl");
  return Value();
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         bool isWarpSync) const {
  llvm_unreachable("not impl");
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  llvm_unreachable("not impl");
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred, Operation *localLoadOp) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  return LLVM::XPU::llGetPid(loc, rewriter, moduleOp, static_cast<int>(axis));
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  llvm_unreachable("not impl");
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "_ZN3xpu6umulhiEjj" : "Unsupported";
  return funcName;
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int formatStrByteCount, ValueRange args,
                        ArrayRef<bool> isSigned) const {
#if defined(TRITON_XPU_PRINTF_MLIR_LOWER)
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef fmtStr =
      getFormatStringFromGlobal(formatStrStart, formatStrByteCount, moduleOp);
  emitXPUPrintf(rewriter, fmtStr, args, isSigned);
#else
  auto *ctx = rewriter.getContext();
  auto loc = UnknownLoc::get(ctx);
  LLVM::LLVMFuncOp printFn = getPrintfDeclaration(rewriter);
  SmallVector<Value, 16> operands = {formatStrStart};
  for (auto [i, arg] : llvm::enumerate(args)) {
    bool argIsSigned = isSigned.empty() ? true : isSigned[i];
    operands.push_back(printfPromoteValue(rewriter, arg, argIsSigned));
  }
  LLVM::CallOp::create(rewriter, loc, printFn, operands);
#endif
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
#if defined(TRITON_XPU_PRINTF_MLIR_LOWER)
  emitXPUPrintf(rewriter, msgNewline, args, isSigned);
#else
  Value msgValue =
      LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()), rewriter,
                              "printfFormat_", msgNewline);
  printf(rewriter, msgValue, msgNewline.size_in_bytes(), args, isSigned);
#endif
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  // Was `llvm_unreachable("not impl")`, which aborts the compiler for any
  // kernel carrying a tl.device_assert -- and inductor emits one per indirect
  // index unless config.assert_indirect_indexing is off. Turning that config
  // off is not a free workaround: it also changes inductor's prologue-fusion
  // decisions, which is how the mm template stopped receiving a materialized A
  // operand and started reading it through a permuted index expression that
  // the SDNN access analysis cannot turn into a DMA. Restore the lowering
  // instead; same shape as the one on xputc-8016-vectorize-unroll-fixes.
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  LLVM::LLVMFuncOp assertFn = getAssertFailDeclaration(rewriter);

  while (auto callLoc = dyn_cast<CallSiteLoc>(loc))
    loc = callLoc.getCallee();
  if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(loc)) {
    file = fileLineColLoc.getFilename();
    line = fileLineColLoc.getLine();
  }

  // __assert_fail reads C strings.
  llvm::SmallString<64> messageString(message), fileString(file),
      funcString(func);
  messageString.push_back('\0');
  fileString.push_back('\0');
  funcString.push_back('\0');
  Value messageVal =
      LLVM::addStringToModule(loc, rewriter, "assertMessage_", messageString);
  Value fileVal =
      LLVM::addStringToModule(loc, rewriter, "assertFile_", fileString);
  Value funcVal =
      LLVM::addStringToModule(loc, rewriter, "assertFunc_", funcString);
  SmallVector<Value> operands = {messageVal, fileVal, b.i32_val(line), funcVal};
  LLVM::CallOp::create(rewriter, loc, assertFn, operands);
}

int TargetInfo::getSharedAddressSpace() const { return 2; }

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  // XPU uses address space 2 for shared/local memory (LM).
  return 2;
}

bool TargetInfo::supportVectorizedAtomics() const { return false; }

uint32_t TargetInfo::getXPUArch() const { return this->xpu_arch; }
uint32_t TargetInfo::getXPUBufferSize() const { return this->buffer_size; }
bool TargetInfo::getXPUIsUseMaskZero() const { return this->isUseMaskZero; }

} // namespace xpu
} // namespace triton
} // namespace mlir
