// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::tle {

// census/v1: host-only 64B header, occurrence-major u64 timestamps, then
// [role=2, CTA, entry/exit=2] u32 SM IDs. The driver reserves per-CTA bytes;
// addresses below use the root pointer and the separately declared grid size.
class PublicationCensus {
public:
  static LogicalResult configure(LLVM::LLVMFuncOp function,
                                 unsigned threadsPerWarp) {
    auto module = function->getParentOfType<ModuleOp>();
    if (!module->hasAttr("tle.census.occurrences"))
      return success();
    auto occurrences = module->getAttrOfType<IntegerAttr>("tle.census.occurrences");
    auto ctas = module->getAttrOfType<IntegerAttr>("tle.census.ctas");
    if (!occurrences || !ctas || occurrences.getInt() <= 0 || ctas.getInt() <= 0 ||
        occurrences.getInt() > 65536 || ctas.getInt() > 65536)
      return function.emitError("census requires positive i32 occurrence/CTA counts up to 65536");
    if (module->hasAttr("tle.ws_trace.capacity") || threadsPerWarp != 32 ||
        triton::gpu::TritonGPUDialect::getNumCTAs(module) != 1)
      return function.emitError("census requires a separate profile ABI, 32-thread warps, and one CTA per cluster");
    uint64_t total = 64 + uint64_t(occurrences.getInt()) * ctas.getInt() * 32 + uint64_t(ctas.getInt()) * 16;
    if (total > (uint64_t(1) << 30))
      return function.emitError("census fixed buffer exceeds 1 GiB");
    if (function.getNumArguments() == 0 ||
        function.getArgumentTypes().back() != LLVM::LLVMPointerType::get(module.getContext(), 1))
      return function.emitError("census requires the normal address-space-1 profile pointer");
    if (auto size = module->getAttrOfType<IntegerAttr>("ttg.profile_scratch_memory_size"))
      if (size.getInt() != 0)
        return function.emitError("census cannot share scratch with another recorder");
    Builder builder(module.getContext());
    uint64_t perCTA = llvm::alignTo(llvm::divideCeil(total, uint64_t(ctas.getInt())), uint64_t(64));
    module->setAttr("ttg.profile_scratch_memory_size", builder.getI64IntegerAttr(perCTA));
    module->setAttr("ttg.profile_scratch_memory_alignment", builder.getI32IntegerAttr(64));
    module->setAttr("tle.census.configured", builder.getUnitAttr());
    module->setAttr("tle.census.total_bytes", builder.getI64IntegerAttr(total));
    function->setAttr("tle.census.profile_arg", builder.getI32IntegerAttr(function.getNumArguments() - 1));
    return success();
  }

  // logicalTid is produced before WS thread-context rewriting. Never derive
  // an owner by taking physical tid modulo a guessed producer width here.
  static void emit(TritonLLVMIRRewriter &builder, LLVM::LLVMFuncOp function,
                   Value scratch, Value occurrence, Value logicalTid,
                   uint32_t point, bool mapping) {
    auto module = function->getParentOfType<ModuleOp>();
    uint64_t occurrences = module->getAttrOfType<IntegerAttr>("tle.census.occurrences").getInt();
    uint64_t ctas = module->getAttrOfType<IntegerAttr>("tle.census.ctas").getInt();
    std::string code;
    llvm::raw_string_ostream os(code);
    os << "{\n.reg .pred owner, enabled, valid;\n"
          ".reg .b32 en, cx, cy, cz, nx, ny, nz, count, sm;\n"
          ".reg .b64 timestamp, cta, tmp, offset, address;\n"
          "setp.eq.u32 owner, $2, 0;\n@!owner bra CENSUS_DONE;\n"
          "ld.global.u32 en, [$0];\nsetp.ne.u32 enabled, en, 0;\n"
          "@!enabled bra CENSUS_DONE;\n"
          "mov.u64 timestamp, %globaltimer;\n"
          "mov.u32 cx, %ctaid.x;\nmov.u32 cy, %ctaid.y;\nmov.u32 cz, %ctaid.z;\n"
          "mov.u32 nx, %nctaid.x;\nmov.u32 ny, %nctaid.y;\nmov.u32 nz, %nctaid.z;\n"
          "mul.lo.u32 count, nx, ny;\nmul.lo.u32 count, count, nz;\n"
       << "setp.eq.u32 valid, count, " << ctas << ";\n@!valid bra CENSUS_DONE;\n"
       << "setp.lt.u32 valid, $1, " << (mapping ? 2 : occurrences) << ";\n@!valid bra CENSUS_DONE;\n"
          "mul.wide.u32 cta, cz, ny;\ncvt.u64.u32 tmp, cy;\nadd.u64 cta, cta, tmp;\n"
          "cvt.u64.u32 tmp, nx;\nmul.lo.u64 cta, cta, tmp;\n"
          "cvt.u64.u32 tmp, cx;\nadd.u64 cta, cta, tmp;\n"
       << "mul.wide.u32 offset, $1, " << ctas << ";\nadd.u64 offset, offset, cta;\n";
    if (mapping) {
      os << "mul.lo.u64 offset, offset, 8;\nadd.u64 offset, offset, "
         << 64 + occurrences * ctas * 32 + point * 4 << ";\n"
            "add.u64 address, $0, offset;\nmov.u32 sm, %smid;\nst.global.u32 [address], sm;\n";
    } else {
      os << "mul.lo.u64 offset, offset, 32;\nadd.u64 offset, offset, " << 64 + point * 8 << ";\n"
            "add.u64 address, $0, offset;\nst.global.u64 [address], timestamp;\n";
    }
    os << "CENSUS_DONE:\n}\n";
    LLVM::InlineAsmOp::create(builder, builder.getLoc(), Type(),
        ValueRange{scratch, occurrence, logicalTid}, code, "l,r,r,~{memory}",
        true, false, LLVM::TailCallKind::None,
        LLVM::AsmDialectAttr::get(builder.getContext(), LLVM::AsmDialect::AD_ATT), ArrayAttr());
  }
};
} // namespace mlir::triton::tle
