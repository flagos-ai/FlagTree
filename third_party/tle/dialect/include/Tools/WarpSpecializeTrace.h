// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Tools/PublicationCensus.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::tle {

// Opt-in diagnostic ABI, v1. Each physical warp owns one 64-byte header and
// capacity 64-byte records in profile_scratch. No circular overwrite, atomics,
// added synchronization, or model-buffer aliasing. The host initializes enable
// (header +36) before each invocation and reads only after kernel completion.
class WarpSpecializeTrace {
public:
  enum Event : uint32_t {
    KernelEntry = 1,
    DefaultDispatch,
    WorkerWaitBegin,
    WorkerWaitEnd,
    WorkerDispatch,
    UnassignedWaitBegin,
    UnassignedWaitMiddle,
    UnassignedWaitEnd,
    DefaultReleaseBegin,
    DefaultReleaseMiddle,
    DefaultBodyBegin,
    DefaultBodyEnd,
    DefaultJoinEnd,
    WorkerCaptureWaitBegin,
    WorkerBodyBegin,
    WorkerBodyEnd,
    WorkerJoinEnd,
    DefaultExitWaitBegin,
    KernelExit,
  };

  static LogicalResult configure(LLVM::LLVMFuncOp func, unsigned numWarps,
                                 unsigned defaultWarps, unsigned threadsPerWarp) {
    auto mod = func->getParentOfType<ModuleOp>();
    if (mod->hasAttr("tle.census.occurrences"))
      return PublicationCensus::configure(func, threadsPerWarp);
    if (!mod->hasAttr("tle.ws_trace.capacity")) {
      if (mod->hasAttr("tle.ws_trace.typed_ctas"))
        return func.emitError("tle.ws_trace.typed_ctas requires enabled trace capacity");
      return success();
    }
    auto attr = mod->getAttrOfType<IntegerAttr>("tle.ws_trace.capacity");
    if (!attr)
      return func.emitError("tle.ws_trace.capacity must be an integer attribute");
    if (attr.getInt() <= 0 || attr.getInt() > (1 << 20))
      return func.emitError("tle.ws_trace.capacity must be in [1, 1048576]");
    if (mod->hasAttr("tle.ws_trace.typed_ctas")) {
      auto ctas = mod->getAttrOfType<DenseI32ArrayAttr>("tle.ws_trace.typed_ctas");
      if (!ctas)
        return func.emitError("tle.ws_trace.typed_ctas must be a dense i32 array");
      int32_t previous = -1;
      bool valid = !ctas.empty();
      for (int32_t cta : ctas.asArrayRef()) {
        valid &= cta >= 0 && cta > previous;
        previous = cta;
      }
      if (!valid)
        return func.emitError("tle.ws_trace.typed_ctas must be nonempty, strictly increasing and nonnegative");
    }
    if (threadsPerWarp != 32 || numWarps > 32 ||
        triton::gpu::TritonGPUDialect::getNumCTAs(mod) != 1)
      return func.emitError("WS trace v1 requires 32-thread warps and one CTA per cluster");
    if (func.getNumArguments() == 0 ||
        !isa<LLVM::LLVMPointerType>(func.getArgumentTypes().back()))
      return func.emitError("WS trace requires the profile_scratch pointer ABI");
    if (auto existing = mod->getAttrOfType<IntegerAttr>("ttg.profile_scratch_memory_size"))
      if (existing.getInt() != 0)
        return func.emitError("WS trace cannot share profile_scratch with another recorder");
    Builder b(mod.getContext());
    // Generic call observations originate before pipe lowering. Establish the
    // typed header before the native runtime emits its first entry record.
    mod.walk([&](TraceEventOp) {
      mod->setAttr("tle.ws_trace.typed", b.getUnitAttr());
    });
    uint64_t bytes = uint64_t(numWarps) * (attr.getInt() + 1) * 64;
    mod->setAttr("ttg.profile_scratch_memory_size", b.getI64IntegerAttr(bytes));
    mod->setAttr("ttg.profile_scratch_memory_alignment", b.getI32IntegerAttr(64));
    mod->setAttr("tle.ws_trace.total_num_warps", b.getI32IntegerAttr(numWarps));
    mod->setAttr("tle.ws_trace.default_num_warps", b.getI32IntegerAttr(defaultWarps));
    func->setAttr("tle.ws_trace.profile_arg", b.getI32IntegerAttr(func.getNumArguments() - 1));
    return success();
  }

  // These hooks are reached by complete physical warps in the generated WS
  // runtime. Elect the lowest active lane and retain activemask to audit that
  // assumption. globaltimer is read before address/count/store bookkeeping.
  // Records are leader observations; intervals are not hardware-engine activity.
  static void emit(TritonLLVMIRRewriter &b, LLVM::LLVMFuncOp func, Event event,
                   Value state = {}, ValueRange payload = {}, Value scratchOverride = {}) {
    auto mod = func->getParentOfType<ModuleOp>();
    auto capacity = mod->getAttrOfType<IntegerAttr>("tle.ws_trace.capacity");
    if (!capacity)
      return;
    auto warps = mod->getAttrOfType<IntegerAttr>("tle.ws_trace.total_num_warps").getInt();
    auto defaultWarps = mod->getAttrOfType<IntegerAttr>("tle.ws_trace.default_num_warps").getInt();
    uint64_t streamBytes = (capacity.getInt() + 1) * 64;
    Value scratch = scratchOverride ? scratchOverride : LLVM::getProfileScratchPtr(b.getLoc(), b, func);
    if (!state)
      state = b.i32_val(-1);
    std::string code;
    llvm::raw_string_ostream os(code);
    os << "{\n.reg .pred leader, enabled, room;\n"
          ".reg .b32 mask, lane, bit, lower, warp, tid, sx, cx, cy, cz, nx, ny, count, next, dropped, en;\n"
          ".reg .b64 timestamp, cta, temp, stream, record, ordinal;\n"
          "mov.u64 timestamp, %globaltimer;\n"
          "activemask.b32 mask;\nmov.u32 lane, %laneid;\n"
          "shl.b32 bit, 1, lane;\nsub.u32 lower, bit, 1;\nand.b32 lower, lower, mask;\n"
          "setp.eq.u32 leader, lower, 0;\n@!leader bra TRACE_DONE;\n"
          "mov.u32 tid, %tid.x;\nshr.u32 warp, tid, 5;\n"
          "mov.u32 cx, %ctaid.x;\nmov.u32 cy, %ctaid.y;\nmov.u32 cz, %ctaid.z;\n"
          "mov.u32 nx, %nctaid.x;\nmov.u32 ny, %nctaid.y;\n"
          "mul.wide.u32 cta, cz, ny;\ncvt.u64.u32 temp, cy;\nadd.u64 cta, cta, temp;\n"
          "cvt.u64.u32 temp, nx;\nmul.lo.u64 cta, cta, temp;\n"
          "cvt.u64.u32 temp, cx;\nadd.u64 cta, cta, temp;\n";
    if (!payload.empty()) {
      if (auto ctas = mod->getAttrOfType<DenseI32ArrayAttr>("tle.ws_trace.typed_ctas")) {
        // configure validated this sorted nonnegative i32 set. Widen before
        // forming exclusive endpoints, including INT32_MAX + 1.
        SmallVector<std::pair<uint64_t, uint64_t>> runs;
        for (int32_t cta : ctas.asArrayRef()) {
          uint64_t value = uint64_t(cta);
          if (!runs.empty() && runs.back().second == value)
            runs.back().second = value + 1;
          else
            runs.emplace_back(value, value + 1);
        }
        auto emitRange = [&](StringRef predicate, uint64_t start, uint64_t end) {
          if (start == 0)
            os << "setp.lt.u64 " << predicate << ", cta, " << end << ";\n";
          else
            // temp's flattening use has ended; leave cta intact for addressing.
            os << "sub.u64 temp, cta, " << start << ";\n"
               << "setp.lt.u64 " << predicate << ", temp, " << end - start << ";\n";
        };
        os << ".reg .pred selected, match;\n";
        if (runs.size() == 1 && runs.front().second - runs.front().first > 1)
          emitRange("selected", runs.front().first, runs.front().second);
        else {
          os << "setp.eq.u32 selected, 0, 1;\n";
          for (auto [start, end] : runs) {
            if (end - start == 1)
              os << "setp.eq.u64 match, cta, " << start << ";\n";
            else
              emitRange("match", start, end);
            os << "or.pred selected, selected, match;\n";
          }
        }
        os << "@!selected bra TRACE_DONE;\n";
      }
    }
    os << "mul.lo.u64 stream, cta, " << warps << ";\n"
          "cvt.u64.u32 temp, warp;\nadd.u64 stream, stream, temp;\n"
       << "mul.lo.u64 stream, stream, " << streamBytes << ";\n"
          "add.u64 stream, stream, $0;\n"
          "ld.global.u32 en, [stream+36];\nsetp.ne.u32 enabled, en, 0;\n"
          "@!enabled bra TRACE_DONE;\n";
    if (event == KernelEntry) {
      os << "st.global.u32 [stream+0], 1465078833;\n" // 'WST1'
         << "st.global.u32 [stream+4], " << (mod->hasAttr("tle.ws_trace.typed") ? 2 : 1) << ";\n"
         << "st.global.u32 [stream+8], " << capacity.getInt() << ";\n"
         << "st.global.u32 [stream+12], " << warps << ";\n"
            "st.global.u32 [stream+16], 0;\nst.global.u32 [stream+20], 0;\n"
            "st.global.u32 [stream+24], 0;\nst.global.u32 [stream+28], 0;\n"
         << "st.global.u32 [stream+32], " << defaultWarps << ";\n";
    }
    os << "ld.global.u32 count, [stream+16];\nadd.u32 next, count, 1;\n"
          "st.global.u32 [stream+16], next;\n"
       << "setp.lt.u32 room, count, " << capacity.getInt() << ";\n"
          "@!room bra TRACE_OVERFLOW;\n"
          "mul.wide.u32 record, count, 64;\nadd.u64 record, record, stream;\n"
          "add.u64 record, record, 64;\ncvt.u64.u32 ordinal, count;\n"
          "mov.u32 sx, %smid;\n"
          "st.global.v2.u64 [record+0], {timestamp, ordinal};\n"
          ".reg .b32 event_value, record_cx, record_cy;\n"
       << "mov.u32 event_value, " << uint32_t(event) << ";\n"
          // CTA coordinates are immutable. Read record-only aliases at their
          // use so the stream-index values need not remain live for this store.
          "mov.u32 record_cx, %ctaid.x;\nmov.u32 record_cy, %ctaid.y;\n"
          "st.global.v4.u32 [record+16], {event_value, $1, record_cx, record_cy};\n"
          "st.global.v4.u32 [record+32], {cz, warp, sx, mask};\n";
    if (payload.empty())
      os << ".reg .b32 zero;\nmov.u32 zero, 0;\n"
            "st.global.v4.u32 [record+48], {zero, zero, zero, zero};\n";
    else
      os << "st.global.v4.u32 [record+48], {$2, $3, $4, $5};\n";
    os <<
          "st.global.u32 [stream+20], next;\nbra TRACE_FINISH;\n"
          "TRACE_OVERFLOW:\nld.global.u32 dropped, [stream+24];\n"
          "add.u32 dropped, dropped, 1;\nst.global.u32 [stream+24], dropped;\n"
          "TRACE_FINISH:\n";
    if (event == KernelExit)
      os << "st.global.u32 [stream+28], 1;\n";
    os << "TRACE_DONE:\n}\n";
    SmallVector<Value> inputs{scratch, state};
    llvm::append_range(inputs, payload);
    LLVM::InlineAsmOp::create(
        b, b.getLoc(), Type(), inputs, code,
        payload.empty() ? "l,r,~{memory}" : "l,r,r,r,r,r,~{memory}", true, false, LLVM::TailCallKind::None,
        LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr());
  }
};

} // namespace mlir::triton::tle
