// Copyright 2018-2020 Philippe Tillet
// Copyright 2020-2022 OpenAI
// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TRITONINSTRUMENT_PASSES
#define TRITONINSTRUMENT_PASSES

include "mlir/Pass/PassBase.td"

def TritonInstrumentConcurrencySanitizer: Pass<"tritoninstrument-concurrency-sanitizer", "mlir::ModuleOp"> {
  let summary = "Add runtime verification of asynchronous operations";

  let description = "Instrument the program with runtime verification of asynchronous operations.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::instrument::TritonInstrumentDialect"];
}

def TritonInstrumentPrepareConSanCaptures: Pass<"tritoninstrument-prepare-consan-captures", "mlir::ModuleOp"> {
  let summary = "Reserve shared memory for ConSan WarpSpecialize captures";

  let description = [{
    Estimates how many explicit captures the ConcurrencySanitizer pass will add
    to WarpSpecializeOps and records the corresponding byte count on each
    WarpSpecializeOp. Backend shared-memory allocation passes consume this
    reservation before ConSan mutates the capture list.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::instrument::TritonInstrumentDialect"];

  let options = [
    Option<"target", "target", "std::string", /*default*/"std::string{}",
           "ConSan target hook key, e.g. amd or nvidia">
  ];
}

def TritonInstrumentFpSanitizer: Pass<"tritoninstrument-fp-sanitizer", "mlir::ModuleOp"> {
  let summary = "Replace floating-point ops with integer-payload equivalents";

  let description = "Rewrite selected floating-point operations to use integer payload semantics for fpsan.";

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::math::MathDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::instrument::TritonInstrumentDialect"];
}

def TritonInstrumentGlobalSanitizer: Pass<"tritoninstrument-global-sanitizer", "mlir::ModuleOp"> {
  let summary = "Add gsan instrumentation to memory access operations.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::instrument::TritonInstrumentDialect"];
}

#endif // TRITON_INSTRUMENT_PASSES
