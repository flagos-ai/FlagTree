// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITONINSTRUMENT_OPS
#define TRITONINSTRUMENT_OPS

include "triton/Dialect/TritonInstrument/IR/TritonInstrumentDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton/Dialect/TritonInstrument/IR/TritonInstrumentAttrDefs.td"

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;

//
// Ops
//

class TTI_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonInstrument_Dialect, mnemonic, traits> {
}

def TTI_ExperimentalAssertInThreadOp : TTI_Op<"experimental_assert_in_thread", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "assert the condition within the current thread";
  let description = [{
    Assert that the condition is true given all the values are available in the current thread.
    If the condition is false, the message is printed, and the program is aborted.
    If check_any is true, any of the values in the condition must be true. Otherwise, all the
    values in the condition must be true.
  }];
  let arguments = (ins AnyTypeOf<[I1, I1Tensor]>:$condition, StrAttr:$message, BoolAttr:$check_any);
  let assemblyFormat = "$condition `,` $message attr-dict `:` type($condition)";
}


def TTI_ExperimentalBufferPointersOp : TTI_Op<"experimental_buffer_pointers", [Pure]> {
  let summary = "definte an array of pointers to shared memory buffers";
  let description = [{
    Create a tensor of pointers to shared memory buffers.
  }];
  let arguments = (ins DenseI32ArrayAttr:$offsets, TT_MemTypeAttr:$memType);
  let results = (outs TT_Tensor:$result);
  let assemblyFormat = [{
    $offsets `,` $memType attr-dict `:` type($result)
  }];
}

def TTI_ExperimentalMemDescToI64Op : TTI_Op<"experimental_memdesc_to_i64", [Pure]> {
  let summary = "Convert a memdesc into its base pointer as i64";
  let description = [{
    Extract the base pointer from the given memdesc and return it as a 64-bit
    integer. This can be used to compare the memdesc against tensors of barrier
    pointers maintained by the concurrency sanitizer.
  }];
  let arguments = (ins TTG_MemDescType:$memdesc);
  let results = (outs I64:$result);
  let builders = [
    OpBuilder<(ins "Value":$memdesc), [{
      build($_builder, $_state, $_builder.getI64Type(), memdesc);
    }]>
  ];
  let assemblyFormat = "$memdesc attr-dict `:` type($memdesc)";
}


// ===== Critical section lock ops =====


def TTI_ExperimentalLockAcquireOp : TTI_Op<"experimental_lock_acquire", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Acquire a lock.";
  let description = [{
    Enter a critical section by acquiring a lock with single thread.
  }];
  let arguments = (ins TT_PtrLike:$lock, Optional<I1>:$pred);
  let assemblyFormat = [{
    $lock (`,` $pred^)? attr-dict `:` type($lock)
  }];
}


def TTI_ExperimentalLockReleaseOp : TTI_Op<"experimental_lock_release", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Release a lock.";
  let description = [{
    Leave a critical section by releasing a lock with single thread.
  }];
  let arguments = (ins TT_PtrLike:$lock, Optional<I1>:$pred);
  let assemblyFormat = [{
    $lock (`,` $pred^)? attr-dict `:` type($lock)
  }];
}

#endif // TRITONINSTRUMENT_OPS
