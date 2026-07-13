#ifndef FLAGTREE_DEBUG_OPS
#define FLAGTREE_DEBUG_OPS

include "Debugger/IR/FlagTreeDebugDialect.td"
include "mlir/IR/OpBase.td"

class FlagTreeDebug_Op<string mnemonic, list<Trait> traits = []> :
    Op<FlagTreeDebug_Dialect, mnemonic, traits>;

def FlagTreeDebug_CollectBeginOp : FlagTreeDebug_Op<"collect_begin", []> {
  let summary = "Begin a debug collect region";
  let description = [{
    Records the user-requested record level and optional address collection
    level. `scope_id` is assigned by ResolveDebugScopePass (module B).
  }];
  let arguments = (ins I32Attr:$level,
                       OptionalAttr<I32Attr>:$addr_level,
                       OptionalAttr<I32Attr>:$scope_id);
  let results = (outs);
  let builders = [
    OpBuilder<(ins "IntegerAttr":$level,
                   CArg<"IntegerAttr", "nullptr">:$scope_id), [{
      build($_builder, $_state, level, /*addr_level=*/nullptr, scope_id);
    }]>
  ];
  let assemblyFormat = "`level` `=` $level attr-dict";
}

def FlagTreeDebug_CollectEndOp : FlagTreeDebug_Op<"collect_end", []> {
  let summary = "End a debug collect region";
  let arguments = (ins OptionalAttr<I32Attr>:$scope_id);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def FlagTreeDebug_RecordSummaryOp : FlagTreeDebug_Op<"record_summary", []> {
  let summary = "Record a debugger summary sample";
  let description = [{
    Concrete Module C instrumentation marker for a protocol SUMMARY record.
    Backend lowering consumes this op and writes one or more SummaryRecord slots
    to the F-managed ring buffer through the hidden debug control pointer.
  }];
  let arguments = (ins AnyType:$value,
                       I32Attr:$op_id,
                       I32Attr:$scope_id,
                       I32Attr:$record_level,
                       ArrayAttr:$collectors,
                       OptionalAttr<I32Attr>:$result_index);
  let results = (outs);
}

def FlagTreeDebug_RecordSummaryBundleOp
    : FlagTreeDebug_Op<"record_summary_bundle", []> {
  let summary = "Record debugger summary samples as two bundle records";
  let description = [{
    Concrete Module C instrumentation marker for protocol v2 summary bundle
    records. Backend lowering consumes this op and writes one count bundle and
    one value bundle slot to the F-managed ring buffer through the hidden debug
    control pointer.
  }];
  let arguments = (ins AnyType:$value,
                       I32Attr:$op_id,
                       I32Attr:$scope_id,
                       I32Attr:$record_level,
                       ArrayAttr:$collectors,
                       OptionalAttr<I32Attr>:$result_index);
  let results = (outs);
}

def FlagTreeDebug_RecordMemoryEventOp
    : FlagTreeDebug_Op<"record_memory_event", []> {
  let summary = "Record a debugger memory event";
  let description = [{
    Concrete Module C instrumentation marker for a protocol MEMORY_EVENT
    record.  The optional operand is the pointer/address value selected by the
    metadata pass for the memory access.
  }];
  let arguments = (ins Variadic<AnyType>:$observed,
                       I32Attr:$op_id,
                       I32Attr:$scope_id,
                       StrAttr:$event_kind);
  let results = (outs);
}

def FlagTreeDebug_CaptureMemoryAddressOp
    : FlagTreeDebug_Op<"capture_memory_address", []> {
  let summary = "Capture a debugger memory operand address";
  let description = [{
    High-level Module C marker for memory operand address capture.  This op is
    intentionally debugger-specific instead of extending Triton's native
    ptr_to_int semantics.  Backend lowering consumes it and writes a protocol
    MEMORY_EVENT record.  Its lowering is backend-sensitive and must be
    validated or adapted when adding a new device backend.
  }];
  let arguments = (ins Variadic<AnyType>:$observed,
                       I32Attr:$op_id,
                       I32Attr:$scope_id,
                       I32Attr:$operand_index,
                       StrAttr:$event_kind,
                       StrAttr:$lowering_policy);
  let results = (outs);
}

def FlagTreeDebug_RecordFullValueRefOp
    : FlagTreeDebug_Op<"record_full_value_ref", []> {
  let summary = "Record a debugger full-value payload reference";
  let description = [{
    Concrete Module C instrumentation marker for a protocol FULL_VALUE record.
    Backend lowering allocates or references a payload range and writes the
    FullValueRefRecord slot.
  }];
  let arguments = (ins AnyType:$value,
                       I32Attr:$op_id,
                       I32Attr:$scope_id,
                       I32Attr:$record_level,
                       OptionalAttr<I32Attr>:$result_index);
  let results = (outs);
}

#endif // FLAGTREE_DEBUG_OPS
