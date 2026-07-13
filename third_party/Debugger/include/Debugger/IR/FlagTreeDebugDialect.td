#ifndef FLAGTREE_DEBUG_DIALECT
#define FLAGTREE_DEBUG_DIALECT

include "mlir/IR/OpBase.td"

def FlagTreeDebug_Dialect : Dialect {
  // MLIR dialect namespaces must pass isValidNamespace (no '.'); op names are
  // e.g. flagtree_debug.collect_begin in textual IR.
  let name = "flagtree_debug";
  let cppNamespace = "::mlir::flagtree::debugger";
  let summary = "FlagTree debugger collect-region markers";
  let description = [{
    Marker operations for `tl.debug_collect_start` / `tl.debug_collect_end`.
    Scope pairing and op-id assignment are handled by later passes (module B).
  }];
  let dependentDialects = [];
  let useDefaultAttributePrinterParser = 1;
}

#endif // FLAGTREE_DEBUG_DIALECT
