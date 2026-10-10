#ifndef TRITON_TARGET_LLVMIR_INTRINSICATTRTABLE_H
#define TRITON_TARGET_LLVMIR_INTRINSICATTRTABLE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace llvm {
class Function;
class FunctionType;
class Module;
} // namespace llvm

namespace mlir {
class Location;
class ModuleOp;
class OpBuilder;
class Operation;
class Type;
class UnitAttr;

namespace LLVM {
class LLVMFuncOp;
} // namespace LLVM

namespace intrinsic_attr_table {

/// One intrinsic's attribute set, decoded from a payload line:
///
///     <intrinsic-name> convergent nounwind willreturn mem:0-0-0
///
/// `memory` is (argMem, inaccessibleMem, other), each an `llvm::ModRefInfo`
/// value; `passthrough` carries the atoms the table spells that are not one of
/// the four attributes the models know natively.
struct IntrinsicAttrs {
  bool convergent = false;
  bool noUnwind = false;
  bool willReturn = false;
  bool hasMemory = false;
  unsigned memory[3] = {0, 0, 0};
  llvm::SmallVector<llvm::StringRef, 4> passthrough;
};

using Table = llvm::StringMap<IntrinsicAttrs>;

/// Parse `intrinsic_tables.stamp_payload()` output.  Returns false and sets
/// `reason` on a malformed line.  Every `StringRef` in the table points into
/// `payload`, so `payload` has to outlive the table.
///
/// Kept in the header (not hidden in the .cpp) as the decode entry point a
/// future C++ unit test drives; `lookup` likewise.
bool parseTable(llvm::StringRef payload, Table &table, std::string &reason);

/// Resolve a name the way the tables do: exact first, then by dropping trailing
/// overload-suffix components (`<name>.i32` -> `<name>`,
/// keeping at least two components).
const IntrinsicAttrs *lookup(const Table &table, llvm::StringRef name);

// -- the in-process table -----------------------------------------------------
//
// The payload is the one Python already builds
// (`intrinsic_tables.stamp_payload(stamp_tag())`); it is set once per process
// by `install_intrinsic_attr_table` (see `llvm19_toolchain.py`) and consulted
// by the creation helpers below.  One value per process, like the other two
// pipeline switches: the key and the facts have to come from the same read.

/// Store the payload for this process (idempotent).  An empty payload disables
/// the whole mechanism, which is what `TRITON_LLVM19_STAMP=0` amounts to.
void setPayload(llvm::StringRef payload);

/// The refusal `setPayload` recorded for a second, different table, and clears
/// it.  Empty unless the last `setPayload` call was refused.
std::string takePayloadError();

/// Attribute set of `name` in this process's table, or nullptr.
const IntrinsicAttrs *lookupInProcess(llvm::StringRef name);

// -- applying a table ---------------------------------------------------------

/// The same facts on an *LLVM IR* declaration.
///
/// A name the linked LLVM already knows is skipped: its `Function` constructor
/// has set the facts from the table of the consumer in this process, and this
/// table must not overwrite them (R13 P1②).  That single condition is the
/// per-leg rule -- on the fallback leg (XTDK's LLVM 22) nothing here fires, on
/// the default leg (public LLVM 22, which knows none of the private names) the
/// table is the only source of a fact.  Returns true when the function was
/// stamped.
bool applyTo(llvm::Function &func, const Table &table);

// -- applying this process's table -------------------------------------------
//
// These are what the emitters call, so that a declaration is born carrying the
// facts the table states instead of being patched up later (or, worse, carrying
// a hand-picked subset that the table -- and the `llc` that reads it -- may not
// agree with).  They consult this process's table (`lookupInProcess`); a
// missing table simply means "no attributes", the same as
// `TRITON_LLVM19_STAMP=0`.

/// The MLIR declaration, with this process's facts for its name on it.
/// Returns true when the declaration was stamped.
bool applyTo(LLVM::LLVMFuncOp func);

/// The LLVM IR declaration, with this process's facts for its name on it.
/// Returns true when the declaration was stamped.
bool applyTo(llvm::Function &func);

// -- the one gate that creates a declaration ----------------------------------
//
// Every emitter creates a private-intrinsic declaration through this gate, and
// nobody creates one outside it: the MLIR half below (`getOrCreateDeclaration`)
// for a conversion, the LLVM IR half (`getOrCreateLLVMFunction`) for the
// dialect translation, which runs where there is no MLIR module to write into.
// The point is the order: the table has to be read *at creation*, so the
// declaration never exists in a state the staged `llc` (which re-parses the
// same table) would disagree with.  The later sweep (`applyPayloadToModule`,
// landing B') is the fallback for what a creation site cannot see -- a
// declaration no source literal mentions, or one the linked device libraries
// bring in -- not the source of the fact.  `applyTo` is a no-op for a name the
// table does not list, so this is safe to use for any external declaration.

/// Get or create `name`, stamped with this process's table.
///
/// `anchor` selects where a *new* declaration goes: null means the top of
/// `module` (the plain "declare it at module scope" case); non-null means
/// immediately before the `LLVMFuncOp` that encloses `anchor`, and marks the
/// declaration with the `libname`/`libpath` attributes -- that is the
/// `appendOrGetExternFuncOp` placement, which forwards here.  Returns the
/// existing declaration untouched when the module already has one.
LLVM::LLVMFuncOp getOrCreateDeclaration(OpBuilder &builder, ModuleOp module,
                                        Location loc, llvm::StringRef name,
                                        Type funcType,
                                        Operation *anchor = nullptr,
                                        llvm::StringRef libname = "",
                                        llvm::StringRef libpath = "");

/// The LLVM IR half of the same gate: what a *translation* that issues a call
/// from a `.td` llvmBuilder gets instead of calling `llvm::Function::Create`.
///
/// That path runs inside `llvm.to_module`, after every MLIR pass -- there is no
/// MLIR module to write an `LLVM::LLVMFuncOp` into, so the gate cannot be the
/// function above.  The rule is still the gate's, and this is the one place it
/// is written for that side: reuse the declaration the module already has,
/// otherwise create it, and read this process's table *at creation* (`applyTo`,
/// which is also what stamps a declaration a linked device library brought in
/// -- that one was never born here, and this is the only call site that knows
/// its name).
llvm::Function *getOrCreateLLVMFunction(llvm::Module &module,
                                        llvm::StringRef name,
                                        llvm::FunctionType *funcType);

/// Landing B' -- the same table, stamped on the *LLVM IR* module right after
/// `llvm.to_module`, for the declarations the creation sites cannot see.
///
/// A creation helper only stamps what it creates.  The XPU dialect's ops create
/// their `llvm::Function` inside the MLIR->LLVM *translation*
/// (`createIntrinsicCallByName`), and device libraries bring in declarations of
/// their own -- anything no source literal mentions reaches the optimizer
/// carrying no fact at all -- not even the `memory(none)` the table states for
/// `llvm.xpu.core_id`.  This sweep closes that hole.  (Historical note: it also
/// used to cover what landing B -- a since-retired MLIR pass -- covered.)
///
/// Returns the number of declarations stamped, 0 when the payload is empty, and
/// reports a malformed payload — or one that differs from the table installed
/// in this process — through `error` instead of throwing.
unsigned applyPayloadToModule(llvm::Module &module, llvm::StringRef payload,
                              std::string &error);

} // namespace intrinsic_attr_table
} // namespace mlir

#endif // TRITON_TARGET_LLVMIR_INTRINSICATTRTABLE_H
