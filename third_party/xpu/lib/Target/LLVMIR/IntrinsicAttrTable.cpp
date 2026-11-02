#include "triton/Target/LLVMIR/IntrinsicAttrTable.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ModRef.h"

#include <mutex>

using namespace mlir;
using namespace mlir::intrinsic_attr_table;

//===----------------------------------------------------------------------===//
// Decoding the payload
//===----------------------------------------------------------------------===//

bool mlir::intrinsic_attr_table::parseTable(llvm::StringRef payload,
                                            Table &table, std::string &reason) {
  llvm::SmallVector<llvm::StringRef, 64> lines;
  payload.split(lines, '\n');
  for (llvm::StringRef line : lines) {
    line = line.rtrim();
    if (line.empty())
      continue;
    llvm::SmallVector<llvm::StringRef, 8> fields;
    line.split(fields, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    if (fields.size() < 2) {
      reason = ("an entry carries no attribute at all: " + line).str();
      return false;
    }
    IntrinsicAttrs attrs;
    for (llvm::StringRef token : llvm::drop_begin(fields)) {
      if (token == "convergent") {
        attrs.convergent = true;
      } else if (token == "nounwind") {
        attrs.noUnwind = true;
      } else if (token == "willreturn") {
        attrs.willReturn = true;
      } else if (token.consume_front("mem:")) {
        llvm::SmallVector<llvm::StringRef, 3> parts;
        token.split(parts, '-');
        if (parts.size() != 3) {
          reason =
              "expected mem:<argmem>-<inaccessiblemem>-<other>: " + line.str();
          return false;
        }
        for (auto [i, part] : llvm::enumerate(parts)) {
          unsigned value;
          if (part.getAsInteger(10, value) || value > 3) {
            reason = ("not a ModRefInfo: " + part).str();
            return false;
          }
          attrs.memory[i] = value;
        }
        attrs.hasMemory = true;
      } else {
        attrs.passthrough.push_back(token);
      }
    }
    table.try_emplace(fields.front(), attrs);
  }
  return true;
}

const IntrinsicAttrs *mlir::intrinsic_attr_table::lookup(const Table &table,
                                                         llvm::StringRef name) {
  auto exact = table.find(name);
  if (exact != table.end())
    return &exact->second;
  llvm::StringRef head = name;
  while (true) {
    size_t dot = head.rfind('.');
    if (dot == llvm::StringRef::npos)
      return nullptr;
    head = head.take_front(dot);
    if (head.count('.') < 1)
      return nullptr;
    auto found = table.find(head);
    if (found != table.end())
      return &found->second;
  }
}

//===----------------------------------------------------------------------===//
// The in-process table
//===----------------------------------------------------------------------===//

namespace {
/// The payload and its decoded table, for the creation helpers.  The payload
/// has to stay alive because every entry points into it.
struct InProcessTable {
  std::mutex mutex;
  std::string payload;
  Table table;
  bool parsed = false;
  bool unusable = false;
  /// Set when `setPayload` refused a second, different table; the pybind
  /// binding turns it into a Python `RuntimeError` (this TU has exceptions
  /// disabled, so the refusal travels as data).
  std::string error;
};
InProcessTable &inProcess() {
  static InProcessTable state;
  return state;
}
} // namespace

void mlir::intrinsic_attr_table::setPayload(llvm::StringRef payload) {
  InProcessTable &state = inProcess();
  std::lock_guard<std::mutex> lock(state.mutex);
  // Idempotent on the "same table installed twice" path, which is the normal
  // one: every compilation installs before its emitters run, so a process that
  // compiles N kernels calls this N times with N copies of one string.  The
  // second call must not touch `state`, because a concurrent compilation's
  // emitters may be reading entries of it without the lock (see
  // `lookupInProcess`), and `clear()` here is exactly what would dangle them.
  //
  // A *different* payload is a different table in one process.  The pointer
  // `lookupInProcess` hands out is read without the lock, so replacing the
  // table here could dangle a reader that has already passed its lookup.  This
  // translation unit is compiled with exceptions disabled, so the refusal is
  // returned as an error string that the Python binding turns into
  // `RuntimeError` -- the contract is enforced at the binding boundary, not
  // just documented.  One process = one table.
  if (state.parsed && !state.unusable) {
    if (payload == state.payload)
      return;
    state.error = ("intrinsic attribute table: a different table is already "
                   "installed in this process (" +
                   std::to_string(state.payload.size()) +
                   "-byte payload); one process = one table.  Install the "
                   "table once per process, before the first compilation");
    return;
  }
  state.error.clear();
  state.payload = payload.str();
  state.table.clear();
  state.parsed = false;
  state.unusable = false;
}

std::string mlir::intrinsic_attr_table::takePayloadError() {
  InProcessTable &state = inProcess();
  std::lock_guard<std::mutex> lock(state.mutex);
  std::string error = std::move(state.error);
  state.error.clear();
  return error;
}

const IntrinsicAttrs *
mlir::intrinsic_attr_table::lookupInProcess(llvm::StringRef name) {
  InProcessTable &state = inProcess();
  std::lock_guard<std::mutex> lock(state.mutex);
  // The returned pointer points into `state` and the mutex is released on
  // return, so the caller reads it without the lock.  Sound under the invariant
  // this table is built on: entries are never invalidated after the first
  // successful parse.  `setPayload` is what would invalidate them, and it
  // enforces the contract itself -- the same payload is a no-op, and a
  // different one throws (`std::runtime_error`) instead of clearing entries a
  // caller of this function may still be reading.  One process = one table; if
  // that ever has to change, redesign this as a generation-stamped snapshot.
  if (!state.parsed) {
    state.parsed = true;
    std::string reason;
    if (state.payload.empty() ||
        !parseTable(state.payload, state.table, reason)) {
      state.unusable = true;
      state.payload.clear();
      state.table.clear();
      return nullptr;
    }
  }
  if (state.unusable)
    return nullptr;
  return lookup(state.table, name);
}

//===----------------------------------------------------------------------===//
// Writing an entry
//===----------------------------------------------------------------------===//

namespace {
/// The memory attribute of a table entry.
///
/// The MLIR frontends in this tree model a different number of memory locations
/// -- XTDK's (the fallback leg's) three, the public LLVM 22 one six -- and an
/// entry decoded from the LLVM 19 table says nothing about the extra locations,
/// so they are NoModRef.
LLVM::MemoryEffectsAttr memoryAttr(MLIRContext *ctx,
                                   const IntrinsicAttrs &attrs) {
  auto modRef = [&](unsigned value) {
    return static_cast<LLVM::ModRefInfo>(value);
  };
  // XTDK's frontend models three memory locations, the public LLVM 22 one six;
  // the build states which one is linked (CMakeLists.txt:
  // add_compile_definitions(TRITON_HAVE_XTDKDL)).
#if defined(TRITON_HAVE_XTDKDL)
  return LLVM::MemoryEffectsAttr::get(ctx, modRef(attrs.memory[2]),
                                      modRef(attrs.memory[0]),
                                      modRef(attrs.memory[1]));
#else
  return LLVM::MemoryEffectsAttr::get(
      ctx, modRef(attrs.memory[2]), modRef(attrs.memory[0]),
      modRef(attrs.memory[1]), LLVM::ModRefInfo::NoModRef,
      LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::NoModRef);
#endif
}

/// The declaration's `passthrough` list with the table's atoms added.
///
/// `passthrough` is a single attribute holding a list, so "write what the table
/// declares, delete nothing" has to merge by name here: an entry the table does
/// not spell stays (it is the emitter's, not ours), and a token the list
/// already carries is not added twice.  Entries that are not plain strings are
/// the emitter's too and are kept verbatim.
ArrayAttr mergePassthrough(MLIRContext *ctx, LLVM::LLVMFuncOp func,
                           ArrayRef<StringRef> tokens) {
  SmallVector<Attribute, 4> merged;
  if (ArrayAttr existing = func.getPassthroughAttr())
    llvm::append_range(merged, existing);
  for (StringRef token : tokens) {
    bool present = llvm::any_of(merged, [&](Attribute attr) {
      auto name = dyn_cast<StringAttr>(attr);
      return name && name.getValue() == token;
    });
    if (!present)
      merged.push_back(StringAttr::get(ctx, token));
  }
  return ArrayAttr::get(ctx, merged);
}

/// The one place an entry is written onto an MLIR declaration.
///
/// Only what the entry declares is written, and nothing is cleared: an atom a
/// line omits is the table having nothing to say about an attribute the emitter
/// wrote for its own reason.  The memory effect *is* the table's statement
/// about those locations, so it replaces whatever the emitter put there --
/// inventing a merge of two ModRef triples would be a fact neither side stated.
void writeAttrsTo(LLVM::LLVMFuncOp func, const IntrinsicAttrs &attrs) {
  MLIRContext *ctx = func.getContext();
  if (attrs.convergent)
    func.setConvergentAttr(UnitAttr::get(ctx));
  if (attrs.noUnwind)
    func.setNoUnwindAttr(UnitAttr::get(ctx));
  if (attrs.willReturn)
    func.setWillReturnAttr(UnitAttr::get(ctx));
  if (attrs.hasMemory)
    func.setMemoryEffectsAttr(memoryAttr(ctx, attrs));
  if (!attrs.passthrough.empty())
    func.setPassthroughAttr(mergePassthrough(ctx, func, attrs.passthrough));
}

/// The one place an entry is written onto an LLVM IR declaration.
void writeAttrsTo(llvm::Function &func, const IntrinsicAttrs &attrs) {
  auto modRef = [](unsigned value) {
    return static_cast<llvm::ModRefInfo>(value);
  };
  if (attrs.convergent)
    func.addFnAttr(llvm::Attribute::Convergent);
  if (attrs.noUnwind)
    func.addFnAttr(llvm::Attribute::NoUnwind);
  if (attrs.willReturn)
    func.addFnAttr(llvm::Attribute::WillReturn);
  if (attrs.hasMemory) {
    // (argMem, inaccessibleMem, other); the payload already folded `errno` into
    // `other` (`stamp_payload`), so errno itself stays NoModRef, and the
    // target-specific locations keep the default NoModRef.  `setModRef` is
    // private, so the public path is a single-location value per location,
    // unioned with `|=` (exactly the per-field ModRef union).
    llvm::MemoryEffects effects = llvm::MemoryEffects::none();
    effects |= llvm::MemoryEffects(llvm::IRMemLocation::ArgMem,
                                   modRef(attrs.memory[0]));
    effects |= llvm::MemoryEffects(llvm::IRMemLocation::InaccessibleMem,
                                   modRef(attrs.memory[1]));
    effects |= llvm::MemoryEffects(llvm::IRMemLocation::Other,
                                   modRef(attrs.memory[2]));
    func.addFnAttr(
        llvm::Attribute::getWithMemoryEffects(func.getContext(), effects));
  }
  // The atoms the LLVM IR model does not carry natively travel as names; an
  // atom this LLVM has no kind for is skipped rather than invented.
  for (llvm::StringRef token : attrs.passthrough) {
    llvm::Attribute::AttrKind kind =
        llvm::Attribute::getAttrKindFromName(token);
    if (kind != llvm::Attribute::None)
      func.addFnAttr(kind);
  }
}
} // namespace

//===----------------------------------------------------------------------===//
// Applying
//===----------------------------------------------------------------------===//

bool mlir::intrinsic_attr_table::applyTo(llvm::Function &func,
                                         const Table &table) {
  // The per-leg rule: a name the linked LLVM already knows has had its facts
  // set by `Function`'s constructor from the table of the consumer in this
  // process, and this table must not overwrite them (R13 P1②).
  if (func.getIntrinsicID() != llvm::Intrinsic::not_intrinsic)
    return false;
  const IntrinsicAttrs *attrs = lookup(table, func.getName());
  if (!attrs)
    return false;
  writeAttrsTo(func, *attrs);
  return true;
}

bool mlir::intrinsic_attr_table::applyTo(LLVM::LLVMFuncOp func) {
  const IntrinsicAttrs *found = lookupInProcess(func.getName());
  if (!found)
    return false;
  writeAttrsTo(func, *found);
  return true;
}

bool mlir::intrinsic_attr_table::applyTo(llvm::Function &func) {
  if (func.getIntrinsicID() != llvm::Intrinsic::not_intrinsic)
    return false;
  const IntrinsicAttrs *attrs = lookupInProcess(func.getName());
  if (!attrs)
    return false;
  writeAttrsTo(func, *attrs);
  return true;
}

LLVM::LLVMFuncOp mlir::intrinsic_attr_table::getOrCreateDeclaration(
    OpBuilder &builder, ModuleOp module, Location loc, llvm::StringRef name,
    Type funcType, Operation *anchor, llvm::StringRef libname,
    llvm::StringRef libpath) {
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return existing;
  OpBuilder::InsertionGuard guard(builder);
  if (anchor) {
    // The `appendOrGetExternFuncOp` placement: immediately before the enclosing
    // function, so the declaration sits at module scope next to the kernel, and
    // carries that helper's libname/libpath convention.
    Operation *parent = anchor;
    if (!isa<LLVM::LLVMFuncOp>(anchor))
      parent = anchor->getParentOfType<LLVM::LLVMFuncOp>();
    builder.setInsertionPoint(parent);
  } else {
    builder.setInsertionPointToStart(module.getBody());
  }
  auto fn = builder.create<LLVM::LLVMFuncOp>(loc, name, funcType,
                                             LLVM::Linkage::External);
  if (anchor) {
    MLIRContext *ctx = module.getContext();
    fn.getOperation()->setAttr("libname", StringAttr::get(ctx, libname));
    fn.getOperation()->setAttr("libpath", StringAttr::get(ctx, libpath));
  }
  // Born with the facts: this is the one place an external declaration comes
  // into existence, and the table is read here rather than patched up by a
  // later sweep (see the header).
  applyTo(fn);
  return fn;
}

llvm::Function *mlir::intrinsic_attr_table::getOrCreateLLVMFunction(
    llvm::Module &module, llvm::StringRef name, llvm::FunctionType *funcType) {
  llvm::Function *fn = module.getFunction(name);
  if (!fn)
    fn = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, name,
                                module);
  // Born with the facts.  For a declaration that was already in the module --
  // an earlier call site's, or one a linked device library brought in -- this
  // is the stamping site instead: it is the only point that knows which name it
  // is, and `applyTo` is a no-op for a name this table does not list (and for
  // one the linked LLVM already knows, per the per-leg rule).
  applyTo(*fn);
  return fn;
}

unsigned mlir::intrinsic_attr_table::applyPayloadToModule(
    llvm::Module &module, llvm::StringRef payload, std::string &error) {
  error.clear();
  // Every call site (`stamp_llvm_ir`) installs the process table *first*
  // (`install_intrinsic_attr_table`), so the table is usually already decoded
  // here -- re-parsing the ~1 MB payload per kernel would throw away that
  // work (~9 ms per compilation, measured).  The `payload` argument is the
  // "the caller means the same table" witness: it must be byte-identical to
  // whatever is installed, or the sweep is refused.  A payload that cannot be
  // the installed one is still decoded on its own first, so a malformed
  // payload is reported as malformed (the louder diagnosis) rather than as a
  // table mismatch.
  if (payload.empty())
    return 0;
  InProcessTable &state = inProcess();
  std::lock_guard<std::mutex> lock(state.mutex);
  // The witness check is against what was *installed* (`state.payload`), not
  // against "the table has been parsed yet": an install that nobody has
  // consulted still owns the process, and sweeping a different table over it
  // would be the exact two-tables-in-one-process hole this guard exists for.
  if (!state.payload.empty() && payload != state.payload) {
    // Decode their payload into a scratch table anyway so the error names the
    // real defect.  The scratch table's StringRefs point into `payload`, which
    // outlives this call -- safe, and never handed out.
    Table scratch;
    std::string reason;
    if (!parseTable(payload, scratch, reason)) {
      error = reason;
      return 0;
    }
    error = ("intrinsic attribute stamp (LLVM IR): the sweep's payload differs "
             "from the table installed in this process; one process = one "
             "table");
    return 0;
  }
  if (state.unusable) {
    error = "the installed payload is unusable (a previous parse failed)";
    return 0;
  }
  if (!state.parsed) {
    // Parse lazily *through the process state* so a backend that forgot the
    // install still gets the table (the previous behavior), and so the decoded
    // entries are cached for the emitters too.  Copy first, parse second:
    // every `StringRef` in the decoded table points into the buffer it was
    // parsed from (see the header), so the table has to point into
    // `state.payload`'s copy -- parsing straight from the pybind temporary
    // would leave `state.table` dangling the moment this call returns.  Under
    // the lock: this is the one path that may populate `state.table`, and
    // populating it outside the lock would race `lookupInProcess`'s lock-free
    // readers.
    state.parsed = true;
    state.payload = payload.str();
    state.table.clear();
    std::string reason;
    if (!parseTable(state.payload, state.table, reason)) {
      state.unusable = true;
      state.payload.clear();
      state.table.clear();
      error = reason;
      return 0;
    }
  }
  unsigned stamped = 0;
  for (llvm::Function &fn : module) {
    // Declarations only: the module's bodies are the kernel and its helpers.
    if (!fn.isDeclaration())
      continue;
    if (const IntrinsicAttrs *attrs = lookup(state.table, fn.getName())) {
      // The per-leg rule lives in `applyTo(llvm::Function&)`; apply the same
      // check here without taking the lock again per name.
      if (fn.getIntrinsicID() == llvm::Intrinsic::not_intrinsic) {
        writeAttrsTo(fn, *attrs);
        ++stamped;
      }
    }
  }
  return stamped;
}
