/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_CONVERSION_TRITON_GPU_TO_LLVM_TIX_ASM_FORMAT_H_
#define TRITON_CONVERSION_TRITON_GPU_TO_LLVM_TIX_ASM_FORMAT_H_

#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir {
class ConversionPatternRewriter;
class Location;

namespace triton {
namespace ppu {
using llvm::StringRef;

struct TIXInstr;
struct TIXInstrCommon;
struct TIXInstrExecution;

// TIXBuilder helps to manage a TIX asm program consists of one or multiple
// instructions.
//
// A helper for building an ASM program, the objective of TIXBuilder is to give
// a thin encapsulation and make the ASM code for MLIR LLVM Dialect more clear.
// Currently, several factors are introduced to reduce the need for mixing
// string and C++ if-else code.
//
// Usage:
// To build: @$3 asm("@%3 add.s32 %0, %1, %2;" : "=r"(i) : "r"(j), "r"(k),
// "b"(p));
//
// TIXBuilder builder;
// auto& add = ::create(builder, );
// add.predicate(pVal).o("lo").o("u32"); // add any suffix
// // predicate here binds %0 to pVal, pVal is a mlir::Value
//
// auto* iOpr = builder.newOperand(iVal, "r"); // %1 bind to iVal
// auto* jOpr = builder.newOperand(jVal, "r"); // %2 bind to jVal
// auto* kOpr = builder.newOperand(kVal, "r"); // %3 bind to kVal
// add(iOpr, jOpr, kOpr).predicate(predVal);   // set operands and predicate
//
// To get the asm code:
// builder.dump()
//
// To get all the mlir::Value used in the TIX code,
//
// builder.getAllMlirArgs() // get {pVal, iVal, jVal, kVal}
//
// To get the string containing all the constraints with "," separated,
// builder.getConstraints() // get "=r,r,k"
//
// TIXBuilder can build a TIX asm with multiple instructions, sample code:
//
// TIXBuilder builder;
// auto& mov = builder.create("mov");
// auto& cp = builder.create("cp");
// mov(...);
// cp(...);
// This will get a TIX code with two instructions.
//
// Similar to a C function, a declared TIXInstr instance can be launched
// multiple times with different operands, e.g.
//
//   auto& mov = builder.create("mov");
//   mov(... some operands ...);
//   mov(... some different operands ...);
//
// Finally, we will get a TIX code with two mov instructions.
//
// There are several derived instruction type for typical instructions, for
// example, the TIXIOInstr for ld and st instructions.
struct TIXBuilder {
  struct Operand {
    std::string constraint;
    Value value;
    int idx{-1};
    llvm::SmallVector<Operand *> list;
    std::function<std::string(int idx)> repr;

    // for list
    Operand() = default;
    Operand(const Operation &) = delete;
    Operand(Value value, StringRef constraint)
        : constraint(constraint), value(value) {}

    bool isList() const { return !value && constraint.empty(); }

    Operand *listAppend(Operand *arg) {
      list.push_back(arg);
      return this;
    }

    Operand *listGet(size_t nth) const {
      assert(nth < list.size());
      return list[nth];
    }

    std::string dump() const;
  };

  template <typename INSTR = TIXInstr, typename... Args>
  INSTR *create(Args &&...args) {
    instrs.emplace_back(std::make_unique<INSTR>(this, args...));
    return static_cast<INSTR *>(instrs.back().get());
  }

  // Create a list of operands.
  Operand *newListOperand() { return newOperand(); }

  Operand *newListOperand(ArrayRef<std::pair<mlir::Value, std::string>> items) {
    auto *list = newOperand();
    for (auto &item : items) {
      list->listAppend(newOperand(item.first, item.second));
    }
    return list;
  }

  Operand *newListOperand(unsigned count, mlir::Value val,
                          const std::string &constraint) {
    auto *list = newOperand();
    for (unsigned i = 0; i < count; ++i) {
      list->listAppend(newOperand(val, constraint));
    }
    return list;
  }

  Operand *newListOperand(unsigned count, const std::string &constraint) {
    auto *list = newOperand();
    for (unsigned i = 0; i < count; ++i) {
      list->listAppend(newOperand(constraint));
    }
    return list;
  }

  // Create a new operand. It will not add to operand list.
  // @value: the MLIR value bind to this operand.
  // @constraint: ASM operand constraint, .e.g. "=r"
  // @formatter: extra format to represent this operand in ASM code, default is
  //             "%{0}".format(operand.idx).
  Operand *newOperand(mlir::Value value, StringRef constraint,
                      std::function<std::string(int idx)> formatter = nullptr);

  // Create a new operand which is written to, that is, the constraint starts
  // with "=", e.g. "=r".
  // If the operand will be used in predicated execution,
  // users may want to initialize it before use.
  // Otherwise if the register is only used in the true branch or the false
  // branch but not both, the register is undefined and ppu-llc can perform
  // aggressive optimizations that may lead to incorrect results.
  Operand *newOperand(StringRef constraint, bool init = false);

  // Create a new operand that is tied to a previous operand. In this case the
  // asm would be permitted to write to an input register. Instead of providing
  // constraint code for this operand, the constraint code of the tied operand
  // is used.
  Operand *newOperand(unsigned operandIndex);

  // Create a constant integer operand.
  Operand *newConstantOperand(int64_t v);
  // Create a constant operand with explicit code specified.
  Operand *newConstantOperand(const std::string &v);

  Operand *newAddrOperand(mlir::Value addr, StringRef constraint, int off = 0);

  llvm::SmallVector<Operand *, 4> getAllArgs() const;

  llvm::SmallVector<Value, 4> getAllMLIRArgs() const;

  std::string getConstraints() const;

  std::string dump() const;

  mlir::Value launch(OpBuilder &rewriter, Location loc, Type resTy,
                     bool hasSideEffect = true, bool isAlignStack = false,
                     ArrayRef<Attribute> attrs = {}) const;

private:
  Operand *newOperand() {
    argArchive.emplace_back(std::make_unique<Operand>());
    return argArchive.back().get();
  }

  void initOperand(Operand *opr);

  // Make the operands in argArchive follow the provided \param order.
  void reorderArgArchive(ArrayRef<Operand *> order) {
    assert(order.size() == argArchive.size());
    // The order in argArchive is unnecessary when onlyAttachMLIRArgs=false, but
    // it does necessary when onlyAttachMLIRArgs is true for the $0, $1... are
    // determined by TIX code snippet passed from external.
    sort(argArchive.begin(), argArchive.end(),
         [&](std::unique_ptr<Operand> &a, std::unique_ptr<Operand> &b) {
           auto ida = std::find(order.begin(), order.end(), a.get());
           auto idb = std::find(order.begin(), order.end(), b.get());
           assert(ida != order.end());
           assert(idb != order.end());
           return ida < idb;
         });
  }

  friend struct TIXInstr;
  friend struct TIXInstrCommon;

protected:
  llvm::SmallVector<std::unique_ptr<Operand>, 6> argArchive;
  llvm::SmallVector<std::unique_ptr<TIXInstrCommon>, 2> instrs;
  llvm::SmallVector<std::unique_ptr<TIXInstrExecution>, 4> executions;
  int oprCounter{};
};

// TIX instruction common interface.
// Put the generic logic for all the instructions here.
struct TIXInstrCommon {
  explicit TIXInstrCommon(TIXBuilder *builder) : builder(builder) {}

  using Operand = TIXBuilder::Operand;

  // clang-format off
  TIXInstrExecution& operator()() { return call({}); }
  TIXInstrExecution& operator()(Operand* a) { return call({a}); }
  TIXInstrExecution& operator()(Operand* a, Operand* b) { return call({a, b}); }
  TIXInstrExecution& operator()(Operand* a, Operand* b, Operand* c) { return call({a, b, c}); }
  TIXInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d) { return call({a, b, c, d}); }
  TIXInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e) { return call({a, b, c, d, e}); }
  TIXInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e, Operand* f) { return call({a, b, c, d, e, f}); }
  TIXInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e, Operand* f, Operand* g) { return call({a, b, c, d, e, f, g}); }
  TIXInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e, Operand* f, Operand* g, Operand* h) { return call({a, b, c, d, e, f, g, h}); }
  TIXInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e, Operand* f, Operand* g, Operand* h, Operand* i) { return call({a, b, c, d, e, f, g, h, i}); }
  // clang-format on

  // Set operands of this instruction.
  TIXInstrExecution &operator()(llvm::ArrayRef<Operand *> oprs,
                                bool onlyAttachMLIRArgs = false);

protected:
  // "Call" the instruction with operands.
  // \param oprs The operands of this instruction.
  // \param onlyAttachMLIRArgs Indicate that it simply attach the MLIR Arguments
  // to the inline Asm without generating the operand ids(such as $0, $1) in TIX
  // code.
  TIXInstrExecution &call(llvm::ArrayRef<Operand *> oprs,
                          bool onlyAttachMLIRArgs = false);

  TIXBuilder *builder{};
  llvm::SmallVector<std::string, 4> instrParts;

  friend struct TIXInstrExecution;
};

template <class ConcreteT> struct TIXInstrBase : public TIXInstrCommon {
  using Operand = TIXBuilder::Operand;

  explicit TIXInstrBase(TIXBuilder *builder, const std::string &name)
      : TIXInstrCommon(builder) {
    o(name);
  }

  // Append a suffix to the instruction.
  // e.g. TIXInstr("add").o("s32") get a add.s32.
  // A predicate is used to tell whether to apply the suffix, so that no if-else
  // code needed. e.g. `TIXInstr("add").o("s32", isS32).o("u32", !isS32);` will
  // get a `add.s32` if isS32 is true.
  ConcreteT &o(const std::string &suffix, bool predicate = true) {
    if (predicate)
      instrParts.push_back(suffix);
    return *static_cast<ConcreteT *>(this);
  }
};

struct TIXInstr : public TIXInstrBase<TIXInstr> {
  using TIXInstrBase<TIXInstr>::TIXInstrBase;

  // Append a ".global" to the instruction.
  TIXInstr &global();

  // Append a ".shared" to the instruction.
  TIXInstr &shared();

  // Append a ".v[0-9]+" to the instruction
  TIXInstr &v(int vecWidth, bool predicate = true);

  // Append a".b[0-9]+" to the instruction
  TIXInstr &b(int width);
};

// Record the operands and context for "launching" a TIXInstr.
struct TIXInstrExecution {
  using Operand = TIXBuilder::Operand;

  llvm::SmallVector<Operand *> argsInOrder;

  TIXInstrExecution() = default;
  explicit TIXInstrExecution(TIXInstrCommon *instr,
                             llvm::ArrayRef<Operand *> oprs,
                             bool onlyAttachMLIRArgs)
      : argsInOrder(oprs.begin(), oprs.end()), instr(instr),
        onlyAttachMLIRArgs(onlyAttachMLIRArgs) {}

  // Prefix a predicate to the instruction.
  TIXInstrExecution &predicate(mlir::Value value, StringRef constraint = "b") {
    assert(value);
    pred = instr->builder->newOperand(value, constraint);
    return *this;
  }

  // Prefix a predicate to the instruction, if non-null
  TIXInstrExecution &maybePredicate(mlir::Value value,
                                    StringRef constraint = "b") {
    if (value)
      predicate(value, constraint);
    return *this;
  }

  // Prefix a !predicate to the instruction.
  TIXInstrExecution &predicateNot(mlir::Value value, StringRef constraint) {
    pred = instr->builder->newOperand(value, constraint);
    pred->repr = [](int idx) { return "@!$" + std::to_string(idx); };
    return *this;
  }

  std::string dump() const;

  SmallVector<Operand *> getArgList() const;

  TIXInstrCommon *instr{};
  Operand *pred{};
  bool onlyAttachMLIRArgs{};
};

/// ====== Some instruction wrappers ======
// We add the wrappers to make the usage more intuitive by avoiding mixing the
// TIX code with some trivial C++ code.

struct TIXCpAsyncLoadInstr : TIXInstrBase<TIXCpAsyncLoadInstr> {
  explicit TIXCpAsyncLoadInstr(TIXBuilder *builder,
                               triton::CacheModifier modifier)
      : TIXInstrBase(builder, "cp.async") {
    o(triton::stringifyCacheModifier(modifier).str());
    o("shared");
    o("global");
  }
};

} // namespace ppu
} // namespace triton
} // namespace mlir

#endif
