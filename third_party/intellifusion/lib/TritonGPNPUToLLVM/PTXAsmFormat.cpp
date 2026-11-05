#include "PTXAsmFormat.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/AsmFormat.h"
#include "llvm/Support/raw_ostream.h"
// TODO(Superjomn): unify to llvm::raw_string_ostream
#include <sstream>

namespace mlir {
namespace triton {

GPNPUPTXInstr::Operand *
GPNPUPTXBuilder::newOperand(mlir::Value value, StringRef constraint,
                            std::function<std::string(int)> formatter) {
  argArchive.emplace_back(std::make_unique<Operand>(value, constraint));
  auto *opr = argArchive.back().get();
  opr->repr = formatter;
  opr->idx = oprCounter++;
  return opr;
}

void GPNPUPTXBuilder::initOperand(Operand *opr) {
  auto numBits = 0;
  // Derive numBits from the constraint.
  if (opr->constraint[1] == 'c' || opr->constraint[1] == 'h')
    numBits = 16;
  else if (opr->constraint[1] == 'r')
    numBits = 32;
  else if (opr->constraint[1] == 'l')
    numBits = 64;
  else
    llvm_unreachable(("Unknown constraint: " + opr->constraint).c_str());
  // If numBits is less than 16, we use 16 as default because PTX does not
  // support 8-bit mov.
  numBits = numBits < 16 ? 16 : numBits;
  auto *zero = newConstantOperand(0);
  auto &init = create<>("mov")->o("u" + std::to_string(numBits));
  init(opr, zero);
}

GPNPUPTXBuilder::Operand *GPNPUPTXBuilder::newOperand(StringRef constraint,
                                                      bool init) {
  // Constraint should be something like "=r"
  assert(constraint.size() == 2 && constraint[0] == '=');
  auto *opr = newOperand();
  opr->idx = oprCounter++;
  opr->constraint = constraint;
  if (init) {
    initOperand(opr);
  }
  return opr;
}

GPNPUPTXBuilder::Operand *GPNPUPTXBuilder::newOperand(unsigned operandIndex) {
  assert(operandIndex < oprCounter && "operand index out of range");
  auto *opr = newOperand();
  opr->idx = oprCounter++;
  opr->constraint = std::to_string(operandIndex);
  return opr;
}

GPNPUPTXBuilder::Operand *
GPNPUPTXBuilder::newConstantOperand(const std::string &v) {
  argArchive.emplace_back(std::make_unique<Operand>());
  argArchive.back()->repr = [v](int idx) { return v; };
  return argArchive.back().get();
}

GPNPUPTXBuilder::Operand *GPNPUPTXBuilder::newConstantOperand(int64_t v) {
  std::stringstream ss;
  ss << "0x" << std::hex << v;
  return newConstantOperand(ss.str());
}

std::string GPNPUPTXBuilder::getConstraints() const {
  auto args = getAllArgs();
  llvm::SmallVector<std::string, 4> argReprs;
  for (auto arg : args)
    argReprs.push_back(arg->constraint);
  return strJoin(argReprs, ",");
}

llvm::SmallVector<Value, 4> GPNPUPTXBuilder::getAllMLIRArgs() const {
  llvm::SmallVector<Value, 4> res;
  for (auto &arg : argArchive) {
    if (!arg->isList() && arg->value)
      res.push_back(arg->value);
  }
  return res;
}

SmallVector<GPNPUPTXBuilder::Operand *, 4> GPNPUPTXBuilder::getAllArgs() const {
  llvm::SmallVector<Operand *, 4> res;
  for (auto &x : argArchive)
    if (!x->isList())
      res.push_back(x.get());
  return res;
}

mlir::Value GPNPUPTXBuilder::launch(OpBuilder &rewriter, Location loc,
                                    Type resTy, bool hasSideEffect,
                                    bool isAlignStack,
                                    ArrayRef<Attribute> attrs) const {
  auto *ctx = rewriter.getContext();
  auto inlineAsm = LLVM::InlineAsmOp::create(
      rewriter, loc, resTy, getAllMLIRArgs(), // operands
      dump(),                                 // asm_string
      getConstraints(),                       // constraints
      hasSideEffect,                          // has_side_effects
      isAlignStack,                           // is_align_stack
      LLVM::TailCallKind::None,
      LLVM::AsmDialectAttr::get(ctx,
                                LLVM::AsmDialect::AD_ATT), // asm_dialect
      ArrayAttr::get(ctx, attrs)                           // operand_attrs
  );

  return inlineAsm.getRes();
}

std::string GPNPUPTXInstr::Operand::dump() const {
  if (repr)
    return repr(idx);
  if (!isList())
    return "$" + std::to_string(idx);

  llvm::SmallVector<std::string> oprs;
  for (auto *opr : list)
    oprs.push_back(opr->dump());
  return "{ " + strJoin(oprs, ", ") + " }";
}

GPNPUPTXInstr::Operand *GPNPUPTXBuilder::newAddrOperand(mlir::Value addr,
                                                        StringRef constraint,
                                                        int off) {
  auto *opr = newOperand(addr, constraint);
  opr->repr = [off](int idx) -> std::string {
    std::stringstream ss;
    ss << "[ $" << idx << " + " << off << " ]";
    return ss.str();
  };

  return opr;
}

std::string GPNPUPTXBuilder::dump() const {
  llvm::SmallVector<std::string> lines;
  for (auto &exec : executions) {
    lines.push_back(exec->dump());
  }

  return strJoin(lines, "\n\t");
}

GPNPUPTXInstrExecution &GPNPUPTXInstrCommon::call(ArrayRef<Operand *> oprs,
                                                  bool onlyAttachMLIRArgs) {
  if (onlyAttachMLIRArgs) {
    // Nearly impossible to make the $0,$1 in two PTX code snippets to point to
    // the same MLIR values in onlyAttachMLIRArgs mode.
    assert(builder->executions.empty() &&
           "builder can only hold a single execution when onlyAttachMIIRArgs "
           "is true.");
    builder->reorderArgArchive(oprs);
  }

  builder->executions.emplace_back(
      std::make_unique<GPNPUPTXInstrExecution>(this, oprs, onlyAttachMLIRArgs));

  return *builder->executions.back();
}

GPNPUPTXInstrExecution &
GPNPUPTXInstrCommon::operator()(ArrayRef<Operand *> oprs,
                                bool onlyAttachMLIRArgs) {
  return call(oprs, onlyAttachMLIRArgs);
}

std::string GPNPUPTXInstrExecution::dump() const {
  std::string osStr;
  llvm::raw_string_ostream os(osStr);

  if (pred) {
    if (!pred->repr)
      os << "@" << pred->dump() << " ";
    else
      os << pred->repr(pred->idx) << " ";
  }

  std::string instrRepr = strJoin(instr->instrParts, ".");
  if (onlyAttachMLIRArgs) {
    os << instrRepr;
    os.flush();
    return osStr;
  }

  llvm::SmallVector<std::string, 4> argReprs;
  for (auto *arg : argsInOrder) {
    argReprs.push_back(arg->dump());
  }

  std::string argsRepr = strJoin(argReprs, ", ");

  os << instrRepr << " " << argsRepr << ";";
  os.flush();
  return osStr;
}

SmallVector<GPNPUPTXInstrExecution::Operand *>
GPNPUPTXInstrExecution::getArgList() const {
  SmallVector<Operand *> args;
  for (auto *arg : argsInOrder) {
    if (arg->isList())
      args.insert(args.end(), arg->list.begin(), arg->list.end());
    else
      args.push_back(arg);
  }
  return args;
}

GPNPUPTXInstr &GPNPUPTXInstr::global() {
  o("global");
  return *this;
}

GPNPUPTXInstr &GPNPUPTXInstr::shared() {
  o("shared");
  return *this;
}

GPNPUPTXInstr &GPNPUPTXInstr::v(int vecWidth, bool predicate) {
  if (vecWidth > 1) {
    o("v" + std::to_string(vecWidth), predicate);
  }
  return *this;
}

GPNPUPTXInstr &GPNPUPTXInstr::b(int width) {
  o("b" + std::to_string(width));
  return *this;
}

} // namespace triton
} // namespace mlir
