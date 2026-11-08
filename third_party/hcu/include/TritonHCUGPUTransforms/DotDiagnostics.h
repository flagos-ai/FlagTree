#ifndef TRITON_HCU_DOT_DIAGNOSTICS_H
#define TRITON_HCU_DOT_DIAGNOSTICS_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::HCU {

constexpr char SoftwareDotDiagnostics[] = "hcug.software_dot_diagnostics";
constexpr char OriginalDotInputs[] = "hcug.original_dot_inputs";
constexpr char FP4DotLoop[] = "hcug.fp4_dot_loop";
constexpr char FP4DecodeLoad[] = "hcug.fp4_decode_load";

inline std::string describeDotInputs(Type lhs, Type rhs) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  auto printType = [&](Type type) {
    if (auto integer = dyn_cast<IntegerType>(type))
      stream << "int" << integer.getWidth();
    else
      stream << type;
  };
  printType(lhs);
  stream << " x ";
  printType(rhs);
  return stream.str();
}

inline void recordSoftwareDot(Operation *op, StringRef detail) {
  auto module = op->getParentOfType<ModuleOp>();
  auto target = module->getAttrOfType<StringAttr>("ttg.target");
  std::string message = (target ? target.getValue().str() : "HCU") + ": " +
                        detail.str() + " (native=false)";
  auto previous = module->getAttrOfType<StringAttr>(SoftwareDotDiagnostics);
  std::string combined = previous ? previous.getValue().str() : "";
  if (llvm::StringRef(combined).contains(message))
    return;
  if (!combined.empty())
    combined += "\n";
  combined += message;
  module->setAttr(SoftwareDotDiagnostics,
                  StringAttr::get(op->getContext(), combined));
}

} // namespace mlir::triton::HCU

#endif
