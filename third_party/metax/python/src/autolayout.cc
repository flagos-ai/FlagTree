#include "triton/Conversion/CommonIRToTTGIR/DotLayoutPlanner.h"

#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/raw_ostream.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

py::list enumerateDotLayoutPlanDescriptions(mlir::ModuleOp &module,
                                            int capability, int numWarps) {
  using namespace mlir;
  using namespace pybind11::literals;
  std::string diagnostics;
  ScopedDiagnosticHandler handler(module.getContext(), [&](Diagnostic &diag) {
    llvm::raw_string_ostream os(diagnostics);
    os << diag << '\n';
    return success();
  });
  auto domains =
      triton::metax::enumerateDotLayoutPlans(module, capability, numWarps);
  if (failed(domains))
    throw py::value_error("failed to enumerate dot layout plans: " +
                          diagnostics);

  auto context =
      py::cast(module.getContext(), py::return_value_policy::reference);
  auto attribute = [&](Attribute layout) {
    auto result = py::cast(layout);
    py::detail::keep_alive_impl(result, context);
    return result;
  };
  py::list result;
  for (const auto &domain : *domains) {
    py::list plans;
    for (const auto &plan : domain.plans)
      plans.append(
          py::dict("plan_id"_a = triton::metax::getDotLayoutPlanId(plan),
                   "profile_id"_a = triton::metax::getDotLayoutProfileId(plan),
                   "mma"_a = attribute(plan.mma),
                   "operand_a"_a = attribute(plan.operandA),
                   "operand_b"_a = attribute(plan.operandB)));
    result.append(py::dict("dot_id"_a = domain.functionName + "/dot/" +
                                        std::to_string(domain.dotIndex),
                           "plans"_a = std::move(plans)));
  }
  return result;
}

} // namespace

void init_triton_metax_autolayout(py::module &&m) {
  m.def(
      "enumerate_dot_layout_plans", &enumerateDotLayoutPlanDescriptions,
      py::arg("module"), py::arg("capability"), py::arg("num_warps"),
      "Return per-dot candidates with native MLIR attributes for this "
      "context.\n"
      "Leave the module unchanged; apply candidates to it or its clones.\n"
      "Raise ValueError for unsupported dots; return [] if there are no dots.");
}
