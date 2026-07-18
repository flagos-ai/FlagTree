#include "RPU/IR/Dialect.h"
#include "RPU/IR/ExecutableKind.h"
#include "RPUPlan/IR/Dialect.h"
#include "RPUTransforms/Passes.h"
#include "lib/RPUTransforms/RPUDSLEmitter.h"
#include "lib/RPUTransforms/RPUExecutableLowering.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

void init_triton_rpu(py::module &&m) {
  m.doc() = "Python bindings to the RPU FlagTree backend";

  auto passes = m.def_submodule("passes");
  passes.def("add_recognize_plan", [](mlir::PassManager &pm) {
    pm.addPass(mlir::rpu::createRecognizeRPUPlanPass());
  });
  passes.def(
      "add_lower_supported_ttir_to_executable", [](mlir::PassManager &pm) {
        pm.addPass(mlir::rpu::createLowerSupportedTTIRToRPUExecutablePass());
        mlir::rpu::addRPUExecutableHighLevelLegalizationPipeline(pm);
      });
  passes.def("add_legalize_executable_high_level_ops",
             [](mlir::PassManager &pm) {
               mlir::rpu::addRPUExecutableHighLevelLegalizationPipeline(pm);
             });
  passes.def(
      "add_legalize_executable_compact_elementwise1d",
      [](mlir::PassManager &pm) {
        pm.addPass(
            mlir::rpu::createLegalizeRPUExecutableCompactElementwise1DPass());
      });
  passes.def("add_legalize_executable_dot", [](mlir::PassManager &pm) {
    pm.addPass(mlir::rpu::createLegalizeRPUExecutableDotPass());
  });
  passes.def("add_legalize_executable_softmax", [](mlir::PassManager &pm) {
    pm.addPass(mlir::rpu::createLegalizeRPUExecutableSoftmaxPass());
  });
  passes.def("add_legalize_executable_value_maps", [](mlir::PassManager &pm) {
    pm.addPass(mlir::rpu::createLegalizeRPUExecutableValueMapsPass());
  });
  passes.def("add_verify_executable_renderable", [](mlir::PassManager &pm) {
    pm.addPass(mlir::rpu::createVerifyRPUExecutableRenderablePass());
  });
  passes.def("add_emit_executable_to_rpurc", [](mlir::PassManager &pm) {
    pm.addPass(mlir::rpu::createVerifyRPUExecutableRenderablePass());
    pm.addPass(mlir::rpu::createEmitRPUExecutableToRPURCPass());
  });
  passes.def("add_convert_plan_to_executable", [](mlir::PassManager &pm) {
    pm.addPass(mlir::rpu::createConvertRPUPlanToRPUExecutablePass());
  });

  m.def("get_module_str_attr", [](mlir::ModuleOp &module,
                                  const std::string &name) {
    auto attr = module->getAttrOfType<mlir::StringAttr>(name);
    if (!attr)
      throw std::runtime_error("missing module string attribute: " + name);
    return attr.getValue().str();
  });

  m.def("_direct_rpurc_supported_patterns",
        []() { return mlir::rpu::directRPUDSLSupportedPatterns(); });

  m.def("_supported_executable_kernel_kinds", []() {
    std::vector<std::string> kinds;
    for (const mlir::rpu::exec::ExecutableKernelKindContract &contract :
         mlir::rpu::exec::supportedExecutableKernelKindContracts())
      kinds.push_back(contract.kind.str());
    return kinds;
  });

  m.def("_get_rpuplan_kernel_summary", [](mlir::ModuleOp &module) {
    mlir::FailureOr<mlir::rpu::RPUPlanKernelSummary> summary =
        mlir::rpu::getRPUPlanKernelSummaryFromModule(module);
    if (failed(summary))
      throw std::runtime_error(
          "RPU rpu_plan.kernel summary failed; see MLIR diagnostics");
    py::dict dict;
    dict["kernel_name"] = summary->kernelName;
    dict["pattern"] = summary->pattern;
    return dict;
  });

  m.def("_get_rpuexec_kernel_summary", [](mlir::ModuleOp &module) {
    mlir::FailureOr<mlir::rpu::RPUExecutableKernelSummary> summary =
        mlir::rpu::getRPUExecutableKernelSummaryFromModule(module);
    if (failed(summary))
      throw std::runtime_error(
          "RPU rpuexec kernel summary failed; see MLIR diagnostics");
    py::dict dict;
    dict["kernel_name"] = summary->kernelName;
    dict["pattern"] = summary->pattern;
    return dict;
  });

  m.def("_emit_rpurc_from_plan", [](mlir::ModuleOp &module) {
    mlir::FailureOr<mlir::rpu::RPUDSLEmissionResult> result =
        mlir::rpu::emitRPUDSLFromModule(module);
    if (failed(result))
      throw std::runtime_error(
          "RPU direct .rc emission failed; see MLIR diagnostics");
    py::dict dict;
    dict["kernel_name"] = result->kernelName;
    dict["pattern"] = result->pattern;
    dict["source"] = result->source;
    return dict;
  });

  m.def("_lower_rpuplan_to_executable", [](mlir::ModuleOp &module) {
    mlir::FailureOr<std::string> text =
        mlir::rpu::lowerRPUPlanToExecutableModule(module);
    if (failed(text))
      throw std::runtime_error(
          "RPU executable lowering failed; see MLIR diagnostics");
    return *text;
  });

  m.def(
      "_lower_rpuplan_to_executable_module",
      [](mlir::ModuleOp &module) {
        mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> executable =
            mlir::rpu::lowerRPUPlanToExecutableModuleOp(module);
        if (failed(executable))
          throw std::runtime_error(
              "RPU executable module lowering failed; see MLIR diagnostics");
        return (*executable)->clone();
      },
      py::return_value_policy::take_ownership);

  m.def("_rpuplan_supports_executable_lowering", [](mlir::ModuleOp &module) {
    return mlir::rpu::supportsRPUPlanExecutableLowering(module);
  });

  m.def("_rpuplan_executable_lowering_failure_reason",
        [](mlir::ModuleOp &module) {
          mlir::FailureOr<std::string> reason =
              mlir::rpu::describeRPUPlanExecutableLoweringFailure(module);
          if (failed(reason))
            throw std::runtime_error("RPU executable lowering failure reason "
                                     "unavailable; see MLIR diagnostics");
          return *reason;
        });

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    context.getOrLoadDialect<mlir::rpu::plan::RPUPlanDialect>();
    context.getOrLoadDialect<mlir::rpu::exec::RPUDialect>();
  });
}
