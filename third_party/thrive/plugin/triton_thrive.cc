#include <pybind11/pybind11.h>

namespace py = pybind11;
#include "Dialect/ThvTile/IR/Dialect.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "thrive_ir.h"

namespace mlir::thrive {
std::unique_ptr<Pass> createChipletToThvTilePass();
}

void init_thrive_passes(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_chiplet_to_thvtile",
                     mlir::thrive::createChipletToThvTilePass);
}

void init_triton_thrive(py::module &&m) {
  init_thrive_ir(m);
  init_thrive_passes(m.def_submodule("passes"));

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::thvtile::ThvTileDialect>();

    mlir::func::registerInlinerExtension(registry);

    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}
