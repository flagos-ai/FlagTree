#include "Gluon/GluonPythonBindings.h"
#include "Gluon/Passes.h"
#include "Gluon/GluonLayoutCandidate.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace py = pybind11;
using namespace pybind11::literals;

namespace mlir::triton::gpu::metax::gluon {

void registerPassBindings(py::module_ &m) {
  ADD_PASS_WRAPPER_1("add_tritonmetaxgpu_gluon_insert_require_layout",
                     createTritonMETAXGPUGluonInsertRequireLayoutPass, int);
  ADD_PASS_WRAPPER_0("add_tritonmetaxgpu_gluon_propagate_layout",
                     createTritonMETAXGPUGluonPropagateLayoutPass);
  ADD_PASS_WRAPPER_0("add_tritonmetaxgpu_gluon_resolve_placeholder_layouts",
                     createTritonMETAXGPUGluonResolvePlaceholderLayoutsPass);
  ADD_PASS_WRAPPER_0(
      "add_tritonmetaxgpu_gluon_legalize_c500_async_copy_layout",
      createTritonMETAXGPUGluonLegalizeC500AsyncCopyLayoutPass);
  ADD_PASS_WRAPPER_0("add_tritonmetaxgpu_gluon_legalize_register_slices",
                     createTritonMETAXGPUGluonLegalizeRegisterSlicesPass);
  ADD_PASS_WRAPPER_0(
      "add_tritonmetaxgpu_gluon_legalize_local_dot_staging",
      createTritonMETAXGPUGluonLegalizeLocalDotStagingPass);
  ADD_PASS_WRAPPER_0(
      "add_tritonmetaxgpu_gluon_select_c500_layout_transfers",
      createTritonMETAXGPUGluonSelectC500LayoutTransfersPass);
  ADD_PASS_WRAPPER_0("add_tritonmetaxgpu_gluon_reorder_instructions",
                     createTritonMETAXGPUGluonReorderInstructionsPass);
  ADD_PASS_WRAPPER_0(
      "add_tritonmetaxgpu_insert_gvm_arrive_barrier_shared",
      createTritonMETAXGPUGluonInsertGvmArriveBarrierSharedPass);
  ADD_PASS_WRAPPER_0("add_tritonmetaxgpu_gluon_verify_synchronization",
                     createTritonMETAXGPUGluonVerifySynchronizationPass);
  ADD_PASS_WRAPPER_0("add_tritonmetaxgpu_gluon_verify_layout_contracts",
                     createTritonMETAXGPUGluonVerifyLayoutContractsPass);
}

void registerCandidateBundleBinding(py::module_ &module) {
  registerTransformsPasses();
  mlir::triton::gpu::registerTritonGPUPasses();
  registerTritonMETAXGPUGluonPasses();

  module.def("build_gluon_layout_candidate_bundle",
             [](ModuleOp &input, int32_t capability) {
               FailureOr<CandidateBundle> bundle =
                   buildC500LayoutCandidateBundle(input, capability);
               if (failed(bundle))
                 throw std::runtime_error(
                     "failed to build the closed C500 Gluon layout "
                     "candidate bundle");

               py::list variants;
               for (const CandidateVariant &variant : bundle->variants)
                 variants.append(py::dict("digest"_a = variant.digest,
                                          "source"_a = variant.source));
               return py::dict(
                   "version"_a = 1, "digest"_a = bundle->domainDigest,
                   "fallback"_a = bundle->fallbackDigest,
                   "fallback_only"_a = (bundle->variants.size() == 1),
                   "runtime_contract"_a = bundle->runtimeContract,
                   "variants"_a = std::move(variants));
             });
}

} // namespace mlir::triton::gpu::metax::gluon
