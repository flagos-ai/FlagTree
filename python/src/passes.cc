#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if FLAGTREE_ENABLE_DEBUGGER
#include "Debugger/Instrumentation/Passes.h"
#include "Debugger/Metadata/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include <cstdint>
#endif

namespace py = pybind11;

void init_triton_analysis(py::module &&m) {
  py::class_<mlir::ModuleAllocation>(m, "allocation", py::module_local())
      .def(py::init<mlir::ModuleOp>());
  py::class_<mlir::ModuleMembarAnalysis>(m, "membar", py::module_local())
      .def(py::init<mlir::ModuleAllocation *>())
      .def("run", &mlir::ModuleMembarAnalysis::run);
}

void init_triton_passes_common(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_sccp", createSCCPPass);
  ADD_PASS_WRAPPER_0("add_symbol_dce", createSymbolDCEPass);
  ADD_PASS_WRAPPER_0("add_inliner", createInlinerPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer", createCanonicalizerPass);
  ADD_PASS_WRAPPER_0("add_cse", createCSEPass);
  ADD_PASS_WRAPPER_0("add_licm", createLoopInvariantCodeMotionPass);
  ADD_PASS_WRAPPER_0("print_ir", createPrintIRPass);
}

void init_triton_passes_ttir(py::module &&m) {
  using namespace mlir::triton;
  ADD_PASS_WRAPPER_0("add_combine", createTritonCombineOps);
  ADD_PASS_WRAPPER_0("add_reorder_broadcast", createTritonReorderBroadcast);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_pointer",
                     createTritonRewriteTensorPointer);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_descriptor_to_pointer",
                     createTritonRewriteTensorDescriptorToPointer);
  ADD_PASS_WRAPPER_0("add_loop_unroll", createTritonLoopUnroll);
  ADD_PASS_WRAPPER_0("add_triton_licm", createTritonLoopInvariantCodeMotion);
  ADD_PASS_WRAPPER_0("add_loop_aware_cse", createTritonLoopAwareCSE);
  ADD_PASS_OPTION_WRAPPER_4("add_convert_to_ttgpuir",
                            createConvertTritonToTritonGPU, const std::string &,
                            int, int, int);
}

void init_triton_passes_ttgpuir(py::module &&m) {
  using namespace mlir;
  using namespace mlir::triton::gpu;
  using namespace mlir::triton::instrument;
  ADD_PASS_WRAPPER_0("add_process_shared_memory_hint",
                     createTritonGPUProcessSharedMemoryHint); // flagtree hints
  ADD_PASS_WRAPPER_0("add_coalesce", createTritonGPUCoalesce);
  ADD_PASS_WRAPPER_0("add_optimize_thread_locality",
                     createTritonGPUOptimizeThreadLocality);
  ADD_PASS_OPTION_WRAPPER_1("add_hoist_tmem_alloc",
                            createTritonGPUHoistTMEMAlloc, bool);
  ADD_PASS_OPTION_WRAPPER_1("add_assign_latencies",
                            createTritonGPUAssignLatencies, int);
  ADD_PASS_WRAPPER_0("add_schedule_loops", createTritonGPUScheduleLoops);
  ADD_PASS_OPTION_WRAPPER_2("add_pipeline", createTritonGPUPipeline, int, bool);
  ADD_PASS_OPTION_WRAPPER_1("add_warp_specialize",
                            createTritonGPUAutomaticWarpSpecialization, int);
  ADD_PASS_WRAPPER_0("add_prefetch", createTritonGPUPrefetch);
  ADD_PASS_WRAPPER_0("add_accelerate_matmul", createTritonGPUAccelerateMatmul);
  ADD_PASS_WRAPPER_0("add_reorder_instructions",
                     createTritonGPUReorderInstructions);
  ADD_PASS_WRAPPER_0("add_f32_dot_tc", createTritonGPUF32DotTC);
  ADD_PASS_OPTION_WRAPPER_1("add_optimize_dot_operands",
                            createTritonGPUOptimizeDotOperands, bool);
  ADD_PASS_WRAPPER_0("add_remove_layout_conversions",
                     createTritonGPURemoveLayoutConversions);
  ADD_PASS_WRAPPER_0("add_reduce_data_duplication",
                     createTritonGPUReduceDataDuplication);
  ADD_PASS_WRAPPER_0("add_allocate_warp_groups",
                     createTritonGPUAllocateWarpGroups);
  ADD_PASS_WRAPPER_0("add_allocate_shared_memory", createAllocateSharedMemory);
  ADD_PASS_WRAPPER_0("add_allocate_global_scratch_memory",
                     createTritonGPUGlobalScratchAllocationPass);
  ADD_PASS_WRAPPER_0("add_combine_tensor_select_and_if",
                     createTritonGPUCombineTensorSelectAndIf);
  ADD_PASS_WRAPPER_0("add_optimize_accumulator_init",
                     createTritonGPUOptimizeAccumulatorInit);
  ADD_PASS_WRAPPER_0("add_fuse_nested_loops", createTritonGPUFuseNestedLoops);
  ADD_PASS_WRAPPER_0("add_coalesce_async_copy",
                     createTritonGPUCoalesceAsyncCopy);
  ADD_PASS_WRAPPER_0("add_concurrency_sanitizer",
                     createTritonInstrumentConcurrencySanitizer);
}

void init_triton_passes_convert(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_scf_to_cf", createSCFToControlFlowPass);
  ADD_PASS_WRAPPER_0("add_cf_to_llvmir", createConvertControlFlowToLLVMPass);
  ADD_PASS_WRAPPER_0("add_index_to_llvmir", createConvertIndexToLLVMPass);
  ADD_PASS_WRAPPER_0("add_arith_to_llvmir", createArithToLLVMConversionPass);
  ADD_PASS_WRAPPER_0("add_nvvm_to_llvm", createConvertNVVMToLLVMPass);
}

void init_triton_passes_llvmir(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_di_scope", mlir::createLLVMDIScope);
}

void init_gluon_passes(py::module &&m) {
  using namespace mlir;
  namespace gluon = mlir::triton::gluon;
  ADD_PASS_WRAPPER_0("add_resolve_auto_encodings",
                     gluon::createGluonResolveAutoEncodingsPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer", gluon::createGluonCanonicalize);
  ADD_PASS_WRAPPER_0("add_inliner", gluon::createGluonInline);
}

#if FLAGTREE_ENABLE_DEBUGGER
void init_flagtree_debug_passes(py::module &&m) {
  using namespace mlir;
  using namespace mlir::flagtree::debugger;
  m.def("has_debug_collect_markers",
        [](ModuleOp mod) { return hasDebugCollectMarkers(mod); });
  m.def("insert_default_debug_collect_markers", [](ModuleOp mod, int32_t level,
                                                   int32_t addrLevel) {
    return succeeded(insertDefaultDebugCollectMarkers(mod, level, addrLevel));
  });
  m.def("get_debug_tracked_op_table_json",
        [](ModuleOp mod) { return getDebugTrackedOpTableJson(mod); });
  m.def("get_debug_kernel_metadata_json",
        [](ModuleOp mod) { return getDebugKernelMetadataJson(mod); });
  m.def("get_debug_kernel_id",
        [](ModuleOp mod) { return getDebugKernelId(mod); });
  m.def("get_debug_records_per_instance",
        [](ModuleOp mod) { return getDebugRecordsPerInstance(mod); });
  m.def("get_debug_record_size",
        [](ModuleOp mod) { return getDebugRecordSize(mod); });
  m.def("get_debug_record_layout",
        [](ModuleOp mod) { return getDebugRecordLayout(mod); });
  m.def("get_debug_record_plan_json",
        [](ModuleOp mod) { return getDebugRecordPlanJson(mod); });
  m.def("get_debug_full_dump_payload_bytes_per_instance", [](ModuleOp mod) {
    return getDebugFullDumpPayloadBytesPerInstance(mod);
  });
  m.def("get_debug_full_dump_plan_json",
        [](ModuleOp mod) { return getDebugFullDumpPlanJson(mod); });
  m.def("set_debug_kernel_id_seed", [](ModuleOp mod, const std::string &seed) {
    setDebugKernelIdSeed(mod, seed);
  });
  m.def("set_debug_hidden_arg_abi_enabled", [](ModuleOp mod, bool enabled) {
    setDebugHiddenArgAbiEnabled(mod, enabled);
  });
  m.def("set_debug_addr_level", [](ModuleOp mod, int32_t addrLevel) {
    setDebugAddrLevel(mod, addrLevel);
  });
  m.def("set_debug_timeline_enabled", [](ModuleOp mod, bool enabled) {
    setDebugTimelineEnabled(mod, enabled);
  });
  m.def("set_debug_timeline_only",
        [](ModuleOp mod, bool enabled) { setDebugTimelineOnly(mod, enabled); });
  m.def("assign_debug_collect_scope_ids_without_erase", [](ModuleOp mod) {
    return succeeded(assignDebugCollectScopeIdsWithoutErase(mod));
  });
  m.def("assign_debug_op_ids_and_metadata_without_pass_manager",
        [](ModuleOp mod) {
          return succeeded(assignDebugOpIdsAndMetadataWithoutPassManager(mod));
        });
  m.def("erase_debug_collect_markers",
        [](ModuleOp mod) { eraseDebugCollectMarkers(mod); });
  m.def("has_triton_tensor_pointer_types",
        [](ModuleOp mod) { return hasTritonTensorPointerTypes(mod); });
  ADD_PASS_WRAPPER_0("add_resolve_debug_scope", createResolveDebugScopePass);
  ADD_PASS_WRAPPER_0("add_assign_debug_op_id", createAssignOpIdPass);
  ADD_PASS_WRAPPER_0("add_insert_instrumentation",
                     createInsertInstrumentationPass);
  ADD_PASS_WRAPPER_0("add_simplify_record_memref_writes",
                     createSimplifyRecordMemrefWritesPass);
}
#endif

void init_triton_passes(py::module &&m) {
  init_triton_analysis(m.def_submodule("analysis"));
  init_triton_passes_common(m.def_submodule("common"));
  init_triton_passes_convert(m.def_submodule("convert"));
  init_triton_passes_ttir(m.def_submodule("ttir"));
  init_triton_passes_ttgpuir(m.def_submodule("ttgpuir"));
  init_triton_passes_llvmir(m.def_submodule("llvmir"));
  init_gluon_passes(m.def_submodule("gluon"));
#if FLAGTREE_ENABLE_DEBUGGER
  init_flagtree_debug_passes(m.def_submodule("flagtree_debug"));
#endif
}
