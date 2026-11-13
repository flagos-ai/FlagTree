#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/Passes.h" // createInlinerPass

#include "ascend/include/AutoBlockify/Passes.h"
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "ascend/include/DiscreteMaskAccessConversion/Passes.h"
#include "ascend/include/TritonControlFlowOpt/Passes.h"
#include "ascend/include/TritonToAnnotation/Passes.h"
#include "ascend/include/TritonToGraph/GraphOptimization.h"
#include "ascend/include/TritonToGraph/LayoutMemoryOptimization.h"
#include "ascend/include/TritonToGraph/Passes.h"
#include "ascend/include/TritonToHFusion/Passes.h"
#include "ascend/include/TritonToHIVM/Passes.h"
#include "ascend/include/TritonToLLVM/Passes.h"
#include "ascend/include/TritonToLinalg/Passes.h"
#include "ascend/include/TritonToStructured/Passes.h"
#include "ascend/include/TritonToUnstructure/Passes.h"

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/Passes.h"

#include "ascend/include/CommonIRToHIVM/Passes.h"
#include "ascend/include/CommonIRToHIVM/ReduceSumStrengthPass.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "mlir-ext/Dialect/CommonIR/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"

#include "ir.h" // TritonOpBuilder
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <limits>
#include <string>

#if TRITON_ASCEND_HAS_INPROC_COSTMODEL
#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>
#endif

namespace py = pybind11;
using namespace mlir;

void init_triton_ascend_passes_ttir(py::module &&m) {
  m.def("add_lane_vectorize", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::tile::createTritonLaneVectorize());
  });

  m.def("add_commonir_to_hivm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createCommonIRToHIVMPass());
  });

  m.def("add_reduce_sum_strength", [](mlir::PassManager &pm, bool enable,
                                      int32_t splitFactor) {
    pm.addPass(mlir::triton::createReduceSumStrengthPass(enable, splitFactor));
  });

  m.def("add_auto_blockify", [](mlir::PassManager &pm, int autoBlockifySize) {
    AutoBlockifyOptions opts;
    opts.autoBlockifySize = autoBlockifySize;
    pm.addPass(mlir::triton::createAutoBlockifyPass(opts));
  });

  m.def("add_triton_to_structure",
        [](mlir::PassManager &pm, bool enableMaskFallbackConversion,
           bool optimizeDynamicOffset) {
          pm.addPass(mlir::triton::createTritonToStructuredPass(
              enableMaskFallbackConversion, optimizeDynamicOffset));
        });

  m.def("add_triton_control_flow_opt", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonControlFlowOptPass());
  });

  m.def("add_triton_to_annotation", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToAnnotationPass());
  });

  m.def("add_triton_to_linalg",
        [](mlir::PassManager &pm, bool globalKernel, bool namedOps,
           bool enableNd2nzOnVector, bool enableSelectAnalysis,
           bool compileOn91095, const std::string &compileMode) {
          pm.addPass(mlir::triton::createTritonToLinalgPass(
              globalKernel, namedOps, enableNd2nzOnVector, enableSelectAnalysis,
              compileOn91095, compileMode));
        });

  m.def("add_merge_concat_load_buffer", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cfg::createMergeConcatLoadBufferPass());
  });

  m.def("add_triton_to_unstructure",
        [](mlir::PassManager &pm, bool compileOn91095,
           const std::string &compileMode) {
          TritonToUnstructureOptions opts;
          opts.compileOn91095 = compileOn91095;
          opts.compileMode = compileMode;
          pm.addPass(mlir::triton::createTritonToUnstructurePass(opts));
        });

  m.def("add_triton_to_hfusion",
        [](mlir::PassManager &pm, bool compileOn91095) {
          pm.addPass(mlir::triton::createTritonToHFusionPass(compileOn91095));
        });

  m.def("add_discrete_mask_access_conversion",
        [](mlir::PassManager &pm, bool compileOn91095,
           const std::string &compileMode) {
          DiscreteMaskAccessConversionOptions opts;
          opts.compileOn91095 = compileOn91095;
          opts.compileMode = compileMode;
          pm.addPass(
              mlir::triton::createDiscreteMaskAccessConversionPass(opts));
        });

  m.def("add_triton_to_hivm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToHIVMPass());
  });

  m.def("add_triton_to_llvm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToLLVMPass());
  });

  m.def("add_normalize_debug_line_locations", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createNormalizeDebugLineLocationsPass());
  });

  m.def("add_bubble_up_operation", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createBubbleUpOperationPass());
  });

  m.def("add_dynamic_cv_pipeline",
        [](mlir::PassManager &pm, bool compileOn91095) {
          AddDynamicCVPipelineOptions opts;
          opts.compileOn91095 = compileOn91095;
          pm.addPass(mlir::triton::createAddDynamicCVPipelinePass(opts));
        });

  m.def(
      "add_graph_optimize",
      [](mlir::PassManager &pm, std::uint64_t ruleMask,
         std::uint64_t maxRewritesPerFunction, std::uint64_t ubCapacityBytes,
         std::uint64_t mappingUBCapacityBytes,
         std::uint64_t storeCoalescingUBBudgetBytes,
         const std::string &compileMode, std::uint64_t ubSafetyPercent,
         std::uint64_t reservedUBBytes) {
        if (ruleMask > std::numeric_limits<std::uint32_t>::max())
          throw py::value_error("rule_mask must fit in uint32_t");
        const auto graphRuleMask =
            static_cast<mlir::triton::cfg::GraphOptimizationRuleMask>(ruleMask);
        if (!mlir::triton::cfg::isValidGraphOptimizationRuleMask(
                graphRuleMask)) {
          throw py::value_error("rule_mask contains unknown graph rule bits");
        }
        if (maxRewritesPerFunction > std::numeric_limits<unsigned>::max())
          throw py::value_error(
              "max_rewrites_per_function must fit in unsigned");
        if (ubCapacityBytes > std::numeric_limits<unsigned>::max())
          throw py::value_error("ub_capacity_bytes must fit in unsigned");
        if (mappingUBCapacityBytes > std::numeric_limits<unsigned>::max())
          throw py::value_error(
              "mapping_ub_capacity_bytes must fit in unsigned");
        if (storeCoalescingUBBudgetBytes > std::numeric_limits<unsigned>::max())
          throw py::value_error(
              "store_coalescing_ub_budget_bytes must fit in unsigned");
        if (ubSafetyPercent == 0 || ubSafetyPercent > 100 ||
            ubSafetyPercent > std::numeric_limits<unsigned>::max())
          throw py::value_error(
              "ub_safety_percent must be an unsigned value in [1, 100]");
        const std::uint64_t effectiveMappingUBCapacity =
            mappingUBCapacityBytes ? mappingUBCapacityBytes : ubCapacityBytes;
        if (reservedUBBytes > std::numeric_limits<unsigned>::max() ||
            reservedUBBytes > effectiveMappingUBCapacity)
          throw py::value_error("reserved_ub_bytes must fit in unsigned and "
                                "not exceed mapping UB");

        mlir::triton::cfg::GraphOptimizationOptions options;
        options.enabledRuleMask = graphRuleMask;
        options.maxRewritesPerFunction =
            static_cast<unsigned>(maxRewritesPerFunction);
        options.ubCapacityBytes = static_cast<unsigned>(ubCapacityBytes);
        options.mappingUBCapacityBytes =
            static_cast<unsigned>(mappingUBCapacityBytes);
        options.storeCoalescingUBBudgetBytes =
            static_cast<unsigned>(storeCoalescingUBBudgetBytes);
        options.ubSafetyPercent = static_cast<unsigned>(ubSafetyPercent);
        options.reservedUBBytes = static_cast<unsigned>(reservedUBBytes);
        options.compileMode = compileMode;
        pm.addPass(mlir::triton::cfg::createGraphOptimizePass(options));
      },
      py::arg("pm"),
      py::arg("rule_mask") = static_cast<std::uint64_t>(
          mlir::triton::cfg::kDefaultGraphOptimizationRuleMask),
      py::arg("max_rewrites_per_function") = 64,
      py::arg("ub_capacity_bytes") = 0,
      py::arg("mapping_ub_capacity_bytes") = 0,
      py::arg("store_coalescing_ub_budget_bytes") = 0,
      py::arg("compile_mode") = "simd_simt_template",
      py::arg("ub_safety_percent") = 80, py::arg("reserved_ub_bytes") = 0);

  m.def("set_buffer_count", [](mlir::ModuleOp &module, const std::string &type,
                               int count) {
    mlir::triton::BufferCountManager mgr(module);
    if (type == "INTRA") {
      mgr.setBufferCount(mlir::triton::BufferCountManager::DepType::IntraCore,
                         count);
    } else if (type == "INTER") {
      mgr.setBufferCount(mlir::triton::BufferCountManager::DepType::InterCore,
                         count);
    } else if (type == "LOAD") {
      mgr.setBufferCount(mlir::triton::BufferCountManager::DepType::LoadStore,
                         count);
    }
  });

  m.def("set_enable_ub_refine_opt",
        [](mlir::ModuleOp &moduleop, bool enable) { return; });
  m.def("set_enable_buffer_insert_optimization", [](mlir::ModuleOp &moduleop) {
    OpBuilder builder(moduleop.getContext());
    moduleop->setAttr(CVPipeline::kInsertionOptimization,
                      builder.getUnitAttr());
  });
}

#if TRITON_ASCEND_HAS_INPROC_COSTMODEL
namespace {
struct AscendCostModelRuntimeOptions {
  bool allowUnregisteredDialect = false;
  bool enableCostModelPipeline = false;
  std::string argBindings;
  std::string hardwareConfig;
};

static void
parseAscendCostModelOptionString(const std::string &option,
                                 AscendCostModelRuntimeOptions &opts) {
  // Parse known keys while allowing commas inside values, e.g.
  // arg-bindings=arg8=256,arg9=80,arg10=32,pid_x=0
  auto startsWith = [](const std::string &s, const std::string &p) {
    return s.rfind(p, 0) == 0;
  };

  auto trim = [](std::string v) {
    while (!v.empty() && (v.back() == ' ' || v.back() == ',')) {
      v.pop_back();
    }
    size_t i = 0;
    while (i < v.size() && v[i] == ' ') {
      ++i;
    }
    return v.substr(i);
  };

  auto extractValue = [&](const std::string &key) -> std::string {
    const std::string needle = key + "=";
    size_t keyPos = option.find(needle);
    if (keyPos == std::string::npos) {
      return "";
    }

    size_t valueStart = keyPos + needle.size();
    size_t valueEnd = option.size();

    const std::string keys[] = {"arg-bindings", "hardware-config"};
    for (const auto &otherKey : keys) {
      if (otherKey == key) {
        continue;
      }
      const std::string otherNeedle = otherKey + "=";
      const std::string commaBoundary = "," + otherNeedle;
      const std::string spaceBoundary = " " + otherNeedle;

      size_t p1 = option.find(commaBoundary, valueStart);
      if (p1 != std::string::npos) {
        valueEnd = std::min(valueEnd, p1);
      }
      size_t p2 = option.find(spaceBoundary, valueStart);
      if (p2 != std::string::npos) {
        valueEnd = std::min(valueEnd, p2);
      }
    }

    return trim(option.substr(valueStart, valueEnd - valueStart));
  };

  // Fast path for single-key payloads passed as next argv token.
  if (startsWith(option, "arg-bindings=")) {
    opts.argBindings = trim(option.substr(std::string("arg-bindings=").size()));
    return;
  }
  if (startsWith(option, "hardware-config=")) {
    opts.hardwareConfig =
        trim(option.substr(std::string("hardware-config=").size()));
    return;
  }

  auto argBindings = extractValue("arg-bindings");
  if (!argBindings.empty()) {
    opts.argBindings = argBindings;
  }
  auto hardwareConfig = extractValue("hardware-config");
  if (!hardwareConfig.empty()) {
    opts.hardwareConfig = hardwareConfig;
  }
}

static AscendCostModelRuntimeOptions
parseAscendCostmodelArgs(const std::vector<std::string> &extraArgs) {
  AscendCostModelRuntimeOptions opts;
  for (size_t i = 0; i < extraArgs.size(); ++i) {
    const auto &arg = extraArgs[i];
    if (arg.empty()) {
      continue;
    }
    if (arg == "-allow-unregistered-dialect") {
      opts.allowUnregisteredDialect = true;
      continue;
    }
    if (arg == "-ascend-perf-model" || arg == "--ascend-perf-model") {
      opts.enableCostModelPipeline = true;
      if (i + 1 < extraArgs.size() && !extraArgs[i + 1].empty() &&
          extraArgs[i + 1][0] != '-') {
        parseAscendCostModelOptionString(extraArgs[++i], opts);
      }
      continue;
    }
    constexpr const char *kShortPrefix = "-ascend-perf-model=";
    constexpr const char *kLongPrefix = "--ascend-perf-model=";
    if (arg.rfind(kShortPrefix, 0) == 0) {
      opts.enableCostModelPipeline = true;
      parseAscendCostModelOptionString(arg.substr(std::strlen(kShortPrefix)),
                                       opts);
      continue;
    }
    if (arg.rfind(kLongPrefix, 0) == 0) {
      opts.enableCostModelPipeline = true;
      parseAscendCostModelOptionString(arg.substr(std::strlen(kLongPrefix)),
                                       opts);
      continue;
    }
  }
  return opts;
}

static double extractEstimatedTimeUs(mlir::ModuleOp module) {
  constexpr double CYCLES_PER_US = 1850.0;

  auto getCycleAttr = [&](llvm::StringRef name) -> std::optional<int64_t> {
    if (auto attr = module->getAttrOfType<mlir::IntegerAttr>(name))
      return attr.getInt();
    return std::nullopt;
  };

  // PipelineAnalysisPass computes loop-multiplied roofline cycles.  Older
  // callers looked for ascend.scheduled_cycles, so keep that first for
  // compatibility, then fall back to the weighted roofline summary.
  if (auto cycles = getCycleAttr("ascend.scheduled_cycles"))
    return static_cast<double>(*cycles) / CYCLES_PER_US;
  if (auto cycles = getCycleAttr("ascend.roofline_cycles"))
    return static_cast<double>(*cycles) / CYCLES_PER_US;
  if (auto cycles = getCycleAttr("ascend.simple_sum_cycles"))
    return static_cast<double>(*cycles) / CYCLES_PER_US;

  int64_t fallbackCycles = 0;
  module.walk([&](mlir::Operation *op) {
    if (auto cyclesAttr =
            op->getAttrOfType<mlir::IntegerAttr>("estimated_cycles")) {
      int64_t multiplier = 1;
      if (auto multiplierAttr =
              op->getAttrOfType<mlir::IntegerAttr>("loop_multiplier"))
        multiplier = std::max<int64_t>(1, multiplierAttr.getInt());
      fallbackCycles += cyclesAttr.getInt() * multiplier;
    }
  });
  return static_cast<double>(fallbackCycles) / CYCLES_PER_US;
}
} // namespace

static std::string
runAscendCostModelInProcess(const std::string &mlirText,
                            const std::vector<std::string> &extraArgs) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::ascend::AscendModelDialect>();
  registry.insert<mlir::triton::TritonDialect>();

  auto opts = parseAscendCostmodelArgs(extraArgs);
  if (!opts.enableCostModelPipeline) {
    throw std::runtime_error(
        "in-process costmodel requires '-ascend-perf-model'");
  }

  mlir::MLIRContext context(registry);
  if (opts.allowUnregisteredDialect) {
    context.allowUnregisteredDialects(true);
  }

  auto module = mlir::parseSourceString<mlir::ModuleOp>(mlirText, &context);
  if (!module) {
    throw std::runtime_error(
        "in-process costmodel failed to parse input MLIR module");
  }

  mlir::PassManager pm(&context);
  // Inline tt.call helpers (e.g. softmax's @max/@sum reduce helpers) into the
  // calling kernel BEFORE conversion/estimation. Without this, ops in a helper
  // invoked from inside a loop are estimated at the helper's top level (outside
  // the loop) and get loopMultiplier=1 instead of the caller's trip count, so
  // per-iteration ops like reductions are under-counted. TritonDialect
  // registers TritonInlinerInterface, so the generic MLIR inliner handles
  // tt.call (CallOpInterface). SymbolDCE then erases the now-dead private
  // helpers so their ops are not double-counted alongside the inlined copies.
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::ascend::createConvertTritonToAscendPass());
  pm.addPass(mlir::ascend::createInsertDataTransfersPass());
  pm.addPass(mlir::ascend::createAssignOpIDsPass());
  {
    mlir::ascend::EstimateCyclesPassOptions estimateOpts;
    estimateOpts.argBindingsStr = opts.argBindings;
    estimateOpts.hardwareConfigPath = opts.hardwareConfig;
    pm.addPass(mlir::ascend::createEstimateCyclesPass(estimateOpts));
  }
  {
    mlir::ascend::PipelineAnalysisPassOptions pipelineOpts;
    pipelineOpts.argBindingsStr = opts.argBindings;
    pipelineOpts.hardwareConfigPath = opts.hardwareConfig;
    pm.addPass(mlir::ascend::createPipelineAnalysisPass(pipelineOpts));
  }

  if (mlir::failed(pm.run(*module))) {
    throw std::runtime_error("in-process costmodel pass pipeline failed");
  }

  const double timeUs = extractEstimatedTimeUs(*module);
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(3);
  os << "Estimated Time: " << timeUs << " us\n";
  return os.str();
}
#endif

// Forward declaration for ascend_ir bindings (defined in ascend_ir.cc)
void init_ascend_ir(py::module &&m);
void init_buffer_ir(py::module &&m);
// 3.5 CommonIR DSA builder bindings (defined in tle_dsa/tle_ir.cc).
void init_tle_dsa_ir(py::module &&m);

void init_triton_ascend(py::module &&m) {
  py::module_ libtriton = py::module_::import("triton._C.libtriton");
  init_buffer_ir(libtriton.def_submodule("buffer_ir"));

  auto passes = m.def_submodule("passes");
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::ascend::TritonAscendDialect,
                    mlir::triton::tile::CommonIRDialect>();
    // commonir-to-hivm is followed by the Inliner, which aborts unless the func
    // dialect's inliner interface is registered.
    mlir::func::registerInlinerExtension(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  init_triton_ascend_passes_ttir(passes.def_submodule("ttir"));

#if TRITON_ASCEND_HAS_INPROC_COSTMODEL
  m.def(
      "run_costmodel_inproc",
      [](const std::string &mlirText,
         const std::vector<std::string> &extraArgs) {
        py::gil_scoped_release release;
        return runAscendCostModelInProcess(mlirText, extraArgs);
      },
      py::arg("mlir_text"), py::arg("extra_args") = std::vector<std::string>{});
#else
  m.def(
      "run_costmodel_inproc",
      [](const std::string &, const std::vector<std::string> &) {
        throw std::runtime_error(
            "in-process costmodel bridge is not enabled in this build");
        return std::string();
      },
      py::arg("mlir_text"), py::arg("extra_args") = std::vector<std::string>{});
#endif

  // Initialize ascend IR bindings (ascendnpu_ir_builder, scope/hivm dialects)
  init_ascend_ir(m.def_submodule("ir"));
  // Attach the 3.5 CommonIR DSA builders to the main TritonOpBuilder class.
  init_tle_dsa_ir(m.def_submodule("tle_dsa"));
}
