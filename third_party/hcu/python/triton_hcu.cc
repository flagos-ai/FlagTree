#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "TritonHCUGPUToLLVM/LLVMModuleTransforms.h"
#include "TritonHCUGPUToLLVM/Passes.h"
#include "TritonHCUGPUToLLVM/TargetUtils.h"
#include "TritonHCUGPUTransforms/Passes.h"
#include "hcu/include/hipblas_instance.h"
#include "hcu/include/hipblas_types.h"
#include "lib/TritonHCUGPUToLLVM/TargetInfo.h"
#include "lld/Common/Driver.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TargetParser/TargetParser.h"
#include <algorithm>
#include <array>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <sstream>
#include <stdexcept>

#ifdef __TLE__
#include "hcu/tle_raw/include/DeferredRawSourceRegistry.h"
#include "hcu/tle_raw/include/Passes.h"
#endif

namespace py = pybind11;

using mlir::triton::HCU::annotateLdsObjectScopes;
using mlir::triton::HCU::interleaveGlobalLdsReads;
using mlir::triton::HCU::lowerTaggedLocalBarriers;
using mlir::triton::HCU::preferReadySharedLoads;
using mlir::triton::HCU::preserveBufferLoadByteOffsets;
using mlir::triton::HCU::preserveLdsBarrierOrder;
using mlir::triton::HCU::preserveLdsDmaAliasScopes;
using mlir::triton::HCU::preserveUniformResourceIndices;
using mlir::triton::HCU::stripSynchronousBarrierMarkers;
using mlir::triton::HCU::tupleRegisterFences;

namespace {
const char *const hcuTargetTriple = "amdgcn-amd-amdhsa";

void init_triton_hcu_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  m.def("add_to_llvmir",
        [](mlir::PassManager &pm, const std::string &arch, bool ftz) {
          pm.addPass(createConvertTritonHCUGPUToLLVMPass(arch, ftz));
        });
  m.def("add_builtin_func_to_llvmir", [](mlir::PassManager &pm, bool ftz) {
    pm.addPass(createConvertBuiltinFuncToLLVMPass(ftz));
  });
  m.def("insert_instruction_sched_hints", [](mlir::PassManager &pm,
                                             const std::string &variant) {
    pm.addPass(createTritonHCUGPUInsertInstructionSchedHintsPass(variant));
  });
  m.def("lower_instruction_sched_hints",
        [](mlir::PassManager &pm, const std::string &arch, int32_t numStages) {
          pm.addPass(createTritonHCUGPULowerInstructionSchedHintsPass(
              arch, numStages));
        });
  ADD_PASS_WRAPPER_2("add_optimize_lds_usage",
                     mlir::triton::HCU::createOptimizeLDSUsagePass,
                     const std::string &, int32_t);
  ADD_PASS_WRAPPER_0("add_allocate_shared_memory",
                     mlir::triton::createAllocateHCUGPUSharedMemory);
  ADD_PASS_OPTION_WRAPPER_4("add_accelerate_matmul",
                            mlir::createTritonHCUGPUAccelerateMatmul,
                            const std::string, int, int, int);
  ADD_PASS_WRAPPER_0("add_optimize_epilogue",
                     mlir::createTritonHCUGPUOptimizeEpilogue);
  ADD_PASS_OPTION_WRAPPER_1(
      "add_optimize_dot_operands",
      mlir::triton::hcugpu::createTritonHCUGPUOptimizeDotOperands,
      const std::string &);
  m.def("add_hoist_layout_conversions", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonHCUGPUHoistLayoutConversions());
  });
  m.def("add_canonicalize_pointers", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonHCUGPUCanonicalizePointers());
  });
  m.def(
      "add_convert_to_buffer_ops",
      [](mlir::PassManager &pm, const std::string &arch, bool allowAtomics,
         bool analyzeSmallTensorOfst, bool asyncOnly) {
        pm.addPass(mlir::createTritonHCUGPUConvertToBufferOps(
            {arch, allowAtomics, analyzeSmallTensorOfst, asyncOnly}));
      },
      py::arg("pm"), py::arg("arch"), py::arg("allow_buffer_atomics"),
      py::arg("analyze_small_tensor_ofst"), py::arg("async_only") = false);
  ADD_PASS_WRAPPER_0("add_reorder_instructions",
                     mlir::createTritonHCUGPUReorderInstructions);
  ADD_PASS_WRAPPER_0("add_fold_true_cmpi", mlir::createTritonHCUFoldTrueCmpI);
  ADD_PASS_OPTION_WRAPPER_1("add_block_pingpong",
                            mlir::createTritonHCUGPUBlockPingpong, int32_t);
  ADD_PASS_OPTION_WRAPPER_1("add_schedule_loops",
                            mlir::createTritonHCUGPUScheduleLoops, int);
  ADD_PASS_OPTION_WRAPPER_2("add_pipeline", mlir::createTritonHCUGPUPipeline,
                            bool, bool);
  ADD_PASS_OPTION_WRAPPER_1("add_coalesce_async_copy",
                            mlir::createTritonHCUGPUCoalesceAsyncCopy,
                            std::string);
  ADD_PASS_OPTION_WRAPPER_1("add_update_async_wait_count",
                            mlir::createTritonHCUGPUUpdateAsyncWaitCount,
                            std::string);
  m.def("add_in_thread_transpose", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonHCUGPUInThreadTranspose());
  });
  ADD_PASS_OPTION_WRAPPER_3("add_mls_stream_pipeline",
                            mlir::createTritonHCUGPUMlsStreamPipeline, int, int,
                            int);
  ADD_PASS_WRAPPER_0("add_mls_encoding_insertion",
                     mlir::createTritonHCUGPUMlsEncodingInsertion);
  ADD_PASS_WRAPPER_0("add_mls_lowering_pass",
                     mlir::createTritonHCUGPUMlsLowering);
  m.def("add_warp_specialize_to_llvm",
        [](mlir::PassManager &pm, const std::string &arch, int waspNumLoadWarps,
           int waspNumMmaWarps, bool wdraEnabled, int wdraNumLoadRegs,
           int wdraNumMmaRegsMain, int wdraNumMmaRegsTail) {
          pm.addPass(createHCUGPUConvertWarpSpecializeToLLVM(
              arch, waspNumLoadWarps, waspNumMmaWarps, wdraEnabled,
              wdraNumLoadRegs, wdraNumMmaRegsMain, wdraNumMmaRegsTail));
        });
  ADD_PASS_WRAPPER_1(
      "set_func_inline_attrs",
      mlir::triton::HCU::createTritonHCUGPUSetFuncInlineAttrsPass,
      const std::string &);
}

#ifdef __TLE__
static void setDeferredRawPendingSources(py::dict sources) {
  mlir::triton::HCU::tle_raw::clearDeferredRawSourceRegistry();
  for (auto item : sources) {
    std::string key = py::cast<std::string>(item.first);
    py::dict entry = py::cast<py::dict>(item.second);

    mlir::triton::HCU::tle_raw::DeferredRawSourceEntry rawEntry;
    rawEntry.sourceId = key;
    rawEntry.regionDialect = entry["region_dialect"].cast<std::string>();
    if (entry.contains("extern_func_name") &&
        !entry["extern_func_name"].is_none()) {
      rawEntry.externFuncName = entry["extern_func_name"].cast<std::string>();
    }
    rawEntry.source = entry["source"].cast<std::string>();
    if (entry.contains("llvm_ir"))
      rawEntry.llvmIr = entry["llvm_ir"].cast<std::string>();
    if (entry.contains("hint"))
      rawEntry.hint = entry["hint"].cast<std::string>();
    mlir::triton::HCU::tle_raw::getDeferredRawSourceRegistry()[key] =
        std::move(rawEntry);
  }
}

void init_hcu_tle_raw_passes(py::module &&m) {
  m.def("deferred_raw_materialize",
        [](py::dict sources, mlir::PassManager &pm) {
          setDeferredRawPendingSources(sources);
          pm.addPass(mlir::createHCUMaterializeDeferredRaw());
        });
}
#endif

void addControlConstant(llvm::Module *module, const char *name,
                        uint32_t bitwidth, uint32_t value) {
  using llvm::GlobalVariable;

  llvm::IntegerType *type =
      llvm::IntegerType::getIntNTy(module->getContext(), bitwidth);
  auto *initializer = llvm::ConstantInt::get(type, value, /*isSigned=*/false);
  auto *constant = new llvm::GlobalVariable(
      *module, type, /*isConstant=*/true,
      GlobalVariable::LinkageTypes::LinkOnceODRLinkage, initializer, name,
      /*before=*/nullptr, GlobalVariable::ThreadLocalMode::NotThreadLocal,
      /*addressSpace=*/4);
  constant->setAlignment(llvm::MaybeAlign(bitwidth / 8));
  constant->setUnnamedAddr(GlobalVariable::UnnamedAddr::Local);
  constant->setVisibility(GlobalVariable::VisibilityTypes::ProtectedVisibility);
}

} // namespace

LLD_HAS_DRIVER(elf)

static void checkMatmulConstraints(const std::string &A_dtype,
                                   const std::string &B_dtype,
                                   const std::string &C_dtype,
                                   const std::vector<int> &A_shape,
                                   const std::vector<int> &B_shape,
                                   const std::vector<int> &C_shape) {
  // Support FP32/FP16/BF16 and 8-bit FP8 (e4m3fn/e4m3fnuz) and BF8
  // (e5m2fn/e5m2fnuz).
  auto is_fp8 = [](const std::string &dtype) {
    return dtype == "torch.float8_e4m3fn" || dtype == "torch.float8_e5m2fn" ||
           dtype == "torch.float8_e4m3fnuz" || dtype == "torch.float8_e5m2fnuz";
  };
  auto is_fp16_family = [](const std::string &dtype) {
    return dtype == "torch.float16" || dtype == "torch.bfloat16";
  };
  const bool A_is_fp8 = is_fp8(A_dtype);
  const bool B_is_fp8 = is_fp8(B_dtype);
  const bool A_supported =
      (A_is_fp8 || is_fp16_family(A_dtype) || A_dtype == "torch.float32");
  const bool B_supported =
      (B_is_fp8 || is_fp16_family(B_dtype) || B_dtype == "torch.float32");
  const bool C_supported = (is_fp16_family(C_dtype) ||
                            C_dtype == "torch.float32" || is_fp8(C_dtype));

  if (!A_supported || !B_supported || !C_supported) {
    std::ostringstream oss;
    oss << "Unsupported data type. Got A=" << A_dtype << ", B=" << B_dtype
        << ", C=" << C_dtype
        << ". Supported: float32, float16, bfloat16, float8_e4m3fn, "
           "float8_e5m2fn, float8_e4m3fnuz, float8_e5m2fnuz.";
    throw std::runtime_error(oss.str());
  }

  if (A_is_fp8 && B_is_fp8) {
    if (C_dtype != "torch.float16" && C_dtype != "torch.float32" &&
        C_dtype != "torch.bfloat16") {
      std::ostringstream oss;
      oss << "When A/B are 8-bit (float8_e4m3fn/e4m3fnuz or "
             "float8_e5m2fn/e5m2fnuz), C must"
          << " be torch.float16, torch.float32, or torch.bfloat16.";
      throw std::runtime_error(oss.str());
    }
  } else {
    if (!(A_dtype == B_dtype && A_dtype == C_dtype)) {
      std::ostringstream oss;
      oss << "Data types do not match: A=" << A_dtype << ", B=" << B_dtype
          << ", C=" << C_dtype << ". Expected all equal when not using 8-bit"
          << " inputs.";
      throw std::runtime_error(oss.str());
    }
  }

  if (A_shape.size() != 2 || B_shape.size() != 2 || C_shape.size() != 2) {
    throw std::runtime_error("Only 2D matrices are supported.");
  }

  int k = A_shape[1];
  if (k != B_shape[1]) {
    std::ostringstream oss;
    oss << "Matrix dimensions do not match. A is [" << A_shape[0] << ", "
        << A_shape[1] << "], B is [" << B_shape[0] << ", " << B_shape[1]
        << "]. Expected A.shape[1] == B.shape[1]. Note that B needs to be "
           "transposed.";
    throw std::runtime_error(oss.str());
  }

  int m = A_shape[0];
  if (m != C_shape[0]) {
    std::ostringstream oss;
    oss << "Matrix dimensions do not match. A is [" << A_shape[0] << ", "
        << A_shape[1] << "], C is [" << C_shape[0] << ", " << C_shape[1]
        << "]. Expected A.shape[0] == C.shape[0].";
    throw std::runtime_error(oss.str());
  }

  int n = B_shape[0];
  if (n != C_shape[1]) {
    std::ostringstream oss;
    oss << "Matrix dimensions do not match. B is [" << B_shape[0] << ", "
        << B_shape[1] << "], C is [" << C_shape[0] << ", " << C_shape[1]
        << "]. Expected B.shape[0] == C.shape[1]. Note that B needs to be "
           "transposed.";
    throw std::runtime_error(oss.str());
  }
}

struct HipBlasInit {
  int m;
  int n;
  int k;
  hipDataType dtype;
  hipDataType out_dtype;
};

static HipBlasInit initialize_hipblas_op(py::object &A, py::object &B,
                                         py::object &out,
                                         std::optional<py::object> accumOpt) {
  auto A_shape = A.attr("shape").cast<std::vector<int>>();
  auto B_shape = B.attr("shape").cast<std::vector<int>>();
  auto OUT_shape = out.attr("shape").cast<std::vector<int>>();

  auto A_dtype = A.attr("dtype").attr("__str__")().cast<std::string>();
  auto B_dtype = B.attr("dtype").attr("__str__")().cast<std::string>();
  auto OUT_dtype = out.attr("dtype").attr("__str__")().cast<std::string>();

  if (accumOpt.has_value()) {
    auto C = accumOpt.value();
    auto C_shape = C.attr("shape").cast<std::vector<int>>();
    auto C_dtype = C.attr("dtype").attr("__str__")().cast<std::string>();

    checkMatmulConstraints(A_dtype, B_dtype, OUT_dtype, A_shape, B_shape,
                           OUT_shape);
    if (C_dtype != OUT_dtype) {
      throw std::runtime_error("C dtype must match output dtype, got C=" +
                               C_dtype + ", D=" + OUT_dtype);
    }
    if (C_shape != OUT_shape) {
      throw std::runtime_error("C and D shapes must match");
    }
  } else {
    checkMatmulConstraints(A_dtype, B_dtype, OUT_dtype, A_shape, B_shape,
                           OUT_shape);
  }

  hipDataType dtype;
  if (A_dtype == "torch.float8_e4m3fn") {
    // Supported for GFX950.
    dtype = HIP_R_8F_E4M3;
  } else if (A_dtype == "torch.float8_e5m2fn") {
    // supported for GFX950.
    dtype = HIP_R_8F_E5M2;
  } else if (A_dtype == "torch.float8_e4m3fnuz") {
    // Supported for GFX942.
    dtype = HIP_R_8F_E4M3_FNUZ;
  } else if (A_dtype == "torch.float8_e5m2fnuz") {
    // Supported for GFX942.
    dtype = HIP_R_8F_E5M2_FNUZ;
  } else if (A_dtype == "torch.float16") {
    dtype = HIP_R_16F;
  } else if (A_dtype == "torch.float32") {
    dtype = HIP_R_32F;
  } else if (A_dtype == "torch.bfloat16") {
    dtype = HIP_R_16BF;
  } else {
    throw std::runtime_error("Unsupported dtype for hipblasLt: " + A_dtype);
  }

  hipDataType out_dtype;
  if (OUT_dtype == "torch.float16") {
    out_dtype = HIP_R_16F;
  } else if (OUT_dtype == "torch.float32") {
    out_dtype = HIP_R_32F;
  } else if (OUT_dtype == "torch.bfloat16") {
    out_dtype = HIP_R_16BF;
  } else {
    throw std::runtime_error("Unsupported output dtype for hipblasLt: " +
                             OUT_dtype);
  }

  int m = A_shape[0];
  int n = B_shape[0];
  int k = A_shape[1];

  return HipBlasInit{m, n, k, dtype, out_dtype};
}

static std::optional<std::string> lldInvoke(const char *inPath,
                                            const char *outPath) {
  // Workaround: Disable parallelism to avoid hangs caused by LLVM's thread pool
  // when the following code is executed in a forked child process.
  // Context: lld::elf::LinkerDriver::link uses parallelFor which uses the
  // LLVM's thread pool. During cleanup at ~TaskGroup() the child process hangs
  // waiting.
  std::array args{"ld.lld", "--threads=1", "-shared", inPath, "-o", outPath};
  std::string errString;
  llvm::raw_string_ostream errStream(errString);
  auto lldRes = lld::lldMain(args, llvm::outs(), llvm::errs(),
                             {{lld::Gnu, &lld::elf::link}});
  bool noErrors = (!lldRes.retCode && lldRes.canRunAgain);
  if (!noErrors) {
    errStream.flush();
    return errString;
  }
  return {};
}

// DTK LLVM 18 MachineSink can move an LDS read past s.barrier into a
// successor block. Reuse the empty memory-clobber fence from Triton's AMD
// workaround (PRs 9624 and 11096; LLVM issue 181708). This emits no ISA.
void init_triton_hcu(py::module &&m) {
  m.def("create_accumulator_fence",
        [](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value acc,
           int32_t priority) -> mlir::Value {
          auto type = mlir::dyn_cast<mlir::RankedTensorType>(acc.getType());
          if (!type || (!type.getElementType().isF32() &&
                        !type.getElementType().isInteger(32)))
            throw std::invalid_argument(
                "accumulator fence requires an f32 or i32 tensor");
          if (priority != 0 && priority != 1)
            throw std::invalid_argument(
                "accumulator fence priority must be 0 or 1");
          return builder
              .create<mlir::triton::hcugpu::AccumulatorFenceOp>(
                  loc, acc.getType(), acc, builder.getI32IntegerAttr(priority))
              .getResult();
        });
  m.def("create_local_memory_wait",
        [](mlir::OpBuilder &builder, mlir::Location loc) {
          // Reuse the target counter operation so native wait insertion
          // recognizes the completed LDS responses; an opaque asm leaves
          // redundant waits.
          builder.create<mlir::triton::hcugpu::MemoryCounterWaitOp>(
              loc, /*load=*/nullptr, /*store=*/nullptr,
              /*ds=*/builder.getI32IntegerAttr(0));
        });
  m.def("has_warp_specialize", [](mlir::ModuleOp mod) {
    bool found = false;
    mod.walk([&](mlir::triton::gpu::WarpSpecializeOp) { found = true; });
    return found;
  });
  m.def("has_cta_barrier", [](mlir::ModuleOp mod) {
    bool found = false;
    mod.walk([&](mlir::triton::gpu::LocalBarrierOp op) {
      found |= op->hasAttr("hcu.cta_scope");
    });
    return found;
  });
  m.def("create_cta_barrier", [](mlir::OpBuilder &builder, mlir::Location loc) {
    // GPU dialect's adjacent-barrier folding is invalid for independently
    // phased WS actors. The standard TTG local barrier retains each rendezvous.
    auto barrier = builder.create<mlir::triton::gpu::LocalBarrierOp>(loc);
    barrier->setAttr("hcu.cta_scope", builder.getUnitAttr());
  });
  m.def("create_warp_group_id",
        [](mlir::OpBuilder &builder, mlir::Location loc,
           int32_t size) -> mlir::Value {
          if (size <= 0 || !llvm::isPowerOf2_32(size))
            throw std::invalid_argument(
                "warp group size must be a positive power of two");
          return builder
              .create<mlir::triton::hcugpu::WarpGroupIdOp>(
                  loc, builder.getI32Type(), builder.getI32IntegerAttr(size))
              .getResult();
        });
  // Target scheduling boundary, independent of any shared-memory protocol.
  m.def(
      "create_sched_barrier",
      [](mlir::OpBuilder &builder, mlir::Location loc, int32_t mask) {
        if (mask < 0)
          throw std::invalid_argument(
              "scheduler barrier mask must be nonnegative");
        mlir::ROCDL::SchedBarrier::create(builder, loc, mask);
      },
      py::arg("builder"), py::arg("loc"), py::arg("mask") = 0);
  using namespace mlir;
  namespace tt = mlir::triton;
  namespace tg = mlir::triton::gpu;
  auto scalarTypes = [](ValueRange values, unsigned width, unsigned count) {
    if (values.size() != count)
      throw std::invalid_argument(
          "structured register coordinate/extent/stride count mismatch");
    for (Value value : values)
      if (!value || !value.getType().isInteger(width))
        throw std::invalid_argument(
            "structured register coordinate/extent/stride has wrong scalar "
            "type");
  };
  m.def("create_structured_register_load",
        [scalarTypes](OpBuilder &b, Location loc, Value base, Type resultType,
                      const std::vector<Value> &shape,
                      const std::vector<Value> &strides,
                      const std::vector<Value> &indices) -> Value {
          auto result = dyn_cast<RankedTensorType>(resultType);
          auto ptr = dyn_cast<tt::PointerType>(base.getType());
          if (!result || result.getRank() != 2 ||
              !(result.getElementType().isF16() ||
                result.getElementType().isBF16()) ||
              !isa<tg::DistributedEncodingTrait>(result.getEncoding()) ||
              !ptr || ptr.getAddressSpace() != 1 ||
              ptr.getPointeeType() != result.getElementType())
            throw std::invalid_argument(
                "structured register load needs matching AS1 f16/bf16 pointer "
                "and rank-two distributed result");
          scalarTypes(shape, 32, 2);
          scalarTypes(strides, 64, 2);
          scalarTypes(indices, 32, 3);
          SmallVector<int32_t> tile;
          for (int64_t d : result.getShape()) {
            if (d < 1 || d > INT32_MAX)
              throw std::invalid_argument("invalid structured register tile");
            tile.push_back(int32_t(d));
          }
          auto op = b.create<tt::MatrixLoadOp>(
              loc, resultType, base, shape, strides, tile, indices,
              ArrayRef<int32_t>{}, tt::CacheModifier::NONE,
              tt::EvictionPolicy::NORMAL, false);
          op->setAttr("hcu.structured_register", b.getUnitAttr());
          return op.getResult();
        });
  // Reuse MLIR's standard LLVM loop metadata, as other backends do. Preserve
  // other loop annotations; explicit LLVM unroll hints take precedence.
  m.def("disable_llvm_loop_unroll", [](mlir::ModuleOp mod) {
    auto *ctx = mod.getContext();
    mod.walk([&](mlir::scf::ForOp loop) {
      auto old = loop->getAttrOfType<mlir::LLVM::LoopAnnotationAttr>(
          "llvm.loop_annotation");
      if (old && old.getUnroll())
        return;
      if (!old)
        old = mlir::LLVM::LoopAnnotationAttr::get(
            ctx, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {});
      auto unroll = mlir::LLVM::LoopUnrollAttr::get(
          ctx, mlir::BoolAttr::get(ctx, true), {}, {}, {}, {}, {}, {});
      auto annotation = mlir::LLVM::LoopAnnotationAttr::get(
          ctx, old.getDisableNonforced(), old.getVectorize(),
          old.getInterleave(), unroll, old.getUnrollAndJam(), old.getLicm(),
          old.getDistribute(), old.getPipeline(), old.getPeeled(),
          old.getUnswitch(), old.getMustProgress(), old.getIsVectorized(),
          old.getStartLoc(), old.getEndLoc(), old.getParallelAccesses());
      loop->setAttr("llvm.loop_annotation", annotation);
    });
  });

  m.def("preserve_buffer_load_byte_offsets", &preserveBufferLoadByteOffsets);
  m.def("preserve_uniform_resource_indices", &preserveUniformResourceIndices);
  m.def("preserve_lds_barrier_order", &preserveLdsBarrierOrder);
  m.def("strip_synchronous_barrier_markers",
        [](llvm::Module *module, const std::string &arch) {
          auto result = stripSynchronousBarrierMarkers(module, arch);
          if (!result)
            throw std::invalid_argument(llvm::toString(result.takeError()));
          return *result;
        });
  m.def("lower_tagged_local_barriers",
        [](llvm::Module *module, const std::string &arch) {
          auto result = lowerTaggedLocalBarriers(module, arch);
          if (!result)
            throw std::invalid_argument(llvm::toString(result.takeError()));
          return *result;
        });
  m.def("annotate_lds_object_scopes", &annotateLdsObjectScopes);
  m.def("preserve_lds_dma_alias_scopes", &preserveLdsDmaAliasScopes);
  m.def("prefer_ready_shared_loads", &preferReadySharedLoads);
  m.def("interleave_global_lds_reads",
        [](llvm::Module *module, const std::string &arch, bool addFences) {
          auto s = interleaveGlobalLdsReads(module, arch, addFences);
          py::dict d;
          d["transformed_windows"] = s.windows;
          d["moved_global_reads"] = s.globalReads;
          d["moved_pure_dependents"] = s.pureDependents;
          d["split_lds_reads"] = s.ldsReads;
          d["sched_barriers_added"] = s.fences;
          d["rejected_dependencies"] = s.rejectedDependencies;
          d["invalid_functions"] = s.invalidFunctions;
          d["iglp_functions"] = s.iglpFunctions;
          return d;
        });
  m.def("tuple_register_fences",
        [](llvm::Module *module, const std::string &arch) {
          auto expected = tupleRegisterFences(module, arch);
          if (!expected)
            throw std::runtime_error(llvm::toString(expected.takeError()));
          auto stats = *expected;
          py::dict result;
          result["sites"] = stats.sites;
          result["scalar_lanes"] = stats.scalarLanes;
          result["rejected"] = stats.rejected;
          return result;
        });
  m.doc() = "Python bindings to the HCU Triton backend";

  auto passes = m.def_submodule("passes");
  init_triton_hcu_passes_ttgpuir(passes.def_submodule("ttgpuir"));

#ifdef __TLE__
  init_hcu_tle_raw_passes(passes.def_submodule("tle_raw"));
#endif

  m.attr("TARGET_TRIPLE") = hcuTargetTriple;
  m.attr("CALLING_CONV_AMDGPU_KERNEL") =
      (unsigned)llvm::CallingConv::AMDGPU_KERNEL;

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::hcugpu::TritonHCUGPUDialect>();
    // registry.insert<mlir::ROCDL::ROCDLDialect>();
    mlir::registerROCDLDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("attach_target_triple", [](llvm::Module *module) {
    module->setTargetTriple(llvm::Triple(hcuTargetTriple));
  });

  // Set target architecture ISA version
  m.def("set_isa_version", [](llvm::Module *module, const std::string &arch) {
    llvm::AMDGPU::IsaVersion version = llvm::AMDGPU::getIsaVersion(arch);
    addControlConstant(module, "__oclc_ISA_version", /*bitwidth=*/32,
                       version.Major * 1000 + version.Minor * 100 +
                           version.Stepping);
  });

  // Set boolean control constant
  m.def("set_bool_control_constant",
        [](llvm::Module *module, const std::string &name, bool enable) {
          addControlConstant(module, name.c_str(), /*bitwidth=*/8, enable);
        });

  // Set code object ABI version
  m.def("set_abi_version", [](llvm::Module *module, int version) {
    // Inject the control constant into the LLVM module so that device libraries
    // linked against module can resolve their references to it.
    llvm::Type *i32Ty = llvm::Type::getInt32Ty(module->getContext());
    llvm::GlobalVariable *abi = new llvm::GlobalVariable(
        *module, i32Ty, /*isConstant=*/true,
        llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
        llvm::ConstantInt::get(i32Ty, version), "__oclc_ABI_version", nullptr,
        llvm::GlobalValue::ThreadLocalMode::NotThreadLocal, 4);
    abi->setVisibility(llvm::GlobalValue::VisibilityTypes::ProtectedVisibility);
    abi->setAlignment(llvm::MaybeAlign(4));
    abi->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);

    // Also attach the control attribute on the LLVM module. This is also needed
    // in addition to the above for various transformations to know what code
    // object version we are targeting at.
    module->addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version",
                          version);
  });

  m.def("cleanup_bitcode_metadata", [](llvm::Module *module) {
    // We can have Clang version metadata from device libraries linked in. We
    // don't care about them so drop them.
    if (auto *ident = module->getNamedMetadata("llvm.ident"))
      module->eraseNamedMetadata(ident);
    // Also various OpenCL version details.
    if (auto *openclVersion = module->getNamedMetadata("opencl.ocl.version"))
      module->eraseNamedMetadata(openclVersion);
  });

  m.def("disable_print_inline", [](llvm::Module *module) {
    // List of functions name prefixes we want to forbid inline.
    std::array<const char *, 2> prefixes = {"__ockl_fprintf", "__ockl_printf"};

    for (llvm::Function &f : module->functions()) {
      if (!f.hasName())
        continue;
      llvm::StringRef name = f.getName();

      auto isNamePrefixed = [&name](const char *prefix) {
        return name.starts_with(prefix);
      };

      if (llvm::any_of(prefixes, isNamePrefixed))
        f.addFnAttr(llvm::Attribute::NoInline);
    }
  });

  m.def(
      "assemble_amdgcn",
      [](const std::string &assembly, const std::string &arch,
         const std::string &features) {
        std::string error;

        llvm::Triple triple(hcuTargetTriple);
        const llvm::Target *target =
            llvm::TargetRegistry::lookupTarget(triple, error);
        if (!target)
          throw std::runtime_error("target lookup error: " + error);

        llvm::SourceMgr srcMgr;
        srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(assembly),
                                  llvm::SMLoc());

        const llvm::MCTargetOptions mcOptions;
        std::unique_ptr<llvm::MCRegisterInfo> mri(
            target->createMCRegInfo(triple));
        std::unique_ptr<llvm::MCAsmInfo> mai(
            target->createMCAsmInfo(*mri, triple, mcOptions));
        std::unique_ptr<llvm::MCSubtargetInfo> sti(
            target->createMCSubtargetInfo(triple, arch, features));

        llvm::MCContext ctx(triple, mai.get(), mri.get(), sti.get(), &srcMgr,
                            &mcOptions);
        std::unique_ptr<llvm::MCObjectFileInfo> mofi(
            target->createMCObjectFileInfo(ctx, /*PIC=*/false,
                                           /*LargeCodeModel=*/false));
        ctx.setObjectFileInfo(mofi.get());

        llvm::SmallString<128> cwd;
        if (!llvm::sys::fs::current_path(cwd))
          ctx.setCompilationDir(cwd);

        llvm::SmallVector<char, 0> result;
        llvm::raw_svector_ostream svos(result);

        std::unique_ptr<llvm::MCStreamer> mcStreamer;
        std::unique_ptr<llvm::MCInstrInfo> mcii(target->createMCInstrInfo());

        std::unique_ptr<llvm::MCCodeEmitter> ce(
            target->createMCCodeEmitter(*mcii, ctx));
        std::unique_ptr<llvm::MCAsmBackend> mab(
            target->createMCAsmBackend(*sti, *mri, mcOptions));
        std::unique_ptr<llvm::MCObjectWriter> ow(mab->createObjectWriter(svos));
        mcStreamer.reset(target->createMCObjectStreamer(
            triple, ctx, std::move(mab), std::move(ow), std::move(ce), *sti));

        std::unique_ptr<llvm::MCAsmParser> parser(
            createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
        std::unique_ptr<llvm::MCTargetAsmParser> tap(
            target->createMCAsmParser(*sti, *parser, *mcii, mcOptions));
        if (!tap)
          throw std::runtime_error("assembler initializtion error");

        parser->setTargetParser(*tap);
        parser->Run(/*NoInitialTextSection=*/false);

        return py::bytes(std::string(result.begin(), result.end()));
      },
      py::return_value_policy::take_ownership);

  m.def("has_architected_sgprs", [](const std::string &arch) {
    std::string error;
    llvm::Triple triple(hcuTargetTriple);
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target)
      throw std::runtime_error("target lookup error: " + error);
    std::unique_ptr<llvm::MCSubtargetInfo> sti(
        target->createMCSubtargetInfo(triple, arch, ""));
    return sti->checkFeatures("+architected-sgprs");
  });

  m.def("supports_multi_cta_launch", [](const std::string &arch) {
    return mlir::triton::HCU::TargetInfo(arch).supportsMultiCTALaunch();
  });

  m.def("need_extern_lib", [](llvm::Module *module, const std::string &lib) {
    for (llvm::Function &f : module->functions()) {
      if (f.hasExternalLinkage() && f.hasName() && !f.hasExactDefinition()) {
        llvm::StringRef funcName = f.getName();
        // The rule for linking the extern lib:
        //    if the function name includes ocml or ockl, link
        //    ocml or ockl accordingly.
        if (funcName.contains(lib))
          return true;
        if (funcName.contains("__nv_")) {
          std::stringstream message;
          message << "Implicit conversion of CUDA " << funcName.str()
                  << " device function has been dropped; "
                  << "please, update your source program to use "
                     "triton.language.extra.<op> "
                  << "to replace triton.language.extra.cuda.<op>";
          throw std::runtime_error(message.str());
        }
      }
    }
    return false;
  });

  m.def("set_all_fn_arg_inreg", [](llvm::Function *fn) {
    for (llvm::Argument &arg : fn->args()) {
      // Check for incompatible attributes.
      if (arg.hasByRefAttr() || arg.hasNestAttr())
        continue;
      arg.addAttr(llvm::Attribute::InReg);
    }
  });

  m.def("link_hsaco",
        [](const std::string &inPath, const std::string &outPath) {
          if (auto errString = lldInvoke(inPath.c_str(), outPath.c_str()))
            throw std::runtime_error("LLD failed to link hsaco source " +
                                     inPath + " into object file " + outPath +
                                     " because " + errString.value());
        });

  m.def("add_scalarize_packed_fops_llvm_pass", [](llvm::Function *fn) {
    mlir::triton::HCU::runScalarizePackedFOpsPass(*fn);
  });

  auto hipBlas = m.def_submodule("hipblas");
  py::class_<HipblasLtInstance>(hipBlas, "HipblasLt")
      .def(py::init<>([&](py::object &workspace) {
        auto wrk_ptr = workspace.attr("data_ptr")().cast<uint64_t>();
        auto wrk_size = workspace.attr("numel")().cast<size_t>() *
                        workspace.attr("element_size")().cast<size_t>();
        return new HipblasLtInstance(wrk_ptr, wrk_size);
      }))
      .def("matmul",
           [](HipblasLtInstance &self, py::object &A, py::object &B,
              py::object &C) {
             auto A_ptr = A.attr("data_ptr")().cast<uint64_t>();
             auto B_ptr = B.attr("data_ptr")().cast<uint64_t>();
             auto C_ptr = C.attr("data_ptr")().cast<uint64_t>();
             auto init = initialize_hipblas_op(A, B, C, std::nullopt);
             self.matmul(init.m, init.n, init.k, A_ptr, B_ptr, C_ptr,
                         init.dtype, init.out_dtype);
           })
      .def("gemm", [](HipblasLtInstance &self, py::object &A, py::object &B,
                      py::object &C, py::object &D, float alpha, float beta) {
        auto A_ptr = A.attr("data_ptr")().cast<uint64_t>();
        auto B_ptr = B.attr("data_ptr")().cast<uint64_t>();
        auto C_ptr = C.attr("data_ptr")().cast<uint64_t>();
        auto D_ptr = D.attr("data_ptr")().cast<uint64_t>();
        auto init = initialize_hipblas_op(A, B, D, C);
        self.gemm(init.m, init.n, init.k, A_ptr, B_ptr, C_ptr, D_ptr,
                  init.dtype, init.out_dtype, alpha, beta);
      });
}
