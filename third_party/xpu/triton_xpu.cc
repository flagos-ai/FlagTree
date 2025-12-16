//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include <cmath>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// clang-format off
#include "mlir/Dialect/Linalg/Passes.h"             // mlir::createLinalgElementwiseOpFusionPass()
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"        // LLVM::LLVMFuncOp
#include "mlir/Dialect/MemRef/Transforms/Passes.h"  // mlir::memref::createExpandStridedMetadataPass
#include "mlir/IR/BuiltinOps.h"                     // mlir::ModuleOp
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/SmallVector.h" // llvm::SmallVector
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"    // llvm::ConstantInt
#include "llvm/IR/LLVMContext.h"  // llvm::LLVMContext

#include "passes.h"

#include "triton/Conversion/TritonToTritonXPU/Passes.h"    // mlir::triton::createConvertTritonToTritonXPUPass
#include "triton/Conversion/TritonXPUToLLVM/Passes.h"      // mlir::triton::createConvertTritonXPUToLLVMPass
#include "triton/Dialect/TritonXPU/IR/Dialect.h"           // mlir::triton::xpu::TritonXPUDialect
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"    // mlir::createTritonXPUGM2LMPass()

#include "triton/Target/LLVMXPU/LLVMXPUToLLVMIRTranslation.h"  // registerLLVMXPUDialectTranslation
// clang-format on

namespace py = pybind11;

std::string translateLLVMIRToASM(llvm::Module &module,
                                 const std::string &triple,
                                 const std::string &proc,
                                 const std::string &features,
                                 const std::vector<std::string> &flags,
                                 bool enable_fp_fusion, bool isObject);

void init_triton_xpu_passes_conversion(py::module &&m) {
  m.def("add_convert_triton_to_tritonxpu_pass",
        [](mlir::PassManager &self, uint32_t xpu_arch, uint32_t buffer_size,
           uint32_t core_num) {
          self.addPass(mlir::triton::createConvertTritonToTritonXPUPass(
              xpu_arch, buffer_size, core_num));
        });

  m.def("add_convert_tritonxpu_to_llvm_pass",
        [](mlir::PassManager &self, uint32_t xpu_arch, uint32_t buffer_size,
           bool isUseMaskZero) {
          self.addPass(mlir::triton::createConvertTritonXPUToLLVMPass(
              xpu_arch, buffer_size, isUseMaskZero));
        });
}

void init_triton_xpu_passes_transform(py::module &&m) {
  // Function Pass
  m.def("add_tritonxpu_gm2lm_pass",
        [](mlir::PassManager &self, uint32_t xpu_arch, bool atomicSim,
           bool oneCoreActOnly, bool isUseMaskZero) {
          self.addPass(mlir::triton::xpu::createTritonXPUCreateGM2LM(
              {xpu_arch, atomicSim, oneCoreActOnly, isUseMaskZero}));
        });

  m.def("add_tritonxpu_legalize_pass",
        [](mlir::PassManager &self, uint32_t buffer_size, uint32_t core_num,
           uint32_t groups_per_cluster, bool isUseMaskZero) {
          self.addPass(mlir::triton::xpu::createTritonXPULegalize(
              {buffer_size, core_num, groups_per_cluster, isUseMaskZero}));
        });

  m.def("add_tritonxpu_mask_pass",
        [](mlir::PassManager &self, bool oneCoreActOnly, bool isUseMaskZero) {
          self.addPass(mlir::triton::xpu::createTritonXPUMask(
              {oneCoreActOnly, isUseMaskZero}));
        });

  m.def("add_tritonxpu_alloca_pass",
        [](mlir::PassManager &self, uint32_t buffer_size, uint32_t core_num) {
          self.addPass(mlir::triton::xpu::createTritonXPUAlloca(
              {buffer_size, core_num}));
        });

  m.def("add_tritonxpu_dtype_convert_pass", [](mlir::PassManager &self,
                                               uint32_t xpu_arch) {
    self.addPass(mlir::triton::xpu::createTritonXPUDtypeConvert({xpu_arch}));
  });

  m.def("add_tritonxpu_loop_grid_pass", [](mlir::PassManager &self) {
    self.addPass(mlir::triton::xpu::createTritonXPULoopGrid());
  });

  m.def("add_tritonxpu_print_pass", [](mlir::PassManager &self) {
    self.addPass(mlir::triton::xpu::createTritonXPUPrint());
  });

  m.def("add_tritonxpu_unroll_control_pass",
        [](mlir::PassManager &self, uint32_t buffer_size, uint32_t core_num,
           bool isUseMaskZero, uint32_t unroll_num) {
          self.addPass(mlir::triton::xpu::createTritonXPUUnrollControl(
              {buffer_size, core_num, isUseMaskZero, unroll_num}));
        });

  m.def("add_tritonxpu_other_sim_pass",
        [](mlir::PassManager &self, uint32_t buffer_size, uint32_t core_num) {
          self.addPass(mlir::triton::xpu::createTritonXPUOtherSim(
              {buffer_size, core_num}));
        });

  // Optimization Pass
  m.def("add_tritonxpu_offset_state_pass",
        [](mlir::PassManager &self, bool dump_flag, uint32_t buffer_size,
           bool isUseMaskZero) {
          self.addPass(mlir::triton::xpu::createTritonXPUOffsetAnalysis(
              {dump_flag, buffer_size, isUseMaskZero}));
        });

  m.def("add_tritonxpu_core_tiling_pass",
        [](mlir::PassManager &self, bool dump_flag, uint32_t buffer_size,
           uint32_t core_num, uint32_t groups_per_cluster) {
          self.addPass(mlir::triton::xpu::createTritonXPUCoreTiling(
              {dump_flag, buffer_size, core_num, groups_per_cluster}));
        });

  m.def("add_tritonxpu_vectorize_pass",
        [](mlir::PassManager &self, bool dump_flag, bool compare_fusion) {
          self.addPass(mlir::triton::xpu::createTritonXPUVectorize(
              {dump_flag, compare_fusion}));
        });

  m.def("add_tritonxpu_memory_async_pass", [](mlir::PassManager &self,
                                              bool dump_flag) {
    self.addPass(mlir::triton::xpu::createTritonXPUMemoryAsync({dump_flag}));
  });

  m.def("add_tritonxpu_interleave_pass", [](mlir::PassManager &self) {
    self.addPass(mlir::triton::xpu::createTritonXPUInterleave());
  });

  m.def("add_tritonxpu_store_control_pass", [](mlir::PassManager &self) {
    self.addPass(mlir::triton::xpu::createTritonXPUStoreControl());
  });

  m.def("add_tritonxpu_memory_cache_pass",
        [](mlir::PassManager &self, uint32_t buffer_size, uint32_t core_num) {
          self.addPass(mlir::triton::xpu::createTritonXPUMemoryCache(
              {buffer_size, core_num}));
        });

  m.def("add_tritonxpu_memory_inplace_pass",
        [](mlir::PassManager &self, uint32_t buffer_size, uint32_t core_num) {
          self.addPass(mlir::triton::xpu::createTritonXPUMemoryInplace(
              {buffer_size, core_num}));
        });

  m.def("add_tritonxpu_func_convert_pass", [](mlir::PassManager &self,
                                              bool to_mlir) {
    self.addPass(mlir::triton::xpu::createTritonXPUFuncConvert({to_mlir}));
  });

  m.def("add_tritonxpu_cf_to_scf_pass", [](mlir::PassManager &self) {
    self.addPass(mlir::triton::xpu::createTritonXPUCFToSCF());
  });
}

void init_triton_sdnn_passes_conversion(py::module &&m);
void init_triton_sdnn_passes_transform(py::module &&m);

namespace mlir::triton::xpu {

template <typename T> static bool hasUserOfType(Operation *op) {
  if (isa<T>(op))
    return true;

  for (auto user : op->getUsers()) {
    if (hasUserOfType<T>(user))
      return true;
  }

  return false;
}

// Describes XPU Metadata. It is used to record the XPU related meta
// information from mlir module.
struct XPUMetadata {
  int maxntidx{-1};
  bool isKernel{};
  // Free to extend with other information.
};

static void
extractXPUMetadata(mlir::ModuleOp module,
                   llvm::DenseMap<llvm::StringRef, XPUMetadata> *dic) {
  for (auto op : module.getOps<LLVM::LLVMFuncOp>()) {
    XPUMetadata meta;

    bool hasMetadata{};

    // maxntid
    if (op->hasAttr("xpu.maxntid")) {
      auto attr = op->getAttr("xpu.maxntid");
      meta.maxntidx = mlir::dyn_cast<IntegerAttr>(attr).getInt();
      hasMetadata = true;
    }

    // kernel
    if (op->hasAttr("xpu.kernel")) {
      meta.isKernel = true;
      hasMetadata = true;
    }

    if (hasMetadata)
      dic->try_emplace(op.getNameAttr().strref(), std::move(meta));
  }
}

// Add the xpu related metadata to LLVM IR.
static void amendLLVMFunc(llvm::Function *func, const XPUMetadata &metadata,
                          int xpu_arch) {
  auto *module = func->getParent();
  auto &ctx = func->getContext();
  auto targetArch = std::string("xpu") + std::to_string(xpu_arch);

  if (metadata.maxntidx > 0) {
    auto warps = llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32),
                                        llvm::APInt(32, metadata.maxntidx));

    llvm::Metadata *md_args[] = {llvm::ValueAsMetadata::get(func),
                                 llvm::MDString::get(ctx, "maxntidx"),
                                 llvm::ValueAsMetadata::get(warps)};

    module->getOrInsertNamedMetadata("xpu.annotations")
        ->addOperand(llvm::MDNode::get(ctx, md_args));
  }

  if (metadata.isKernel) {
    func->setDSOLocal(true);
    func->setCallingConv(llvm::CallingConv::XPU_KERNEL);
    llvm::Metadata *mdArgs[] = {
        llvm::ValueAsMetadata::get(func), llvm::MDString::get(ctx, "kernel"),
        llvm::ValueAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1))};
    module->getOrInsertNamedMetadata("xpu.annotations")
        ->addOperand(llvm::MDNode::get(ctx, mdArgs));
  }

  llvm::AttrBuilder funcAttrs(ctx);
  funcAttrs.addAttribute("correctly-rounded-divide-sqrt-fp-math", "false");
  funcAttrs.addAttribute("disable-tail-calls", "false");
  funcAttrs.addAttribute("frame-pointer", "all");
  funcAttrs.addAttribute("less-precise-fpmad", "false");
  funcAttrs.addAttribute("min-legal-vector-width", "0");
  funcAttrs.addAttribute("no-infs-fp-math", "false");
  funcAttrs.addAttribute("no-jump-tables", "false");
  funcAttrs.addAttribute("no-nans-fp-math", "false");
  funcAttrs.addAttribute("no-signed-zeros-fp-math", "false");
  funcAttrs.addAttribute("no-trapping-math", "false");
  funcAttrs.addAttribute("stack-protector-buffer-size", "8");
  funcAttrs.addAttribute("target-cpu", targetArch);
  funcAttrs.addAttribute("unsafe-fp-math", "false");
  funcAttrs.addAttribute("use-soft-float", "false");
  func->addFnAttrs(funcAttrs);
}

} // namespace mlir::triton::xpu

using ret = py::return_value_policy;

void init_triton_xpu_llvm(py::module &&m) {

  m.def("get_kernel_name", [](llvm::Module &mod) {
    for (auto &F : mod) {
      if (F.getCallingConv() == llvm::CallingConv::XPU_KERNEL) {
        std::string name = F.getName().str();
        return py::str(name);
      }
    }

    auto MD = mod.getNamedMetadata("xpu.annotations");
    std::string name;
    for (auto *Op : MD->operands()) {
      if (Op->getNumOperands() != 3)
        continue;
      auto *Prop = llvm::dyn_cast<llvm::MDString>(Op->getOperand(1));
      name = Prop->getString();
    }
    return py::str(name);
  });

  m.def("amend_func", [](llvm::Module *llvm_mod, mlir::ModuleOp mlir_mod,
                         llvm::LLVMContext &ctx, int xpu_arch,
                         bool isSdnnKernel = false) {
    llvm::DenseMap<llvm::StringRef, mlir::triton::xpu::XPUMetadata> XPUMetadata;
    extractXPUMetadata(mlir_mod, &XPUMetadata);

    for (auto &func : llvm_mod->functions()) {
      auto it = XPUMetadata.find(func.getName());
      if (it != XPUMetadata.end())
        mlir::triton::xpu::amendLLVMFunc(&func, it->second, xpu_arch);
      if (isSdnnKernel) {
        auto targetFeat = std::string("+cdnn") + std::to_string(xpu_arch);
        func.addFnAttr("target-features", targetFeat);
      }
    }
  });

  m.def("need_extern_lib", [](mlir::ModuleOp module) {
    llvm::SmallVector<mlir::LLVM::LLVMFuncOp> funcs;
    module.walk([&](mlir::LLVM::LLVMFuncOp func) {
      if (func.isExternal())
        funcs.push_back(func);
    });

    return funcs.empty() ? false : true;
  });

  m.def(
      "translate_to_asm",
      [](llvm::Module &module, std::string triple, std::string proc,
         std::string features, std::vector<std::string> flags,
         bool enable_fp_fusion, bool isObject) -> py::object {
        std::string obj;
        {
          // when allow_threads goes out of scope, gil will be released
          py::gil_scoped_release allow_threads;
          // create LLVM module from C++
          obj = translateLLVMIRToASM(module, triple, proc, features, flags,
                                     enable_fp_fusion, isObject);
        }
        if (isObject)
          return py::bytes(obj);
        else
          return py::str("");
      },
      ret::take_ownership);
}

extern void insertTritonSDNNDialect(mlir::DialectRegistry &registry);
extern void defineIsSDNNKernel(py::module &m);
extern void defineGetLutInfo(py::module &m);

void init_triton_xpu(py::module &&m) {
  m.doc() = "Python bindings to the XPU Triton backend";

  auto passes = m.def_submodule("passes");
  init_triton_xpu_passes_conversion(passes.def_submodule("ttxpuir"));
  init_triton_xpu_passes_transform(passes.def_submodule("ttxpuir"));
  init_triton_sdnn_passes_conversion(passes.def_submodule("ttsdnnir"));
  init_triton_sdnn_passes_transform(passes.def_submodule("ttsdnnir"));
  init_triton_xpu_llvm(m.def_submodule("llvm"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::xpu::TritonXPUDialect>();
    insertTritonSDNNDialect(registry);
    registerLLVMXPUDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  // check if it is a sdnn kernel
  defineIsSDNNKernel(m); // is_sdnn_kernel

  m.def("get_tensor_args", [](mlir::ModuleOp &mod,
                              std::vector<int64_t> &tensorArgs) {
    mod.walk([&](mlir::triton::FuncOp funcOp) {
      tensorArgs.clear();
      for (auto arg : funcOp.getArguments()) {
        auto ptrTy = mlir::dyn_cast<mlir::triton::PointerType>(arg.getType());
        if (!ptrTy || !ptrTy.getPointeeType().isBF16())
          continue;

        llvm::SmallVector<mlir::Value, 8> worklist = {arg};
        llvm::DenseSet<mlir::Value> visited;
        while (!worklist.empty()) {
          mlir::Value current = worklist.pop_back_val();
          if (!visited.insert(current).second)
            continue;

          for (auto &use : current.getUses()) {
            llvm::TypeSwitch<mlir::Operation *>(use.getOwner())
                .Case([&](mlir::triton::LoadOp loadOp) {
                  if (mlir::triton::xpu::hasUserOfType<mlir::triton::DotOp>(
                          loadOp))
                    tensorArgs.push_back(arg.getArgNumber());
                })
                .Case([&](mlir::RegionBranchOpInterface regionOp) {
                  for (mlir::Region &region : regionOp->getRegions()) {
                    for (mlir::Block &block : region) {
                      auto idx = block.getNumArguments() -
                                 regionOp->getNumOperands() +
                                 use.getOperandNumber();
                      worklist.push_back(block.getArgument(idx));
                    }
                  }
                })
                .Case([&](mlir::scf::YieldOp yieldOp) {
                  auto res =
                      yieldOp->getParentOp()->getResult(use.getOperandNumber());
                  worklist.push_back(res);
                })
                .Default([&](mlir::Operation *op) {
                  for (auto res : op->getResults()) {
                    worklist.push_back(res);
                  }
                });
          }
        }
      }
    });

    mod.walk([&](mlir::func::FuncOp funcOp) {
      std::vector<int64_t> _tensorArgs;
      for (auto argIdx : tensorArgs) {
        auto arg = funcOp.getArgument(argIdx);
        auto memrefTy = mlir::cast<mlir::UnrankedMemRefType>(arg.getType());
        if (memrefTy.getElementType().isF16()) {
          _tensorArgs.push_back(argIdx);
        }
      }
      tensorArgs = _tensorArgs;
    });

    return tensorArgs;
  });

  defineGetLutInfo(m);

  m.def("get_buffer_len", [](mlir::ModuleOp &mod, unsigned maxBufferSize,
                             int elemBytes) {
    unsigned bufferLen = maxBufferSize;

    auto _get_buffer_len = [&](mlir::Type &ptrTy, unsigned maxBufferSize,
                               unsigned &bufferLen) {
      mlir::Type ptrdataTy;
      if (auto ptrTensorTy = mlir::dyn_cast<mlir::RankedTensorType>(ptrTy)) {
        ptrdataTy =
            mlir::cast<mlir::triton::PointerType>(ptrTensorTy.getElementType())
                .getPointeeType();
      } else {
        ptrdataTy =
            mlir::cast<mlir::triton::PointerType>(ptrTy).getPointeeType();
      }
      if (ptrdataTy.isBF16() || ptrdataTy.isF16() || ptrdataTy.isF32()) {
        unsigned bitWidth = ptrdataTy.getIntOrFloatBitWidth();
        bufferLen = std::min(bufferLen, maxBufferSize / (bitWidth / 8));
      }
    };

    mod.walk([&](mlir::triton::LoadOp loadOp) {
      auto ptrTy = loadOp.getPtr().getType();
      _get_buffer_len(ptrTy, maxBufferSize, bufferLen);
    });
    mod.walk([&](mlir::triton::StoreOp storeOp) {
      auto ptrTy = storeOp.getPtr().getType();
      _get_buffer_len(ptrTy, maxBufferSize, bufferLen);
    });
    bufferLen = elemBytes == 0 ? bufferLen : maxBufferSize / elemBytes;
    return bufferLen;
  });
}
