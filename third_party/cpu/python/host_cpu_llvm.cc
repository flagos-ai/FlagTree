//===- host_cpu_llvm.cc - host-CPU LLVM helpers (FlagTree plugin) -*- C++ -*-=//
//
// Pybind plugin that injects host-CPU LLVM helper bindings
// (`translate_to_host_asm`, `get_cpu_name`, `get_cpu_features`,
// `set_host_target`) into the existing `triton._C.libtriton.llvm`
// submodule, so they appear as if defined in upstream python/src/llvm.cc
// without touching that file.
//
// Compiled and linked into libtriton.so only when FLAGTREE_BACKEND=cpu
// (gated by an `if(FLAGTREE_BACKEND STREQUAL "cpu")` block in the root
// CMakeLists.txt that registers this plugin via `add_triton_plugin`).
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using ret = py::return_value_policy;

// Defined in python/src/llvm.cc — external linkage, no header declaration,
// so we forward-declare it here and rely on both translation units landing
// in the same libtriton.so.
extern std::string translateLLVMIRToASM(llvm::Module &module,
                                        const std::string &triple,
                                        const std::string &proc,
                                        const std::string &features,
                                        const std::vector<std::string> &flags,
                                        bool enable_fp_fusion, bool isObject);

// Called from main.cc via the FlagTree plugin FOR_EACH_P(INIT_BACKEND, ...)
// dispatch. The plugin macro hands us our own submodule
// (`triton._C.libtriton.host_cpu_llvm`); we ignore it and inject these
// helpers into the already-initialized `triton._C.libtriton.llvm` so the
// public Python API matches the legacy in-tree placement.
void init_triton_host_cpu_llvm(py::module &&m_own) {
  (void)m_own;
  auto llvm_mod = py::module::import("triton._C.libtriton.llvm");

  llvm_mod.def(
      "translate_to_host_asm",
      [](std::string llvmIR, bool enable_fp_fusion) -> py::object {
        std::string res;
        {
          py::gil_scoped_release allow_threads;
          llvm::LLVMContext context;
          std::unique_ptr<llvm::MemoryBuffer> buffer =
              llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
          llvm::SMDiagnostic error;
          std::unique_ptr<llvm::Module> module =
              llvm::parseIR(buffer->getMemBufferRef(), error, context);
          if (!module) {
            llvm::report_fatal_error(
                "failed to parse IR: " + error.getMessage() +
                "lineno: " + std::to_string(error.getLineNo()));
          }
          std::string triple = llvm::sys::getDefaultTargetTriple();
          if (triple.empty()) {
            triple = llvm::sys::getProcessTriple();
          }
          res = translateLLVMIRToASM(*module, triple,
                                     llvm::sys::getHostCPUName().str(), "", {},
                                     enable_fp_fusion, false);
        }
        return py::str(res);
      },
      ret::take_ownership);

  llvm_mod.def("get_cpu_name",
               []() { return llvm::sys::getHostCPUName().str(); });

  llvm_mod.def("get_cpu_features", []() {
    auto features = llvm::sys::getHostCPUFeatures();
    std::set<std::string> res;
    for (auto &f : features) {
      if (f.second)
        res.insert(f.first().str());
    }
    return res;
  });

  llvm_mod.def("set_host_target", [](llvm::Module *mod) {
    std::string triple = llvm::sys::getDefaultTargetTriple();
    if (triple.empty()) {
      triple = llvm::sys::getProcessTriple();
    }
    mod->setTargetTriple(triple);
    std::string error;
    auto target =
        llvm::TargetRegistry::lookupTarget(mod->getTargetTriple(), error);
    if (!target) {
      throw std::runtime_error("target lookup error: " + error);
    }
    std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
        mod->getTargetTriple(), llvm::sys::getHostCPUName(), "", {},
        llvm::Reloc::PIC_));
    mod->setDataLayout(machine->createDataLayout());
  });
}
