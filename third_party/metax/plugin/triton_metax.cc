#include "TritonMETAXGPUToLLVM/Passes.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#include "mlir/Dialect/LLVMIR/MACADialect.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#ifdef _WIN32
#define PLUGIN_EXPORT __declspec(dllexport)
#else
#define PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

PLUGIN_EXPORT void init_triton_metax_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  m.def("add_to_llvmir", [](mlir::PassManager &pm, int32_t capability) {
    pm.addPass(
        mlir::triton::METAX::createConvertTritonGPUToLLVMPass(capability));
  });
  m.def("add_builtin_func_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::METAX::createConvertBuiltinFuncToLLVMPass());
  });
  ADD_PASS_WRAPPER_4("add_accelerate_matmul",
                     mlir::createTritonMETAXGPUAccelerateMatmulPass, int, bool,
                     bool, int);
  ADD_PASS_WRAPPER_4("add_pipeline_maca",
                     mlir::createTritonMETAXGPUPipelineMACAPass, int, int, bool,
                     bool);
  ADD_PASS_WRAPPER_3("add_pipeline_async_base",
                     mlir::createTritonMETAXGPUPipelineAsyncBasePass, int, bool,
                     bool);
  ADD_PASS_WRAPPER_3("add_pipeline_async_tn",
                     mlir::createTritonMETAXGPUPipelineAsyncTNPass, int, int,
                     int);
  ADD_PASS_WRAPPER_1("add_pipeline_async_tt",
                     mlir::createTritonMETAXGPUPipelineAsyncTTPass, int);
  ADD_PASS_WRAPPER_0("add_tritonmetaxgpu_change_layout_from_repn_to_elemn_pass",
                     mlir::createTritonMETAXGPUChangeLayoutFromRepNToElemNPass);
  ADD_PASS_WRAPPER_1("add_tritonmetaxgpu_optimize_cstore_pass",
                     mlir::createTritonMETAXGPUOptimizeCStorePass, int);
  ADD_PASS_WRAPPER_2("add_tritonmetaxgpu_change_layout_for_int8_pass",
                     mlir::createTritonMETAXGPUChangeLayoutForInt8Pass, int,
                     const std::string);
  ADD_PASS_WRAPPER_0(
      "add_tritonmetaxgpu_change_layout_for_constancy_load_layout",
      mlir::createTritonMETAXGPUChangeLayoutForConstancyLoadPass);
  ADD_PASS_WRAPPER_0("add_tritonmetaxgpu_change_transop_graph_pass",
                     mlir::createTritonMETAXGPUChangeTransOpGraphPass);
  ADD_PASS_WRAPPER_3("add_tritonmetaxgpu_addptr_opt_pass",
                     mlir::createTritonMETAXGPUAddPtrOptPass, int, bool, bool);
  ADD_PASS_WRAPPER_1("add_tritonmetaxgpu_optimize_smem_usage",
                     mlir::createTritonMETAXGPUOptimizeSmemUsage, bool);
}

PLUGIN_EXPORT void init_triton_metax(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_metax_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::MACA::MACADialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def(
      "mlir_opt",
      [](const std::string mlir_src, std::string mlir_opt) -> py::object {
        llvm::SmallString<64> fsrc;
        llvm::sys::fs::createTemporaryFile("compile-maca-src", "ll", fsrc);
        const char *_fsrc = fsrc.c_str();
        llvm::FileRemover srcRemover(fsrc);

        std::ofstream ofs(_fsrc);
        ofs << mlir_src << std::endl;
        ofs.close();

        llvm::SmallString<64> flog;
        llvm::sys::fs::createTemporaryFile("compile-maca-log", "", flog);
        const char *_flog = flog.c_str();
        llvm::FileRemover logRemover(flog);

        llvm::SmallString<64> fdst;
        llvm::sys::fs::createTemporaryFile("compile-maca-dst", "", fdst);
        const char *_fdst = fdst.c_str();
        llvm::FileRemover dstRemover(fdst);
        std::string mlir_opt_options =
            " -split-input-file -allow-unregistered-dialect -eliminate-empty-tensors \
                              --convert-scf-to-cf --convert-cf-to-llvm --buffer-hoisting --buffer-loop-hoisting --buffer-results-to-out-params  \
                              --drop-equivalent-buffer-results --promote-buffers-to-stack --convert-linalg-to-parallel-loops ";

        std::string cmd = mlir_opt + mlir_opt_options + _fsrc + " -o " + _fdst +
                          " 2> " + _flog;
        if (std::getenv("TRITON_PRINT_COMPILE_OPTIONS")) {
          llvm::outs() << "mlir_opt compile options:" << cmd << "\n";
          srcRemover.releaseFile(); // don't remove src file for debug
        }
        int err;
        err = system(cmd.c_str());
        if (err != 0) {
          std::ifstream _log(_flog);
          std::string log(std::istreambuf_iterator<char>(_log), {});
          throw std::runtime_error("Internal Triton mlir opt error: \n" + log);
        }

        std::ifstream _fdst_stream(_fdst);
        std::string result(std::istreambuf_iterator<char>(_fdst_stream), {});
        _fdst_stream.close();
        py::str result_py(result);
        return std::move(result_py);
      });

  m.def(
      "translate_mlir_to_llir",
      [](const std::string mlir_src, std::string mlir_translate) -> py::object {
        llvm::SmallString<64> fsrc;
        llvm::sys::fs::createTemporaryFile("compile-maca-src", "ll", fsrc);
        const char *_fsrc = fsrc.c_str();
        llvm::FileRemover srcRemover(fsrc);

        std::ofstream ofs(_fsrc);
        ofs << mlir_src << std::endl;
        ofs.close();

        llvm::SmallString<64> flog;
        llvm::sys::fs::createTemporaryFile("compile-maca-log", "", flog);
        const char *_flog = flog.c_str();
        llvm::FileRemover logRemover(flog);

        llvm::SmallString<64> fdst;
        llvm::sys::fs::createTemporaryFile("compile-maca-dst", "", fdst);
        const char *_fdst = fdst.c_str();
        llvm::FileRemover dstRemover(fdst);

        std::string cmd =
            mlir_translate +
            " -mlir-to-llvmir -split-input-file -verify-diagnostics " + _fsrc +
            " -o " + _fdst + " 2> " + _flog;
        if (std::getenv("TRITON_PRINT_COMPILE_OPTIONS")) {
          llvm::outs() << "mlir_translate compile options:" << cmd << "\n";
          srcRemover.releaseFile(); // don't remove src file for debug
        }
        int err;
        err = system(cmd.c_str());
        if (err != 0) {
          std::ifstream _log(_flog);
          std::string log(std::istreambuf_iterator<char>(_log), {});
          throw std::runtime_error(
              "Internal Triton translate mlir to llir error: \n" + log);
        }

        std::ifstream _fdst_stream(_fdst);
        std::string result(std::istreambuf_iterator<char>(_fdst_stream), {});
        _fdst_stream.close();
        py::str result_py(result);
        return std::move(result_py);
      });

  m.def("translate_llvmir_to_mcfatbin",
        [](const std::string llvmIR, std::string mxcc_arch,
           std::string maca_path, std::string extra_option) -> py::object {
          py::gil_scoped_release allow_threads;

          // compile llvmir with mxcc
          llvm::SmallString<64> fsrc;
          llvm::sys::fs::createTemporaryFile("compile-maca-src", "ll", fsrc);
          const char *_fsrc = fsrc.c_str();
          llvm::FileRemover srcRemover(fsrc);

          llvm::SmallString<64> flog;
          llvm::sys::fs::createTemporaryFile("compile-maca-log", "", flog);
          const char *_flog = flog.c_str();
          llvm::FileRemover logRemover(flog);

          llvm::SmallString<64> fbin;
          llvm::sys::fs::createTemporaryFile("compile-maca-dst", "fatbin",
                                             fbin);
          const char *_fbin = fbin.c_str();
          llvm::FileRemover binRemover(fbin);

          std::ofstream ofs(_fsrc);
          ofs << llvmIR << std::endl;
          ofs.close();

          std::string cmd;
          std::string dump_str;
          std::string opt_str;
          if (std::getenv("TRITON_COMPILER_DUMP_ALL")) {
            dump_str = " --keep --fatbin ";
          } else {
            dump_str = " --fatbin ";
          }
          if (std::getenv("TRITON_ENABLE_COMPILER_OPT")) {
            opt_str = " -mllvm -metaxgpu-sched-regpressure=false -mllvm "
                      "-metaxgpu-PostRA-Scheduler=false -mllvm "
                      "-metaxgpu-mma-sched=true ";
          } else {
            opt_str = "";
          }

          if (std::getenv("TRITON_DISABLE_BSM_OFFSET_0") != nullptr) {
            cmd = mxcc_arch + dump_str + _fsrc + extra_option + opt_str +
                  " -maca-link -mllvm -metaxgpu-disable-bsm-offset=0 "
                  "-input-is-device --maca-path " +
                  maca_path + " -mllvm -inputFuncBC=" + maca_path +
                  "/lib/maca_kernellib.bc" + " -o " + _fbin + " 2> " + _flog;
          } else {
            cmd = mxcc_arch + dump_str + _fsrc + extra_option + opt_str +
                  " -maca-link -mllvm -metaxgpu-disable-bsm-offset=1 "
                  "-input-is-device --maca-path " +
                  maca_path + " -mllvm -inputFuncBC=" + maca_path +
                  "/lib/maca_kernellib.bc" + " -o " + _fbin + " 2> " + _flog;
          }
          if (std::getenv("TRITON_PRINT_COMPILE_OPTIONS")) {
            llvm::outs() << "mxcc compile options:" << cmd << "\n";
            srcRemover.releaseFile(); // don't remove src file for debug
          }
          int err;
          err = system(cmd.c_str());
          if (err != 0) {
            std::ifstream _log(_flog);
            std::string log(std::istreambuf_iterator<char>(_log), {});
            throw std::runtime_error("Internal Triton llir codegen error: \n" +
                                     log);
          }
          std::ifstream _fatbin(_fbin, std::ios::binary);
          std::string fatbin(std::istreambuf_iterator<char>(_fatbin), {});
          _fatbin.close();
          py::bytes bytes(fatbin);
          return std::move(bytes);
        });

  m.def("link_extern_libs",
        [](const std::string llvmIR, const std::vector<std::string> &paths,
           const std::string maca_path) -> py::object {
          std::string path;
          for (auto p : paths) {
            path += p + " ";
          }

          llvm::SmallString<64> fsrc;
          llvm::sys::fs::createTemporaryFile("compile-maca-src", "ll", fsrc);
          const char *_fsrc = fsrc.c_str();
          llvm::FileRemover srcRemover(fsrc);

          std::ofstream ofs(_fsrc);
          ofs << llvmIR << std::endl;
          ofs.close();

          llvm::SmallString<64> flog;
          llvm::sys::fs::createTemporaryFile("compile-maca-log", "", flog);
          const char *_flog = flog.c_str();
          llvm::FileRemover logRemover(flog);

          llvm::SmallString<64> fbc;
          llvm::sys::fs::createTemporaryFile("compile-maca-bc", "bc", fbc);
          const char *_fbc = fbc.c_str();
          llvm::FileRemover bcRemover(fbc);

          llvm::SmallString<64> fdst;
          llvm::sys::fs::createTemporaryFile("compile-maca-dst", "ll", fdst);
          const char *_fdst = fdst.c_str();
          llvm::FileRemover dstRemover(fdst);

          std::string link = maca_path + "/mxgpu_llvm/bin/llvm-link ";
          std::string dis = maca_path + "/mxgpu_llvm/bin/llvm-dis ";

          // link as bc
          std::string link_cmd = link + _fsrc + " " + path +
                                 " --only-needed -o " + _fbc + " 2> " + _flog;
          int err;
          err = system(link_cmd.c_str());
          if (err != 0) {
            std::ifstream _log(_flog);
            std::string log(std::istreambuf_iterator<char>(_log), {});
            throw std::runtime_error("Internal Triton link error: \n" + log);
          }

          // dis as llir
          std::string dis_cmd = dis + _fbc + " -o " + _fdst + " 2> " + _flog;
          err = system(dis_cmd.c_str());
          if (err != 0) {
            std::ifstream _log(_flog);
            std::string log(std::istreambuf_iterator<char>(_log), {});
            throw std::runtime_error("Internal Triton dis error: \n" + log);
          }

          std::ifstream _fdst_stream(_fdst);
          std::string result(std::istreambuf_iterator<char>(_fdst_stream), {});
          _fdst_stream.close();
          py::str result_py(result);
          return std::move(result_py);
        });
}
