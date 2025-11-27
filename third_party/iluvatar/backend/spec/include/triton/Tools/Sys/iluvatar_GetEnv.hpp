#ifndef ILUVATAR_TRITON_TOOLS_SYS_GETENV_HPP
#define ILUVATAR_TRITON_TOOLS_SYS_GETENV_HPP

#define FLAGTREE_SPEC_Tools_Sys_GetEnv_BACKEND_IR_ENABLE_DUMP                  \
  "ILUIR_ENABLE_DUMP"
#define FLAGTREE_SPEC_Tools_Sys_GetEnv_funtions

#include <dlfcn.h>
#include <filesystem>
#include <optional>

namespace fs = std::filesystem;

namespace mlir::triton {
namespace tools {

static fs::path &getCudaPath(void) {
  static fs::path cuda_path = [] {
    void *handle = dlopen("libnvrtc.so", RTLD_LAZY);
    if (!handle) {
      std::fprintf(stderr, "%s\n", dlerror());
      exit(EXIT_FAILURE);
    }
    void *pfunc = dlsym(handle, "nvrtcCompileProgram");
    Dl_info info;
    if (dladdr(pfunc, &info) == 0) {
      std::fprintf(stderr, "Failed to get symbol information: %s\n", dlerror());
      exit(EXIT_FAILURE);
    }
    return fs::path(info.dli_fname).parent_path().parent_path();
  }();
  return cuda_path;
}

static fs::path &getLinkerPath(void) {
  static fs::path linker_path = [] {
    fs::path cuda_path = getCudaPath();
    fs::path linker_path1 = cuda_path / "bin/ld.lld";
    fs::path linker_path2 = cuda_path / "../bin/ld.lld";
    if (!fs::exists(linker_path1)) {
      if (fs::exists(linker_path2)) {
        linker_path1 = linker_path2;
      } else {
        fprintf(stderr, "iluvatar linker not found in %s and %s\n",
                linker_path1.c_str(), linker_path2.c_str());
        exit(EXIT_FAILURE);
      }
    }
    return linker_path1;
  }();
  return linker_path;
}

} // namespace tools
} // namespace mlir::triton

#endif // ILUVATAR_TRITON_TOOLS_SYS_GETENV_HPP
