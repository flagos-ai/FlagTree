#include <cuda.h>
#include <elf.h>

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

static const char *errorMessage(CUresult error) {
  const char *message = nullptr;
  cuGetErrorString(error, &message);
  return message ? message : "unknown driver error";
}

static void check(CUresult error, const char *operation) {
  if (error != CUDA_SUCCESS) {
    std::fprintf(stderr, "%s: %s (%d)\n", operation, errorMessage(error),
                 static_cast<int>(error));
    std::exit(2);
  }
}

static int nonnegativeInteger(const char *text, const char *name) {
  int value = 0;
  const char *end = text + std::strlen(text);
  auto result = std::from_chars(text, end, value);
  if (result.ec != std::errc{} || result.ptr != end || value < 0) {
    std::fprintf(stderr, "%s must be a nonnegative decimal integer\n", name);
    std::exit(2);
  }
  return value;
}

static int deviceAttribute(CUdevice device, CUdevice_attribute attribute) {
  int value = 0;
  check(cuDeviceGetAttribute(&value, attribute, device), "cuDeviceGetAttribute");
  return value;
}

static int functionAttribute(CUfunction function, CUfunction_attribute attribute) {
  int value = 0;
  check(cuFuncGetAttribute(&value, attribute, function), "cuFuncGetAttribute");
  return value;
}

int main(int argc, char **argv) {
  const bool deviceOnly = argc == 2 && std::strcmp(argv[1], "--device-info") == 0;
  if (!deviceOnly && argc != 5) {
    std::fprintf(stderr,
                 "usage: %s --device-info | CUBIN SYMBOL THREADS DYNAMIC_SHARED_BYTES\n",
                 argv[0]);
    return 2;
  }
  int threads = 0;
  int dynamicShared = 0;
  std::vector<char> cubin;
  if (!deviceOnly) {
    threads = nonnegativeInteger(argv[3], "threads");
    dynamicShared = nonnegativeInteger(argv[4], "dynamic shared bytes");
    if (!threads) {
      std::fprintf(stderr, "threads must be positive\n");
      return 2;
    }
    std::ifstream file(argv[1], std::ios::binary);
    if (!file) {
      std::fprintf(stderr, "cannot read cubin: %s\n", argv[1]);
      return 2;
    }
    if (file.peek() == std::ifstream::traits_type::eof()) {
      std::fprintf(stderr, "cubin is empty or unreadable: %s\n", argv[1]);
      return 2;
    }
    Elf64_Ehdr header{};
    file.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!file || std::memcmp(header.e_ident, ELFMAG, SELFMAG) != 0 ||
        header.e_ident[EI_CLASS] != ELFCLASS64 || header.e_ident[EI_DATA] != ELFDATA2LSB) {
      std::fprintf(stderr, "cubin is not a little-endian ELF64 file: %s\n", argv[1]);
      return 2;
    }
    file.seekg(0);
    cubin.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (file.bad() || cubin.empty()) {
      std::fprintf(stderr, "cubin is empty or unreadable: %s\n", argv[1]);
      return 2;
    }
  }
  if (setenv("CUDA_MODULE_LOADING", "0", 1) != 0) {
    std::perror("setenv CUDA_MODULE_LOADING");
    return 2;
  }
  const CUresult init = cuInit(0);
  if (init == CUDA_ERROR_NO_DEVICE) {
    std::fprintf(stderr, "GPU_UNAVAILABLE cuInit: %s (%d)\n",
                 errorMessage(init), static_cast<int>(init));
    return 77;
  }
  check(init, "cuInit");
  int count = 0;
  check(cuDeviceGetCount(&count), "cuDeviceGetCount");
  if (!count) {
    std::fprintf(stderr, "GPU_UNAVAILABLE cuDeviceGetCount: count=0\n");
    return 77;
  }
  CUdevice device;
  check(cuDeviceGet(&device, 0), "cuDeviceGet");
  char name[256] = {};
  check(cuDeviceGetName(name, sizeof(name), device), "cuDeviceGetName");
  const int warpSize = deviceAttribute(device, CU_DEVICE_ATTRIBUTE_WARP_SIZE);
  const int maxThreads = deviceAttribute(device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
  const int registersPerSm =
      deviceAttribute(device, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR);
  std::printf("query_only=1\nmodule_loading=0\ndevice_count=%d\ndevice_name=%s\n", count, name);
  std::printf("warp_size=%d\ndevice_max_threads_per_block=%d\nregisters_per_sm=%d\n",
              warpSize, maxThreads, registersPerSm);
  std::printf("sm_count=%d\nshared_bytes_per_sm=%d\n",
              deviceAttribute(device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
              deviceAttribute(device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR));
  if (deviceOnly)
    return 0;

  CUcontext context;
  check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
  check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
  CUmodule module;
  check(cuModuleLoadData(&module, cubin.data()), "cuModuleLoadData");
  CUfunction function;
  check(cuModuleGetFunction(&function, module, argv[2]), "cuModuleGetFunction");
  const int registers = functionAttribute(function, CU_FUNC_ATTRIBUTE_NUM_REGS);
  const int functionMaxThreads = functionAttribute(function, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
  std::printf("kernel=%s\nnum_regs=%d\nfunction_max_threads_per_block=%d\n",
              argv[2], registers, functionMaxThreads);
  std::printf("local_bytes=%d\nstatic_shared_bytes=%d\n",
              functionAttribute(function, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
              functionAttribute(function, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES));
  std::printf("requested_threads=%d\nrequested_dynamic_shared_bytes=%d\n",
              threads, dynamicShared);
  // Report the flat accounting separately from the driver's occupancy model;
  // disagreement is evidence to investigate, not permission to bypass admission.
  std::printf("raw_register_product=%lld\nthreads_within_device=%d\nthreads_within_function=%d\n",
              static_cast<long long>(registers) * threads,
              threads <= maxThreads, threads <= functionMaxThreads);
  std::fflush(stdout);
  int activeBlocks = 0;
  check(cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &activeBlocks, function, threads, static_cast<size_t>(dynamicShared)),
        "cuOccupancyMaxActiveBlocksPerMultiprocessor");
  std::printf("occupancy_blocks_per_sm=%d\n", activeBlocks);
  check(cuModuleUnload(module), "cuModuleUnload");
  check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
  return 0;
}
