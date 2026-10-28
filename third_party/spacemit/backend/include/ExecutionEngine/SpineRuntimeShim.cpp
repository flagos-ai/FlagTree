//===- SpineRuntimeShim.cpp - spine_malloc/spine_free C-ABI shim ----------===//
//
// Statically linked into each kernel .so so the kernel is self-contained for
// both native and RPC modes — no external .so required.
//
// spine-mlir's SpeRTtoLLVM.cc emits calls to spine_malloc / spine_free for
// non-thread-local memref.alloc. libspert (v0.6.0+) only exports the new-ABI
// spine_alloc_with_id / spine_free_with_id (different signatures), and
// spine-rpc-server links only libspert — so the legacy spine_malloc symbol
// would otherwise be undefined when the kernel .so is dlopen'd.
//
// Implementation mirrors spine-mlir runtime's SpeRTMemoryPool.cc:
// posix_memalign with 64-byte alignment, plain free.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdlib>

#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
#define EXPORT
#endif

// SpeRTtoLLVM.cc declares:
//   spine_malloc : i8Ptr (uintptr_t)
//   spine_free   : void (i8Ptr)
extern "C" {

EXPORT void *spine_malloc(uint64_t size) noexcept {
  void *ptr = nullptr;
  if (posix_memalign(&ptr, 64, size) != 0)
    return nullptr;
  return ptr;
}

EXPORT void spine_free(void *ptr) noexcept { free(ptr); }

} // extern "C"
