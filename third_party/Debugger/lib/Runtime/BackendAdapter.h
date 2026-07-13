#pragma once

#include "Debugger/Runtime/TransferEngine.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace mlir {
namespace flagtree {
namespace debugger {

class RuntimeBackendAdapter {
public:
  virtual ~RuntimeBackendAdapter() = default;

  virtual TransferDriverKind driverKind() const = 0;
  virtual const char *name() const = 0;
  virtual bool isAvailable() const = 0;

  virtual void *allocateDevice(size_t bytes) = 0;
  virtual void freeDevice(void *ptr) = 0;

  virtual void *allocateHost(size_t bytes) = 0;
  virtual void freeHost(void *ptr) = 0;

  virtual void memsetDevice(void *ptr, int value, size_t bytes,
                            uint64_t streamHandle) = 0;
  virtual void copyHostToDevice(void *deviceDst, const void *hostSrc,
                                size_t bytes, uint64_t streamHandle) = 0;
  virtual void copyDeviceToHost(void *hostDst, const void *deviceSrc,
                                size_t bytes, uint64_t streamHandle) = 0;
  virtual void synchronize(uint64_t streamHandle) = 0;
};

std::unique_ptr<RuntimeBackendAdapter>
createRuntimeBackendAdapter(const TransferEngineOptions &options);

} // namespace debugger
} // namespace flagtree
} // namespace mlir
