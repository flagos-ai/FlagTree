// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef PROTON_RUNTIME_HIP_RUNTIME_H_
#define PROTON_RUNTIME_HIP_RUNTIME_H_

#include "Runtime.h"
#include "Utility/Singleton.h"

namespace proton {

class HipRuntime : public Singleton<HipRuntime>, public Runtime {
public:
  HipRuntime() : Runtime(DeviceType::HIP) {}
  ~HipRuntime() = default;

  void launchKernel(void *kernel, unsigned int gridDimX, unsigned int gridDimY,
                    unsigned int gridDimZ, unsigned int blockDimX,
                    unsigned int blockDimY, unsigned int blockDimZ,
                    unsigned int sharedMemBytes, void *stream,
                    void **kernelParams, void **extra) override;
  void memset(void *devicePtr, uint32_t value, size_t size,
              void *stream) override;
  void allocateHostBuffer(uint8_t **buffer, size_t size, bool mapped) override;
  void getHostDevicePointer(uint8_t *hostPtr, uint8_t **devicePtr) override;
  void freeHostBuffer(uint8_t *buffer) override;
  void allocateDeviceBuffer(uint8_t **buffer, size_t size) override;
  void freeDeviceBuffer(uint8_t *buffer) override;
  void copyDeviceToHostAsync(void *dst, const void *src, size_t size,
                             void *stream) override;
  void *getDevice() override;
  void *getPriorityStream() override;
  void synchronizeStream(void *stream) override;
  void synchronizeDevice() override;
  void destroyStream(void *stream) override;
  void
  processHostBuffer(uint8_t *hostBuffer, size_t hostBufferSize,
                    uint8_t *deviceBuffer, size_t deviceBufferSize,
                    void *stream,
                    std::function<void(uint8_t *, size_t)> callback) override;
};

} // namespace proton

#endif // PROTON_RUNTIME_HIP_RUNTIME_H_
