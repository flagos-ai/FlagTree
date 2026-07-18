// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "Profiler/Instrumentation/CudaRuntime.h"

#include "Driver/GPU/CudaApi.h"
#include <stdexcept>
namespace proton {

void CudaRuntime::allocateHostBuffer(uint8_t **buffer, size_t size) {
  cuda::memAllocHost<true>(reinterpret_cast<void **>(buffer), size);
}

void CudaRuntime::freeHostBuffer(uint8_t *buffer) {
  cuda::memFreeHost<true>(buffer);
}

uint64_t CudaRuntime::getDevice() {
  CUdevice device;
  cuda::ctxGetDevice<true>(&device);
  return static_cast<uint64_t>(device);
}

void *CudaRuntime::getPriorityStream() {
  CUstream stream;
  // TODO: Change priority
  int lowestPriority, highestPriority;
  cuda::ctxGetStreamPriorityRange<true>(&lowestPriority, &highestPriority);
  cuda::streamCreateWithPriority<true>(&stream, CU_STREAM_NON_BLOCKING,
                                       highestPriority);
  return reinterpret_cast<void *>(stream);
}

void CudaRuntime::synchronizeStream(void *stream) {
  cuda::streamSynchronize<true>(reinterpret_cast<CUstream>(stream));
}

void CudaRuntime::processHostBuffer(
    uint8_t *hostBuffer, size_t hostBufferSize, uint8_t *deviceBuffer,
    size_t deviceBufferSize, void *stream,
    std::function<void(uint8_t *, size_t)> callback) {
  int64_t chunkSize = std::min(hostBufferSize, deviceBufferSize);
  int64_t sizeLeftOnDevice = deviceBufferSize;
  while (chunkSize > 0) {
    cuda::memcpyDToHAsync<true>(reinterpret_cast<void *>(hostBuffer),
                                reinterpret_cast<CUdeviceptr>(deviceBuffer),
                                chunkSize, reinterpret_cast<CUstream>(stream));
    // We should not use synchronization here in general if we want to copy
    // buffer while the kernel is running. But for the sake of simplicity, we
    // only copy the buffer after the kernel is finished for now.
    cuda::streamSynchronize<true>(reinterpret_cast<CUstream>(stream));
    callback(hostBuffer, chunkSize);
    sizeLeftOnDevice -= chunkSize;
    chunkSize =
        std::min(static_cast<int64_t>(hostBufferSize), sizeLeftOnDevice);
  }
}

} // namespace proton
