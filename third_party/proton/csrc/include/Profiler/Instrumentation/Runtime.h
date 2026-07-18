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

#ifndef PROTON_PROFILER_INSTRUMENTATION_RUNTIME_H_
#define PROTON_PROFILER_INSTRUMENTATION_RUNTIME_H_

#include <cstddef>
#include <cstdlib>
#include <functional>

#include "Device.h"

namespace proton {
class Runtime {
public:
  Runtime(DeviceType deviceType) : deviceType(deviceType) {}
  virtual ~Runtime() = default;

  virtual void allocateHostBuffer(uint8_t **buffer, size_t size) = 0;

  virtual void freeHostBuffer(uint8_t *buffer) = 0;

  virtual uint64_t getDevice() = 0;

  virtual void *getPriorityStream() = 0;

  virtual void synchronizeStream(void *stream) = 0;

  virtual void
  processHostBuffer(uint8_t *hostBuffer, size_t hostBufferSize,
                    uint8_t *deviceBuffer, size_t deviceBufferSize,
                    void *stream,
                    std::function<void(uint8_t *, size_t)> callback) = 0;

  DeviceType getDeviceType() const { return deviceType; }

protected:
  DeviceType deviceType;
};
} // namespace proton

#endif // PROTON_PROFILER_INSTRUMENTATION_RUNTIME_H
