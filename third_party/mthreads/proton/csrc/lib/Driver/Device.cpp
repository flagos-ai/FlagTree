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

#include "Device.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/HipApi.h"

#include "Utility/Errors.h"

namespace proton {

Device getDevice(DeviceType type, uint64_t index) {
  if (type == DeviceType::CUDA) {
    return cuda::getDevice(index);
  }
  if (type == DeviceType::HIP) {
    return hip::getDevice(index);
  }
  throw std::runtime_error("DeviceType not supported");
}

const std::string getDeviceTypeString(DeviceType type) {
  if (type == DeviceType::CUDA) {
    return DeviceTraits<DeviceType::CUDA>::name;
  } else if (type == DeviceType::HIP) {
    return DeviceTraits<DeviceType::HIP>::name;
  }
  throw std::runtime_error("DeviceType not supported");
}

} // namespace proton
