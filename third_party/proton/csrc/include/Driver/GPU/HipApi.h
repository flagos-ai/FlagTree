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

#ifndef PROTON_DRIVER_GPU_HIP_API_H_
#define PROTON_DRIVER_GPU_HIP_API_H_

#include "Device.h"
#include "hip/hip_runtime_api.h"

namespace proton {

namespace hip {

template <bool CheckSuccess> hipError_t ctxGetDevice(hipDevice_t *device);

template <bool CheckSuccess> hipError_t deviceSynchronize();

template <bool CheckSuccess>
hipError_t deviceGetAttribute(int *value, hipDeviceAttribute_t attribute,
                              int deviceId);

template <bool CheckSuccess> hipError_t getDeviceCount(int *count);

template <bool CheckSuccess>
hipError_t getDeviceProperties(hipDeviceProp_t *prop, int deviceId);

Device getDevice(uint64_t index);

template <bool CheckSuccess>
hipError_t ctxGetStreamPriorityRange(int *leastPriority, int *greatestPriority);

template <bool CheckSuccess>
hipError_t streamCreateWithPriority(hipStream_t *pStream, unsigned int flags,
                                    int priority);

template <bool CheckSuccess> hipError_t streamSynchronize(hipStream_t stream);

template <bool CheckSuccess>
hipError_t memcpyDToHAsync(void *dst, hipDeviceptr_t src, size_t count,
                           hipStream_t stream);

const std::string getHipArchName(uint64_t index);

const char *getKernelNameRef(const hipFunction_t f);
const char *getKernelNameRefByPtr(const void *hostFunction, hipStream_t stream);

template <bool CheckSuccess>
hipError_t memAllocHost(void **pp, size_t bytesize);
template <bool CheckSuccess> hipError_t memFreeHost(void *p);

} // namespace hip

} // namespace proton

#endif // PROTON_DRIVER_GPU_HIP_API_H_
