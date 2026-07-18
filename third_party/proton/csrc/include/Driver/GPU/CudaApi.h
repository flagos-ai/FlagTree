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

#ifndef PROTON_DRIVER_GPU_CUDA_API_H_
#define PROTON_DRIVER_GPU_CUDA_API_H_

#include "Device.h"
#include "cuda.h"

namespace proton {

namespace cuda {

template <bool CheckSuccess> CUresult init(int flags);

template <bool CheckSuccess> CUresult ctxSynchronize();

template <bool CheckSuccess> CUresult ctxGetCurrent(CUcontext *pctx);

template <bool CheckSuccess> CUresult ctxGetDevice(CUdevice *device);

template <bool CheckSuccess>
CUresult ctxGetStreamPriorityRange(int *leastPriority, int *greatestPriority);

template <bool CheckSuccess>
CUresult deviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);

template <bool CheckSuccess> CUresult deviceGet(CUdevice *device, int ordinal);

template <bool CheckSuccess>
CUresult streamCreateWithPriority(CUstream *pStream, unsigned int flags,
                                  int priority);

template <bool CheckSuccess> CUresult streamSynchronize(CUstream stream);

template <bool CheckSuccess>
CUresult memcpyDToHAsync(void *dst, CUdeviceptr src, size_t count,
                         CUstream stream);

template <bool CheckSuccess> CUresult memAllocHost(void **pp, size_t bytesize);

template <bool CheckSuccess> CUresult memFreeHost(void *p);

Device getDevice(uint64_t index);

} // namespace cuda

} // namespace proton

#endif // PROTON_DRIVER_GPU_CUDA_API_H_
