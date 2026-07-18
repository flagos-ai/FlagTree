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

#ifndef PROTON_DRIVER_GPU_HSA_API_H_
#define PROTON_DRIVER_GPU_HSA_API_H_

#include "Device.h"
#include "hsa/hsa_ext_amd.h"

namespace proton {

namespace hsa {

template <bool CheckSuccess>
hsa_status_t agentGetInfo(hsa_agent_t agent, hsa_agent_info_t attribute,
                          void *value);

hsa_status_t iterateAgents(hsa_status_t (*callback)(hsa_agent_t agent,
                                                    void *data),
                           void *data);

} // namespace hsa

} // namespace proton

#endif // PROTON_DRIVER_GPU_HSA_API_H_
