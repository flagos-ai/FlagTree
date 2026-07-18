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

#include "Driver/GPU/HsaApi.h"
#include "Driver/Dispatch.h"

namespace proton {

namespace hsa {

struct ExternLibHsa : public ExternLibBase {
  using RetType = hsa_status_t;
  static constexpr const char *name = "libhsa-runtime64.so";
  static constexpr const char *defaultDir = "";
  static constexpr RetType success = HSA_STATUS_SUCCESS;
  static void *lib;
};

void *ExternLibHsa::lib = nullptr;

DEFINE_DISPATCH(ExternLibHsa, agentGetInfo, hsa_agent_get_info, hsa_agent_t,
                hsa_agent_info_t, void *);

hsa_status_t iterateAgents(hsa_status_t (*callback)(hsa_agent_t agent,
                                                    void *data),
                           void *data) {
  typedef hsa_status_t (*hsa_iterate_agents_t)(
      hsa_status_t (*)(hsa_agent_t, void *), void *data);
  static hsa_iterate_agents_t func = nullptr;
  Dispatch<ExternLibHsa>::init(ExternLibHsa::name, &ExternLibHsa::lib);
  if (func == nullptr)
    func = reinterpret_cast<hsa_iterate_agents_t>(
        dlsym(ExternLibHsa::lib, "hsa_iterate_agents"));
  return (func ? func(callback, data) : HSA_STATUS_ERROR_FATAL);
}

} // namespace hsa

} // namespace proton
