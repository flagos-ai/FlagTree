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

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include <dfca/dfca.h>

using TTThriveStream = dfcaStream_t;
using TTThriveFunction = dfcaKernel_t;
using TTThriveDim3 = dim3;

enum ValueKind { VALUE_KIND_BYVALUE = 0, VALUE_KIND_POINTER };

struct Arg_t {
  Arg_t(const void *data, const size_t size, const ValueKind vk)
      : Data(data), Size(size), VK(vk), ElementSize(1) {}
  Arg_t(const void *data, const size_t size, const size_t es,
        const ValueKind vk)
      : Data(data), Size(size), ElementSize(es), VK(vk) {}

  const void *Data;
  const size_t Size;
  const size_t ElementSize;
  const ValueKind VK;
};

#define CHECK_DFCA(expr)                                                       \
  do {                                                                         \
    dfcaError_t res = expr;                                                    \
    if (res != dfcaSuccess) {                                                  \
      fprintf(stderr, "%s:%d: DFCA Error: '%s' failed with result code %d\n",  \
              __FILE__, __LINE__, #expr, res);                                 \
    }                                                                          \
  } while (0)

static bool launchAsync(TTThriveFunction kernel, TTThriveDim3 clusterDim,
                        TTThriveDim3 blockDim, const std::vector<Arg_t> &vArgs,
                        size_t sharedMem, TTThriveStream stream,
                        bool useDshmem) {
  auto numArgs = vArgs.size();
  void *kArgs[numArgs];

  bool isSetCfg = false;
  dfcaDieConfig_t dieConfig({0, 0});
  dieConfig.type = dfcaDieConfigGrid;

  for (size_t i = 0; i < numArgs; i++) {
    const auto &arg = vArgs[i];
    if (arg.VK == VALUE_KIND_BYVALUE) {
      kArgs[i] = (void *)arg.Data;
    } else {
      dfcaMemory_t *memory = (dfcaMemory_t *)arg.Data;
      if (memory == nullptr) {
        kArgs[i] = (void *)&arg.Data;
      } else {
        kArgs[i] = (void *)memory;
        dfcaMemoryAttribute_t attr;
        CHECK_DFCA(dfcaMemGetAttribute(*memory, &attr));
        if (!isSetCfg) {
          isSetCfg = true;
          dieConfig = attr.dies;
        } else {
          assert(dieConfig == attr.dies &&
                 "[thrive] The die_config of all tensors must be consistent.");
        }
      }
    }
  }

  dfcaLaunchConfig_t launch_config;
  dfcaLaunchAttribute_t launch_attr[2];

  launch_config.dies = &dieConfig;
  launch_config.subgrid_dim = clusterDim;
  launch_config.block_dim = blockDim;
  launch_config.stream = stream;
  launch_config.dynamic_smem_bytes = 0;
  launch_config.attrs = launch_attr;
  launch_config.num_attrs = 0;

  launch_attr[0].id = dfcaLaunchAttributeClusterDim;
  launch_attr[0].val.cluster_dim.x = clusterDim.x; // 1
  launch_attr[0].val.cluster_dim.y = clusterDim.y; // 1
  launch_attr[0].val.cluster_dim.z = clusterDim.z; // 1
  launch_config.num_attrs += 1;

  if (useDshmem) {
    launch_attr[1].id = dfcaLaunchAttributeShmem;
    launch_attr[1].val.shmem_option.enable_shmem_routine = true;
    launch_attr[1].val.shmem_option.enable_pass_generic_buffer = true;
    launch_attr[1].val.shmem_option.device_malloc_heap_size_per_die = 0;
    launch_config.num_attrs += 1;
  }

  CHECK_DFCA(dfcaLaunchKernelExC(&launch_config, kernel, kArgs));
  return true;
}
