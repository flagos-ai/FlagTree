/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/* clang-format off */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <hggc.h>


// helpers to check for hggc errors
#define HGGC_CHECK(ans) {{\
    gpuAssert((ans), __FILE__, __LINE__);\
  }}\

static inline void gpuAssert(HGresult code, const char *file, int line) {{
  if (code != HGGC_SUCCESS) {{
    const char *prefix = "Triton Error [HGGC]: ";
    const char *str;
    hgGetErrorString(code, &str);
    char err[1024] = {{0}};
    strcat(err, prefix);
    strcat(err, str);
    printf("%s\\n", err);
    exit(code);
  }}
}}

// globals
#define HGBIN_NAME {kernel_name}_hgbin
HGmodule {kernel_name}_mod = NULL;
HGfunction {kernel_name}_func = NULL;
unsigned char HGBIN_NAME[{bin_size}] = {{ {bin_data} }};


void unload_{kernel_name}(void) {{
    HGGC_CHECK(hgModuleUnload({kernel_name}_mod));
}}

void load_{kernel_name}() {{
    int dev = 0;
    void *bin = (void *)&HGBIN_NAME;
    int shared = {shared};
    HGGC_CHECK(hgModuleLoadData(&{kernel_name}_mod, bin));
    HGGC_CHECK(hgModuleGetFunction(&{kernel_name}_func, {kernel_name}_mod, "{triton_kernel_name}"));
    // set dynamic shared memory if necessary
    int shared_optin;
    HGGC_CHECK(hgDeviceGetAttribute(&shared_optin, HG_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev));
    if (shared > 49152 && shared_optin > 49152) {{
      HGGC_CHECK(hgFuncSetCacheConfig({kernel_name}_func, HG_FUNC_CACHE_PREFER_SHARED));
      HGGC_CHECK(hgFuncSetAttribute({kernel_name}_func, HG_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_optin))
    }}
}}

/*
{kernel_docstring}
*/
HGresult {kernel_name}(HGstream stream, {signature}) {{
    if ({kernel_name}_func == NULL)
       load_{kernel_name}();
    unsigned int gX = {gridX};
    unsigned int gY = {gridY};
    unsigned int gZ = {gridZ};
    HGdeviceptr global_scratch = 0;
    HGdeviceptr profile_scratch = 0;
    void *args[{num_args}] = {{ {arg_pointers} }};
    // TODO: shared memory
    if(gX * gY * gZ > 0)
      return hgLaunchKernel({kernel_name}_func, gX, gY, gZ, {num_warps} * {warp_size}, 1, 1, {shared}, stream, args, NULL);
}}
