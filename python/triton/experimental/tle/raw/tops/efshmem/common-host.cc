// Copyright 2025-     FlagOS Contributors
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
//
// Host-side shim for the efshmem (Enflame GCU NVSHMEM) init flow.
// Counterpart of triton/experimental/tle/raw/nvshmem/common-host.cu, built by
// topscc for the host and linked against libefshmem_host + topsrt.
// efshmem spells its NVSHMEM-derived API with an `efshmem`/`efshmemx` prefix.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <tops_runtime.h>

#include <efshmem.h>
#include <efshmemx.h>

#undef TOPS_CHECK
#define TOPS_CHECK(stmt)                                                       \
  do {                                                                         \
    topsError_t result = (stmt);                                               \
    if (topsSuccess != result) {                                               \
      fprintf(stderr, "[%s:%d] tops failed with %s \n", __FILE__, __LINE__,    \
              topsGetErrorString(result));                                     \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

extern "C" int efshmem_get_unique_id_bytes(void *uid_buffer,
                                           size_t uid_buffer_size) {
  if (uid_buffer_size < sizeof(efshmemx_uniqueid_t)) {
    return -1;
  }

  efshmemx_uniqueid_t uid;
  efshmemx_get_uniqueid(&uid);
  memcpy(uid_buffer, &uid, sizeof(uid));
  return 0;
}

extern "C" int efshmem_init_from_torch_distributed(int rank, int nranks,
                                                   int device,
                                                   void *uid_buffer,
                                                   size_t uid_buffer_size) {
  if (uid_buffer_size < sizeof(efshmemx_uniqueid_t)) {
    return -1;
  }

  TOPS_CHECK(topsSetDevice(device));

  efshmemx_uniqueid_t uid;
  memcpy(&uid, uid_buffer, sizeof(uid));

  efshmemx_init_attr_t attr = EFSHMEMX_INIT_ATTR_INITIALIZER;
  efshmemx_set_attr_uniqueid_args(rank, nranks, &uid, &attr);

  // `efshmemx_hostlib_init_attr` is the host-library entry point (the same one
  // efshmem's own test/host/init/efshmemx_hostlib_init_attr.cpp uses). Unlike
  // the `efshmemx_init_attr` static inline, it is a real symbol of
  // libefshmem_host.so, so the shim does not have to resolve the runtime's
  // `efshmemi_init_thread` -- and therefore does not have to link
  // libefshmem_device.a (whose object drags in the device fatbin reference).
  //
  // It only bootstraps: the device half is completed by efshmemx_cumodule_init()
  // on the loaded kernel, which the TLE-Raw kernel init hook performs.
  return efshmemx_hostlib_init_attr(EFSHMEMX_INIT_WITH_UNIQUEID, &attr);
}

extern "C" int efshmem_finalize_from_torch_distributed() {
  efshmemx_hostlib_finalize();

  return 0;
}
