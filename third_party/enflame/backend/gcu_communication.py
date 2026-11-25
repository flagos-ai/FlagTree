# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""GCU implementation of the TLE FlagCX communication runtime hooks.

`triton.experimental.tle.language.communication` is backend neutral: it asks
`flagcx_rt_conf.communication_runtime` (optional) for the device module,
process-group backend, allocator and memory-pool details. The nvidia/iluvatar
builds leave that attribute unset and keep the CUDA/NCCL path; the enflame
backend publishes this object from its `distributed.py`.
"""

import os
import subprocess
import tempfile


class GcuCommunicationRuntime:
    """Runtime hooks for the GCU backend (torch.gcu + ECCL)."""

    name = "gcu"

    @staticmethod
    def _torch():
        import torch
        return torch

    def device_module(self):
        # torch_gcu exposes its runtime as torch.gcu; torch.cuda is NOT an
        # alias of it, so every device call goes through this module.
        torch = self._torch()
        return torch.gcu if hasattr(torch, "gcu") else torch.cuda

    def dist_backend(self):
        """ProcessGroup backend for the Bootstrap (unique-id broadcast only)."""
        override = os.environ.get("FLAGCX_TORCH_DIST_BACKEND")
        if override:
            return override
        try:
            import ctypes
            ctypes.CDLL("libefml.so")
        except OSError:
            return "gloo"
        return "cpu:gloo,gcu:eccl"

    def set_device(self, local_rank):
        device_module = self.device_module()
        count = device_module.device_count()
        index = local_rank % count if count > 0 else local_rank
        device_module.set_device(index)
        return index

    def pluggable_allocator(self, so_file, alloc_name, free_name):
        return self.device_module().GCUPluggableAllocator(so_file, alloc_name, free_name)

    def make_mem_pool(self, allocator):
        # no_split keeps every allocation its own segment, which preserves the
        # 4KB alignment the ECCL symmetric window needs.
        return self.device_module().MemPool(allocator, no_split=True)

    def prepare_uid_tensor(self, id_bytes):
        # The composite cpu:gloo backend carries the 256-byte unique id, so the
        # tensor stays on the host.
        torch = self._torch()
        return torch.frombuffer(bytearray(id_bytes), dtype=torch.uint8)

    def compile_allocator(self, lib_path, include_path, lib_dir):
        """Build the pluggable allocator .so with gcc (no GCU load_inline path).

        The CCL allocator maps to ecclMemAlloc, whose memory is what ECCL
        symmetric windows can be registered on.
        """
        if os.path.isfile(lib_path):
            return
        tops_home = _tops_home()
        source_path = os.path.join(tempfile.gettempdir(), "flagcx_allocator_gcu.c")
        with open(source_path, "w", encoding="utf-8") as stream:
            stream.write(ALLOCATOR_SOURCE_C)
        compiler = os.environ.get("CC", "gcc")
        cmd = [
            compiler,
            "-shared",
            "-fPIC",
            "-O2",
            "-I",
            include_path,
            "-I",
            os.path.join(tops_home, "include"),
            source_path,
            "-L",
            lib_dir,
            "-lflagcx",
            "-L",
            os.path.join(tops_home, "lib"),
            "-ltopsrt",
            f"-Wl,-rpath,{lib_dir}",
            f"-Wl,-rpath,{os.path.join(tops_home, 'lib')}",
            "-o",
            lib_path,
        ]
        subprocess.check_call(cmd)


def _tops_home():
    try:
        from triton.backends.enflame import toolkit as _toolkit
        return str(_toolkit.get_tops_home())
    except Exception:
        return os.environ.get("TOPS_HOME", "/opt/tops")


ALLOCATOR_SOURCE_C = """
#include <stddef.h>
#include <flagcx.h>

void* flagcx_alloc_plug(size_t size, int device, void* stream) {
  (void)device;
  (void)stream;
  void* ptr = NULL;
  if (flagcxMemAlloc(&ptr, size, flagcxMemCCL) != flagcxSuccess) {
    return NULL;
  }
  return ptr;
}

void flagcx_free_plug(void* ptr, size_t size, int device, void* stream) {
  (void)size;
  (void)device;
  (void)stream;
  if (ptr != NULL) {
    flagcxMemFree(ptr, flagcxMemCCL);
  }
}
"""
