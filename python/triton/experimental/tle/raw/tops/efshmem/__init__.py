# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""efshmem (Enflame GCU NVSHMEM) support for TLE-Raw TOPS kernels.

efshmem is the Enflame GCU port of NVSHMEM with an `efshmem`/`efshmemx` naming
scheme (upstream NVIDIA spells the same API `nvshmem`/`nvshmemx`): same
device/host API, `tops*` runtime calls instead of `cuda*`, and a
`libefshmem_device.bc` built for the TOPS target. This package mirrors
`triton.experimental.tle.raw.nvshmem` for the CUDA backend, and helpers are
imported from `...tops.efshmem.utils` just like the CUDA ones are imported from
`...nvshmem.utils`:

    utils.py         - EFShmem home/library resolution, host shim build+load,
                       torch distributed + efshmem init, tensor views.
    common-host.cc   - C shim around the unique-id init/finalize flow.

Typical use:

    @dialect(name="tops", file=Path("kernel.tops"), library="efshmem",
             extern_func_name="my_kernel", deferred=True)
    def my_kernel(*args, **kwargs):
        ...

    group = init_torch_distributed()
    host = load_host(Path("host.cc"))
    common = load_common_host()
    init_efshmem_by_torch_pg(common, group)
"""
