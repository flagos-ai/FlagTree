import functools
import os
import shutil
import subprocess
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver, expand_signature as expand_descriptor_signature, wrap_handle_tensordesc_impl
from triton.runtime import _allocation
from triton.runtime.build import compile_module_from_src

dirname = os.path.dirname(os.path.realpath(__file__))


def _resolve_ifwa_home() -> str:
    env_path = os.environ.get("IFWA_HOME")
    if env_path:
        return os.path.realpath(env_path)

    vcc = shutil.which("vcc")
    if vcc:
        vcc_path = os.path.realpath(vcc)
        bin_dir = os.path.dirname(vcc_path)
        if os.path.basename(bin_dir) == "bin":
            return os.path.realpath(os.path.join(bin_dir, ".."))

    raise RuntimeError("Cannot resolve IFWA_HOME directory. "
                       "Please set IFWA_HOME, or ensure `vcc` is discoverable via PATH.")


libraries = ["libifwa.so"]

PyIfwaTensorMap = None
PyKernelArg = None
ARG_CONSTEXPR = None
ARG_KERNEL = None
ARG_TUPLE = None


@functools.lru_cache()
def libifwa_dirs():
    ifwa_home = _resolve_ifwa_home()

    candidate_dirs = [os.path.join(ifwa_home, "lib")]
    dirs = [d for d in candidate_dirs if os.path.exists(os.path.join(d, "libifwa.so"))]
    if dirs:
        return dirs

    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
    locs = [line.split()[-1] for line in libs.splitlines() if "libifwa.so" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path and not dirs:
        dirs = [d for d in env_ld_library_path.split(":") if os.path.exists(os.path.join(d, "libifwa.so"))]
    msg = "libifwa.so cannot be found!\n"
    if locs:
        msg += "Possible files are located at %s." % str(locs)
        msg += "Please create a symlink of libifwa.so to one of the files."
    else:
        msg += "Please set IFWA_HOME correctly or refresh loader cache with /sbin/ldconfig."
    assert any(os.path.exists(os.path.join(path, "libifwa.so")) for path in dirs), msg
    return dirs


@functools.lru_cache()
def library_dirs():
    return [*libifwa_dirs()]


class IfwaUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(IfwaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        ifwa_home = _resolve_ifwa_home()
        candidate_include = [os.path.join(ifwa_home, "include")]
        include_dirs = [d for d in candidate_include if os.path.exists(d)]
        if not include_dirs:
            include_dirs = candidate_include[:1]

        rpath_dirs = [os.path.join(ifwa_home, "lib")]
        mod = compile_module_from_src(
            src=Path(os.path.join(dirname, "driver.c")).read_text(),
            name="ifwa_utils",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
            ccflags=[
                *[f"-Wl,-rpath,{p}" for p in rpath_dirs],
            ],
        )
        global PyIfwaTensorMap
        global PyKernelArg
        global ARG_CONSTEXPR
        global ARG_KERNEL
        global ARG_TUPLE
        PyIfwaTensorMap = mod.PyIfwaTensorMap
        PyKernelArg = mod.PyKernelArg
        ARG_CONSTEXPR = mod.ARG_CONSTEXPR
        ARG_KERNEL = mod.ARG_KERNEL
        ARG_TUPLE = mod.ARG_TUPLE
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.get_max_active_clusters = mod.get_max_active_clusters
        self.set_printf_fifo_size = mod.set_printf_fifo_size
        self.launch = mod.launch
        self.build_signature_metadata = mod.build_signature_metadata
        self.fill_tma_descriptor_tiled = mod.fill_tma_descriptor_tiled
        self.fill_tma_descriptor_im2col = mod.fill_tma_descriptor_im2col


def ty_to_cpp(ty):
    if ty[0] == "*":
        return "CUdeviceptr"
    if ty.startswith("tensordesc"):
        return "CUtensorMap"
    return {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "double",
        "bf16": "double",
        "fp32": "double",
        "f32": "double",
        "fp64": "double",
        "TmaDesc": "CUtensorMap",
        "nvTmaDesc": "CUtensorMap",
    }[ty]


def expand_signature(signature, tensordesc_meta):
    return expand_descriptor_signature(signature, tensordesc_meta, "TmaDesc")


def _flatten_signature(signature):

    def _flatten(sig, output):
        if isinstance(sig, tuple):
            for x in sig:
                _flatten(x, output)
        else:
            output.append(sig)

    out = []
    for sig in signature:
        _flatten(sig, out)
    return out


def make_kernel_signature(signature):
    flat_signature = _flatten_signature(signature)
    kernel_signature = [x for x in flat_signature if x != "constexpr"]
    return triton.runtime.driver.active.utils.build_signature_metadata(kernel_signature)


def annotate_arguments(signature):
    annotated_arguments = []
    for sig in signature:
        if isinstance(sig, tuple):
            annotated_arguments.append((PyKernelArg(nested_tuple=annotate_arguments(sig), type=ARG_TUPLE)))
        elif sig != "constexpr":
            annotated_arguments.append(PyKernelArg(nested_tuple=None, type=ARG_KERNEL))
        else:
            annotated_arguments.append(PyKernelArg(nested_tuple=None, type=ARG_CONSTEXPR))
    return annotated_arguments


TMA_DTYPE_DEVICE_TO_HOST = dict((index, index) for index in range(16))
TMA_DTYPE_DEVICE_TO_HOST[8] = 10
TMA_DTYPE_DEVICE_TO_HOST[9] = 8
TMA_DTYPE_DEVICE_TO_HOST[10] = 9
TMA_TF32 = 11


def make_tensordesc_arg(arg, metadata, base_args=None):
    if metadata is None:
        return [
            arg.base,
            *arg.shape,
            *arg.strides,
            arg.padding == "nan",
            arg.round_f32_to_tf32,
            *arg.shape,
            *arg.strides,
        ]

    swizzle = metadata["swizzle"]
    elem_size = metadata["elem_size"]
    elem_type = metadata["elem_type"]
    block_size = metadata["block_size"]
    fp4_padded = metadata["fp4_padded"]
    is_im2col = metadata.get("is_im2col", False)

    shape = arg.shape
    strides = arg.strides
    assert strides[-1] == 1
    padding = 1 if arg.padding == "nan" else 0
    expanded_shape = list(shape)
    if fp4_padded:
        expanded_shape[-1] *= 2
    if arg.round_f32_to_tf32:
        elem_type = TMA_TF32

    if is_im2col:
        element_strides = arg.element_strides if arg.element_strides is not None else [1] * len(shape)
        tensor_map = triton.runtime.driver.active.utils.fill_tma_descriptor_im2col(
            arg.base.data_ptr(),
            swizzle,
            elem_size,
            TMA_DTYPE_DEVICE_TO_HOST[elem_type],
            block_size,
            expanded_shape,
            strides,
            padding,
            arg.pixel_box_lower_corner,
            arg.pixel_box_upper_corner,
            element_strides,
        )
    else:
        tensor_map = triton.runtime.driver.active.utils.fill_tma_descriptor_tiled(
            arg.base.data_ptr(),
            swizzle,
            elem_size,
            TMA_DTYPE_DEVICE_TO_HOST[elem_type],
            block_size,
            expanded_shape,
            strides,
            padding,
        )
    return [tensor_map, *shape, *strides]


def wrap_handle_tensordesc(launcher, signature, tensordesc_meta):
    return wrap_handle_tensordesc_impl(launcher, signature, tensordesc_meta or [], make_tensordesc_arg)


class IfwaLauncher(object):

    def __init__(self, src, metadata):
        signature = {idx: value for idx, value in src.signature.items()}
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)

        launcher = triton.runtime.driver.active.utils.launch
        expanded_signature = expand_signature(signature.values(), tensordesc_meta)
        self.arg_annotations = annotate_arguments(expanded_signature)
        self.kernel_signature = make_kernel_signature(expanded_signature)
        self.num_ctas = getattr(metadata, "num_ctas", 1)
        self.launch = wrap_handle_tensordesc(launcher, signature, tensordesc_meta)
        self.global_scratch_size = getattr(metadata, "global_scratch_size", 0)
        self.global_scratch_align = getattr(metadata, "global_scratch_align", 16)
        self.profile_scratch_size = getattr(metadata, "profile_scratch_size", 0)
        self.profile_scratch_align = getattr(metadata, "profile_scratch_align", 16)
        self.launch_cooperative_grid = getattr(metadata, "launch_cooperative_grid", False)
        self.launch_pdl = getattr(metadata, "launch_pdl", False)

    def __call__(
        self,
        gridX,
        gridY,
        gridZ,
        stream,
        function,
        kernel_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *args,
    ):

        def allocate_scratch(size, align, allocator):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * self.num_ctas * size
                alloc_fn = allocator.get()
                return alloc_fn(alloc_size, align, stream)
            return None

        global_scratch = allocate_scratch(self.global_scratch_size, self.global_scratch_align, _allocation._allocator)
        profile_scratch = allocate_scratch(
            self.profile_scratch_size,
            self.profile_scratch_align,
            _allocation._profile_allocator,
        )

        self.launch(
            gridX,
            gridY,
            gridZ,
            stream,
            function,
            self.launch_cooperative_grid,
            self.launch_pdl,
            kernel_metadata,
            launch_metadata,
            launch_enter_hook,
            launch_exit_hook,
            global_scratch,
            profile_scratch,
            self.arg_annotations,
            self.kernel_signature,
            args,
        )


class IfwaDriver(GPUDriver):

    def __init__(self):
        self.utils = IfwaUtils()
        self.launcher_cls = IfwaLauncher
        super().__init__()
        import torch

        # Replace the base driver's default device helpers with IFWA bindings.
        self.get_device_capability = torch.ifwa.get_device_capability
        self.get_current_device = torch.ifwa.current_device
        self.set_current_device = torch.ifwa.set_device
        self.get_current_stream = lambda idx: torch.ifwa.current_stream(idx).ifwa_stream

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        warp_size = 32
        return GPUTarget("ifwa", str(capability), warp_size)

    def get_device_interface(self):
        import torch
        return torch.ifwa

    @staticmethod
    def is_active():
        try:
            import torch
            return torch.ifwa.is_available()
        except (ImportError, AttributeError):
            return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_active_torch_device(self):
        import torch

        return torch.device("ifwa", self.get_current_device())

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device="ifwa")

    def clear_cache(self, cache):
        cache.zero_()
