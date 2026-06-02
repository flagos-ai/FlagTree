import sysconfig
import os
import shutil
import subprocess

# CPU backend only: ARM march flags, OpenMP linkage, and GCC-compat .s
# rewriting are guarded by this flag so GPU / other backends keep the
# upstream _build() behavior unchanged.
_IS_CPU_BACKEND = os.environ.get("FLAGTREE_BACKEND") == "cpu"
if _IS_CPU_BACKEND:
    import platform
    import re


def _build(name, src, srcdir, library_dirs, include_dirs, libraries):
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    # try to avoid setuptools if possible
    cc = os.environ.get("CC")
    if cc is None:
        # TODO: support more things here.
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
        if cc is None:
            raise RuntimeError("Failed to find C compiler. Please specify via CC environment variable.")
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    custom_backend_dirs = set(os.getenv(var) for var in ('TRITON_CUDACRT_PATH', 'TRITON_CUDART_PATH'))
    include_dirs = include_dirs + [srcdir, py_include_dir, *custom_backend_dirs]
    # for -Wno-psabi, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111047
    cc_cmd = [cc, src, "-O3", "-shared", "-fPIC", "-Wno-psabi", "-o", so]

    if _IS_CPU_BACKEND:
        # ARM ISA features for kernel assembly emitted by the CPU backend.
        machine = platform.machine()
        if src.endswith(".s") and machine in ("aarch64", "arm64"):
            cc_cmd += [
                "-march=armv9-a+sve2+i8mm+bf16+fp16",
                "-msve-vector-bits=128",
            ]

        # OpenMP for the C++ launcher so run_omp_kernels parallelizes grid
        # blocks across OMP threads (controlled by OMP_NUM_THREADS).
        if src.endswith(".cpp"):
            cc_cmd += ["-fopenmp"]
            # Link against the same libgomp that PyTorch uses.
            torch_lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "torch", "lib")
            if os.path.isdir(torch_lib_dir):
                cc_cmd += [f"-L{torch_lib_dir}", "-lgomp", f"-Wl,-rpath,{torch_lib_dir}"]
            else:
                cc_cmd += ["-lgomp"]

        # LLVM emits DWARF debug directives (.file / .loc / .cfi_*) that the
        # GCC assembler used here does not accept; strip them so the kernel
        # .s assembles cleanly.
        if src.endswith('.s'):
            with open(src, 'r') as f:
                asm_content = f.read()
            new_lines = []
            for line in asm_content.split('\n'):
                stripped = line.strip()
                if stripped.startswith('.file\t') or stripped.startswith('.loc\t') or stripped.startswith('.cfi_'):
                    continue
                line = line.replace("'\\t", "\t")
                new_lines.append(line)
            asm_content = '\n'.join(new_lines)
            asm_content = re.sub(r'\n\s*\n\s*\n', '\n\n', asm_content)
            with open(src, 'w') as f:
                f.write(asm_content)

    cc_cmd += [f'-l{lib}' for lib in libraries]
    cc_cmd += [f"-L{dir}" for dir in library_dirs]
    cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)
    return so
