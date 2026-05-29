"""
FlagTree CPU backend setup helper.

The CPU backend uses the standard host LLVM toolchain (x86_64, AArch64, etc.)
and does not require any special GPU SDK or custom LLVM build.
SLEEF (vectorized math) and OpenMP (parallel dispatch) are used if available.
"""


def get_resources_url(resource_name):
    """CPU backend uses system/pre-installed LLVM; no external downloads needed."""
    return ""


def get_resources_hash(resource_name):
    """No external resource hashes for the CPU backend."""
    return ""


def get_backend_cmake_args(*args, **kargs):
    """Return extra CMake arguments for the CPU backend build."""
    cmake_args = []
    # Disable proton (GPU-specific profiler) for CPU builds
    cmake_args.append("-DTRITON_BUILD_PROTON=OFF")
    return cmake_args


def install_extension(*args, **kargs):
    """No extra install steps needed for the CPU backend."""
    pass


def post_install():
    """No post-install patching needed for the CPU backend."""
    pass


def get_extra_install_packages():
    """Return extra packages for CPU backend."""
    return []


def get_package_data_tools():
    """Return package data for CPU backend."""
    return []


def get_package_dir():
    """Return package directory mapping for CPU backend."""
    return {}


def skip_package_dir(package):
    """Whether to skip a package directory."""
    return False


def configure_packages_and_data(packages, package_dir, package_data, deps_dir):
    """Configure packages and data for CPU backend."""
    return packages, package_dir, package_data
