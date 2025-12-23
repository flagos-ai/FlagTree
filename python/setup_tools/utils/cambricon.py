import os
import shutil
import glob


def get_resources_url(resource_name):
    ...


def get_resources_hash(resource_name):
    ...


def get_package_data_tools():
    return ["mlu_compile.h", "mlu_compile.mlu", "mlu_compile.py", "mlu_link.py"]


def configure_packages_and_data(packages, package_dir, package_data, deps_dir):
    backend_triton_path = os.path.join("third_party", "cambricon", "python")

    libtriton_dst = os.path.join(backend_triton_path, "triton", "_C", "libtriton.so")
    os.makedirs(os.path.dirname(libtriton_dst), exist_ok=True)
    libtriton_src = os.path.join(deps_dir, "libtriton.so")
    shutil.copy(libtriton_src, libtriton_dst)

    if "triton._C" not in packages:
        packages.append("triton._C")
    package_dir["triton._C"] = os.path.join(backend_triton_path, "triton", "_C")
    package_data.setdefault("triton._C", []).append("libtriton.so")

    libdevice_bc_dst = os.path.join(backend_triton_path, "triton", "backends", "mlu", "lib")
    os.makedirs(libdevice_bc_dst, exist_ok=True)
    bc_files = glob.glob(os.path.join(deps_dir, "libdevice*.bc"))
    for bc_file in bc_files:
        shutil.copy(bc_file, libdevice_bc_dst)
    copied_bc_filenames = [os.path.basename(f) for f in bc_files]

    libdevice_package = "triton.backends.mlu.lib"
    if libdevice_package not in packages:
        packages.append(libdevice_package)
    package_dir[libdevice_package] = os.path.join(backend_triton_path, "triton", "backends", "mlu", "lib")
    for bc_filename in copied_bc_filenames:
        package_data.setdefault(libdevice_package, []).append(bc_filename)

    return packages, package_dir, package_data
