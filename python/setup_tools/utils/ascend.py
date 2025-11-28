import os
import shutil
from pathlib import Path
from setup_tools.utils.tools import flagtree_root_dir, Module, flagtree_submodule_dir, DownloadManager

def precompile_hook_flir(*args, **kargs):
    default_backends = kargs["default_backends"]
    if 'amd' in default_backends:
        default_backends.remove('amd')
    default_backends.append('flir')

downloader = DownloadManager()

submodules = (Module(name="ascendnpu-ir", url="https://gitee.com/ascend/ascendnpu-ir.git",
                     commit_id="1922371c42749fda534d6395b7ed828b5c9f36d4",
                     dst_path=os.path.join(flagtree_submodule_dir, "ascend/third_party/ascendnpu-ir")), )

'''
def get_backend_cmake_args(*args, **kargs):
    build_ext = kargs['build_ext']
    src_ext_path = build_ext.get_ext_fullpath("triton-adapter-opt")
    src_ext_path = os.path.abspath(os.path.dirname(src_ext_path))
    return ["-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=" + src_ext_path]


def install_extension(*args, **kargs):
    build_ext = kargs['build_ext']
    src_ext_path = build_ext.get_ext_fullpath("triton-adapter-opt")
    src_ext_path = os.path.join(os.path.abspath(os.path.dirname(src_ext_path)), "triton-adapter-opt")
    python_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dst_ext_path = os.path.join(python_root_dir, "triton/backends/ascend/triton-adapter-opt")
    shutil.copy(src_ext_path, dst_ext_path)
'''
'''
def create_symlink_for_triton(link_map):
    for target, source in link_map.items():
        target_path = Path(os.path.join(flagtree_root_dir, "python", target))
        source_path = Path(os.path.join(flagtree_root_dir, source))
        if target_path.exists():
            if target_path.is_file():
                os.unlink(target_path)
            elif target_path.is_dir():
                shutil.rmtree(target_path)

        if source_path.is_dir():
            os.makedirs(target_path, exist_ok=True)
            for src_file in source_path.glob("*"):
                if src_file.is_file():
                    dest_file = target_path / src_file.name
                    os.symlink(src_file, dest_file)
                    print(f"Created symlink: {dest_file} -> {src_file}")
        elif source_path.is_file():
            if target_path.exists():
                os.unlink(target_path)
            os.symlink(source_path, target_path)
            print(f"Created symlink: {target_path} -> {source_path}")
        else:
            print("[ERROR]: wrong file mapping")


def cmake_patch_copy():
    src_path = os.path.join(flagtree_root_dir, "python/setup_tools/utils/src/ascend/CMakeLists.txt")
    if os.path.exists(os.path.join(flagtree_submodule_dir, "ascend")):
        dst_path = os.path.join(flagtree_submodule_dir, "ascend/triton-adapter/CMakeLists.txt")
    else:
        dst_path = os.path.join(flagtree_submodule_dir, "triton_ascend/ascend/triton-adapter/CMakeLists.txt")
    if not os.path.exists(src_path):
        raise RuntimeError(f"Source file {src_path} does not exist.")
    shutil.copyfile(src_path, dst_path)
    print(f"Copied {src_path} to {dst_path}")


def get_package_dir():
    package_dict = {}
    triton_patch_prefix_dir = os.path.join(flagtree_root_dir, "third_party/ascend/triton_patch/python/triton_patch")
    package_dict["triton/triton_patch"] = f"{triton_patch_prefix_dir}"
    package_dict["triton/triton_patch/language"] = f"{triton_patch_prefix_dir}/language"
    package_dict["triton/triton_patch/compiler"] = f"{triton_patch_prefix_dir}/compiler"
    package_dict["triton/triton_patch/runtime"] = f"{triton_patch_prefix_dir}/runtime"
    patch_paths = {
        "language/_utils.py",
        "compiler/compiler.py",
        "compiler/code_generator.py",
        "compiler/errors.py",
        "runtime/autotuner.py",
        "runtime/autotiling_tuner.py",
        "runtime/jit.py",
        "runtime/tile_generator.py",
        "runtime/utils.py",
        "runtime/libentry.py",
        "runtime/code_cache.py",
        "testing.py",
    }

    for path in patch_paths:
        package_dict[f"triton/{path}"] = f"{triton_patch_prefix_dir}/{path}"
    create_symlink_for_triton(package_dict)
    raise RuntimeError("will Fixed")
    return package_dict
'''
'''
def get_extra_install_packages():
    return [
        "triton/triton_patch",
        "triton/triton_patch/language",
        "triton/triton_patch/compiler",
        "triton/triton_patch/runtime",
    ]
'''

def is_compile_ascend_npu_ir():
    return os.getenv("ASCEND_NPU_IR_COMPILE", "1") == "1"

'''
def precompile_hook(*args, **kargs):
    third_party_base_dir = Path(kargs['third_party_base_dir'])
    ascend_path = Path(third_party_base_dir) / "ascend"
    patch_path = Path(ascend_path) / "triton_patch"
    project_path = Path(third_party_base_dir) / "triton_ascend"
    if os.path.exists(ascend_path):
        shutil.rmtree(ascend_path)
    if not os.path.exists(project_path):
        raise RuntimeError(f"{project_path} can't be found. It might be due to a network issue")
    ascend_src_path = Path(project_path) / "ascend"
    patch_src_path = Path(project_path) / "triton_patch"
    shutil.copytree(ascend_src_path, ascend_path, dirs_exist_ok=True)
    shutil.copytree(patch_src_path, patch_path, dirs_exist_ok=True)
    shutil.rmtree(project_path)
    [downloader.download(module=submodule, required=False) for submodule in submodules]
    cmake_patch_copy()
    patched_code = """  set(triton_abs_dir "${TRITON_ROOT_DIR}/include/triton/Dialect/Triton/IR") """
    src_code = """set(triton_abs_dir"""

    filepath = os.path.join(patch_path, "include/triton/Dialect/Triton/IR/CMakeLists.txt")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as tmp_file:
            with open(filepath, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if src_code in line:
                        tmp_file.writelines(patched_code)
                    else:
                        tmp_file.writelines(line)
        backup_path = str(filepath) + '.bak'
        if os.path.exists(backup_path):
            os.remove(backup_path)
        shutil.move(filepath, backup_path)
        shutil.move(tmp_file.name, filepath)
        print(f"[INFO]: {filepath} is patched")
        return True
    except PermissionError:
        print(f"[ERROR]: No permission to write to {filepath}!")
    except FileNotFoundError:
        print(f"[ERROR]: {filepath} does not exist!")
    except Exception as e:
        print(f"[ERROR]: Unknown error: {str(e)}")
    return False
'''
