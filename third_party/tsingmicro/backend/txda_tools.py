import os
import shutil
import subprocess
import hashlib
from posixpath import dirname

from triton.backends.tsingmicro.logger_config import setup_logger

logger = setup_logger("tsingmicro_launch")

_dump_dir_cache = None
dump_cmd_count = 0


def _get_dump_env_path():
    path = os.getenv("TRITON_DUMP_PATH", "")
    if not path:
        return ""
    os.makedirs(path, exist_ok=True)
    return path


def get_dump_dir():
    global _dump_dir_cache
    # 如果已缓存，直接返回
    if _dump_dir_cache is not None:
        return _dump_dir_cache

    base_dir = _get_dump_env_path()
    # 查找第一个不存在的dumpN目录
    index = 1
    while True:
        dir_name = f"dump{index}"
        full_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            _dump_dir_cache = full_path  # 缓存结果
            logger.debug(f"make dump dir:{full_path}")
            break
        index += 1
    return _dump_dir_cache


def runLoweringCmd(destFile: str, args: list):
    isAlwaysCompile = os.getenv("TRITON_ALWAYS_COMPILE", "0").lower() in ("1", "true", "yes")
    if isAlwaysCompile or not os.path.exists(destFile):
        if os.getenv("MLIR_ENABLE_DUMP", "0") == "1":
            subprocess.check_call(args, stderr=subprocess.STDOUT)
        else:
            subprocess.check_call(args, stdout=subprocess.DEVNULL)
    else:
        logger.debug(f"Skip lowering {destFile}")


def is_use_profile():
    return os.getenv("ENABLE_PROFILING", "0").lower() in ("1", "true", "yes")


def is_enable_kernel_file_cache():
    return os.getenv("ENABLE_KERNEL_FILE_CACHE", "1").lower() in ("1", "true", "yes")


def get_kernel_cache_size():
    if is_enable_kernel_file_cache():
        kernel_size_str = os.getenv("KERNEL_FILE_SIZE", "1024")
        try:
            kernel_size = int(kernel_size_str)
            return str(kernel_size)
        except ValueError:
            raise ValueError(f"Illegal input KERNEL_FILE_SIZE '{kernel_size}', need Integer number.")
    else:
        raise ValueError("Must set ENABLE_KERNEL_FILE_CACHE=1 first")


def dump_ir_if_needed(files):
    path = get_dump_dir()
    for f in files:
        shutil.copy(f, os.path.join(path, os.path.basename(f)))


def dump_file_if_needed(src_file, dest_file_name):
    path = get_dump_dir()
    shutil.copy(src_file, os.path.join(path, dest_file_name))


def dump_cmd_if_needed(cmd: list, flag: str):
    path = get_dump_dir()
    global dump_cmd_count
    if dump_cmd_count == 0:
        open_type = 'w'
    else:
        open_type = 'a'
    file_path = os.path.join(path, "cmds.txt")
    str_cmd = ' '.join(map(str, cmd))
    dump_cmd = f"{flag}:{str_cmd}\n\n"

    # 将字符串写入指定文件
    with open(file_path, open_type, encoding='utf-8') as f:
        f.write(dump_cmd)
    dump_cmd_count += 1


def get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)


def get_tsm_opt_path() -> str:
    return os.path.join(os.path.dirname(__file__), "bin", "tsingmicro-opt")


def get_tx8_deps_path(sub_name: str) -> str:
    path = os.getenv("TX8_DEPS_ROOT", "")
    if path == "":
        raise Exception("TX8_DEPS_ROOT is not set.")
    return os.path.join(path, sub_name)


def get_kuiper_path(sub_name: str) -> str:
    kuiper_path = "/usr/local/kuiper"
    return os.path.join(kuiper_path, sub_name)


def get_tx8_profiler_path() -> str:
    path = os.path.join(dirname(get_tsm_opt_path()), "tx-profiler")
    return path


def is_dump_args_profile():
    env_value = os.getenv("DUMP_KERNEL_ARGS", "0").strip()
    try:
        num_value = int(env_value)
    except ValueError:
        num_value = 0
    return num_value


def is_debug():
    debug_value = os.getenv("DEBUG", "OFF").strip().upper()
    return debug_value in ["ON", "TRUE", "1", "YES"]


def calculate_str_md5(string: str):
    str_hash = hashlib.md5(string).hexdigest()
    return str_hash


def calculate_file_md5(file_path):
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    return calculate_str_md5(file_bytes)
