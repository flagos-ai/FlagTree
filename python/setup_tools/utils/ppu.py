import os
import shutil
from pathlib import Path


def get_llvm_irformatter():
    '''
    Get irformatter
    '''
    binary = "llvm-irformatter"
    paths = [
        os.environ.get("TRITON_IR_FORMATTER_PATH", ""),
        os.path.join(os.environ.get("PPU_SDK"), "bin", binary)
    ]
    for bin in paths:
        if os.path.exists(bin) and os.path.isfile(bin):
            return bin
    raise RuntimeError("Cannot find IR Formatter")


def copy_llvm_irformatter():
    binary = "llvm-irformatter"
    src_path = get_llvm_irformatter()
    base_dir = os.path.dirname(__file__)
    dst_path = os.path.join(base_dir, "third_party", "ppu", "backend", f'bin/{binary}')  # final binary path
    os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
    print(f'copy {src_path} to {dst_path} ...')
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        shutil.copy(src_path, dst_path)

def install_extension(*args, **kargs):
    # Modify nvidia driver's is_active() to return False for ppu backend
    # This prevents nvidia driver from being activated when using ppu
    drvfile = Path(__file__).parent.parent.parent.parent / 'third_party' / 'nvidia' / 'backend' / 'driver.py'
    if drvfile.exists():
        with open(drvfile, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if 'def is_active():' in line:
                if i + 1 < len(lines) and 'return False' not in lines[i + 1]:
                    lines.insert(i + 1, '        return False\n')
                break
        with open(drvfile, 'w') as f:
            f.writelines(lines)
    
    copy_llvm_irformatter()
