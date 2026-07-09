import os
import shutil
from pathlib import Path


# sunrise
def sunrise_cp_bc_files(path):
    # mkdir -p third_party/sunrise/backend/lib
    lib_dir = Path("third_party/sunrise/backend/lib")
    os.makedirs(lib_dir, exist_ok=True)
    # cp ${LLVM_SYSPATH}/stpu/bitcode/*.bc third_party/sunrise/backend/lib
    bc_dir = Path(path) / "stpu" / "bitcode"
    for bc_file in bc_dir.glob("*.bc"):
        shutil.copy(bc_file, lib_dir)
