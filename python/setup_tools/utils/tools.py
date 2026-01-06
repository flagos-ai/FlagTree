import os
import shutil
from pathlib import Path
from dataclasses import dataclass

flagtree_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
flagtree_submoduel_dir = os.path.join(flagtree_root_dir, "third_party")

network_configs = {
    "MAX_RETRY_COUNT": 4,
    "GIT_CLONE_TIMEOUT": 60,
}


@dataclass
class Module:
    name: str
    url: str
    commit_id: str = None
    dst_path: str = None


def dir_rollback(deep, base_path):
    while (deep):
        base_path = os.path.dirname(base_path)
        deep -= 1
    return Path(base_path)


def remove_triton_in_modules(model):
    model_path = model.dst_path
    triton_path = os.path.join(model_path, "triton")
    if os.path.exists(triton_path):
        shutil.rmtree(triton_path)


def py_clone(module):
    import git
    retry_count = network_configs["MAX_RETRY_COUNT"]
    has_specialization_commit = module.commit_id is not None
    while (retry_count):
        try:
            repo = git.Repo.clone_from(module.url, module.dst_path)
            if has_specialization_commit:
                repo.git.checkout(module.commit_id)
            return True
        except Exception:
            retry_count -= 1
            print(
                f"\n[{network_configs['MAX_RETRY_COUNT'] - retry_count}] retry to clone {module.name} to  {module.dst_path}"
            )
    return False


def sys_clone(module):
    retry_count = network_configs["MAX_RETRY_COUNT"]
    has_specialization_commit = module.commit_id is not None
    while (retry_count):
        try:
            os.system(f"git clone {module.url} {module.dst_path}")
            if has_specialization_commit:
                os.system("cd module.dst_path")
                os.system(f"git checkout {module.commit_id}")
                os.system("cd -")
            return True
        except Exception:
            retry_count -= 1
            print(
                f"\n[{network_configs['MAX_RETRY_COUNT'] - retry_count}] retry to clone {module.name} to  {module.dst_path}"
            )
    return False


def clone_module(module):
    succ = True if py_clone(module) else sys_clone(module)
    if not succ:
        print(f"[ERROR]: Failed to clone {module.name} from {module.url}")
        return False
    print(f"[INFO]: Successfully cloned {module.name} to {module.dst_path}")
    return True


def download_module(module, required=False):
    if module is None:
        return
    if not os.path.exists(module.dst_path):
        succ = clone_module(module)
    else:
        print(f'Found third_party {module.name} at {module.dst_path}\n')
        return True
    if not succ and required:
        raise RuntimeError(
            f"[ERROR]: Failed to download {module.name} from {module.url}, It's most likely the network!")
    remove_triton_in_modules(module)
