import os


def get_temp_dir(fileCacheManager, pid, rnd_id):
    return os.path.join(fileCacheManager.cache_dir, f"tmp.pid_{pid}_{rnd_id}")


def get_temp_path_in_FileCacheManager_put(fileCacheManager, pid, rnd_id, filename):
    temp_dir = get_temp_dir(fileCacheManager, pid, rnd_id)
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, filename)
    return temp_path


def remove_temp_dir_in_FileCacheManager_put(fileCacheManager, pid, rnd_id):
    temp_dir = get_temp_dir(fileCacheManager, pid, rnd_id)
    os.removedirs(temp_dir)
