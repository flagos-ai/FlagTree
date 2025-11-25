def add_Autotuner_attributes(autotuner):
    # cache_fn_map fmt: {"fn_cache_key: [hash_cache_file_0, hash_cache_file_1, ...], [so_path_0, so_path_1, ...]]"}
    autotuner.cache_fn_map = dict()


def get_jit_func(autotuner):
    if hasattr(autotuner.fn, "cache_key"):
        # for autotune + jit
        return autotuner.fn
    elif hasattr(autotuner.fn.fn, "cache_key"):
        # for autotune + heuristics + jit
        return autotuner.fn.fn
    else:
        msg = f'Current {autotuner.fn} or {autotuner.fn.fn} has no attribute cache_key.'
        raise RuntimeError(msg)


def ext_Autotuner_bench(autotuner):
    cache_key = str(get_jit_func(autotuner).cache_key)
    check_key = autotuner.cache_fn_map.get(str(cache_key), None)
    if not check_key:
        autotuner.cache_fn_map.setdefault(cache_key, [[], []])
    hash_cache_file = str(get_jit_func(autotuner).hash_cache_file)
    so_path = ''
    if get_jit_func(autotuner).so_path:
        so_path = get_jit_func(autotuner).so_path.split('/')[-2]
    autotuner.cache_fn_map[cache_key][0].append(hash_cache_file)
    autotuner.cache_fn_map[cache_key][1].append(so_path)


def ext_Autotuner_key(autotuner, _args, *args):
    key = [_args[i] for i in autotuner.key_idx]
    divisibility = 16
    for arg in args:
        if hasattr(arg, "data_ptr"):
            key.append(arg.dtype)
            key.append(arg.data_ptr() % divisibility == 0)
        elif isinstance(arg, int):
            key.append(arg)
    key = tuple(key)
    return key


def build_best_config_hash(args_names, key):
    import os
    import hashlib
    from triton.runtime.cache import default_cache_dir
    cache_dir = os.environ.get('TRITON_CACHE_DIR', default_cache_dir())
    hasher = hashlib.sha256()
    hasher.update(f"{'_'.join(args_names) + str(key)}\n".encode())
    cfg_hash = hasher.hexdigest()
    cfg_hash_dir = os.path.join(cache_dir, cfg_hash)
    cfg_hash_file = os.path.splitext(cfg_hash)[0] + ".best_config"
    cfg_hash_file = os.path.join(cfg_hash_dir, cfg_hash_file)
    return cfg_hash_dir, cfg_hash_file


def load_best_config(args_names, key):
    import os
    import json
    _, cfg_hash_file = build_best_config_hash(args_names, key)
    if os.path.exists(cfg_hash_file):
        with open(cfg_hash_file) as fd:
            best_config = json.loads(fd.read())
            num_warps = best_config.pop('num_warps') if 'num_warps' in best_config else 4
            num_stages = best_config.pop('num_stages') if 'num_stages' in best_config else 1
            return best_config, num_warps, num_stages
    return None


def save_best_config(cfg, args_names, key):
    import os
    import filelock
    import json
    cfg_hash_dir, cfg_hash_file = build_best_config_hash(args_names, key)
    if os.path.exists(cfg_hash_dir):
        return
    os.makedirs(cfg_hash_dir, exist_ok=True)
    lock = filelock.FileLock(f"{cfg_hash_file}.lock")
    with lock:
        if os.path.exists(cfg_hash_file):
            return
        with open(cfg_hash_file, "w") as fd:
            fd.write(json.dumps({
                **cfg.kwargs,
                "num_warps": cfg.num_warps,
                "num_stages": cfg.num_stages,
            }))


def handle_only_save_best_config_cache(autotuner, key, *args, **kwargs):
    import os
    import time
    import builtins
    from triton.runtime.autotuner import Config
    from triton.runtime.cache import default_cache_dir
    only_save_best_config_cache = os.environ.get("TRITON_ONLY_SAVE_BEST_CONFIG_CACHE", "0") == "1"
    if only_save_best_config_cache:
        load_config = load_best_config(autotuner.arg_names, key)
        if load_config:
            best_config, num_warps, num_stages = load_config
            config = Config(best_config, num_warps, num_stages)
            autotuner.cache[key] = config
            autotuner.pre_hook(args, reset_only=True)
        else:
            pruned_configs = autotuner.prune_configs(kwargs)
            bench_start = time.time()
            timings = {config: autotuner._bench(*args, config=config, **kwargs) for config in pruned_configs}
            bench_end = time.time()
            autotuner.bench_time = bench_end - bench_start
            autotuner.cache[key] = builtins.min(timings, key=timings.get)
            list_keys = list(timings.keys())
            best_key_index = list_keys.index(builtins.min(timings, key=timings.get))
            save_best_config(autotuner.cache[key], autotuner.arg_names, key)
            autotuner.pre_hook(args, reset_only=True)
            autotuner.configs_timings = timings
            cache_key = str(get_jit_func(autotuner).cache_key)
            check_key = autotuner.cache_fn_map.get(cache_key, None)
            if check_key:
                best_cache_file = autotuner.cache_fn_map[cache_key][0][best_key_index]
                best_so_path = autotuner.cache_fn_map[cache_key][1][best_key_index]
                ck_list = [best_cache_file, best_so_path]
                for i in range(len(ck_list)):
                    for tmp_key in check_key[i]:
                        if ck_list[i] != tmp_key:
                            del_cache_file = os.path.join(os.environ.get('TRITON_CACHE_DIR', default_cache_dir()),
                                                          tmp_key)
                            import shutil
                            shutil.rmtree(del_cache_file, ignore_errors=True)
            autotuner.cache_fn_map.clear()
