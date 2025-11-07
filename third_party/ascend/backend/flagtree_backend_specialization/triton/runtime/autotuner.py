def set_Autotuner_auto_profile_dir(autotuner, auto_profile_dir):
    autotuner.auto_profile_dir = auto_profile_dir

def has_spec_default_Autotuner_configs():
    return True

def get_spec_default_Autotuner_configs():
    from triton.runtime.autotuner import Config
    return Config({})

def ext_Autotuner_do_bench_MLIRCompilationError(exception_types):
    from triton.compiler.errors import MLIRCompilationError
    return (MLIRCompilationError)

def _profile(autotuner, *args, config, **meta):
    from triton.testing import do_bench_npu

    # check for conflicts, i.e. meta-parameters both provided
    # as kwargs and by the autotuner
    conflicts = meta.keys() & config.kwargs.keys()
    if conflicts:
        raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}."
                            " Make sure that you don't re-define auto-tuned symbols.")
    # augment meta-parameters with tunable ones
    current = dict(meta, **config.all_kwargs())
    full_nargs = {**autotuner.nargs, **current}

    def kernel_call():
        if config.pre_hook:
            config.pre_hook(full_nargs)
        autotuner.pre_hook(full_nargs)
        try:
            autotuner.fn.run(
                *args,
                **current,
            )
        except Exception as e:
            try:
                autotuner.post_hook(full_nargs, exception=e)
            finally:
                # Throw exception raised by `autotuner.fn.run`
                raise

        autotuner.post_hook(full_nargs, exception=None)

    do_bench_npu(
        kernel_call, prof_dir=autotuner.auto_profile_dir, keep_res=True
    )

def ext_Autotuner_profile(autotuner, used_cached_result, args, kwargs):
    if not used_cached_result and autotuner.auto_profile_dir is not None:
        _profile(*args, config=autotuner.best_config, **kwargs)

def set_Config_BiShengIR_options(config, bishengir_options):
    # BiShengIR Options allowed for autotune
    config.multibuffer = bishengir_options.get("multibuffer", None) # Compiler Default True
    config.unit_flag = bishengir_options.get("unit_flag", None) # Compiler Default False
    config.limit_auto_multi_buffer_only_for_local_buffer = bishengir_options.get("limit_auto_multi_buffer_only_for_local_buffer", None) # Compiler Default False
    config.limit_auto_multi_buffer_of_local_buffer = bishengir_options.get("limit_auto_multi_buffer_of_local_buffer", None) # Compiler Default no-limit
    config.set_workspace_multibuffer = bishengir_options.get("set_workspace_multibuffer", None) # Compiler Default 1
    config.enable_hivm_auto_cv_balance = bishengir_options.get("enable_hivm_auto_cv_balance", None) # Compiler Default True
    config.tile_mix_vector_loop = bishengir_options.get("tile_mix_vector_loop", None) # Compiler Default 1
    config.tile_mix_cube_loop = bishengir_options.get("tile_mix_cube_loop", None) # Compiler Default 1

def ext_Config_all_kwargs(config):
    return (
        ("multibuffer", config.multibuffer),
        ("enable_hivm_auto_cv_balance", config.enable_hivm_auto_cv_balance),
        ("unit_flag", config.unit_flag),
        ("limit_auto_multi_buffer_only_for_local_buffer", \
            config.limit_auto_multi_buffer_only_for_local_buffer),
        ("limit_auto_multi_buffer_of_local_buffer", config.limit_auto_multi_buffer_of_local_buffer),
        ("set_workspace_multibuffer", config.set_workspace_multibuffer),
        ("tile_mix_vector_loop", config.tile_mix_vector_loop),
        ("tile_mix_cube_loop", config.tile_mix_cube_loop)
    )

def ext_Config_to_str(res, config):
    res.append(f"multibuffer: {config.multibuffer}")
    res.append(f"enable_hivm_auto_cv_balance: {config.enable_hivm_auto_cv_balance}")
    res.append(f"unit_flag: {config.unit_flag}")
    res.append(f"limit_auto_multi_buffer_only_for_local_buffer: \
        {config.limit_auto_multi_buffer_only_for_local_buffer}")
    res.append(f"limit_auto_multi_buffer_of_local_buffer: {config.limit_auto_multi_buffer_of_local_buffer}")
    res.append(f"set_workspace_multibuffer: {config.set_workspace_multibuffer}")
    res.append(f"tile_mix_vector_loop: {config.tile_mix_vector_loop}")
    res.append(f"tile_mix_cube_loop: {config.tile_mix_cube_loop}")

def new_AutoTilingTuner(fn, configs, key, reset_to_zero, restore_value, pre_hook,
                        post_hook, prune_configs_by, warmup, rep,
                        use_cuda_graph, do_bench, auto_profile_dir,
                        split_params, tiling_params, low_dims,
                        dual_reduction, persistent_reduction):
    from triton.runtime.autotiling_tuner import AutoTilingTuner
    return AutoTilingTuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                           post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                           use_cuda_graph=use_cuda_graph, do_bench=do_bench, auto_profile_dir=auto_profile_dir,
                           split_params=split_params, tiling_params=tiling_params, low_dims=low_dims,
                           dual_reduction=dual_reduction, persistent_reduction=persistent_reduction)
