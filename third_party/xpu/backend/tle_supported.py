from .distributed import flagcx_rt_conf

TLE_SUPPORTED_PRIMITIVES = (["get_device_id", "n_pes", "shard_id", "distributed_barrier", "remote"]
                            if flagcx_rt_conf.bitcode_path else [])
