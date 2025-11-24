def k_must_be_divisiable_by_bk_sk(K, BLOCK_K, SPLIT_K):
    if (K % (BLOCK_K * SPLIT_K) != 0):
        return True
    return False


def calculate_total_time_ms(compute_ms, load_ms, store_ms):
    return compute_ms + load_ms + store_ms


def get_pruned_configs(capability, v, BLOCK_M, BLOCK_N, BLOCK_K):
    import torch
    pruned_configs = []
    if hasattr(torch, "corex"):
        for stage in range(len(v)):
            random_config = v[stage][0]
            random_config.num_stages = v[stage][1]
            if (capability[0] < 8 and v[stage][1] < 3):
                pruned_configs.append(random_config)
            if capability[0] == 8:
                blocks = BLOCK_M + BLOCK_N + BLOCK_K
                if blocks <= 256:
                    pruned_configs.append(random_config)
                elif v[stage][1] > 2 and blocks > 256:
                    pruned_configs.append(random_config)
        return pruned_configs
    else:
        return None
