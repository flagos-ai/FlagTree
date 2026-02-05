def compute_dq_like_mma_v3():
    return True


def always_support_flash_attention():
    return True


def attention_forward_config():
    BLOCK_M = 64
    BLOCK_N = 64
    num_stages = 1
    return (BLOCK_M, BLOCK_N, num_stages)


def attention_backward_config(BLOCK_DMODEL):
    import torch
    # otherwise shared memory out of resource
    BLOCK = 64  # FIXME: currently BLOCK=128 has issues, BLOCK=64 works for common cases.
    capability = torch.cuda.get_device_capability()
    num_warps = 16 if BLOCK_DMODEL > 64 and capability[0] < 8 else 8
    return (BLOCK, num_warps)
