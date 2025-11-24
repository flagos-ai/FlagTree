def backend_smi_cmd(attrs):
    return ['ixsmi', '-i', '0', '--query-gpu=' + attrs, '--format=csv,noheader,nounits']


def get_mem_clock_khz():
    import torch
    capability = torch.cuda.get_device_capability()
    if capability[0] == 8:
        mem_clock_khz = 1800000
        return mem_clock_khz
    return None


def is_get_tflops_support_capability_lt_8():
    return True
