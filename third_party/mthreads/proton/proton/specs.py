# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

flops_by_device = {
    "CUDA": {
        "80":
        lambda width, **kwargs: 624e12 / (width / 8),
        "89":
        lambda width, **kwargs: (330.3 * 1e12) / (width / 8),  # TODO(Keren): Implement fp16 acc-> 660.6 fp8
        "90":
        lambda width, num_sms, clock_rate, **kwargs: ((num_sms / 114 * clock_rate / (1755 * 1e3) * 1513) * 1e12) /
        (width / 8),
        "100":
        lambda width, num_sms, clock_rate, **kwargs: (num_sms * 16384 * (clock_rate / 1e3) * 1e6) / (width / 8),
    }
}

amd_bps_by_arch = {
    'gfx90a': 3.2 * 1e12,
    'gfx942': 5.3 * 1e12,
    'gfx950': 8.0 * 1e12,
}

# FP8 Matrix Performance(FLOPS/clock/CU)
# For gfx90a we use the performance of INT8 since it doesn't support FP8 matrix operations.
amd_fp8_flops_by_arch = {'gfx90a': 1024, 'gfx942': 4096, 'gfx950': 8192}


def max_flops(device_type, arch, width, num_sms, clock_rate):
    """
    Calculate the maximum FLOPS for a given device type and width.

    Args:
        device_type (str): The type of device (e.g., "CUDA", "HIP").
        arch (str): The architecture of the device (e.g., "80", "90").
        width (int): The width in bits.
        num_sms (int): The number of streaming multiprocessors.
        clock_rate (float): The clock rate in GHz.

    Returns:
        float: The maximum FLOPS for the given device type and width.
    """
    if device_type == "HIP":
        return amd_fp8_flops_by_arch[arch] * num_sms * clock_rate * 1e3 / (width / 8)

    if device_type not in flops_by_device:
        raise ValueError(f"Unsupported device type: {device_type}")

    if arch not in flops_by_device[device_type]:
        raise ValueError(f"Unsupported architecture: {arch}")

    flops_func = flops_by_device[device_type][arch]

    return flops_func(width, num_sms=num_sms, clock_rate=clock_rate)


def max_bps(device_type, arch, bus_width, memory_clock_rate):
    """
    Calculate the maximum bytes per second for a given bus width and memory clock rate.

    Args:
        bus_width (int): The bus width in bits.
        memory_clock_rate (float): The memory clock rate in GHz.

    Returns:
        float: The maximum bytes per second.
    """
    if device_type == "CUDA":
        return 2 * bus_width * memory_clock_rate * 1e3 / 8
    else:
        assert device_type == "HIP"
        return amd_bps_by_arch[arch]
