"""Record BI-V150 state and equalize scenario starts outside request timing."""
import os
import subprocess
import time


def scenario_start(max_temperature):
    if max_temperature is None:
        return None
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not device or "," in device:
        raise ValueError("Temperature-controlled measurement needs one visible physical GPU")
    started = time.monotonic()
    observations = []
    while True:
        values = subprocess.check_output([
            "ixsmi", "-i", device,
            "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True).strip().split(",")
        temperature, sm, memory, used, utilization = map(int, values)
        state = {"temperature_c": temperature, "sm_clock_mhz": sm,
                 "memory_clock_mhz": memory, "memory_used_mib": used,
                 "utilization_percent": utilization}
        observations.append(state)
        if temperature <= max_temperature:
            return {"max_start_temperature_c": max_temperature,
                    "wait_seconds_excluded": time.monotonic() - started,
                    "observations": observations}
        if time.monotonic() - started > 180:
            raise RuntimeError("GPU did not reach the requested scenario start temperature")
        time.sleep(2)
