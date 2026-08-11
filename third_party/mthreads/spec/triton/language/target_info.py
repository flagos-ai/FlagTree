from triton.runtime import driver
from triton.runtime.jit import constexpr_function

__all__ = ["current_target"]


def current_target():
    try:
        active_driver = driver.active
    except RuntimeError:
        # If there is no active driver, return None
        return None
    return active_driver.get_current_target()


current_target.__triton_builtin__ = True


@constexpr_function
def is_cuda():
    target = current_target()
    return target is not None and target.backend == "cuda"


@constexpr_function
def cuda_capability_geq(major, minor=0):
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    target = current_target()
    if target is None or target.backend != "cuda":
        return False
    assert isinstance(target.arch, int)
    return target.arch >= major * 10 + minor


@constexpr_function
def is_musa():
    target = current_target()
    return target is not None and target.backend == "musa"


@constexpr_function
def musa_capability_geq(major, minor=0):
    """
    Determines whether we have MUSA capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain MUSA capability.
    """
    target = current_target()
    if target is None or target.backend != "musa":
        return False
    arch = target.arch
    if isinstance(arch, int):
        capability = arch
    elif isinstance(arch, str):
        normalized = arch.lower()
        if normalized.startswith("ph1"):
            capability = 31
        elif normalized.isdecimal():
            capability = int(normalized)
        elif "." in normalized:
            arch_major, arch_minor = normalized.split(".", 1)
            if arch_major.isdecimal() and arch_minor.isdecimal():
                capability = int(arch_major) * 10 + int(arch_minor)
            else:
                capability = None
        else:
            capability = None
    else:
        capability = None
    return capability is not None and capability >= major * 10 + minor


@constexpr_function
def is_hip():
    target = current_target()
    return target is not None and target.backend == "hip"


@constexpr_function
def is_hip_cdna3():
    target = current_target()
    return target is not None and target.arch == "gfx942"


@constexpr_function
def is_hip_cdna4():
    target = current_target()
    return target is not None and target.arch == "gfx950"
