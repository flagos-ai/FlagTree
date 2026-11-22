# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit native arithmetic contracts and their host-evaluation boundary."""

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.distributed_inference import all_broadcast, tensor_of
from triton.flagmega.ir.model import DType, DistributedType


def normalize_compute_profile(attrs, opname, *, key="compute_profile", default="default", supported):
    profile = attrs.get(key, default)
    choices = (supported,) if isinstance(supported, str) else tuple(supported)
    if not isinstance(profile, str) or profile not in {default, *choices}:
        raise IRSchemaError(f"{opname} {key} must be {default} or {supported}.")
    return {} if profile == default else {key: profile}


def require_host_reference(node, *, key="compute_profile", default="default"):
    profile = node.attrs.get(key, default)
    if profile != default:
        raise EvaluationError(f"{node.op} {key}={profile!r} requires a validated hardware reference; "
                              "no software evaluator is registered.")


def verify_complete_f32(value_type, opname):
    value = tensor_of(value_type)
    if value.dtype != DType.FLOAT32 or any(not dim.is_fixed or dim.fixed_value <= 0 for dim in value.shape):
        raise IRSchemaError(f"{opname} native profile requires scalar F32 tensors with positive fixed dimensions.")
    if isinstance(value_type, DistributedType) and not all_broadcast(value_type):
        raise IRSchemaError(f"{opname} native profile requires complete broadcast tensors.")


def compute_profile_supports_target(attrs, platform, architecture):
    profiles = (("native_compute_profile", "f32", ("k230_q8_fp16_v1", "k230_iq4_act1_f16_experimental_v1")),
                ("compute_profile", "default", ("ggml_f32_v1", "ggml_cpu_f16_1t_v1", "ggml_cpu_f16_rowwise_1t_v1")),
                ("norm_compute_profile", "default", "ggml_f32_v1"))
    native_target = platform == "k230" and architecture == "k230"
    for key, default, supported in profiles:
        profile = attrs.get(key, default)
        if profile != default and (not native_target or profile not in (
                supported if isinstance(supported, tuple) else (supported,))):
            return False
    return attrs.get("math_mode") != "ggml" or native_target
