# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Default typed encoders for the frozen E32 IQ4 streaming ABI."""
from triton.flagmega.errors import CodegenError

SCHEMA = "flagmega.iq4-warp-local-native/r001-v2.1"
PLAN_SHA256 = "7b04b547f07fd1c71e4ecb8242720a631470c2a49953ab115487f3a1f5a6c361"
NATIVE_INTERFACE = "E32-idx32-full-warp/v1"
RESIDENCY_ENCODING = {"size_per_thread": [1, 1], "threads_per_warp": [1, 32],
                      "warps_per_cta": [4, 1], "order": [1, 0]}
FAMILIES = ("iq4_streaming_projection", "iq4_streaming_glu", "iq4_streaming_interleaved")


def validate_raw(raw):
    p = raw["parameters"]
    if p.get("tile_n") != 4 or p.get("num_warps") != 4:
        raise CodegenError("r001 requires R4 and four warps")
    if p.get("native_interface") != NATIVE_INTERFACE or p.get("plan_sha256") != PLAN_SHA256:
        raise CodegenError("r001 requires the frozen E32/idx32 interface and v2.1 plan hash")
    if p.get("plan_hash") != PLAN_SHA256 or p.get("residency_encoding") != RESIDENCY_ENCODING:
        raise CodegenError("r001 requires the checked E32 residency encoding")
    _validate_abi(raw)


def _validate_abi(raw):
    if raw.get("facts", {}).get("internal_grid_barriers", 0):
        raise CodegenError("IQ4 row tasks cannot enter a grid barrier")
    attrs = raw["semantic_attrs"]
    keys = ("gate_weight_encoding", "up_weight_encoding") if raw["family"] == FAMILIES[1] else ("weight_encoding",)
    if any(attrs.get(k) != "IQ4_XS" for k in keys):
        raise CodegenError("IQ4 streaming does not admit other weight encodings")
    if attrs.get("gate_clamp_max") is not None or attrs.get("up_clamp") is not None:
        raise CodegenError("v4 does not add a GLU clamp")
    for section in ("inputs", "outputs"):
        for argument in raw.get(section, ()):
            for binding in argument.get("buffers", ()):
                abi = binding["abi"]
                if abi.get("logical_shape") != abi.get("local_capacity_shape"):
                    raise CodegenError("IQ4 streaming requires a complete chip owner")
                if abi.get("pool_scope_stride_bytes", 0):
                    raise CodegenError("IQ4 streaming cannot receive a per-CTA pool pointer")
                if section == "inputs" and argument["formal"] == "value" and abi.get("scalar_dtype") not in {"bfloat16", "float32"}:
                    raise CodegenError("IQ4 source must retain BF16/F32 precision")
                if section == "inputs" and argument["formal"] in {"weight", "gate_weight", "up_weight"}:
                    strides = tuple(abi["scalar_storage_strides"])
                    shape = tuple(abi["logical_shape"])
                    if (abi.get("scalar_dtype") != "uint8" or abi.get("scalar_lane_count", 1) != 1
                            or len(shape) != 2 or shape[1] % 136 or strides != (shape[1], 1)
                            or abi.get("pool_byte_offset", 0) % 4):
                        raise CodegenError("IQ4 requires contiguous original136B rows and aligned U32 addresses")


def projection(raw):
    from .kernel_call_renderers import _block_fp8_call
    return {"schema": SCHEMA, **_block_fp8_call(raw)}


def glu(raw):
    from .kernel_call_renderers import _matmul_glu_call
    return {"schema": SCHEMA, **_matmul_glu_call(raw)}


def interleaved(raw):
    from .interleaved_projection_renderer import interleaved_projection_call
    return {"schema": SCHEMA, **interleaved_projection_call(raw)}


def register():
    from .kernel_call_renderers import register_family_encoder
    for family, encoder in zip(FAMILIES, (projection, glu, interleaved)):
        register_family_encoder(family, encoder, schema=SCHEMA, validate_raw=validate_raw)


def encode_default(raw):
    """Describe exported R4 packages without a caller's local registration.

    The accepted v4 package has the same scalar physical ABI and no E32 plan
    parameters. Keep its schema intact; new E32 packages require the full pin.
    Explicit local registrations continue to own their versioned encoders.
    """
    parameters = raw["parameters"]
    plan_fields = {"native_interface", "plan_sha256", "plan_hash", "residency_encoding"}
    legacy = (parameters.get("tile_n") == 4 and parameters.get("num_warps") == 4
              and not plan_fields.intersection(parameters))
    if legacy:
        _validate_abi(raw)
    else:
        validate_raw(raw)
    encoder = dict(zip(FAMILIES, (projection, glu, interleaved)))[raw["family"]]
    value = encoder(raw)
    if legacy:
        value["schema"] = "flagmega.iq4-streaming-native/v4"
    return value

