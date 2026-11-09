# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Machine-readable operator dtype configuration for the Ascend backend.
#
# Source of truth: docs/zh/python-api/_ascend_constraints.py. Each documented
# operator carries an ASCII ``dtype_support`` table (one column per scalar
# dtype, one row per Ascend platform); the rules below transcribe that table
# for every operator whose support differs across platforms. Fully supported
# operators have no rule.
#
# ---------------------------------------------------------------------------
# Rule schema
# ---------------------------------------------------------------------------
#   "<rule id>": {
#       # Fully qualified names matched by the *entry layer* (direct
#       # ``tl.<name>(...)`` / @jit composite calls). Only direct
#       # ``triton.language`` operators are guarded.
#       "names": ["triton.language.permute"],
#
#       # Tensor dunder builtins matched by the *builtin-dunder layer*
#       # (``a + b``, ``a >> b``, ``a & b`` ...); reflected dunders included.
#       "dunder": ["__add__", "__radd__"],
#       "dunder_label": "+",
#
#       # Positions (0-based) of tensor-like operands to inspect. When
#       # omitted, every positional operand is inspected (non-tensors and
#       # pointers are skipped automatically).
#       "tensor_args": [0, 1],
#
#       # Positions that are pointers: the pointee element dtype is read.
#       "pointer_args": [0],
#
#       # The dtype blacklist per architecture bucket:
#       #   "a2" -> Ascend A2/A3 ; "a5" -> Ascend 950
#       #   * "unsupported": blacklist -- the interception decision uses this:
#       #     a dtype listed here is rejected at the front end.
#       # The accepted ("supported") dtypes are normally NOT stored; they are
#       # derived at runtime as the complement of this list over
#       # ALL_SCALAR_DTYPES and used in the error message. A rule whose
#       # advertised dtypes are intentionally narrower than that complement
#       # (currently only "dot_scaled") may spell out an explicit "supported"
#       # list, which then overrides the derivation. A missing bucket means
#       # "no dtype restriction on that architecture" (every dtype accepted).
#       "unsupported": {"a2": [...], "a5": [...]},
#   }
#
# Dtype tokens (matched against ``dtype.name``):
#   * regular names: int8/int16/int32/int64/uint8/uint16/uint32/uint64/
#     fp16/bf16/fp32/fp64/bool
#   * "fp8e4" : the E4M3 column, matching fp8e4nv/fp8e4b8/fp8e4b15 at runtime
#   * "fp8e5" : the E5M2 column, matching fp8e5/fp8e5b16 at runtime
#   (``bool`` is represented internally as ``int1``; the matching layer
#   normalizes both. The concrete fp8 variant names live in FP8*_DTYPES.)

FP8E4_DTYPES = ["fp8e4nv", "fp8e4b8", "fp8e4b15"]
FP8E5_DTYPES = ["fp8e5", "fp8e5b16"]

# The 15 scalar-dtype columns of the documented ``dtype_support`` tables, in
# table order. Each rule's "unsupported" list is a subset of this universe; the
# accepted dtypes for a rule are the complement of that list.
ALL_SCALAR_DTYPES = [
    "uint8",
    "int8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "uint64",
    "int64",
    "fp16",
    "fp32",
    "fp64",
    "bf16",
    "fp8e4",
    "fp8e5",
    "bool",
]

OP_DTYPE_RULES = {
    # "abs": {
    #     "names": ["triton.language.abs"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "add": {
    #     "names": ["triton.language.add"],
    #     "dunder": ["__add__", "__radd__"],
    #     "dunder_label": "+",
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "advance": {
    #     "names": ["triton.language.advance"],
    #     "pointer_args": [0],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5", "bool"],
    #         "a5": ["fp64", "bool"],
    #     },
    # },
    # "and": {
    #     "dunder": ["__and__", "__rand__"],
    #     "dunder_label": "&",
    #     "unsupported": {
    #         "a2": [
    #             "uint16",
    #             "uint32",
    #             "uint64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #         "a5": ["fp16", "fp32", "fp64", "bf16", "fp8e4", "fp8e5"],
    #     },
    # },
    # "argmax": {
    #     "names": ["triton.language.argmax"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64", "fp8e4", "fp8e5"],
    #     },
    # },
    # "argmin": {
    #     "names": ["triton.language.argmin"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64", "fp8e4", "fp8e5"],
    #     },
    # },
    # "associative_scan": {
    #     "names": ["triton.language.associative_scan"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64", "bf16", "fp8e4", "fp8e5"],
    #     },
    # },
    "assume": {
        "names": ["triton.language.assume"],
        "unsupported": {
            "a2": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4",
                "fp8e5",
            ],
            "a5": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4",
                "fp8e5",
            ],
        },
    },
    "condition": {
        "names": ["triton.language.condition"],
        "unsupported": {"a2": ["uint64"], "a5": ["uint64"]},
        "supported": {
            "a2": ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"], "a5":
            ["uint8", "int8", "int16", "int32", "int64", "fp16", "fp32", "bf16", "bool"]
        },
    },
    # "atomic_add": {
    #     "names": ["triton.language.atomic_add"],
    #     "tensor_args": [0, 1],
    #     "pointer_args": [0],
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "uint64",
    #             "fp64",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #         "a5": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "fp64",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #     },
    # },
    # "atomic_and": {
    #     "names": ["triton.language.atomic_and"],
    #     "tensor_args": [0, 1],
    #     "pointer_args": [0],
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "uint64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #         "a5": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #     },
    # },
    # "atomic_cas": {
    #     "names": ["triton.language.atomic_cas"],
    #     "tensor_args": [0, 1, 2],
    #     "pointer_args": [0],
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "uint32",
    #             "uint64",
    #             "fp64",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #         "a5": ["uint8", "int8", "fp64", "fp8e4", "fp8e5", "bool"],
    #     },
    # },
    # "atomic_max": {
    #     "names": ["triton.language.atomic_max"],
    #     "tensor_args": [0, 1],
    #     "pointer_args": [0],
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "uint64",
    #             "fp16",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #         "a5": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "fp16",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #     },
    # },
    # "atomic_min": {
    #     "names": ["triton.language.atomic_min"],
    #     "tensor_args": [0, 1],
    #     "pointer_args": [0],
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "uint64",
    #             "fp16",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #         "a5": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "fp16",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #     },
    # },
    # "atomic_or": {
    #     "names": ["triton.language.atomic_or"],
    #     "tensor_args": [0, 1],
    #     "pointer_args": [0],
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "uint64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #         "a5": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #     },
    # },
    # "atomic_xchg": {
    #     "names": ["triton.language.atomic_xchg"],
    #     "tensor_args": [0, 1],
    #     "pointer_args": [0],
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "uint64",
    #             "fp16",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #         "a5": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "fp16",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #     },
    # },
    # "atomic_xor": {
    #     "names": ["triton.language.atomic_xor"],
    #     "tensor_args": [0, 1],
    #     "pointer_args": [0],
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "uint64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #         "a5": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #     },
    # },
    "bitonic_merge": {
        "names": ["triton.language.bitonic_merge"],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
            "a5": ["uint16", "uint32", "uint64", "fp64"],
        },
    },
    # "broadcast": {
    #     "names": ["triton.language.broadcast"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "broadcast_to": {
    #     "names": ["triton.language.broadcast_to"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "cast": {
    #     "names": ["triton.language.cast"],
    #     "tensor_args": [1],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    "cat": {
        "names": ["triton.language.cat"],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
            "a5": ["fp64"],
        },
    },
    "cdiv": {
        "names": ["triton.language.cdiv"],
        "unsupported": {
            "a2": [
                "uint16",
                "uint32",
                "uint64",
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4",
                "fp8e5",
                "bool",
            ],
            "a5": ["fp16", "fp32", "fp64", "bf16", "fp8e4", "fp8e5", "bool"],
        },
    },
    "ceil": {
        "names": ["triton.language.ceil"],
        "unsupported": {
            "a2": [
                "fp64",
            ],
            "a5": [
                "fp64",
            ],
        },
        "supported": {
            "a2": ["fp32"],
            "a5": [
                "fp32",
            ],
        },
    },
    "clamp": {
        "names": ["triton.language.clamp"],
        "unsupported": {
            "a2": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "uint64",
                "int64",
                "fp64",
                "fp8e4",
                "fp8e5",
                "bool",
            ],
            "a5": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "uint64",
                "int64",
                "fp64",
                "bool",
            ],
        },
        "supported": {
            "a2": [
                "fp16",
                "fp32",
                "bf16",
            ],
            "a5": [
                "fp16",
                "fp32",
                "bf16",
                "fp8e4",
                "fp8e5",
            ],
        },
    },
    "cos": {
        "names": ["triton.language.cos"],
        "unsupported": {
            "a2": [
                "fp64",
            ],
            "a5": [
                "fp64",
            ],
        },
        "supported": {
            "a2": [
                "fp32",
            ],
            "a5": [
                "fp32",
            ],
        },
    },
    "cumprod": {
        "names": ["triton.language.cumprod"],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
            "a5": ["fp64", "fp8e4", "fp8e5"],
        },
    },
    # "cumsum": {
    #     "names": ["triton.language.cumsum"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "device_assert": {
    #     "names": ["triton.language.device_assert"],
    #     "tensor_args": [0],
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "int32",
    #             "uint64",
    #             "int64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #         "a5": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "int32",
    #             "uint64",
    #             "int64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #     },
    # },
    # "device_print": {
    #     "names": ["triton.language.device_print"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "div": {
    #     "dunder": ["__truediv__", "__rtruediv__"],
    #     "dunder_label": "/",
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "dot": {
    #     "names": ["triton.language.dot"],
    #     "tensor_args": [0, 1],
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "int32",
    #             "uint64",
    #             "int64",
    #             "fp64",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #         "a5": [
    #             "uint8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "int32",
    #             "uint64",
    #             "int64",
    #             "fp64",
    #             "bool",
    #         ],
    #     },
    # },
    # "dot_scaled": {
    #     "names": ["triton.language.dot_scaled"],
    #     "tensor_args": [0, 3],
    #     "unsupported": {
    #         "a2": [
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #         "a5": [],
    #     },
    #     # Exception to the complement rule: the advertised input dtypes for
    #     # dot_scaled are narrower than the blacklist alone, so the message
    #     # list is kept explicit instead of being derived.
    #     "supported": {
    #         "a2": ["fp16", "bf16"],
    #         # uint = e2m1 / fp8 as uint8
    #         "a5": ["fp8e4", "fp8e5", "fp16", "bf16", "uint8"],
    #     },
    # },
    # "eq": {
    #     "dunder": ["__eq__", "__req__"],
    #     "dunder_label": "==",
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    "erf": {
        "names": ["triton.language.erf"],
        "unsupported": {
            "a2": [
                "fp64",
            ],
            "a5": [
                "fp64",
            ],
        },
        "supported": {
            "a2": [
                "fp32",
            ],
            "a5": [
                "fp32",
            ],
        },
    },
    "exp": {
        "names": ["triton.language.exp"],
        "unsupported": {
            "a2": [
                "fp64",
            ],
            "a5": [
                "fp64",
            ],
        },
        "supported": {
            "a2": [
                "fp16",
                "fp32",
            ],
            "a5": [
                "fp16",
                "fp32",
            ],
        },
    },
    "exp2": {
        "names": ["triton.language.exp2"],
        "unsupported": {
            "a2": [
                "fp64",
            ],
            "a5": [
                "fp64",
            ],
        },
        "supported": {
            "a2": [
                "fp32",
            ],
            "a5": [
                "fp32",
            ],
        },
    },
    # "expand_dims": {
    #     "names": ["triton.language.expand_dims"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "fdiv": {
    #     "names": ["triton.language.fdiv"],
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "int32",
    #             "uint64",
    #             "int64",
    #             "fp64",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #         "a5": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "int32",
    #             "uint64",
    #             "int64",
    #             "fp64",
    #             "bool",
    #         ],
    #     },
    # },
    "flip": {
        "names": ["triton.language.flip"],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
            "a5": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
        },
    },
    "floor": {
        "names": ["triton.language.floor"],
        "unsupported": {
            "a2": [
                "fp64",
            ],
            "a5": [
                "fp64",
            ],
        },
        "supported": {
            "a2": [
                "fp32",
            ],
            "a5": [
                "fp32",
            ],
        },
    },
    # "floordiv": {
    #     "dunder": ["__floordiv__", "__rfloordiv__"],
    #     "dunder_label": "//",
    #     "unsupported": {
    #         "a2": [
    #             "uint16",
    #             "uint32",
    #             "uint64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #         "a5": ["fp16", "fp32", "fp64", "bf16", "fp8e4", "fp8e5", "bool"],
    #     },
    # },
    "fma": {
        "names": ["triton.language.fma"],
        "unsupported": {
            "a2": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "fp64",
                "fp8e4",
                "fp8e5",
                "bool",
            ],
            "a5": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "fp64",
                "bool",
            ],
        },
    },
    # "full": {
    #     "names": ["triton.language.full"],
    #     "tensor_args": [1, 2],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64", "fp8e4", "fp8e5"],
    #     },
    # },
    # "gather": {
    #     "names": ["triton.language.gather"],
    #     "tensor_args": [0],
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "int32",
    #             "uint64",
    #             "int64",
    #             "fp64",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #         "a5": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "int32",
    #             "uint64",
    #             "int64",
    #             "fp64",
    #         ],
    #     },
    # },
    # "ge": {
    #     "dunder": ["__ge__", "__rge__"],
    #     "dunder_label": ">=",
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "gt": {
    #     "dunder": ["__gt__", "__rgt__"],
    #     "dunder_label": ">",
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64", "fp8e4", "fp8e5"],
    #     },
    # },
    "histogram": {
        "names": ["triton.language.histogram"],
        "tensor_args": [0],
        "unsupported": {
            "a2": [
                "uint16",
                "fp16",
                "fp64",
                "bf16",
                "fp8e4",
                "fp8e5",
                "bool",
            ],
            "a5": ["fp16", "fp32", "fp64", "bf16", "fp8e4", "fp8e5", "bool"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
            ],
            "a5": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "int64",
                "uint64",
            ],
        },
    },
    # "inline_asm_elementwise": {
    #     "names": ["triton.language.inline_asm_elementwise"],
    #     "tensor_args": [3],
    #     "unsupported": {
    #         "a2": [
    #             "uint16",
    #             "uint32",
    #             "uint64",
    #             "fp16",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #         "a5": ["uint64", "fp16", "fp64", "bf16", "fp8e4", "fp8e5", "bool"],
    #     },
    # },
    # "interleave": {
    #     "names": ["triton.language.interleave"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "invert": {
    #     "dunder": ["__invert__"],
    #     "dunder_label": "~",
    #     "unsupported": {
    #         "a2": [
    #             "uint16",
    #             "uint32",
    #             "uint64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #         "a5": [
    #             "uint16",
    #             "uint32",
    #             "uint64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #     },
    # },
    # "join": {
    #     "names": ["triton.language.join"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "le": {
    #     "dunder": ["__le__", "__rle__"],
    #     "dunder_label": "<=",
    #     "unsupported": {
    #         "a2": ["uint8", "uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "load": {
    #     "names": ["triton.language.load"],
    #     "tensor_args": [0, 2],
    #     "pointer_args": [0],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    "load_tensor_descriptor": {
        "names": ["triton.language.load_tensor_descriptor"],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5", "bool"],
            "a5": ["fp64", "bool"],
        },
    },
    "log": {
        "names": ["triton.language.log"],
        "unsupported": {
            "a2": [
                "fp64",
            ],
            "a5": [
                "fp64",
            ],
        },
        "supported": {
            "a2": [
                "fp32",
            ],
            "a5": [
                "fp32",
            ],
        },
    },
    "log2": {
        "names": ["triton.language.log2"],
        "unsupported": {
            "a2": [
                "fp64",
            ],
            "a5": [
                "fp64",
            ],
        },
        "supported": {
            "a2": [
                "fp32",
            ],
            "a5": [
                "fp32",
            ],
        },
    },
    "logical_and": {
        "names": ["triton.language.tensor.logical_and"],
        "unsupported": {
            "a2": [
                "uint16",
                "uint32",
                "uint64",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4",
                "fp8e5",
            ],
            "a5": ["fp32", "fp64", "bf16", "fp8e4", "fp8e5"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "bool",
            ],
            "a5": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "bool",
            ],
        },
    },
    "logical_or": {
        "names": ["triton.language.tensor.logical_or"],
        "unsupported": {
            "a2": [
                "uint16",
                "uint32",
                "uint64",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4",
                "fp8e5",
            ],
            "a5": ["fp32", "fp64", "bf16", "fp8e4", "fp8e5"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "bool",
            ],
            "a5": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "bool",
            ],
        },
    },
    "lshift": {
        "dunder": ["__lshift__", "__rlshift__"],
        "dunder_label": "<<",
        "unsupported": {
            "a2": [
                "fp64",
                "fp8e4",
                "fp8e5",
            ],
            "a5": [
                "fp64",
                "fp8e4",
                "fp8e5",
            ],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "bool",
            ],
            "a5": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "bool",
            ],
        },
    },
    # "lt": {
    #     "dunder": ["__lt__", "__rlt__"],
    #     "dunder_label": "<",
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "make_block_ptr": {
    #     "names": ["triton.language.make_block_ptr"],
    #     "pointer_args": [0],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    "make_tensor_descriptor": {
        "names": ["triton.language.make_tensor_descriptor"],
        "pointer_args": [0],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5", "bool"],
            "a5": ["fp64", "bool"],
        },
    },
    # "max": {
    #     "names": ["triton.language.max"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64", "fp8e4", "fp8e5"],
    #     },
    # },
    # "max_constancy": {
    #     "names": ["triton.language.max_constancy"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "max_contiguous": {
    #     "names": ["triton.language.max_contiguous"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "maximum": {
    #     "names": ["triton.language.maximum"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["uint16", "uint32", "fp64"],
    #     },
    # },
    # "min": {
    #     "names": ["triton.language.min"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64", "fp8e4", "fp8e5"],
    #     },
    # },
    # "minimum": {
    #     "names": ["triton.language.minimum"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["uint16", "uint32", "fp64"],
    #     },
    # },
    # "mod": {
    #     "dunder": ["__mod__", "__rmod__"],
    #     "dunder_label": "%",
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "mul": {
    #     "names": ["triton.language.mul"],
    #     "dunder": ["__mul__", "__rmul__"],
    #     "dunder_label": "*",
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "multiple_of": {
    #     "names": ["triton.language.multiple_of"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "ne": {
    #     "dunder": ["__ne__", "__rne__"],
    #     "dunder_label": "!=",
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "neg": {
    #     "dunder": ["__neg__"],
    #     "dunder_label": "unary -",
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "uint16",
    #             "uint32",
    #             "uint64",
    #             "fp64",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #         "a5": [
    #             "uint8",
    #             "uint16",
    #             "uint32",
    #             "uint64",
    #             "fp64",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #     },
    # },
    # "not": {
    #     "dunder": ["__not__"],
    #     "dunder_label": "not",
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "int32",
    #             "uint64",
    #             "int64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #         "a5": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "int32",
    #             "uint64",
    #             "int64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #     },
    # },
    # "or": {
    #     "dunder": ["__or__", "__ror__"],
    #     "dunder_label": "|",
    #     "unsupported": {
    #         "a2": [
    #             "uint16",
    #             "uint32",
    #             "uint64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #         "a5": [
    #             "uint16",
    #             "uint32",
    #             "uint64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #     },
    # },
    # "permute": {
    #     "names": ["triton.language.permute"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "ravel": {
    #     "names": ["triton.language.ravel"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "reduce": {
    #     "names": ["triton.language.reduce"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64", "fp8e4", "fp8e5"],
    #     },
    # },
    "reduce_or": {
        "names": ["triton.language.reduce_or"],
        "unsupported": {
            "a2": [
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4",
                "fp8e5",
            ],
            "a5": ["fp16", "fp32", "fp64", "bf16", "fp8e4", "fp8e5"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
            ],
            "a5": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "int32",
                "uint64",
                "int64",
            ],
        },
    },
    # "reshape": {
    #     "names": ["triton.language.reshape"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    "rshift": {
        "dunder": ["__rshift__", "__rrshift__"],
        "dunder_label": ">>",
        "unsupported": {
            "a2": [
                "uint8",
                "fp64",
                "fp8e4",
                "fp8e5",
            ],
            "a5": [
                "fp64",
                "fp8e4",
                "fp8e5",
            ],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "bool",
            ],
            "a5": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "bool",
            ],
        },
    },
    "rsqrt": {
        "names": ["triton.language.rsqrt"],
        "unsupported": {
            "a2": [
                "fp64",
            ],
            "a5": [
                "fp64",
            ],
        },
        "supported": {
            "a2": [
                "fp32",
            ],
            "a5": [
                "fp32",
            ],
        },
    },
    "sigmoid": {
        "names": ["triton.language.sigmoid"],
        "unsupported": {
            "a2": [
                "fp64",
            ],
            "a5": [
                "fp64",
            ],
        },
        "supported": {
            "a2": [
                "fp32",
            ],
            "a5": [
                "fp32",
            ],
        },
    },
    "sin": {
        "names": ["triton.language.sin"],
        "unsupported": {
            "a2": [
                "fp64",
            ],
            "a5": [
                "fp64",
            ],
        },
        "supported": {
            "a2": [
                "fp32",
            ],
            "a5": [
                "fp32",
            ],
        },
    },
    "softmax": {
        "names": ["triton.language.softmax"],
        "unsupported": {
            "a2": [
                "fp64",
            ],
            "a5": [
                "fp64",
            ],
        },
        "supported": {
            "a2": [
                "fp32",
            ],
            "a5": [
                "fp32",
            ],
        },
    },
    "sort": {
        "names": ["triton.language.sort"],
        "unsupported": {
            "a2": [
                "uint16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "fp64",
                "fp8e4",
                "fp8e5",
                "bool",
            ],
            "a5": ["uint16", "uint32", "uint64", "fp64"],
        },
        "supported": {
            "a2": [
                "int8",
                "int16",
                "fp16",
                "fp32",
                "bf16",
            ],
            "a5": [
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
                "fp8e4",
                "fp8e5",
                "bool",
            ],
        },
    },
    # "split": {
    #     "names": ["triton.language.split"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    "sqrt": {
        "names": ["triton.language.sqrt"],
        "unsupported": {
            "a2": [
                "fp64",
            ],
            "a5": [
                "fp64",
            ],
        },
        "supported": {
            "a2": [
                "fp32",
            ],
            "a5": [
                "fp32",
            ],
        },
    },
    "sqrt_rn": {
        "names": ["triton.language.sqrt_rn"],
        "unsupported": {
            "a2": [
                "fp64",
            ],
            "a5": [
                "fp64",
            ],
        },
        "supported": {
            "a2": [
                "fp32",
            ],
            "a5": [
                "fp32",
            ],
        },
    },
    # "static_assert": {
    #     "names": ["triton.language.static_assert"],
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "int32",
    #             "uint64",
    #             "int64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #         "a5": [
    #             "uint8",
    #             "int8",
    #             "uint16",
    #             "int16",
    #             "uint32",
    #             "int32",
    #             "uint64",
    #             "int64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #     },
    # },
    # "static_print": {
    #     "names": ["triton.language.static_print"],
    #     "unsupported": {
    #         "a2": ["uint8", "uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["uint8", "uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #     },
    # },
    # "store": {
    #     "names": ["triton.language.store"],
    #     "tensor_args": [0, 1],
    #     "pointer_args": [0],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    "store_tensor_descriptor": {
        "names": ["triton.language.store_tensor_descriptor"],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5", "bool"],
            "a5": ["fp64", "bool"],
        },
    },
    # "sub": {
    #     "names": ["triton.language.sub"],
    #     "dunder": ["__sub__", "__rsub__"],
    #     "dunder_label": "-",
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "sum": {
    #     "names": ["triton.language.sum"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["uint8", "uint16", "uint32", "fp64"],
    #     },
    # },
    "swizzle2d": {
        "names": ["triton.language.swizzle2d"],
        "unsupported": {
            "a2": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "uint64",
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4",
                "fp8e5",
                "bool",
            ],
            "a5": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "uint64",
                "fp16",
                "fp32",
                "fp64",
                "bf16",
                "fp8e4",
                "fp8e5",
                "bool",
            ],
        },
    },
    # "topk": {
    #     "names": ["triton.language.topk"],
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "uint16",
    #             "uint32",
    #             "int32",
    #             "uint64",
    #             "int64",
    #             "fp64",
    #             "fp8e4",
    #             "fp8e5",
    #             "bool",
    #         ],
    #         "a5": ["uint8", "uint16", "uint32", "uint64", "fp64"],
    #     },
    # },
    # "trans": {
    #     "names": ["triton.language.trans"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    "umulhi": {
        "names": ["triton.language.umulhi"],
        "unsupported": {
            "a2": [
                "fp64",
            ],
            "a5": [
                "fp64",
            ],
        },
        "supported": {
            "a2": [
                "fp32",
            ],
            "a5": [
                "fp32",
            ],
        },
    },
    "arange": {
        "names": ["triton.language.arange"],
        "unsupported": {
            "a2": ["bf16", "fp16", "fp32", "fp64", "bool"],
            "a5": ["bf16", "fp16", "fp32", "fp64", "bool"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
            ],
            "a5": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "int32",
            ],
        },
    },
    "range": {
        "names": ["triton.language.range"],
        "unsupported": {
            "a2": ["bf16", "fp16", "fp32", "fp64", "bool"],
            "a5": ["bf16", "fp16", "fp32", "fp64", "bool"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
            ],
            "a5": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
            ],
        },
    },
    "static_range": {
        "names": ["triton.language.static_range"],
        "unsupported": {
            "a2": ["bf16", "fp16", "fp32", "fp64", "bool"],
            "a5": ["bf16", "fp16", "fp32", "fp64", "bool"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
            ],
            "a5": [
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
            ],
        },
    },
    "rand": {
        "names": ["triton.language.rand"],
        "unsupported": {
            "a2": [
                "bf16",
                "fp16",
                "fp32",
                "fp64",
            ],
            "a5": [
                "bf16",
                "fp16",
                "fp32",
                "fp64",
            ],
        },
        "supported": {
            "a2": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "bool"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "bool"],
        },
    },
    "randn": {
        "names": ["triton.language.randn"],
        "unsupported": {
            "a2": [
                "bf16",
                "fp16",
                "fp32",
                "fp64",
            ],
            "a5": [
                "bf16",
                "fp16",
                "fp32",
                "fp64",
            ],
        },
        "supported": {
            "a2": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "bool"],
            "a5": ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "bool"],
        },
    },
    # "view": {
    #     "names": ["triton.language.view"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64"],
    #     },
    # },
    # "where": {
    #     "names": ["triton.language.where"],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64", "fp8e4", "fp8e5"],
    #     },
    # },
    # "xor": {
    #     "dunder": ["__xor__", "__rxor__"],
    #     "dunder_label": "^",
    #     "unsupported": {
    #         "a2": [
    #             "uint16",
    #             "uint32",
    #             "uint64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #         "a5": [
    #             "uint16",
    #             "uint32",
    #             "uint64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #     },
    # },
    # "xor_sum": {
    #     "names": ["triton.language.xor_sum"],
    #     "unsupported": {
    #         "a2": [
    #             "uint8",
    #             "uint16",
    #             "uint32",
    #             "uint64",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #         "a5": [
    #             "uint8",
    #             "uint16",
    #             "uint32",
    #             "fp16",
    #             "fp32",
    #             "fp64",
    #             "bf16",
    #             "fp8e4",
    #             "fp8e5",
    #         ],
    #     },
    # },
    # "zeros": {
    #     "names": ["triton.language.zeros"],
    #     "tensor_args": [1],
    #     "unsupported": {
    #         "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
    #         "a5": ["fp64", "fp8e4", "fp8e5"],
    #     },
    # },
    "zeros_like": {
        "names": ["triton.language.zeros_like"],
        "unsupported": {
            "a2": ["uint16", "uint32", "uint64", "fp64", "fp8e4", "fp8e5"],
            "a5": ["fp64", "fp8e4", "fp8e5"],
        },
        "supported": {
            "a2": [
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "fp16",
                "fp32",
                "bf16",
            ],
            "a5": [
                "uint8",
                "int8",
                "int16",
                "uint16",
                "int32",
                "uint32",
                "int64",
                "uint64",
                "fp16",
                "fp32",
                "bf16",
            ],
        },
    },
}


def _build_lookup_indexes():
    by_name = {}
    for rule_id, rule in OP_DTYPE_RULES.items():
        for name in rule.get("names", []):
            by_name[name] = rule_id
    return by_name


RULES_BY_NAME = _build_lookup_indexes()
