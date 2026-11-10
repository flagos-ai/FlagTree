# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Closed native serial program ABI; shared by codegen and ctypes runtime."""
import ctypes
import enum
import hashlib
import json
import struct

from triton.flagmega.errors import RuntimeContractError

PROGRAM_SCHEMA = 'flagmega.cpp-programs/v1'
ABI_VERSION = 3


class ArgTag(enum.IntEnum):
    POINTER = 1
    BOOL = 2
    I32 = 3
    U32 = 4
    I64 = 5
    U64 = 6
    F32 = 7


SCALAR_TAGS = {'bool': ArgTag.BOOL, 'int32': ArgTag.I32, 'uint32': ArgTag.U32,
               'int64': ArgTag.I64, 'uint64': ArgTag.U64, 'float32': ArgTag.F32}
C_TYPES = {ArgTag.POINTER: 'void *', ArgTag.BOOL: 'bool', ArgTag.I32: 'int32_t',
           ArgTag.U32: 'uint32_t', ArgTag.I64: 'int64_t', ArgTag.U64: 'uint64_t', ArgTag.F32: 'float'}


class ArgSlot(ctypes.Structure):
    _fields_ = [('tag', ctypes.c_uint32), ('reserved', ctypes.c_uint32), ('payload', ctypes.c_uint64)]


class ResourceRecord(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        'version', 'record_bytes', 'entry_id', 'device', 'cc_major', 'cc_minor',
        'sm_count', 'warp_size', 'registers_per_thread', 'max_threads_per_block',
        'block_threads', 'resident_blocks_per_sm', 'cooperative_supported', 'reserved')]
    _fields_ += [(name, ctypes.c_uint64) for name in ('static_shared_bytes', 'dynamic_shared_bytes', 'local_bytes')]


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def slot(tag, value):
    """No tensor conversion or device scalar reads; exact closed scalar domain."""
    tag = ArgTag(tag)
    if tag == ArgTag.BOOL:
        if type(value) is not bool:
            raise RuntimeContractError('Native BOOL requires a Python bool.')
        bits = int(value)
    elif tag == ArgTag.F32:
        if type(value) is not float:
            raise RuntimeContractError('Native F32 requires a Python float.')
        try:
            bits = struct.unpack('<I', struct.pack('<f', value))[0]
        except (OverflowError, struct.error) as error:
            raise RuntimeContractError('Native F32 is out of range.') from error
    else:
        if type(value) is not int:
            raise RuntimeContractError('Native integer/pointer requires a Python int.')
        signed = tag in (ArgTag.I32, ArgTag.I64)
        width = 32 if tag in (ArgTag.I32, ArgTag.U32) else 64
        if not (-(1 << (width-1)) if signed else 0) <= value < (1 << (width-1) if signed else 1 << width):
            raise RuntimeContractError('Native argument is out of range.')
        bits = value & ((1 << width)-1)
    return ArgSlot(int(tag), 0, bits)


def validate_slots(slots, signature):
    if len(slots) != len(signature):
        raise RuntimeContractError('Native argument count mismatch.')
    for value, formal in zip(slots, signature, strict=True):
        if value.reserved or value.tag != formal['tag']:
            raise RuntimeContractError('Native argument tag/reserved mismatch.')
        if (value.tag == ArgTag.BOOL and value.payload > 1 or
                value.tag in (ArgTag.I32, ArgTag.U32, ArgTag.F32) and value.payload >> 32):
            raise RuntimeContractError('Native argument has noncanonical payload bits.')


assert ctypes.sizeof(ArgSlot) == 16
assert ctypes.sizeof(ResourceRecord) == 80


COUNTER_EPOCH_ABI_VERSION = 1


class ProgramRow(ctypes.Structure):
    _fields_ = [('args', ctypes.POINTER(ArgSlot)), ('arg_count', ctypes.c_uint32),
                ('reserved', ctypes.c_uint32)]


class PlanStage(enum.IntEnum):
    PREVALIDATE = 1
    RESET = 2
    LAUNCH = 3
    COMPLETE = 4


class PlanResult(ctypes.Structure):
    _fields_ = [('version', ctypes.c_uint32), ('record_bytes', ctypes.c_uint32),
                ('ordinal', ctypes.c_uint32), ('stage', ctypes.c_uint32),
                ('status', ctypes.c_int32), ('submitted', ctypes.c_uint32),
                ('reserved0', ctypes.c_uint32), ('reserved1', ctypes.c_uint32)]


assert ctypes.sizeof(ProgramRow) == 16
assert ctypes.sizeof(PlanResult) == 32
