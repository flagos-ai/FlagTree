# SPDX-License-Identifier: Apache-2.0
# reference https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/pynccl_wrapper.py

import ctypes
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.distributed import ReduceOp

# === export types and functions from flagcx to Python ===
# for the original flagcx definition, please check
# https://github.com/FlagOpen/FlagCX/blob/main/flagcx/include/flagcx.h

flagcxResult_t = ctypes.c_int
flagcxDataType_t = ctypes.c_int
flagcxRedOp_t = ctypes.c_int
flagcxMemcpyType_t = ctypes.c_int
flagcxMemType_t = ctypes.c_int
flagcxEventType_t = ctypes.c_int
flagcxIpcMemHandle_t = ctypes.c_void_p

flagcxComm_t = ctypes.c_void_p
flagcxEvent_t = ctypes.c_void_p
flagcxStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

# Device API types
flagcxDevComm_t = ctypes.c_void_p
flagcxDevMem_t = ctypes.c_void_p
flagcxWindow_t = ctypes.c_void_p

# P2P engine (one-sided RDMA / RPC control plane) types
flagcxP2pEngine_t = ctypes.c_void_p
flagcxP2pConn_t = ctypes.c_void_p
flagcxP2pMr_t = ctypes.c_uint64

# Window flags
FLAGCX_WIN_DEFAULT = 0x00
FLAGCX_WIN_COLL_SYMMETRIC = 0x01


# This wrapper calls the legacy flagcxDevCommCreate symbol, so this ctypes
# layout must remain the frozen legacy ABI prefix.
class flagcxDevCommRequirements(ctypes.Structure):
    _fields_ = [
        ("intraMulticast", ctypes.c_bool),
        ("barrierCount", ctypes.c_int),
        ("intraBarrierCount", ctypes.c_int),
        ("interBarrierCount", ctypes.c_int),
        ("intraLLA2ABlockCount", ctypes.c_int),
        ("intraLLA2ASlotCount", ctypes.c_int),
        ("interForceEnable", ctypes.c_bool),
        ("interContextCount", ctypes.c_int),
        ("interSignalCount", ctypes.c_int),
        ("interCounterCount", ctypes.c_int),
    ]


FLAGCX_UNIQUE_ID_BYTES = 256


class flagcxUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * FLAGCX_UNIQUE_ID_BYTES)]


flagcxUniqueId_t = ctypes.POINTER(flagcxUniqueId)

DEVICE_SYNCHRONIZE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t)
DEVICE_MEMCPY_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                          flagcxMemcpyType_t, flagcxStream_t)
DEVICE_MEMSET_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
                                          flagcxMemType_t, flagcxStream_t)
DEVICE_MALLOC_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                                          flagcxMemType_t, flagcxStream_t)
DEVICE_FREE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.c_void_p, flagcxMemType_t, flagcxStream_t)
SET_DEVICE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.c_int)
GET_DEVICE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(ctypes.c_int))
GET_DEVICE_COUNT_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(ctypes.c_int))
GET_VENDOR_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.c_char_p)
HOST_GET_DEVICE_POINTER_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)

STREAM_CREATE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(flagcxStream_t))
STREAM_DESTROY_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxStream_t)
STREAM_COPY_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(flagcxStream_t), ctypes.c_void_p)
STREAM_FREE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxStream_t)
STREAM_SYNCHRONIZE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxStream_t)
STREAM_QUERY_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxStream_t)
STREAM_WAIT_EVENT_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxStream_t, flagcxEvent_t)

EVENT_CREATE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(flagcxEvent_t), flagcxEventType_t)
EVENT_DESTROY_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxEvent_t)
EVENT_RECORD_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxEvent_t, flagcxStream_t)
EVENT_SYNCHRONIZE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxEvent_t)
EVENT_QUERY_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxEvent_t)
EVENT_ELAPSED_TIME_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(ctypes.c_float), flagcxEvent_t,
                                               flagcxEvent_t)

IPC_MEM_HANDLE_CREATE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(flagcxIpcMemHandle_t),
                                                  ctypes.POINTER(ctypes.c_size_t))
IPC_MEM_HANDLE_GET_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxIpcMemHandle_t, ctypes.c_void_p)
IPC_MEM_HANDLE_OPEN_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxIpcMemHandle_t, ctypes.POINTER(ctypes.c_void_p))
IPC_MEM_HANDLE_CLOSE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.c_void_p)
IPC_MEM_HANDLE_FREE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxIpcMemHandle_t)


class flagcxDeviceHandle(ctypes.Structure):
    _fields_ = [
        # Basic functions
        ("deviceSynchronize", DEVICE_SYNCHRONIZE_FUNCTYPE),
        ("deviceMemcpy", DEVICE_MEMCPY_FUNCTYPE),
        ("deviceMemset", DEVICE_MEMSET_FUNCTYPE),
        ("deviceMalloc", DEVICE_MALLOC_FUNCTYPE),
        ("deviceFree", DEVICE_FREE_FUNCTYPE),
        ("setDevice", SET_DEVICE_FUNCTYPE),
        ("getDevice", GET_DEVICE_FUNCTYPE),
        ("getDeviceCount", GET_DEVICE_COUNT_FUNCTYPE),
        ("getVendor", GET_VENDOR_FUNCTYPE),
        ("hostGetDevicePointer", HOST_GET_DEVICE_POINTER_FUNCTYPE),
        # Stream functions
        ("streamCreate", STREAM_CREATE_FUNCTYPE),
        ("streamDestroy", STREAM_DESTROY_FUNCTYPE),
        ("streamCopy", STREAM_COPY_FUNCTYPE),
        ("streamFree", STREAM_FREE_FUNCTYPE),
        ("streamSynchronize", STREAM_SYNCHRONIZE_FUNCTYPE),
        ("streamQuery", STREAM_QUERY_FUNCTYPE),
        ("streamWaitEvent", STREAM_WAIT_EVENT_FUNCTYPE),
        # Event functions
        ("eventCreate", EVENT_CREATE_FUNCTYPE),
        ("eventDestroy", EVENT_DESTROY_FUNCTYPE),
        ("eventRecord", EVENT_RECORD_FUNCTYPE),
        ("eventSynchronize", EVENT_SYNCHRONIZE_FUNCTYPE),
        ("eventQuery", EVENT_QUERY_FUNCTYPE),
        ("eventElapsedTime", EVENT_ELAPSED_TIME_FUNCTYPE),
        # IpcMemHandle functions
        ("ipcMemHandleCreate", IPC_MEM_HANDLE_CREATE_FUNCTYPE),
        ("ipcMemHandleGet", IPC_MEM_HANDLE_GET_FUNCTYPE),
        ("ipcMemHandleOpen", IPC_MEM_HANDLE_OPEN_FUNCTYPE),
        ("ipcMemHandleClose", IPC_MEM_HANDLE_CLOSE_FUNCTYPE),
        ("ipcMemHandleFree", IPC_MEM_HANDLE_FREE_FUNCTYPE),
    ]


flagcxDeviceHandle_t = ctypes.POINTER(flagcxDeviceHandle)


class flagcxHandlerGroup(ctypes.Structure):
    _fields_ = [
        ("uniqueId", flagcxUniqueId_t),
        ("comm", flagcxComm_t),
        ("devHandle", flagcxDeviceHandle_t),
    ]


flagcxHandlerGroup_t = ctypes.POINTER(flagcxHandlerGroup)


class flagcxDataTypeEnum:
    flagcxInt8 = 0
    flagcxChar = 0
    flagcxUint8 = 1
    flagcxInt32 = 2
    flagcxInt = 2
    flagcxUint32 = 3
    flagcxInt64 = 4
    flagcxUint64 = 5
    flagcxFloat16 = 6
    flagcxHalf = 6
    flagcxFloat32 = 7
    flagcxFloat = 7
    flagcxFloat64 = 8
    flagcxDouble = 8
    flagcxBfloat16 = 9
    flagcxNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.flagcxInt8
        if dtype == torch.uint8:
            return cls.flagcxUint8
        if dtype == torch.int32:
            return cls.flagcxInt32
        if dtype == torch.int64:
            return cls.flagcxInt64
        if dtype == torch.float16:
            return cls.flagcxFloat16
        if dtype == torch.float32:
            return cls.flagcxFloat32
        if dtype == torch.float64:
            return cls.flagcxFloat64
        if dtype == torch.bfloat16:
            return cls.flagcxBfloat16
        # Same convention as PyTorch's NCCL backend (bool -> ncclUint8).
        if dtype == torch.bool:
            return cls.flagcxUint8
        raise ValueError(f"Unsupported dtype: {dtype}")


class flagcxRedOpTypeEnum:
    flagcxSum = 0
    flagcxProd = 1
    flagcxMax = 2
    flagcxMin = 3
    flagcxAvg = 4
    flagcxNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.flagcxSum
        if op == ReduceOp.PRODUCT:
            return cls.flagcxProd
        if op == ReduceOp.MAX:
            return cls.flagcxMax
        if op == ReduceOp.MIN:
            return cls.flagcxMin
        if op == ReduceOp.AVG:
            return cls.flagcxAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class FLAGCXLibrary:
    exported_functions = [
        Function("flagcxDeviceHandleInit", flagcxResult_t, [ctypes.POINTER(flagcxDeviceHandle_t)]),
        Function("flagcxDeviceHandleFree", flagcxResult_t, [flagcxDeviceHandle_t]),
        Function("flagcxGetErrorString", ctypes.c_char_p, [flagcxResult_t]),
        Function("flagcxGetVersion", flagcxResult_t, [ctypes.POINTER(ctypes.c_int)]),
        Function("flagcxGetUniqueId", flagcxResult_t, [ctypes.POINTER(flagcxUniqueId)]),
        # Note that flagcxComm_t is a pointer type, so the first argument
        # is a pointer to a pointer
        Function("flagcxCommInitRank", flagcxResult_t,
                 [ctypes.POINTER(flagcxComm_t), ctypes.c_int,
                  ctypes.POINTER(flagcxUniqueId), ctypes.c_int]),
        # Note that flagcxStream_t is a pointer type, so the last argument
        # is a pointer
        Function(
            "flagcxAllReduce", flagcxResult_t,
            [buffer_type, buffer_type, ctypes.c_size_t, flagcxDataType_t, flagcxRedOp_t, flagcxComm_t, flagcxStream_t]),

        # Note that flagcxStream_t is a pointer type, so the last argument
        # is a pointer
        Function("flagcxReduce", flagcxResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, flagcxDataType_t, flagcxRedOp_t, ctypes.c_int, flagcxComm_t,
            flagcxStream_t
        ]),

        # Note that flagcxStream_t is a pointer type, so the last argument
        # is a pointer
        Function("flagcxAllGather", flagcxResult_t,
                 [buffer_type, buffer_type, ctypes.c_size_t, flagcxDataType_t, flagcxComm_t, flagcxStream_t]),

        # Note that flagcxStream_t is a pointer type, so the last argument
        # is a pointer
        Function(
            "flagcxReduceScatter", flagcxResult_t,
            [buffer_type, buffer_type, ctypes.c_size_t, flagcxDataType_t, flagcxRedOp_t, flagcxComm_t, flagcxStream_t]),
        Function("flagcxSend", flagcxResult_t,
                 [buffer_type, ctypes.c_size_t, flagcxDataType_t, ctypes.c_int, flagcxComm_t, flagcxStream_t]),
        Function("flagcxRecv", flagcxResult_t,
                 [buffer_type, ctypes.c_size_t, flagcxDataType_t, ctypes.c_int, flagcxComm_t, flagcxStream_t]),
        Function(
            "flagcxBroadcast", flagcxResult_t,
            [buffer_type, buffer_type, ctypes.c_size_t, flagcxDataType_t, ctypes.c_int, flagcxComm_t, flagcxStream_t]),
        Function("flagcxGroupStart", flagcxResult_t, [flagcxComm_t]),
        Function("flagcxGroupEnd", flagcxResult_t, [flagcxComm_t]),

        # be cautious! this is a collective call, it will block until all
        # processes in the communicator have called this function.
        # because Python object destruction can happen in random order,
        # it is better not to call it at all.
        # flagcxResult_t flagcxCommDestroy(flagcxComm_t comm);
        Function("flagcxCommDestroy", flagcxResult_t, [flagcxComm_t]),
        Function("flagcxCommRegister", flagcxResult_t,
                 [flagcxComm_t, ctypes.c_void_p, ctypes.c_size_t,
                  ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]),
        Function("flagcxOneSideRegister", flagcxResult_t, [flagcxComm_t, ctypes.c_void_p, ctypes.c_size_t]),
        Function("flagcxOneSideSignalRegister", flagcxResult_t,
                 [flagcxComm_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]),
        Function("flagcxOneSideStagingRegister", flagcxResult_t, [flagcxComm_t, ctypes.c_void_p, ctypes.c_size_t]),
        Function("flagcxOneSideStagingDeregister", flagcxResult_t, [flagcxComm_t]),
        Function(
            "flagcxGet", flagcxResult_t,
            [flagcxComm_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_int
             ]),
        Function(
            "flagcxPut", flagcxResult_t,
            [flagcxComm_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_int
             ]),
        Function("flagcxBatchPut", flagcxResult_t, [
            flagcxComm_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
        ]),
        Function("flagcxPutSignal", flagcxResult_t, [
            flagcxComm_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64
        ]),
        Function("flagcxSignal", flagcxResult_t, [flagcxComm_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_uint64]),
        Function("flagcxWaitSignal", flagcxResult_t,
                 [flagcxComm_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_uint64, flagcxStream_t]),
        Function("flagcxReadCounter", flagcxResult_t, [flagcxComm_t, ctypes.POINTER(ctypes.c_uint64)]),
        Function("flagcxWaitCounter", flagcxResult_t, [flagcxComm_t, ctypes.c_uint64]),

        # Device API — Memory Management
        Function("flagcxMemAlloc", flagcxResult_t, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]),
        Function("flagcxMemFree", flagcxResult_t, [ctypes.c_void_p, ctypes.c_int]),

        # Device API — Window Registration
        Function("flagcxCommWindowRegister", flagcxResult_t, [
            flagcxComm_t, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(flagcxWindow_t), ctypes.c_int, ctypes.c_int
        ]),
        Function("flagcxCommWindowDeregister", flagcxResult_t, [flagcxComm_t, flagcxWindow_t, ctypes.c_int]),

        # Device API — DevComm Lifecycle
        Function(
            "flagcxDevCommCreate", flagcxResult_t,
            [flagcxComm_t, ctypes.POINTER(flagcxDevCommRequirements),
             ctypes.POINTER(flagcxDevComm_t)]),
        Function("flagcxDevCommDestroy", flagcxResult_t, [flagcxComm_t, flagcxDevComm_t]),

        # Device API — DevMem Lifecycle
        Function("flagcxDevMemCreate", flagcxResult_t,
                 [flagcxComm_t, ctypes.c_void_p, ctypes.c_size_t, flagcxWindow_t,
                  ctypes.POINTER(flagcxDevMem_t)]),
        Function("flagcxDevMemDestroy", flagcxResult_t, [flagcxComm_t, flagcxDevMem_t]),

        # Device API — Device Pointer Retrieval
        Function("flagcxDevCommGetDevicePtr", flagcxResult_t,
                 [flagcxDevComm_t, ctypes.POINTER(ctypes.c_void_p)]),
        Function("flagcxDevCommFreeDevicePtr", flagcxResult_t, [flagcxDevComm_t]),
        Function("flagcxDevMemGetDevicePtr", flagcxResult_t,
                 [flagcxDevMem_t, ctypes.POINTER(ctypes.c_void_p)]),
        Function("flagcxDevMemFreeDevicePtr", flagcxResult_t, [flagcxDevMem_t]),
    ]

    p2p_engine_functions = [
        Function("flagcxP2pRpcEngineCreate", flagcxP2pEngine_t, []),
        Function("flagcxP2pRpcEngineDestroy", None, [flagcxP2pEngine_t]),
        Function("flagcxP2pRpcGetPort", ctypes.c_int, [flagcxP2pEngine_t]),
        Function("flagcxP2pRpcStartServer", ctypes.c_int, [flagcxP2pEngine_t]),
        Function("flagcxP2pRpcRegister", ctypes.c_int,
                 [flagcxP2pEngine_t, ctypes.c_uint64, ctypes.c_uint64,
                  ctypes.POINTER(ctypes.c_uint64)]),
        Function("flagcxP2pRpcRegisterHost", ctypes.c_int,
                 [flagcxP2pEngine_t, ctypes.c_uint64, ctypes.c_uint64,
                  ctypes.POINTER(ctypes.c_uint64)]),
        Function("flagcxP2pRpcGetConn", flagcxP2pConn_t, [flagcxP2pEngine_t, ctypes.c_char_p]),
        Function("flagcxP2pRpcBatchWriteSync", ctypes.c_int, [
            flagcxP2pConn_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _find_default_library() -> str:
        import ctypes.util
        import os

        searched = []

        def first_existing(paths):
            for path in paths:
                searched.append(path)
                if os.path.isfile(path):
                    return path
            return None

        # 1. Check FLAGCX_PATH env var. A source tree installs into lib/ or
        #    build/lib/, a packaged (deb/rpm) install ships the versioned
        #    soname only.
        flagcx_path = os.environ.get("FLAGCX_PATH")
        if flagcx_path:
            found = first_existing([
                os.path.join(flagcx_path, "lib", "libflagcx.so"),
                os.path.join(flagcx_path, "build", "lib", "libflagcx.so"),
                os.path.join(flagcx_path, "lib", "libflagcx.so.0"),
            ])
            if found:
                return found
        # An unusable FLAGCX_PATH is not fatal: keep looking, and only report
        # it if nothing anywhere resolves.
        # 2. Check alongside the installed flagcx Python package
        #    (setup.py copies libflagcx.so into the package's lib/ directory)
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        found = first_existing([os.path.join(pkg_dir, "lib", "libflagcx.so")])
        if found:
            return found
        # 3. Fall back to <repo_root>/build/lib/libflagcx.so
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        found = first_existing([os.path.join(repo_root, "build", "lib", "libflagcx.so")])
        if found:
            return found
        # 4. Fall back to a system-wide install, resolved the way the dynamic
        #    loader would resolve it (find_library consults ldconfig).
        system_so = ctypes.util.find_library("flagcx")
        if system_so:
            return system_so
        raise FileNotFoundError("Cannot find libflagcx.so. Searched:\n" +
                                "".join(f"  - {path}\n" for path in searched) +
                                "  - system library lookup for 'flagcx' (not found)\n" +
                                "Set FLAGCX_PATH to a FlagCX build tree or install " + "the FlagCX runtime package.")

    def __init__(self, so_file: Optional[str] = None):
        if so_file is None:
            so_file = FLAGCXLibrary._find_default_library()

        try:
            if so_file not in FLAGCXLibrary.path_to_library_cache:
                lib = ctypes.CDLL(so_file)
                FLAGCXLibrary.path_to_library_cache[so_file] = lib
            self.lib = FLAGCXLibrary.path_to_library_cache[so_file]
        except Exception as e:
            raise e

        if so_file not in FLAGCXLibrary.path_to_dict_mapping:
            _funcs: Dict[str, Any] = {}
            for func in FLAGCXLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            # Best-effort: P2P engine symbols may be absent in older libs.
            for func in FLAGCXLibrary.p2p_engine_functions:
                try:
                    f = getattr(self.lib, func.name)
                except AttributeError:
                    continue
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            FLAGCXLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = FLAGCXLibrary.path_to_dict_mapping[so_file]

        # init flagcx device handle to call device-related apis
        self.devHandle = flagcxDeviceHandle_t()
        self.FLAGCX_CHECK(self._funcs["flagcxDeviceHandleInit"](ctypes.byref(self.devHandle)))

    def __del__(self):
        # free flagcx device handle
        if hasattr(self, '_funcs') and hasattr(self, 'devHandle'):
            self.FLAGCX_CHECK(self._funcs["flagcxDeviceHandleFree"](self.devHandle))

    def flagcxGetErrorString(self, result: flagcxResult_t) -> str:
        return self._funcs["flagcxGetErrorString"](result).decode("utf-8")

    def FLAGCX_CHECK(self, result: flagcxResult_t) -> None:
        if result != 0:
            error_str = self.flagcxGetErrorString(result)
            raise RuntimeError(f"FLAGCX error: {error_str}")

    def flagcxGetVersion(self) -> str:
        version = ctypes.c_int()
        self.FLAGCX_CHECK(self._funcs["flagcxGetVersion"](ctypes.byref(version)))
        code = version.value
        major = code // 10000
        minor = (code // 100) % 100
        patch = code % 100
        return f"{major}.{minor}.{patch}"

    def flagcxGetUniqueId(self) -> flagcxUniqueId:
        unique_id = flagcxUniqueId()
        self.FLAGCX_CHECK(self._funcs["flagcxGetUniqueId"](ctypes.byref(unique_id)))
        return unique_id

    def unique_id_from_bytes(self, data: bytes) -> flagcxUniqueId:
        """
        Reconstructs a flagcxUniqueId object from bytes data.
        Args:
            data: Must match the size of FlagCX's unique_id.
        Returns:
            flagcxUniqueId: The reconstructed FlagCX Unique ID object.
        Raises:
            ValueError: If the input data length does not match unique_id.
        """
        if len(data) != FLAGCX_UNIQUE_ID_BYTES:
            raise ValueError(f"Expected {FLAGCX_UNIQUE_ID_BYTES} bytes for flagcxUniqueId, "
                             f"got {len(data)} bytes")

        unique_id = flagcxUniqueId()
        ctypes.memmove(ctypes.addressof(unique_id.internal), data, FLAGCX_UNIQUE_ID_BYTES)
        return unique_id

    def flagcxCommInitRank(self, world_size: int, unique_id: flagcxUniqueId, rank: int) -> flagcxComm_t:
        comm = flagcxComm_t()
        self.FLAGCX_CHECK(self._funcs["flagcxCommInitRank"](ctypes.byref(comm), world_size, ctypes.byref(unique_id),
                                                            rank))
        return comm

    def flagcxAllReduce(self, sendbuff: buffer_type, recvbuff: buffer_type, count: int, datatype: int, op: int,
                        comm: flagcxComm_t, stream: flagcxStream_t) -> None:
        # `datatype` actually should be `flagcxDataType_t`
        # and `op` should be `flagcxRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.FLAGCX_CHECK(self._funcs["flagcxAllReduce"](sendbuff, recvbuff, count, datatype, op, comm, stream))

    def flagcxReduce(self, sendbuff: buffer_type, recvbuff: buffer_type, count: int, datatype: int, op: int, root: int,
                     comm: flagcxComm_t, stream: flagcxStream_t) -> None:
        # `datatype` actually should be `flagcxDataType_t`
        # and `op` should be `flagcxRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.FLAGCX_CHECK(self._funcs["flagcxReduce"](sendbuff, recvbuff, count, datatype, op, root, comm, stream))

    def flagcxReduceScatter(self, sendbuff: buffer_type, recvbuff: buffer_type, count: int, datatype: int, op: int,
                            comm: flagcxComm_t, stream: flagcxStream_t) -> None:
        # `datatype` actually should be `flagcxDataType_t`
        # and `op` should be `flagcxRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.FLAGCX_CHECK(self._funcs["flagcxReduceScatter"](sendbuff, recvbuff, count, datatype, op, comm, stream))

    def flagcxAllGather(self, sendbuff: buffer_type, recvbuff: buffer_type, count: int, datatype: int,
                        comm: flagcxComm_t, stream: flagcxStream_t) -> None:
        # `datatype` actually should be `flagcxDataType_t`
        # which is an aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.FLAGCX_CHECK(self._funcs["flagcxAllGather"](sendbuff, recvbuff, count, datatype, comm, stream))

    def flagcxSend(self, sendbuff: buffer_type, count: int, datatype: int, dest: int, comm: flagcxComm_t,
                   stream: flagcxStream_t) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxSend"](sendbuff, count, datatype, dest, comm, stream))

    def flagcxRecv(self, recvbuff: buffer_type, count: int, datatype: int, src: int, comm: flagcxComm_t,
                   stream: flagcxStream_t) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxRecv"](recvbuff, count, datatype, src, comm, stream))

    def flagcxBroadcast(self, sendbuff: buffer_type, recvbuff: buffer_type, count: int, datatype: int, root: int,
                        comm: flagcxComm_t, stream: flagcxStream_t) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxBroadcast"](sendbuff, recvbuff, count, datatype, root, comm, stream))

    def flagcxGroupStart(self, comm: flagcxComm_t) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxGroupStart"](comm))

    def flagcxGroupEnd(self, comm: flagcxComm_t) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxGroupEnd"](comm))

    def flagcxCommDestroy(self, comm: flagcxComm_t) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxCommDestroy"](comm))

    def flagcxCommRegister(self, comm: flagcxComm_t, buff: int, size: int, allocator: int = 0) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        self.FLAGCX_CHECK(self._funcs["flagcxCommRegister"](comm, ctypes.c_void_p(buff), ctypes.c_size_t(size),
                                                            ctypes.byref(handle), ctypes.c_int(allocator)))
        return handle

    def flagcxOneSideRegister(self, comm: flagcxComm_t, buff: int, size: int) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxOneSideRegister"](comm, ctypes.c_void_p(buff), ctypes.c_size_t(size)))

    def flagcxOneSideSignalRegister(self, comm: flagcxComm_t, buff: int, size: int, ptrType: int = 1) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxOneSideSignalRegister"](comm, ctypes.c_void_p(buff), ctypes.c_size_t(size),
                                                                     ctypes.c_int(ptrType)))

    def flagcxOneSideStagingRegister(self, comm: flagcxComm_t, buff: int, size: int) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxOneSideStagingRegister"](comm, ctypes.c_void_p(buff),
                                                                      ctypes.c_size_t(size)))

    def flagcxGet(self, comm: flagcxComm_t, peer: int, srcOffset: int, dstOffset: int, size: int, srcMrIdx: int,
                  dstMrIdx: int) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxGet"](comm, peer, ctypes.c_size_t(srcOffset), ctypes.c_size_t(dstOffset),
                                                   ctypes.c_size_t(size), srcMrIdx, dstMrIdx))

    def flagcxPut(self, comm: flagcxComm_t, peer: int, srcOffset: int, dstOffset: int, size: int, srcMrIdx: int,
                  dstMrIdx: int) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxPut"](comm, peer, ctypes.c_size_t(srcOffset), ctypes.c_size_t(dstOffset),
                                                   ctypes.c_size_t(size), srcMrIdx, dstMrIdx))

    def flagcxBatchPut(self, comm: flagcxComm_t, peer: int, srcOffsets: list[int], dstOffsets: list[int],
                       sizes: list[int], srcMrIdxs: list[int], dstMrIdxs: list[int]) -> None:
        count = len(sizes)
        if count == 0:
            return
        if (len(srcOffsets) != count or len(dstOffsets) != count or len(srcMrIdxs) != count or len(dstMrIdxs) != count):
            raise ValueError("flagcxBatchPut argument lengths do not match")
        size_array = ctypes.c_size_t * count
        int_array = ctypes.c_int * count
        self.FLAGCX_CHECK(self._funcs["flagcxBatchPut"](comm, peer, size_array(*srcOffsets), size_array(*dstOffsets),
                                                        size_array(*sizes), int_array(*srcMrIdxs),
                                                        int_array(*dstMrIdxs), ctypes.c_size_t(count)))

    def flagcxPutSignal(self, comm: flagcxComm_t, peer: int, srcOffset: int, dstOffset: int, size: int,
                        signalOffset: int, srcMrIdx: int, dstMrIdx: int, signalValue: int) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxPutSignal"](comm, peer, ctypes.c_size_t(srcOffset),
                                                         ctypes.c_size_t(dstOffset), ctypes.c_size_t(size),
                                                         ctypes.c_size_t(signalOffset), srcMrIdx, dstMrIdx,
                                                         ctypes.c_uint64(signalValue)))

    def flagcxSignal(self, comm: flagcxComm_t, peer: int, signalOffset: int, signalValue: int) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxSignal"](comm, peer, ctypes.c_size_t(signalOffset),
                                                      ctypes.c_uint64(signalValue)))

    def flagcxWaitSignal(self, comm: flagcxComm_t, peer: int, signalOffset: int, expected: int,
                         stream: flagcxStream_t) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxWaitSignal"](comm, peer, ctypes.c_size_t(signalOffset),
                                                          ctypes.c_uint64(expected), stream))

    def flagcxReadCounter(self, comm: flagcxComm_t) -> int:
        count = ctypes.c_uint64(0)
        self.FLAGCX_CHECK(self._funcs["flagcxReadCounter"](comm, ctypes.byref(count)))
        return count.value

    def flagcxWaitCounter(self, comm: flagcxComm_t, target: int) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxWaitCounter"](comm, ctypes.c_uint64(target)))

    def flagcxMemAlloc(self, size: int, allocator: int = 0) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self.FLAGCX_CHECK(self._funcs["flagcxMemAlloc"](ctypes.byref(ptr), ctypes.c_size_t(size),
                                                        ctypes.c_int(allocator)))
        return ptr

    def flagcxMemFree(self, ptr: ctypes.c_void_p, allocator: int = 0) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxMemFree"](ptr, ctypes.c_int(allocator)))

    def flagcxCommWindowRegister(self, comm: flagcxComm_t, buff: int, size: int, flags: int = 0,
                                 allocator: int = 0) -> flagcxWindow_t:
        win = flagcxWindow_t()
        self.FLAGCX_CHECK(self._funcs["flagcxCommWindowRegister"](comm, ctypes.c_void_p(buff), ctypes.c_size_t(size),
                                                                  ctypes.byref(win), ctypes.c_int(flags),
                                                                  ctypes.c_int(allocator)))
        return win

    def flagcxCommWindowDeregister(self, comm: flagcxComm_t, win: flagcxWindow_t, allocator: int = 0) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxCommWindowDeregister"](comm, win, ctypes.c_int(allocator)))

    def flagcxDevCommCreate(self, comm: flagcxComm_t, reqs: flagcxDevCommRequirements) -> flagcxDevComm_t:
        dev_comm = flagcxDevComm_t()
        self.FLAGCX_CHECK(self._funcs["flagcxDevCommCreate"](comm, ctypes.byref(reqs), ctypes.byref(dev_comm)))
        return dev_comm

    def flagcxDevCommDestroy(self, comm: flagcxComm_t, dev_comm: flagcxDevComm_t) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxDevCommDestroy"](comm, dev_comm))

    def flagcxDevMemCreate(self, comm: flagcxComm_t, buff: int, size: int,
                           win: Optional[flagcxWindow_t] = None) -> flagcxDevMem_t:
        dev_mem = flagcxDevMem_t()
        self.FLAGCX_CHECK(self._funcs["flagcxDevMemCreate"](comm, ctypes.c_void_p(buff), ctypes.c_size_t(size),
                                                            win if win is not None else flagcxWindow_t(),
                                                            ctypes.byref(dev_mem)))
        return dev_mem

    def flagcxDevMemDestroy(self, comm: flagcxComm_t, dev_mem: flagcxDevMem_t) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxDevMemDestroy"](comm, dev_mem))

    def flagcxDevCommGetDevicePtr(self, dev_comm: flagcxDevComm_t) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self.FLAGCX_CHECK(self._funcs["flagcxDevCommGetDevicePtr"](dev_comm, ctypes.byref(ptr)))
        return ptr

    def flagcxDevCommFreeDevicePtr(self, dev_comm: flagcxDevComm_t) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxDevCommFreeDevicePtr"](dev_comm))

    def flagcxDevMemGetDevicePtr(self, dev_mem: flagcxDevMem_t) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self.FLAGCX_CHECK(self._funcs["flagcxDevMemGetDevicePtr"](dev_mem, ctypes.byref(ptr)))
        return ptr

    def flagcxDevMemFreeDevicePtr(self, dev_mem: flagcxDevMem_t) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxDevMemFreeDevicePtr"](dev_mem))

    def flagcxP2pEngineCreate(self) -> flagcxP2pEngine_t:
        if "flagcxP2pRpcEngineCreate" not in self._funcs:
            raise RuntimeError("libflagcx.so lacks P2P engine symbols; rebuild FlagCX with "
                               "USE_IBUC=1 (or the IB-capable backend).")
        engine = self._funcs["flagcxP2pRpcEngineCreate"]()
        if not engine:
            raise RuntimeError("flagcxP2pEngineCreate returned NULL")
        return ctypes.c_void_p(engine)

    def flagcxP2pEngineDestroy(self, engine: flagcxP2pEngine_t) -> None:
        self._funcs["flagcxP2pRpcEngineDestroy"](engine)

    def flagcxP2pGetRpcPort(self, engine: flagcxP2pEngine_t) -> int:
        port = self._funcs["flagcxP2pRpcGetPort"](engine)
        if port < 0:
            raise RuntimeError("flagcxP2pGetRpcPort failed")
        return int(port)

    def flagcxP2pStartRpcServer(self, engine: flagcxP2pEngine_t) -> None:
        if self._funcs["flagcxP2pRpcStartServer"](engine) != 0:
            raise RuntimeError("flagcxP2pStartRpcServer failed")

    def flagcxP2pRegister(self, engine: flagcxP2pEngine_t, addr: int, size: int) -> int:
        mr_id = ctypes.c_uint64(0)
        rc = self._funcs["flagcxP2pRpcRegister"](engine, ctypes.c_uint64(addr), ctypes.c_uint64(size),
                                                 ctypes.byref(mr_id))
        if rc != 0:
            raise RuntimeError(f"flagcxP2pRegister failed (addr={hex(addr)}, size={size})")
        return mr_id.value

    def flagcxP2pRegisterHost(self, engine: flagcxP2pEngine_t, addr: int, size: int) -> int:
        mr_id = ctypes.c_uint64(0)
        rc = self._funcs["flagcxP2pRpcRegisterHost"](engine, ctypes.c_uint64(addr), ctypes.c_uint64(size),
                                                     ctypes.byref(mr_id))
        if rc != 0:
            raise RuntimeError(f"flagcxP2pRegisterHost failed (addr={hex(addr)}, size={size})")
        return mr_id.value

    def flagcxP2pGetConn(self, engine: flagcxP2pEngine_t, session: str) -> flagcxP2pConn_t:
        conn = self._funcs["flagcxP2pRpcGetConn"](engine, session.encode("utf-8"))
        if not conn:
            raise RuntimeError(f"flagcxP2pGetConn failed for session {session}")
        return ctypes.c_void_p(conn)

    def flagcxP2pBatchWriteSync(self, conn: flagcxP2pConn_t, src_vaddrs: list[int], dst_vaddrs: list[int],
                                sizes: list[int]) -> None:
        count = len(sizes)
        if count == 0:
            return
        if len(src_vaddrs) != count or len(dst_vaddrs) != count:
            raise ValueError("flagcxP2pBatchWriteSync argument lengths do not match")
        u64 = ctypes.c_uint64 * count
        rc = self._funcs["flagcxP2pRpcBatchWriteSync"](conn, ctypes.c_int(count), u64(*src_vaddrs), u64(*dst_vaddrs),
                                                       u64(*sizes))
        if rc != 0:
            raise RuntimeError("flagcxP2pBatchWriteSync failed")

    def adaptor_stream_create(self):
        new_stream = flagcxStream_t()
        self.FLAGCX_CHECK(self.devHandle.contents.streamCreate(ctypes.byref(new_stream)))
        return new_stream

    def adaptor_stream_copy(self, old_stream):
        new_stream = flagcxStream_t()
        attr_names = ['txda_stream', 'musa_stream', 'npu_stream', 'cuda_stream']
        raw_stream = 0
        for name in attr_names:
            if hasattr(old_stream, name):
                raw_stream = getattr(old_stream, name)
                break
        self.FLAGCX_CHECK(self.devHandle.contents.streamCopy(ctypes.byref(new_stream), ctypes.c_void_p(raw_stream)))
        return new_stream

    def adaptor_stream_free(self, stream):
        self.FLAGCX_CHECK(self.devHandle.contents.streamFree(stream))

    def adaptor_stream_destroy(self, stream):
        self.FLAGCX_CHECK(self.devHandle.contents.streamDestroy(stream))

    def sync_stream(self, stream):
        self.FLAGCX_CHECK(self.devHandle.contents.streamSynchronize(stream))


__all__ = [
    "FLAGCX_UNIQUE_ID_BYTES", "FLAGCXLibrary", "flagcxDataTypeEnum", "flagcxRedOpTypeEnum", "flagcxUniqueId",
    "flagcxDeviceHandle_t", "flagcxComm_t", "flagcxStream_t", "flagcxEvent_t", "buffer_type"
]
