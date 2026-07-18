////////////////////////////////////////////////////////
// @file tang.h
// tang DRIVER INTERFACE
// @author linan<linan@sensetime.com>
////////////////////////////////////////////////////////

#ifndef _TANG_H_
#define _TANG_H_
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define TA_VERSION_MAJOR 0
#define TA_VERSION_MINOR 13
#define TA_VERSION_PATCH 0

#define TA_VERSION \
  ((TA_VERSION_MAJOR * 1000) + (TA_VERSION_MINOR * 10) + TA_VERSION_PATCH)

#if defined(_MSC_VER)
#define TA_DEPRECATED __declspec(deprecated)
#define TA_API_EXPORT __declspec(dllexport)
#define TA_API_IMPORT __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#define TA_DEPRECATED __attribute__((deprecated))
#define TA_API_EXPORT __attribute__((visibility("default")))
#define TA_API_IMPORT __attribute__((visibility("default")))
#else
#define TA_DEPRECATED
#define TA_API_EXPORT
#define TA_API_IMPORT
#endif  //! UNKNOWN COMPILER

#if defined(tang_EXPORTS)
#define TA_API TA_API_EXPORT
#else
#define TA_API TA_API_IMPORT
#endif  //! For user

#define COMMAND_MAGIC 0x100d58ba
enum COMMAND_SA {
  COMMAND_ASYNC = 0,
  COMMAND_SYNC,
};

enum MODE_TYPE {
  PT_MGR_TYPE = 1,
  PT_COLL_TYPE,
  OTHER_TYPE,
};

enum OPERATIONS {
  OPS_LINK_EN = 1,
  OPS_LINK_DIS,
  OPS_LINK_DETECT,
  OPS_LINK_PORTADDR = 4,
  OPS_LINK_BIF,
  OPS_LINK_P2P_ATTR,
  OPS_PEER_ACCESS_CAN = 7,
  OPS_PEER_ACCESS_EN,
  OPS_PEER_ACCESS_DIS,
  OPS_LINK_INIT,
  OPS_LINK_PORT_INIT,
};

enum COLL_OPS {
  COLL_BROADCAST = 0,
  COLL_REDUCE,
  COLL_ALLGATHER,
  COLL_REDUCESCATTER,
  COLL_ALLREDUCE,
  COLL_MAX_OPS
};

enum COMMANDS {
  CMD_MAGIC = 0,
  CMD_SYNC,
  MODE_TYPE,
  CMD_ID,
  PORT,
  DEV_ID,
  RDEV_ID,
  MSGIN_LEN,
  MSGOUT_LEN,
};

struct scp_msg_ack {
  int  retval;      // FW irq return value
  int  status;      // simple status value
  char payload[0];  // complex struct return
};

#define C2CSCP_MSG_HEAD (8)

typedef uint64_t TAdeviceptr;  //!< TANG device pointer

#define TAdevice_nullptr (TAdeviceptr)0

typedef struct TAdevice_s*    TAdevice;     //!< TANG device
typedef struct TActx_s*       TAcontext;    //!< TANG context
typedef struct TAfunc_s*      TAfunction;   //!< TANG function handle
typedef struct TAevent_s*     TAevent;      //!< TANG event handle
typedef struct TAstream_s*    TAstream;     //!< TANG stream handle
typedef struct TAmodule_s*    TAmodule;     //!< TANG module handle
typedef struct TAvariable_s*  TAvariable;   //!< TANG variable
typedef struct TAgraph_s*     TAgraph;      //!< TANG graph handle
typedef struct TAgraphExec_s* TAgraphExec;  //!< TANG graph exec handle
typedef struct TAgraphNode_s* TAgraphNode;  //!< TANG graph node

typedef struct TAdsoWrapper_s {
  uintptr_t data[20];
} TAdsoWrapper_t;

/**
 * @brief Stream flags.
 * @sa __s2StreamFlags.
 */
typedef enum TAstream_flags_e {
  TA_STREAM_DEFAULT      = 0x0,  //!< The default stream creation flag.
  TA_STREAM_NON_BLOCKING = 0x1,  //!< The non blocking stream creation flag.
  //! TA_STREAM_LEGACY    = 0x2,  //!< The legacy stream creation flag.
  //!                            //!< This flag can only be used internally.
  //!                            //!< User use this flag will cause
  //!                            //!< ::TANG_ERROR_INVALID_VALUE error.
} TAstream_flags;

typedef enum TAstreamCaptureMode_e {
  TA_STREAM_CAPTURE_MODE_GLOBAL       = 0,
  TA_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
  TA_STREAM_CAPTURE_MODE_RELAXED      = 2,
} TAstreamCaptureMode;

typedef enum TAstreamCaptureStatus_e {
  TA_STREAM_CAPTURE_STATUS_NONE        = 0,
  TA_STREAM_CAPTURE_STATUS_ACTIVE      = 1,
  TA_STREAM_CAPTURE_STATUS_INVALIDATED = 2,
} TAstreamCaptureStatus;

typedef struct TAgraphInfo_s {
  int nr_nodes;
} TAgraphInfo;

typedef struct TAeventTimestamp_s {
  uint64_t comp;
  uint64_t comp_sw;
  uint64_t create;
  uint64_t enqueue;
  uint64_t writeq_beg;
  uint64_t writeq_end;
} TAeventTimestamp;

typedef enum TAevent_record_flags_e {
  //!< The default record flag
  TA_EVENT_RECORD_DEFAULT = 0,

  //!< Require hardware event
  TA_EVENT_RECORD_HW = 0x0100,

  //!< Require software event
  TA_EVENT_RECORD_SW = 0x0200,

  //!< Allow waiting while allocating hardware event.
  TA_EVENT_RECORD_ALLOW_BLOCKING = 0x0400,
} TAevent_record_flags;

typedef enum TAevent_flags_e {
  TA_EVENT_DISABLE_TIMING = 0x02,
  TA_EVENT_INTERPROCESS   = 0x04,
} TAevent_flags;

typedef enum TAevent_sync_flags_e {
  //!< The default synchronization behaviour.
  //!< 1. If the event has not been recorded,
  //!< taEventSynchronize will return imediately.
  //!< 2. If the event has been recorded,
  //!< taEventSynchronize will block until the
  //!< event is done.
  TA_EVENT_SYNC_DEFAULT = 0x00,

  //!< Block until the event is recorded and done.
  TA_EVENT_SYNC_RECORDED_AND_DONE = 0x01,
} TAevent_sync_flags;

#define TA_IPC_HANDLE_SIZE   64U

struct TAipcMemHandle_s {
  unsigned long reserved[TA_IPC_HANDLE_SIZE / sizeof(unsigned long)];
};
typedef struct TAipcMemHandle_s TAipcMemHandle;

struct TAipcEventHandle_s {
  unsigned long reserved[TA_IPC_HANDLE_SIZE / sizeof(unsigned long)];
};
typedef struct TAipcEventHandle_s TAipcEventHandle;

enum TAipcMem_flags_e {
  TA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x01,
};
typedef enum TAipcMem_flags_e TAipcMem_flags;

#define TA_LAUNCH_PARAM_END 0
#define TA_LAUNCH_PARAM_INVALIDATE_L1P5 1
#define TA_LAUNCH_PARAM_ICACHE_FLUSH 2
#define TA_LAUNCH_PARAM_WORK_MODE 3
#define TA_LAUNCH_PARAM_MAX_ACTIVE_BLOCK_COUNT_PER_CU 4
#define TA_LAUNCH_PARAM_SHARE_MEM_MIRROR 5
#define TA_LAUNCH_PARAM_CLST_DIMX 6
#define TA_LAUNCH_PARAM_CLST_DIMY 7
#define TA_LAUNCH_PARAM_CLST_DIMZ 8

struct TAextraLaunchParam_s {
  unsigned long type;
  union {
    unsigned long val;
    void *ptr;
  };
};
typedef struct TAextraLaunchParam_s TAextraLaunchParam;

typedef enum TAmemorytype_e {
  TA_MEMORYTYPE_HOST    = 0x01,
  TA_MEMORYTYPE_DEVICE  = 0x02,
  TA_MEMORYTYPE_ARRAY   = 0x04,
  TA_MEMORYTYPE_UNIFIED = 0x05,
} TAmemorytype;

typedef enum TApointer_attribute_e {
  /**< The ::TAcontext on which a pointer is allocated and registered */
  TA_POINTER_ATTRIBUTE_CONTEXT = 1,

  /**< The ::TAmemorytype describing the physical location of a pointer */
  TA_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,

  TA_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
  TA_POINTER_ATTRIBUTE_HOST_POINTER   = 4,
  TA_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
} TApointer_attribute;

typedef enum TAfunction_attribute_e {
  // The maximum number of threads per block, beyond which
  // a lanuch of the function would fail.
  // This value depends on the function and the device.
  TA_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,

  // The number of bytes statically allocated shared memory.
  TA_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,

  // The number of bytes of user allocated constant memory.
  // This attribute is not implemented in pt200
  TA_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,

  // The number of bytes of local memory used by each thread of the function.
  TA_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,

  // The number of registers used by each thread of this function.
  TA_FUNC_ATTRIBUTE_NUM_REGS = 4,

  // The maximum size of dynamically allocated shared memory that
  // can be used by this function.
  TA_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
} TAfunction_attribute;

typedef enum TAgraphNodeType_enum {
  TA_GRAPH_NODE_TYPE_KERNEL       = 0,
  TA_GRAPH_NODE_TYPE_MEMCPY       = 1,
  TA_GRAPH_NODE_TYPE_MEMSET       = 2,
  TA_GRAPH_NODE_TYPE_HOST         = 3,
  TA_GRAPH_NODE_TYPE_GRAPH        = 4,
  TA_GRAPH_NODE_TYPE_EMPTY        = 5,
  TA_GRAPH_NODE_TYPE_WAIT_EVENT   = 6,
  TA_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
} TAgraphNodeType;

typedef enum TAmoduleSymbolType_enum {
  TA_MODULE_SYMBOL_FUNCTION = 1,
  TA_MODULE_SYMBOL_VARIABLE = 2,
} TAmoduleSymbolType;

typedef struct TAmoduleSymbolHandle_s {
  union {
    TAfunction function;
    TAvariable variable;
  };
#ifdef __cplusplus
  TAmoduleSymbolHandle_s()
    : function(nullptr) {}

  explicit TAmoduleSymbolHandle_s(TAfunction func)
    : function(func) {}

  explicit TAmoduleSymbolHandle_s(TAvariable var)
    : variable(var) {}
#endif  // __cplusplus
} TAmoduleSymbolHandle;

/**
 * @brief TAmodule symbols iteration call back function type.
 * @note If the function returns true, the iteration will stop.
 * Always returns false to iterate all symbols.
 */
// typedef bool (*TAmoduleSymbolIterateFn)(TAmoduleSymbolType   symbolType,
//                                         const char*          symbolName,
//                                         TAmoduleSymbolHandle symbolHandle,
//                                         void*                userData);

typedef void (*TAhostFn)(void* userData);

typedef struct TANG_HOST_NODE_PARAMS {
  TAhostFn fn;
  void*    userData;
} TANG_HOST_NODE_PARAMS;

typedef struct TANG_KERNEL_NODE_PARAMS_s {
  TAfunction func;

  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;

  void** kernelParams;
  void** extra;
} TANG_KERNEL_NODE_PARAMS;

#define L2_CACHE_MP_CNT 4
#define L2_CACHE_MPX_SUBCNT 16
#define WARP_SIZE 32
#define BUFFER_SIZE 32
#define IPATH_PERF_CNT 10
#define EXT_PERF_CNT 13
#define SHM_PERF_CNT 15
#define CLUSTER_CNT 12
#define SUBCORE_CNT 8
#define CLUSTER_CNT_HALF 6
#define CLUSTER_CNT_3 3

typedef struct ptProfileInfo_s {
  uint32_t blockDim_x;
  uint32_t blockDim_y;
  uint32_t blockDim_z;
  uint32_t gridDim_x;
  uint32_t gridDim_y;
  uint32_t gridDim_z;
  int __s2Stream_t;
  int __s2Context_t;
  int regs;
  int device;
  int SSMem;
  int DSMem;
  uint32_t max_bkcnt;
  uint32_t thread_cnt;
  float    waves_per_sm;
  uint32_t max_warps_per_sched;
  float    perf_blk_occupation;
  uint32_t achieved_active_warps_per_sm;
  uint32_t blk_limit_reg;
  uint32_t blk_limit_shared_mem;
  uint32_t local_memory_size;
  uint32_t blk_shm_size;
  char funcName[512];
  uint64_t knlTime;
  uint32_t knlSubTime[96];
  uint64_t ipathcnt[38]; //!< ipath-10 ext-13 shm-15 perf
  uint64_t l2cache[6]; //!< [1]Monitor Read Hit Counter               [Offset: 0x610]
                       //!< [2]Monitor Cacheable Read Request Counter [Offset: 0x608]
                       //!< [3]Monitor Read Request Counter           [Offset: 0x600]
                       //!< [4]Monitor Cacheable Write Request Counter[Offset: 0x60c]
                       //!< [5]Monitor Write Request Counter          [Offset: 0x604]
                       //!< [6]Monitor Write Hit Counter              [Offset: 0x614]
  uint64_t l1p5cache[6]; //!< [1]Monitor Read Hit Counter               [Offset: 0x610]
                        //!< [2]Monitor Cacheable Read Request Counter [Offset: 0x608]
                        //!< [3]Monitor Read Request Counter           [Offset: 0x600]
                        //!< [4]Monitor Cacheable Write Request Counter[Offset: 0x60c]
                        //!< [5]Monitor Write Request Counter          [Offset: 0x604]
                        //!< [6]Monitor Write Hit Counter              [Offset: 0x614]
  uint32_t clock;
  uint32_t knlInfo[3];  //!< blk_cnt\knl_rcvd_cnt\knl_cmpl_cnt
  uint32_t l2HitRate[L2_CACHE_MP_CNT*L2_CACHE_MPX_SUBCNT][4];
  uint32_t l1p5HitRate[L2_CACHE_MPX_SUBCNT*(CLUSTER_CNT)][4];
  uint64_t extDetail[CLUSTER_CNT*SUBCORE_CNT][EXT_PERF_CNT];
  uint64_t shmDetail[CLUSTER_CNT*SUBCORE_CNT][SHM_PERF_CNT];
  float perf_achieved_warp_occupation;
  uint32_t l1invalid;
  uint32_t warp_regfile_size;
} ptProfileInfo;

#ifndef TA_STREAM_LEGACY
#define TA_STREAM_LEGACY ((TAstream)0x01)
#endif  //! TA_STREAM_LEGACY

#ifndef TA_STREAM_PER_THREAD
#define TA_STREAM_PER_THREAD ((TAstream)0x02)
#endif  //! TA_STREAM_PER_THREAD

/**
 * If set, host memory is page locked.
 * Flag for ::taMemHostAlloc()
 */
#define TA_MEMHOSTALLOC_DEFAULT 0x00

/**
 * If set, host memory is portable between TANG contexts.
 * Flag for ::taMemHostAlloc()
 */
#define TA_MEMHOSTALLOC_PORTABLE 0x01

/**
 * If set, host memory is mapped into TANG address space and
 * ::taMemHostGetDevicePointer() may be called on the host pointer.
 * Flag for ::taMemHostAlloc()
 */
#define TA_MEMHOSTALLOC_DEVICEMAP 0x02

/**
 * If set, host memory is allocated as write-combined - fast to write,
 * faster to DMA, slow to read except via SSE4 streaming load instruction
 * (MOVNTDQA).
 * Flag for ::taMemHostAlloc()
 */
#define TA_MEMHOSTALLOC_WRITECOMBINED 0x04

/**
 * If set allocate memory from device side and map it to the user space.
 *
 */
#define TA_MEMHOSTALLOC_MAP_DEVICE_MEMORY 0x100

/**
 * If set, host memory is page locked.
 * Flag for ::taMemHostRegister()
 */
#define TA_MEMHOSTREGISTER_DEFAULT 0x00

/**
 * If set, host memory is portable between TANG contexts.
 * Flag for ::taMemHostRegister()
 */
#define TA_MEMHOSTREGISTER_PORTABLE 0x01

/**
 * If set, host memory is mapped into TANG address space and
 * ::taMemHostGetDevicePointer() may be called on the host pointer.
 * Flag for ::taMemHostRegister()
 */
#define TA_MEMHOSTREGISTER_DEVICEMAP 0x02

/**
 * If set, the passed memory pointer is treated as pointing to some
 * memory-mapped I/O space, e.g. belonging to a third-party PCIe device.
 * Flag for ::taMemHostRegister()
 */
#define TA_MEMHOSTREGISTER_IOMEMORY 0x04

/**
 * If set, the passed memory pointer is treated as pointing to memory that is
 * considered read-only by the device.
 * Flag for ::taMemHostRegister()
 */
#define TA_MEMHOSTREGISTER_READ_ONLY 0x08

/**
 * @ingroup PT_ERROR error handling
 * @{
 ************************************************/
/**
 * @brief Driver API error codes.
 ************************************************/
typedef enum TAresult_e {
  /**
   * @brief The API call returned with no errors.
   * @note For asynchronous operations, \c TANG_SUCCESS
   * just means the operation is ququed on the \c stream
   * successfully.
   */
  TANG_SUCCESS = 0,

  /**
   * @brief This indicates one or more invalid parameters
   * are passed to the API call.
   */
  TANG_ERROR_INVALID_VALUE = 1,

  /**
   * @brief This indicates the API call failed because
   * it can not allocate enough memory to perform the requested
   * operation.
   */
  TANG_ERROR_OUT_OF_MEMORY = 2,

  /**
   * @brief This indicates that the PT dirver has not been initialized
   * with ::__taInit or thar initialization has failed.
   */
  TANG_ERROR_NOT_INITIALIZED = 3,

  /**
   * @brief This indicates that the PT driver is int the process of shutting
   * down.
   */
  TANG_ERROR_DEINITIALIZED = 4,

  //!< The device is remove for some reason.
  //!< echo "1" > /sys/../remove
  TANG_ERROR_DEVICE_REMOVED = 5,

  //!< The device is reseted.
  //!< Example: enable or disable SR-IOV
  TANG_ERROR_DEVICE_RESET = 6,

  //!< The operation is not allowed.
  TANG_ERROR_NOT_PERMITTED = 7,

  //!< No such file or directroy
  TANG_ERROR_NO_SUCH_FILE = 8,

  /**
   * This indicates that a kernel launch is requesting resources that can
   * never be satisfied by the current device.
   */
  TANG_ERROR_INVALID_CONFIGURATION = 9,

  //!< Null pointer is passed as argument but it is not allowed.
  TANG_ERROR_NULL_POINTER = 10,

  //!< The kernel mode driver is not compatible with current runtime.
  TANG_ERROR_INCOMPATIBLE_DRIVER = 11,

  //!< Can allocate enough resources to perform the requested operation.
  TANG_ERROR_OUT_OF_RESOURCES = 12,

  TANG_ERROR_TIMEOUT = 13,

  /**
   * @brief This indicates the API call is not implemented
   * and just a stub or for the given parameter(s) the function
   * has not been implemented yet.
   */
  TANG_ERROR_NOT_IMPLEMENTED = 99,

  /**
   * @brief No available PT devices.
   */
  TANG_ERROR_NO_DEVICE = 100,

  /**
   * @brief Invalid device.
   */
  TANG_ERROR_INVALID_DEVICE = 101,

  //!< Bad file descriptor.
  TANG_ERROR_BAD_FD = 102,

  //!< Normal indicate some invariant are broken
  TANG_ERROR_UNREACHABLE_CODE = 103,

  //!< More than one function use the same symbol name.
  TANG_ERROR_DUPLICATE_FUNC_NAME = 198,

  //!< More than one global value use the same symbol name.
  TANG_ERROR_DUPLICATE_VAR_NAME = 199,

  /**
   * @brief
   */
  TANG_ERROR_INVALID_IMAGE = 200,

  /**
   * @brief This most frequently indicates there is
   * no context bound to the current thread.
   * This error code is also returned when an invalid
   * context is passed to API call.
   */
  TANG_ERROR_INVALID_CONTEXT = 201,

  /**
   * @brief No context is bound to the calling thread.
   */
  TANG_ERROR_NO_CONTEXT_BOUND = 202,

  /**
   * @brief Invalid host address encountered.
   */
  TANG_ERROR_ILLEGAL_HOST_ADDRESS = 203,

  //!< Context mismatch
  TANG_ERROR_CONTEXT_MISMATCH = 204,

  /**
   * This indicates that the ::TAlimit passed to the API call is not
   * supported by the active device.
   */
  TANG_ERROR_UNSUPPORTED_LIMIT = 215,

  /**
   * @brief The key is not found!
   *
   */
  TANG_ERROR_NOT_FOUND = 301,

  // This indicates that a resource required by the API call
  // is not in a valid state to perform the requested operation.
  TANG_ERROR_ILLEGAL_STATE = 302,

  // This error indicates that the operation is not permitted
  // then the stream is capturing.
  TANG_ERROR_STREAM_CAPTURE_UNSUPPORTED = 303,

  // This error indicates that the current capture
  // sequence on the stream has been invalidated
  // due a previous error.
  TANG_ERROR_STREAM_CAPTURE_INVALIDATED = 304,

  // This error indicates that the operation whould
  // have resulted in a merge of two independent capture
  // sequences.
  TANG_ERROR_STREAM_CAPTURE_MERGE = 305,

  // This error indicates that the capture was not initiated in this stream.
  TANG_ERROR_STREAM_CAPTURE_UNMATCHED = 306,

  // This error indicates that the capture sequence contains a fork that was
  // not joined to the primary stream.
  TANG_ERROR_STREAM_CAPTURE_UNJOINED = 307,

  // This error indicates that a dependency would have been created which
  // crossed the capture sequence boundary. Only implicit in-stream ordering
  // dependencies are allowed to cross the boundary.
  TANG_ERROR_STREAM_CAPTURE_ISOLATION = 308,

  // This error indicates a disallowed implicit dependency on a current
  // capture sequence from TA_STREAM_LEGACY.
  TANG_ERROR_STREAM_CAPTURE_IMPLICIT = 309,

  // A stream capture sequence not initiated with the
  // ::TA_STREAM_CAPTURE_MODE_RELAXED argument to taStreamBeginCapture was
  // passed to ::cuStreamEndCapture in a different thread.
  TANG_ERROR_STREAM_CAPTURE_WRONG_THREAD = 310,

  // This error indicates that the operation is not permitted on an event
  // which was last recorded in a capturing stream.
  TANG_ERROR_CAPTURED_EVENT = 311,

  /**
   * @brief This indicates an invalid resource handle
   * passed to a API call.
   * In general, resource handles are opaque type like
   * ::TAstream and ::TAcontext.
   */
  TANG_ERROR_INVALID_HANDLE = 400,

  /**
   * @brief This error code indicates asynchronous operations issued previously
   * have not been completed yet.
   */
  TANG_ERROR_NOT_READY = 600,

  /**
   * @brief A load or store instruction on an invalid
   * memory address occured when the device executing a
   * kernel.
   * This error makes the process is an inconsitant state.
   * The process should be terminated and relanuched.
   */
  TANG_ERROR_ILLEGAL_ADDRESS = 700,

  /**
   * @brief resouce is not enougn for the kernel
   *
   */
  TANG_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,

  /**
   * @brief lanch kernel timeout
   *
   */
  TANG_ERROR_LAUNCH_TIMEOUT = 702,

  TANG_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
  TANG_ERROR_PEER_ACCESS_NOT_ENABLED = 705,

  /**
   * @brief The premary for a context has been
   * initialized.
   */
  TANG_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,

  /**
   * @brief
   *
   */
  TANG_ERROR_NOT_SUPPORTED = 801,

  /**
   * @brief This indicates that an unknown internal error has occurred.
   */
  TANG_ERROR_UNKNOWN = 999,

  /**
   * @brief context is destroyed or in destroying in kernel
   *
   */
  TANG_ERROR_CONTEXT_IS_DESTROYED = 3000,

  /**
   * @brief context is not valid in kernel
   *
   */
  TANG_ERROR_CONTEXT_INVALID = 3001,

  /**
   * @brief stream is destroyed or in destroying in kernel
   *
   */
  TANG_ERROR_STREAM_IS_DESTROYED = 3002,

  /**
   * @brief stream is not valid in kernel
   *
   */
  TANG_ERROR_STREAM_INVALID = 3003,

  /**
   * @brief event is destroyed or in destroying in kernel
   *
   */
  TANG_ERROR_EVENT_IS_DESTROYED = 3004,

  /**
   * @brief event is not valid in kernel
   *
   */
  TANG_ERROR_EVENT_INVALID = 3005,

  /**
   * @brief device memory is not enough for current operation
   *
   */
  TANG_ERROR_DEVICE_OUT_OF_MEMORY = 3006,

  /**
   * @brief device memory is not found
   *
   */
  TANG_ERROR_DEVICE_MEMORY_NOT_FOUND = 3007,

  /**
   * @brief pcie fatal error occured
   *
   */
  TANG_ERROR_PCIE_FATAL = 3012,

  /**
   * @brief pcie non-fatal unrecovered error occured
   *
   */
  TANG_ERROR_PCIE_NON_FATAL_UNRECOVERED = 3013,

  /**
   * @brief no more event exist
   *
   */
  TANG_ERROR_SCP_EVENT_NOT_EXIST = 3014,

  /**
   * @brief record event failed
   *
   */
  TANG_ERROR_SCP_EVENT_RECORD_FAILED = 3015,

  /**
   * @brief scp packet crc check failed
   *
   */
  TANG_ERROR_SCP_PACKET_CRC_FAILED = 3016,

  /**
   * @brief scp dispatch send failed
   *
   */
  TANG_ERROR_SCP_DISP_SEND_FAILED = 3017,

  /**
   * @brief sq write sequence error
   *
   */
  TANG_ERROR_SCP_SQ_WRITE_INVALID = 3018,

  /**
   * @brief udrc pcie xdma packet invalid
   *
   */
  TANG_ERROR_UDRC_PCIE_DMA_CMD_PACKET_INVALID = 3019,

  /**
   * @brief udrc mp dma packet invalid
   *
   */
  TANG_ERROR_UDRC_MP_DMA_CMD_PACKET_INVALID = 3020,

  /**
   * @brief udrc reg packet invalid
   *
   */
  TANG_ERROR_UDRC_REG_CMD_PACKET_INVALID = 3021,

  /**
   * @brief udrc reg access invalid
   *
   */
  TANG_ERROR_UDRC_REG_ACCESS_INVALID = 3022,

  /**
   * @brief aiss cluster is not configured
   *
   */
  TANG_ERROR_AISS_VF_CTRL_CLUST_USR_NOT_ALLOCATED = 3023,

  /**
   * @brief barrier is destroyed or in destroying in kernel
   *
   */
  TANG_ERROR_BARRIER_IS_DESTROYED = 3024,

  /**
   * @brief barrier is not valid in kernel
   *
   */
  TANG_ERROR_BARRIER_INVALID = 3025,

  /**
   * @brief one obj is destroyed or in destroying in kernel
   *
   */
  TANG_ERROR_IS_DESTROYED = 3026,

  /**
   * @brief xdma C2H align mismath
   *
   */
  TANG_ERROR_XDMA_C2H_ALIGN_MISMATCH = 3300,

  /**
   * @brief xdma C2H invalid magic stopped
   *
   */
  TANG_ERROR_XDMA_C2H_INVALID_MAGIC_STOPPED = 3301,

  /**
   * @brief xdma C2H invalid Len
   *
   */
  TANG_ERROR_XDMA_C2H_INVALID_LEN = 3302,

  /**
   * @brief xdma C2H decode error
   *
   */
  TANG_ERROR_XDMA_C2H_DECODE = 3303,

  /**
   * @brief xdma C2H slave
   *
   */
  TANG_ERROR_XDMA_C2H_SLAVE = 3304,

  /**
   * @brief xdma C2H descriptor unsupport request
   *
   */
  TANG_ERROR_XDMA_C2H_DESC_UNSUPPORT_REQUEST = 3305,

  /**
   * @brief xdma C2H descriptor completer abort
   *
   */
  TANG_ERROR_XDMA_C2H_DESC_COMPLETER_ABORT = 3306,

  /**
   * @brief xdma C2H descriptor parity
   *
   */
  TANG_ERROR_XDMA_C2H_DESC_PARITY = 3307,

  /**
   * @brief xdma C2H descriptor header ep
   *
   */
  TANG_ERROR_XDMA_C2H_DESC_HEADER_EP = 3308,

  /**
   * @brief xdma C2H descriptor unexpected comp
   *
   */
  TANG_ERROR_XDMA_C2H_DESC_UNEXPECTED_COMP = 3309,

  /**
   * @brief xdma C2H timeout
   *
   */
  TANG_ERROR_XDMA_C2H_TIMEOUT = 3310,

  /**
   * @brief xdma C2H unknown
   *
   */
  TANG_ERROR_XDMA_C2H_UNKNOWN = 3311,

  /**
   * @brief xdma H2C align mismatch
   *
   */
  TANG_ERROR_XDMA_H2C_ALIGN_MISMATCH = 3350,

  /**
   * @brief xdma H2C invalid magic stopped
   *
   */
  TANG_ERROR_XDMA_H2C_INVALID_MAGIC_STOPPED = 3351,

  /**
   * @brief xdma H2C invalid len
   *
   */
  TANG_ERROR_XDMA_H2C_INVALID_LEN = 3352,

  /**
   * @brief xdma H2C read unsupport request
   *
   */
  TANG_ERROR_XDMA_H2C_READ_UNSUPPORT_REQUEST = 3353,

  /**
   * @brief xdma H2C read completer abort
   *
   */
  TANG_ERROR_XDMA_H2C_READ_COMPLETER_ABORT = 3354,

  /**
   * @brief xdma H2C read parity
   *
   */
  TANG_ERROR_XDMA_H2C_READ_PARITY = 3355,

  /**
   * @brief xdma H2C read header ep
   *
   */
  TANG_ERROR_XDMA_H2C_READ_HEADER_EP = 3356,

  /**
   * @brief xdma H2C read unexpected comp
   *
   */
  TANG_ERROR_XDMA_H2C_READ_UNEXPECTED_COMP = 3357,

  /**
   * @brief xdma H2C decode error
   *
   */
  TANG_ERROR_XDMA_H2C_DECODE = 3358,

  /**
   * @brief xdma H2C slave
   *
   */
  TANG_ERROR_XDMA_H2C_SLAVE = 3359,

  /**
   * @brief xdma H2C descriptor unsupport request
   *
   */
  TANG_ERROR_XDMA_H2C_DESC_UNSUPPORT_REQUEST = 3360,

  /**
   * @brief xdma H2C descriptor completer abort
   *
   */
  TANG_ERROR_XDMA_H2C_DESC_COMPLETER_ABORT = 3361,

  /**
   * @brief xdma H2C descriptor parity
   *
   */
  TANG_ERROR_XDMA_H2C_DESC_PARITY = 3362,

  /**
   * @brief xdma H2C descriptor header ep
   *
   */
  TANG_ERROR_XDMA_H2C_DESC_HEADER_EP = 3363,

  /**
   * @brief xdma H2C descriptor unexpected com
   *
   */
  TANG_ERROR_XDMA_H2C_DESC_UNEXPECTED_COMP = 3364,

  /**
   * @brief xdma H2C timeout
   *
   */
  TANG_ERROR_XDMA_H2C_TIMEOUT = 3365,

  /**
   * @brief xdma H2C unknown
   *
   */
  TANG_ERROR_XDMA_H2C_UNKNOWN = 3366,
  /**
   * @brief gpu profling share mem out of size
   *
   */
  TANG_ERROR_GPU_PROFLING_SHAMEM_OUT_OF_SIZE = 3367,

  /**
   * @brief The requested ipc mem is destroied.
   * @sa taIpcOpenMemHandle
   */
  TANG_ERROR_IPC_MEM_DESTROIED = 3368,
} TAresult;

/**
 * @brief Get the TANG SDK Runtime version.
 *
 * @param runtimeVersion - Returned runtime version number.
 * @return int
 * ::TANG_SUCCESS - \p runtimeVersion is a non-null poiner.
 * ::TANG_ERROR_INVALID_VALUE - \p runtimeVersion is a null pointer.
 * @deleted Do not use this function.
 ******************************************************/
TAresult TA_API taRuntimeGetVersion(int* runtimeVersion);

/**
 * @brief Get the TANG SDK Driver version.
 *
 * @param driverVersion - Returned driver version number.
 * @return int
 * ::TANG_SUCCESS - \p driverVersion is a non-null poiner.
 * ::TANG_ERROR_INVALID_VALUE - \p driverVersion is a null pointer.
 ******************************************************/
TAresult TA_API taDriverGetVersion(int* driverVersion);

/**
 * @brief Get the kernel mode driver version.
 *
 * @param kernelDriverVersion
 * @return int
 */
TAresult TA_API taKernelDriverGetVersion(int* kernelDriverVersion);

/**
 * @brief Get error description.
 *
 * @param error
 * @param ppstr - Returned Null-terminated string.
 * @return int
 * ::TANG_SUCCESS - \p error is a valid error code.
 * ::TANG_ERROR_INVALID_VALUE - \p error is an invalid
 * error code.
 ******************************************************/
TAresult TA_API taGetErrorString(TAresult error, char const** ppstr);

/**
 * @brief Get the string representation of an error code.
 *
 * @param error
 * @param ppstr - Returned Null-terminated string.
 * @return int
 * ::TANG_SUCCESS - \p error is a valid error code.
 * ::TANG_ERROR_INVALID_VALUE - \p error is an invalid
 * error code.
 ******************************************************/
TAresult TA_API taGetErrorName(TAresult error, char const** ppstr);
/** @} PT_ERROR */

/**
 * @brief Initilaize driver module.
 *
 * This function initialize driver module.
 * @param flags Initialization flags for driver API
 * @return
 * ::TANG_SUCCESS
 *******************************************************/
TAresult TA_API taInit(unsigned int flags);
// TAresult TA_API taDeinit(void);

/**
 * @brief __taDeviceGet
 * Get a handle to a compute device
 * @param device Pointer to a device handle.
 * @param ordinal The device number to get handle for
 * @return int
 ********************************************************/
TAresult TA_API taDeviceGet(TAdevice* device, int ordinal);

/**
 * @brief Wait for all work completed
 *
 * @param device
 * @return int
 ********************************************************/
TAresult TA_API taDeviceSynchronize(TAdevice device);

/**
 * @brief Synchronize with the current device of the calling thread.
 *
 * @warning Not a public API, may change in the future.
 * @return int
 * ::TANG_SUCCESS
 */
TAresult TA_API __taDeviceSynchronizeCurrent(void);

/**
 * @brief Reset the current device.
 * Only the calling process is impacted.
 * @return int
 * @warning This is a dangerous API. The caller must
 * ensure that all resources allocated from the current
 * device will not be used again. The most difficult to
 * handle is TAcontext which may be pushed onto thread's context
 * stack.
 * BE CAREFUL.
 **********************************************************/
TAresult TA_API taDeviceReset(void);

/**
 * @brief Returns a handle to a compute device.
 *
 * @param device - device handle
 * @param pciBusId - PCI Bus ID
 * @return int
 * ::TANG_SUCCESS - \p error is a valid error code.
 * ::TANG_ERROR_INVALID_DEVICE - \p error is an invalid
 * error code.
 * ::TANG_ERROR_INVALID_VALUE - \p error is an invalid
 * error code.
 */
TAresult TA_API taDeviceGetByPCIBusId(TAdevice* device, const char* pciBusId);

/**
 * @brief Returns a PCI Bus Id string for the device.
 *
 * @param pciBusId - PCI Bus ID
 * @param len - Maximum length of pciBusId name string
 * @param device - device handle
 * @return int
 * ::TANG_SUCCESS - \p error is a valid error code.
 * ::TANG_ERROR_INVALID_DEVICE - \p error is an invalid
 * error code.
 * ::TANG_ERROR_INVALID_VALUE - \p error is an invalid
 * error code.
 */
TAresult TA_API taDeviceGetPCIBusId(char* pciBusId, int len, TAdevice device);

/**
 * Limits
 */
typedef enum TAlimit_enum {
  TA_LIMIT_STACK_SIZE       = 0x00, /**< GPU thread stack size */
  TA_LIMIT_PRINTF_FIFO_SIZE = 0x01, /**< GPU printf FIFO size */
  TA_LIMIT_MALLOC_HEAP_SIZE = 0x02, /**< GPU malloc heap size */
  TA_LIMIT_DEV_RUNTIME_SYNC_DEPTH =
    0x03, /**< GPU device runtime launch synchronize depth */
  TA_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT =
    0x04, /**< GPU device runtime pending launch count */
  TA_LIMIT_MAX_L2_FETCH_GRANULARITY =
    0x05, /**< A value between 0 and 128 that indicates the maximum fetch
             granularity of L2 (in Bytes). This is a hint */
  TA_LIMIT_MAX
} TAlimit;

/**
 * @brief Get Resource limits of current context
 *
 * @param [out] pValue
 * @param [in]  limit
 * @return int
 * ::TANG_SUCCESS - \p error is a valid error code.
 * ::TANG_ERROR_INVALID_VALUE - \p error is an invalid
 * error code.
 * ::TANG_ERROR_UNSUPPORTED_LIMIT - \p error is an invalid
 * error code.
 *
 */
TAresult TA_API taCtxGetLimit(size_t* pValue, TAlimit limit);
TAresult TA_API taCtxSetLimit(TAlimit limit, size_t value);

/**
 * @brief Query detail limit information.
 * Not a plublic interface.
 *
 * @param context
 * @param limit
 * @param pCurrent The current value of the \p limit
 * @param pLimit The limit of the \p limit
 * @return int
 * ::TANG_SUCCESS
 * ::TANG_ERROR_INVALID_VALUE
 */
TAresult TA_API __taCtxQueryLimit(TAcontext context,
                             TAlimit   limit,
                             size_t*   pCurrent,
                             size_t*   pLimit);

/**
 * TA device attributes
 */
typedef enum taDeviceAttr {
  TA_DEV_ATTR_SHARED_MEM_PER_BLOCK = 0,        //!< sharedMemPerBlock
  TA_DEV_ATTR_REGS_PER_BLOCK,                  //!< regsPerBlock
  TA_DEV_ATTR_WARP_SIZE,                       //!< warpSize
  TA_DEV_ATTR_MEM_PITCH,                       //!< memPitch
  TA_DEV_ATTR_MAX_THREADS_PER_BLOCK,           //!< maxThreadsPerBlock
  TA_DEV_ATTR_MAX_THREADS_DIM_X,               //!< maxThreadsDimX
  TA_DEV_ATTR_MAX_THREADS_DIM_Y,               //!< maxThreadsDimY
  TA_DEV_ATTR_MAX_THREADS_DIM_Z,               //!< maxThreadsDimZ
  TA_DEV_ATTR_MAX_GRID_SIZE_X,                 //!< maxGridSizeX
  TA_DEV_ATTR_MAX_GRID_SIZE_Y,                 //!< maxGridSizeY
  TA_DEV_ATTR_MAX_GRID_SIZE_Z,                 //!< maxGridSizeZ
  TA_DEV_ATTR_CLOCK_RATE,                      //!< clockRate
  TA_DEV_ATTR_TOTAL_CONST_MEM,                 //!< totalConstMem
  TA_DEV_ATTR_MULTIPROCESSOR_COUNT,            //!< multiProcessorCount
  TA_DEV_ATTR_MAX_BLOCKS_PER_MULTIPROCESSOR,   //!< maxBlocksPerMultiProcessor
  TA_DEV_ATTR_ASYNC_ENGINE_COUNT,              //!< asyncEngineCount
  TA_DEV_ATTR_MEMORY_CLOCK_RATE,               //!< memoryClockRate
  TA_DEV_ATTR_MEMORY_BUS_WIDTH,                //!< memoryBusWidth
  TA_DEV_ATTR_L2_CACHE_SIZE,                   //!< l2CacheSize
  TA_DEV_ATTR_MAX_THREADS_PER_MULTIPROCESSOR,  //!< maxThreadsPerMultiProcessor
  TA_DEV_ATTR_GLOBAL_L1_CACHE_SUPPORTED,       //!< globalL1CacheSupported
  TA_DEV_ATTR_LOCAL_L1_CACHE_SUPPORTED,        //!< localL1CacheSupported
  TA_DEV_ATTR_SHARED_MEM_PER_MULTIPROCESSOR,   //!< sharedMemPerMultiprocessor
  TA_DEV_ATTR_REGS_PER_MULTIPROCESSOR,         //!< regsPerMultiprocessor
  TA_DEV_ATTR_STREAM_PRIORITIES_SUPPORTED,     //!< streamPrioritiesSupported
  TA_DEV_ATTR_CONCURRENT_KERNELS,              //!< concurrentKernels
  TA_DEV_ATTR_COMPUTE_PREEMPTION_SUPPORTED,    //!< computePreemptionSupported
  TA_DEV_ATTR_KERNEL_EXEC_TIMEOUT_ENABLED,     //!< kernelExecTimeoutEnabled
  TA_DEV_ATTR_ECC_ENABLED,                     //!< ECCEnabled
  TA_DEV_ATTR_ACCESS_POLICY_MAX_WINDOW_SIZE,   //!< accessPolicyMaxWindowSize
  TA_DEV_ATTR_TCC_DRIVER,                      //!< tccDriver
  TA_DEV_ATTR_SINGLE_TO_DOUBLE_PRECISION_PER_RATIO,  //!< singleToDoublePrecisionPerfRatio
  TA_DEV_ATTR_COOPERATIVE_LAUNCH,                    //!< cooperativeLaunch
  TA_DEV_ATTR_COOPERATIVE_MULTI_DEVICE_LAUNCH,  //!< cooperativeMultiDeviceLaunch
  TA_DEV_ATTR_PERSISTING_L2_CACHE_MAX_SIZE,     //!< persistingL2CacheMaxSize
  TA_DEV_ATTR_CAN_MAP_HOST_MEMORY,              //!< canMapHostMemory
  TA_DEV_ATTR_UNIFIED_ADDRESSING,               //!< unifiedAddressing
  TA_DEV_ATTR_MANAGED_MEMORY,                   //!< managedMemory
  TA_DEV_ATTR_CONCURRENT_MANAGED_ACCESS,        //!< concurrentManagedAccess
  TA_DEV_ATTR_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST,  //!< directManagedMemAccessFromHost
  TA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS,               //!< pageableMemoryAccess
  TA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,  //!< pageableMemoryAccessUsesHostPageTables
  TA_DEV_ATTR_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM,  //!< canUseHostPointerForRegisteredMem
  TA_DEV_ATTR_HOST_NATIVE_ATOMIC_SUPPORTED,       //!< hostNativeAtomicSupported
  TA_DEV_ATTR_CAN_FLUSH_REMOTE_WRITES,            //!< canFlushRemoteWrites
  TA_DEV_ATTR_GPU_OVERLAP,                        //!< gpuOverlap
  TA_DEV_ATTR_INTEGRATED,                         //!< integrated
  TA_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,  //!< maxSharedMemoryPerBlockOptin
  TA_DEV_ATTR_GPU_DIRECT_RDMA_SUPPORTED,          //!< gpuDirectRDMASupported
  TA_DEV_ATTR_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS,  //!< gpuDirectRDMAFlushWritesOptions
  TA_DEV_ATTR_GPU_DIRECT_RDMA_WRITES_ORDERING,  //!< gpuDirectRDMAWritesOrdering
  TA_DEV_ATTR_MAJOR,                            //!< major
  TA_DEV_ATTR_MINOR,                            //!< minor
  TA_DEV_ATTR_PCI_BUS_ID,                       //!< pciBusID
  TA_DEV_ATTR_PCI_DEVICE_ID,                    //!< pciDeviceID
  TA_DEV_ATTR_PCI_DOMAIN_ID,                    //!< pciDomainID
  TA_DEV_ATTR_IS_MULTI_GPU_BOARD,               //!< isMultiGpuBoard
  TA_DEV_ATTR_GPU_BOARD_GROUP_ID,               //!< multiGpuBoardGroupID
  TA_DEV_ATTR_COMPUTE_MODE,                     //!< computeMode
  TA_DEV_ATTR_RESERVED_SHARED_MEMORY_PER_BLOCK,  //!< reservedSharedMemoryPerBlock
  TA_DEV_ATTR_SPARSE_TANG_ARRAY_SUPPORTED,       //!< sparseTangArraySupported
  TA_DEV_ATTR_HOST_REGISTER_SUPPORTED,           //!< hostRegisterSupported
  TA_DEV_ATTR_HOST_REGISTER_READ_ONLY_SUPPORTED,  //!< hostRegisterReadOnlySupported
  TA_DEV_ATTR_MEMORY_POOLS_SUPPORTED,             //!< memoryPoolsSupported
  TA_DEV_ATTR_MEMORY_POOL_SUPPORTED_HANDLE_TYPES,  //!< memoryPoolSupportedHandleTypes
  TA_DEV_ATTR_MAX
} taDeviceAttr;

TAresult TA_API taDeviceGetAttribute(int* value, taDeviceAttr attr, TAdevice dev);

TAresult TA_API taDeviceGetName(char* name, int len, TAdevice dev);
TAresult TA_API taDeviceGetUuid(char* uuid, TAdevice dev);
TAresult TA_API taDeviceTotalMem(size_t* bytes, TAdevice dev);

/**
 * @brief Get the number of available devices.
 *
 * @param count Returns the count of available devices.
 * @return int
 * ::TANG_SUCCESS
 * ::TANG_ERROR_NOT_INITIALIZED
 ********************************************************/
TAresult TA_API taDeviceGetCount(int* count);

/**
 * @brief Gets free and total memory of the current device.
 *
 * @param free - Returned free memory in bytes
 * @param total - Returned total memory in bytes
 * @return int
 */
TAresult TA_API taMemGetInfo(size_t* free, size_t* total);

/**
 * @brief Gets free and total memory of \p device.
 *
 * @param device
 * @param free
 * @param total
 * @return int
 */
TAresult TA_API taDeviceMemGetInfo(TAdevice device, size_t* free, size_t* total);

/**
 * @brief Create a context.
 *
 * @param pctx - Returned newly created context.
 * @param flags - Flags for creating the context.
 * @param dev - The device ID.
 * @return int
 * ::TANG_SUCCESS
 * ::TANG_ERROR_OUT_OF_MEMORY
 * @note When the context is no longer used, the caller
 * should call \c taCtxDestroy to destroy the context.
 * @sa taCtxDestroy
 ********************************************************/
TAresult TA_API taCtxCreate(TAcontext* pctx, unsigned int flags, TAdevice dev);

/**
 * @brief Destroy the context \p ctx.
 *
 * @param ctx
 * @return int
 * ::TANG_SUCCESS
 * ::TANG_ERROR_INVALID_CONTEXT
 * @note It's the caller's responsibility ensure that the
 * context \p ctx is no longer referenced by other objects.
 * @taCtxCreate
 ********************************************************/
TAresult TA_API taCtxDestroy(TAcontext ctx);

/**
 * @brief Bind the context \p ctx to the calling thread.
 *
 * @param ctx - The context to be bound to the calling thread.
 * @return int
 * @note If \p ctx is NULL, this function just POPs the context
 * stack of the calling thread if the context stack is not
 * empty.
 * If \p ctx is no NULL, this function replace the top of
 * the stack if the context stack is no empty.
 *********************************************************/
TAresult TA_API taCtxSetCurrent(TAcontext ctx);

/**
 * @brief Get the current context bound to the calling
 * thread.
 *
 * @param pctx - Returned context bound to the calling thread.
 * @return int
 * ::TANG_SUCCESS
 * ::TANG_ERROR_NO_CONTEXT_BOUND
 ********************************************************/
TAresult TA_API taCtxGetCurrent(TAcontext* pctx);

/**
 * @brief Query the current context bound to the calling
 * thread.
 * 
 * @param pCtx 
 * @return int 
 */
TAresult TA_API taCtxQueryCurrent(TAcontext* pCtx);

/**
 * @brief Get the device ID of the current context bound
 * to the call thread.
 *
 * @param dev - Returned device ID of the current context.
 * @return int
 * ::TANG_SUCCESS
 * ::TANG_ERROR_NO_CONTEXT_BOUND
 ********************************************************/
TAresult TA_API taCtxGetCurrentDevice(TAdevice* dev);

TAresult TA_API taCtxGetDevice(TAcontext ctx, TAdevice* dev);
TAresult TA_API taCtxGetOrdinal(TAcontext ctx, int* ordinal);

TAresult TA_API taCtxGetFunction(TAcontext   ctx,
                            const void* hostFunc,
                            TAfunction* phFunc);

TAresult TA_API taCtxRegisterFunction(TAcontext   ctx,
                                 const void* hostFunc,
                                 TAfunction  hFunc);

TAresult TA_API taCtxGetBuiltInFunction(TAcontext   ctx,
                                   const char* funcName,
                                   TAfunction* phFunc);

TAresult TA_API taCtxRegisterBuiltInFunction(TAcontext   ctx,
                                        const char* funcName,
                                        TAfunction  hFunc);

TAresult TA_API taCtxGetVariable(TAcontext   ctx,
                            const void* hostVar,
                            TAvariable* hVar);

TAresult TA_API taCtxRegisterVariable(TAcontext   ctx,
                                 const void* hostVar,
                                 TAvariable  hVar);

TAresult TA_API taCtxPushCurrent(TAcontext ctx);
TAresult TA_API taCtxPopCurrent(TAcontext* ctx);
TAresult TA_API taCtxSynchronize(TAcontext ctx);

/**
 * @ingroup Primary Context Management.
 * @{
 */
/**
 * @brief Retain the primary context of \p dev.
 *
 * @param pctx Pointer to receive the primary context handle.
 * @param dev
 * @return int
 ********************************************************/
TAresult TA_API taDevicePrimaryCtxRetain(TAcontext* pctx, TAdevice dev);

/**
 * @brief Release the primary context of \p dev.
 *
 * @param dev
 * @return int
 ********************************************************/
TAresult TA_API taDevicePrimaryCtxRelease(TAdevice dev);

/**
 * @brief Reset the primary context of \p dev.
 *
 * @param dev
 * @return int
 ********************************************************/
TAresult TA_API taDevicePrimaryCtxReset(TAdevice dev);

/**
 * @brief Get the state of primary context of device \p dev.
 *
 * @param dev Device to get primary context's state for.
 * @param flags Pointer to receive the flags.
 * @param active
 * @return int
 **********************************************************/
TAresult TA_API taDevicePrimaryCtxGetState(TAdevice      dev,
                                      unsigned int* flags,
                                      int*          active);

/**
 * @brief Set flags for the primary context of the device \p dev.
 *
 * @param dev
 * @param flags
 * @return int
 */
TAresult TA_API taDevicePrimaryCtxSetFlags(TAdevice dev, unsigned int flags);
/** }@ */

TAresult TA_API taDeviceGetOrdinal(TAdevice dev, int* ordinal);

/**
 * @brief Allocate a block of memory in the device.
 *
 * @param dptr Receives the allocated device memory block handle.
 * @param size
 * @return int
 * On success, zero is returned.
 * @sa taMemFree
 ***********************************************************/
TAresult TA_API taMemAlloc(TAdeviceptr* dptr, size_t size);
TAresult TA_API taMemAlloc(TAdeviceptr* dptr, size_t size);
TAresult TA_API taMemAllocAsync(TAdeviceptr* dptr, size_t size, TAstream stream);
TAresult TA_API taMemAllocAsync_ptsz(TAdeviceptr* dptr,
                                     size_t       size,
                                     TAstream     stream);

/**
 * @brief Free a device memory block.
 *
 * @param dptr Device memory block handle.
 * @return int
 * On success, zero is returned.
 * @sa taMemAlloc
 ************************************************************/
TAresult TA_API taMemFree(TAdeviceptr dptr);
TAresult TA_API taMemFreeAsync(TAdeviceptr dptr, TAstream hStream);
TAresult TA_API taMemFreeAsync_ptsz(TAdeviceptr dptr, TAstream hStream);

/**
 * @brief Allocate page locked host memory.
 *
 * @param hptr Pointer to the allocated page locked host memory
 * @param sizeBytes Requested memory size
 * @return int
 * On success, zero is returned.
 * @sa taMemFreeHost
 ***********************************************************/
TAresult TA_API taMemAllocHost(void** hptr, size_t sizeBytes);

/**
 * @brief Allocate page locked host memory.
 *
 * @param hptr Pointer to the allocated page locked host memory
 * @param sizeBytes Requested memory size
 * @param flags See below.
 * flags:
 * - #TA_MEMHOSTALLOC_PORTABLE      Memory is considered registered by all
 *contexts.
 * - #TA_MEMHOSTALLOC_DEVICEMAP     Map the allocation into the address space
 *for the current device.
 * - #TA_MEMHOSTALLOC_WRITECOMBINED Allocates the memory as write-combined (WC).
 * TANG does not support IOMMU on device side, so flags of
 *TA_MEMHOSTALLOC_DEVICEMAP and TA_MEMHOSTALLOC_WRITECOMBINED will always return
 *false.
 * @return int
 * On success, zero is returned.
 * @sa taMemFreeHost
 ***********************************************************/
TAresult TA_API taMemHostAlloc(void** hptr, size_t sizeBytes, unsigned int flags);
TAresult TA_API taMemHostGetDevicePointer(TAdeviceptr* pdptr,
                                     void*        pHost,
                                     unsigned int flags);

/**
 * @brief Get the flags that were used for allocation.
 * 
 * @param pFlags 
 * @param p 
 * @return int 
 */
TAresult TA_API taMemHostGetFlags(unsigned int* pFlags, void* p);

/**
 * @brief Free page locked host memory.
 *
 * @param hptr Pointer to memory to be freed
 * @return int
 * On success, zero is returned.
 * @sa taMemAllocHost, taMemHostAlloc
 ************************************************************/
TAresult TA_API taMemFreeHost(void* hptr);

/**
 * @brief Register host memory as page locked memory.
 *
 * @param hptr Pointer to host memory to be registered.
 * @param sizeBytes Requested memory size
 * @param flags See below.
 * flags:
 * - #TA_MEMHOSTREGISTER_PORTABLE  Memory is considered registered by all
 *contexts.
 * - #TA_MEMHOSTREGISTER_DEVICEMAP Map the allocation into the address space for
 *the current device.
 * - #TA_MEMHOSTREGISTER_IOMEMORY  The passed memory pointer is treated as
 *pointing to some memory-mapped I/O space.
 * - #TA_MEMHOSTREGISTER_READ_ONLY The passed memory pointer is treated as
 *pointing to memory that is considered read-only by the device. TANG does not
 *support IOMMU on device side, so flags of TA_MEMHOSTREGISTER_DEVICEMAP and
 *TA_MEMHOSTREGISTER_IOMEMORY and TA_MEMHOSTREGISTER_READ_ONLY will always
 *return false.
 * @return int
 * On success, zero is returned.
 * @sa taMemHostUnregister
 ***********************************************************/
TAresult TA_API taMemHostRegister(void* hptr, size_t sizeBytes, unsigned int flags);

/**
 * @brief Un-register host pointer
 *
 * @param hptr Host pointer previously registered
 * @return int
 * On success, zero is returned.
 * @sa taMemHostRegister
 ************************************************************/
TAresult TA_API taMemHostUnregister(void* hptr);

/**
 * @brief Returns information about a pointer.
 * 
 * @param data - Pointer to the returned attribute value.
 * @param attr - Pointer attribute to query
 * @param ptr  - Pointer to be queried.
 * @return int
 * ::TANG_SUCCESS
 * ::TANG_ERROR_INVALID_VALUE
 */
TAresult TA_API taPointerGetAttribute(void*               data,
                                 TApointer_attribute attr,
                                 TAdeviceptr         ptr);

TAresult TA_API taMemset(TAdeviceptr dptr, int value, size_t size);
TAresult TA_API taMemset_ptds(TAdeviceptr dptr, int value, size_t size);
TAresult TA_API taMemsetAsync(TAdeviceptr dptr,
                         int         value,
                         size_t      size,
                         TAstream    stream);
TAresult TA_API taMemsetAsync_ptsz(TAdeviceptr dptr,
                              int         value,
                              size_t      size,
                              TAstream    stream);

/**
 * @brief Copy data from host memory to host memory.
 *
 * @param dstHost - Host destination data address.
 * @param srcHost - Host source data address.
 * @param size - The size in bytes of data to be copied.
 * @return int
 * ::TANG_SUCCESS
 ************************************************************/
TAresult TA_API taMemcpyH2H(void* dstHost, const void* srcHost, size_t size);

/**
 * @brief Copy data from host memory to host memory.
 *
 * @param dstHost - Host destination data address.
 * @param srcHost - Host source data address
 * @param size The size in bytes of data to be copied.
 * @return int
 ************************************************************/
TAresult TA_API taMemcpyH2H_ptds(void*       dstHost,
                            const void* srcHost,
                            size_t      size);


/**
 * @brief Copy data from host memory to device memory.
 *
 * @param dstDevice - Device destination data address.
 * @param srcHost - Host source data address.
 * @param size - The size in bytes of data to be copied.
 * @return int
 * ::TANG_SUCCESS
 ************************************************************/
TAresult TA_API taMemcpyH2D(TAdeviceptr dstDevice, const void* srcHost, size_t size);

/**
 * @brief Copy data from host memory to device memory.
 *
 * @param dstDevice Device destination data address
 * @param srcHost Host source data address
 * @param size The size in bytes of data to be copied.
 * @return int
 ************************************************************/
TAresult TA_API taMemcpyH2D_ptds(TAdeviceptr dstDevice,
                            const void* srcHost,
                            size_t      size);

/**
 * @brief Copy data from device to host.
 *
 * @param dstHost
 * @param srcDevice
 * @param size
 * @return int
 * ::TANG_SUCCESS
 *************************************************************/
TAresult TA_API taMemcpyD2H(void* dstHost, TAdeviceptr srcDevice, size_t size);

/**
 * @brief
 *
 * @param dstHost
 * @param srcDevice
 * @param size
 * @return int
 *************************************************************/
TAresult TA_API taMemcpyD2H_ptds(void* dstHost, TAdeviceptr srcDevice, size_t size);

/**
 * @brief
 *
 * @param dstDevice
 * @param srcDevice
 * @param size
 * @return int
 * ::TANG_ERROR_NOT_IMPLEMENTED.
 *************************************************************/
TAresult TA_API taMemcpyD2D(TAdeviceptr dstDevice,
                       TAdeviceptr srcDevice,
                       size_t      size);

/**
 * @brief
 *
 * @param dstDevice
 * @param srcDevice
 * @param size
 * @return int
 *************************************************************/
TAresult TA_API taMemcpyD2D_ptds(TAdeviceptr dstDevice,
                            TAdeviceptr srcDevice,
                            size_t      size);

TAresult TA_API taMemcpyH2HAsync(void*       dstHost,
                            const void* srcHost,
                            size_t      size,
                            TAstream    stream);

TAresult TA_API taMemcpyH2HAsync_ptsz(void*       dstHost,
                                 const void* srcHost,
                                 size_t      size,
                                 TAstream    stream);

TAresult TA_API taMemcpyH2DAsync(TAdeviceptr dstDevice,
                            const void* srcHost,
                            size_t      size,
                            TAstream    stream);

TAresult TA_API taMemcpyH2DAsync_ptsz(TAdeviceptr dstDevice,
                                 const void* srcHost,
                                 size_t      size,
                                 TAstream    stream);

TAresult TA_API taMemcpyD2HAsync(void*       dstHost,
                            TAdeviceptr srcDevice,
                            size_t      size,
                            TAstream    stream);

TAresult TA_API taMemcpyD2HAsync_ptsz(void*       dstHost,
                                 TAdeviceptr srcDevice,
                                 size_t      size,
                                 TAstream    stream);

TAresult TA_API taMemcpyD2DAsync(TAdeviceptr dstDevice,
                            TAdeviceptr srcDevice,
                            size_t      size,
                            TAstream    stream);

TAresult TA_API taMemcpyD2DAsync_ptsz(TAdeviceptr dstDevice,
                                 TAdeviceptr srcDevice,
                                 size_t      size,
                                 TAstream    stream);

TAresult TA_API taStreamCreate(TAstream* pStream, unsigned int flags);
TAresult TA_API taStreamCreateWithPriority(TAstream*    pstream,
                                      unsigned int flags,
                                      int          priority);
TAresult TA_API taStreamGetPriority(TAstream hstream, int* priority);
TAresult TA_API taStreamGetPriority_ptsz(TAstream hstream, int* priority);
TAresult TA_API taStreamGetFlags(TAstream hstream, unsigned int* priority);
TAresult TA_API taStreamGetFlags_ptsz(TAstream hstream, unsigned int* priority);
TAresult TA_API taStreamGetId(TAstream hstream, int* pId);
TAresult TA_API taStreamGetId_ptsz(TAstream hstream, int* pId);
TAresult TA_API taStreamDestroy(TAstream hStream);

/*********************************************************
 ********************************************************/
#ifndef TANGRT_DEVICE_P2P_ATTR_ENUM
#define TANGRT_DEVICE_P2P_ATTR_ENUM
/**
 * TANG Device P2P attributes
 * TODO: This is design bug fix this. Remove this
 */
typedef enum tangDeviceP2PAttr {
  tangDevP2PAttrPerformanceRank          = 1,
  tangDevP2PAttrAccessSupported          = 2,
  tangDevP2PAttrNativeAtomicSupported    = 3,
  tangDevP2PAttrTangArrayAccessSupported = 4,
} tangDeviceP2PAttr;
#endif  // TANGRT_DEVICE_P2P_ATTR_ENUM

TAresult TA_API taStreamC2Ctransfers(TAstream  hStream,
                                uint32_t* cmd,
                                uint32_t  cmd_count,
                                uint64_t  device_addr,
                                uint32_t  mem_size);

TAresult TA_API taDeviceGetP2PAttribute(int*              value,
                                   tangDeviceP2PAttr attr,
                                   int               srcDevice,
                                   int               dstDevice);
TAresult TA_API taDeviceGetPeerPointer(int    device,
                                  int    port,
                                  void*  peerAddr,
                                  void** accessAddr);
TAresult TA_API taDeviceCanAccessPeer(int* canAccessPeer,
                                 int  device,
                                 int  peerDevice);
TAresult TA_API taDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
TAresult TA_API taDeviceDisablePeerAccess(int peerDevice);
TAresult TA_API taMemcpyPeer(void*       dst,
                        int         dstDevice,
                        const void* src,
                        int         srcDevice,
                        size_t      count);

TAresult TA_API taMemcpyPeerAsync(void*       dst,
                             int         dstDevice,
                             const void* src,
                             int         srcDevice,
                             size_t      count,
                             TAstream    stream);

//!< @ingroup Memory between peers.
//!< Copy data by HDMA
//!< @{{{
TAresult TA_API taMemcpyPeer_v2(TAdeviceptr dst,
                           TAdevice    dstDevice,
                           TAdeviceptr src,
                           TAdevice    srcDevice,
                           size_t      size);

TAresult TA_API taMemcpyPeer_v2_ptds(TAdeviceptr dst,
                                TAdevice    dstDevice,
                                TAdeviceptr src,
                                TAdevice    srcDevice,
                                size_t      size);

TAresult TA_API taMemcpyPeerAsync_v2(TAdeviceptr dst,
                                TAdevice    dstDevice,
                                TAdeviceptr src,
                                TAdevice    srcDevice,
                                size_t      size,
                                TAstream    hStream);

TAresult TA_API taMemcpyPeerAsync_v2_ptsz(TAdeviceptr dst,
                                     TAdevice    dstDevice,
                                     TAdeviceptr src,
                                     TAdevice    srcDevice,
                                     size_t      size,
                                     TAstream    hStream);

TAresult TA_API taMemcpyFromPeerAsync(TAdeviceptr dst,
                                 TAdeviceptr src,
                                 TAdevice    srcDevice,
                                 size_t      size,
                                 TAstream    stream);

TAresult TA_API taMemcpyToPeerAsync(TAdeviceptr dst,
                               TAdevice    dstDevice,
                               TAdeviceptr src,
                               size_t      size,
                               TAstream    stream);

TAresult TA_API taMemcpyFromPeerAsync_ptsz(TAdeviceptr dst,
                                      TAdeviceptr src,
                                      TAdevice    srcDevice,
                                      size_t      size,
                                      TAstream    stream);

TAresult TA_API taMemcpyToPeerAsync_ptsz(TAdeviceptr dst,
                                    TAdevice    dstDevice,
                                    TAdeviceptr src,
                                    size_t      size,
                                    TAstream    stream);
//!< }}}@

TAresult TA_API taStreamWaitEvent(TAstream     hStream,
                             TAevent      hEvent,
                             unsigned int flags);
TAresult TA_API taStreamWaitEvent_ptsz(TAstream     hStream,
                                  TAevent      hEvent,
                                  unsigned int flags);
TAresult TA_API taStreamSynchronize(TAstream hStream);
TAresult TA_API taStreamSynchronize_ptsz(TAstream hStream);
TAresult TA_API taStreamQuery(TAstream hStream);
TAresult TA_API taStreamQuery_ptsz(TAstream hStream);

TAresult TA_API taStreamBeginCapture(TAstream hStream, TAstreamCaptureMode mode);
TAresult TA_API taStreamBeginCapture_ptsz(TAstream            hStream,
                                     TAstreamCaptureMode mode);
TAresult TA_API taThreadExchangeStreamCaptureMode(TAstreamCaptureMode* mode);
TAresult TA_API taStreamEndCapture(TAstream hStream, TAgraph* phGraph);
TAresult TA_API taStreamEndCapture_ptsz(TAstream hStream, TAgraph* phGraph);
TAresult TA_API taStreamIsCapturing(TAstream               hStream,
                               TAstreamCaptureStatus* captureStatus);
TAresult TA_API taStreamIsCapturing_ptsz(TAstream               hStream,
                                    TAstreamCaptureStatus* captureStatus);
TAresult TA_API taStreamGetCaptureInfo(TAstream               hStream,
                                  TAstreamCaptureStatus* pStatus,
                                  unsigned long long*    pId,
                                  TAgraph*               pGraph,
                                  const TAgraphNode**    deps,
                                  size_t*                numDeps);
TAresult TA_API taStreamGetCaptureInfo_ptsz(TAstream               hStream,
                                       TAstreamCaptureStatus* pStatus,
                                       unsigned long long*    pId,
                                       TAgraph*               pGraph,
                                       const TAgraphNode**    deps,
                                       size_t*                numDeps);

TAresult TA_API taGraphInstantiateWithFlags(TAgraphExec*       phGraphExec,
                                       TAgraph            hGraph,
                                       unsigned long long flags);
TAresult TA_API taGraphLaunch(TAgraphExec hGraphExec, TAstream hStream);
TAresult TA_API taGraphLaunch_ptsz(TAgraphExec hGraphExec, TAstream hStream);
TAresult TA_API taGraphDestroy(TAgraph hGraph);
TAresult TA_API taGraphExecDestroy(TAgraphExec hGraphExec);
TAresult TA_API taGraphGetInfo(TAgraph hGraph, TAgraphInfo* pInfo);
TAresult TA_API taGraphCreate(TAgraph* phGraph, unsigned int flags);

TAresult TA_API taGraphAddHostNode(TAgraphNode*                 phGraphNode,
                              TAgraph                      hGraph,
                              const TAgraphNode*           dependencies,
                              size_t                       numDependencies,
                              const TANG_HOST_NODE_PARAMS* nodeParams);

TAresult TA_API taGraphAddKernelNode(TAgraphNode*                   phGraphNode,
                                TAgraph                        hGraph,
                                const TAgraphNode*             dependencies,
                                size_t                         numDependencies,
                                const TANG_KERNEL_NODE_PARAMS* nodeParams);

TAresult TA_API taEventCreate(TAevent* phEvent, unsigned int flags);
TAresult TA_API taEventDestroy(TAevent hEvent);
TAresult TA_API taEventRecord(TAevent hEvent, TAstream hStream);
TAresult TA_API taEventRecord_ptsz(TAevent hEvent, TAstream hStream);
TAresult TA_API taEventRecordWithFlags(TAevent      hEvent,
                                  TAstream     hStream,
                                  unsigned int flags);
TAresult TA_API taEventRecordWithFlags_ptsz(TAevent      hEvent,
                                       TAstream     hStream,
                                       unsigned int flags);
TAresult TA_API taEventSynchronize(TAevent hEvent);
TAresult TA_API taEventElapsedTime(float*  pMilliseconds,
                              TAevent hStart,
                              TAevent hEnd);
TAresult TA_API taEventQuery(TAevent hEvent);
TAresult TA_API taEventQueryTimestamp(TAevent hEvent, TAeventTimestamp* pTs);

TAresult TA_API taEventSynchronizeWithFlags(TAevent hEvent, unsigned int flags);

void TA_API taDsoWrapperInit(TAdsoWrapper_t* dso);
void TA_API taDsoWrapperDeinit(TAdsoWrapper_t* dso);

TAresult TA_API taGetBuiltinModule(TAmodule* phModule, const char* name);

TAresult TA_API taModuleLoad(TAmodule* phModule, const char* filename);
TAresult TA_API taModuleLoadData(TAmodule* phModule, const void* image, size_t size);
TAresult TA_API taModuleUnload(TAmodule hModule);

TAresult TA_API taModuleLoadFatBinaryManaged(TAmodule*       phModule,
                                        const void*     fatbin,
                                        const char*     fatbinInfo,
                                        TAdsoWrapper_t* dso);

TAresult TA_API taModuleUnloadManaged(TAmodule hModule);

/**
 * @brief Get the module symbol type name.
 * 
 * @param name The pointer to receive the name of the type.
 * @param type The symbol type.
 * @return TAresult
 * ::TANG_SUCCESS if the type is a valid value.
 * ::TANG_ERROR_INVALID_VALUE if the type is not a valid type.
 */
TAresult TA_API taModuleSymbolTypeGetName(char const** name, TAmoduleSymbolType type);

/**
 * @brief Iterate symbols in \p hmod.
 * 
 * @param hmod 
 * @param fn The call back function. Return true will cause the
 *        iteration to stop.
 * @param userData 
 * @return TAresult
 */
// TAresult TA_API taModuleIterateSymbols(TAmodule                hmod,
//                                        TAmoduleSymbolIterateFn fn,
//                                        void*                   userData);

TAresult TA_API taModuleGetFunction(TAfunction* hfunc,
                                    TAmodule    hmod,
                                    const char* name);

TAresult TA_API taFuncGetAttribute(int*                 pi,
                                   TAfunction_attribute attr,
                                   TAfunction           hfunc);

TAresult TA_API taFuncGetModule(TAmodule* hmod, TAfunction func);

TAresult TA_API taFunctionGetAddress(TAfunction func, TAdeviceptr* address);

TAresult TA_API taFunctionGetNumArgs(TAfunction func, size_t* numArgs);

TAresult TA_API taFunctionGetInfo(TAfunction   func,
                                  TAdeviceptr* address,
                                  size_t*      lenArgs,
                                  size_t*      thread_regfile_size,
                                  size_t*      shared_regfile_base,
                                  size_t*      shared_regfile_size,
                                  size_t*      warp_regfile_size,
                                  size_t*      local_memory_size,
                                  size_t*      static_shared_mem_size,
                                  size_t*      shared_memory_mirror,
                                  size_t*      max_threads_per_block,
                                  size_t* max_dynamic_shared_mem_size_per_block,
                                  size_t* max_block_count);

TAresult TA_API taModuleGetVariable(TAvariable* hVar,
                               TAmodule    hMod,
                               const char* varName);

TAresult TA_API taVariableGetInfo(TAvariable   hVar,
                             TAdeviceptr* address,
                             size_t*      size);

TAresult TA_API taVariableCopyFromDevice(TAvariable  hVar,
                                    TAdeviceptr src,
                                    size_t      size,
                                    size_t      offset);

TAresult TA_API taVariableCopyFromDevice_ptds(TAvariable  hVar,
                                         TAdeviceptr src,
                                         size_t      size,
                                         size_t      offset);

TAresult TA_API taVariableCopyFromDeviceAsync(TAvariable  hVar,
                                         TAdeviceptr src,
                                         size_t      size,
                                         size_t      offset,
                                         TAstream    stream);

TAresult TA_API taVariableCopyFromDeviceAsync_ptsz(TAvariable  hVar,
                                              TAdeviceptr src,
                                              size_t      size,
                                              size_t      offset,
                                              TAstream    stream);

TAresult TA_API taVariableCopyFromHost(TAvariable  hVar,
                                  const void* src,
                                  size_t      size,
                                  size_t      offset);

TAresult TA_API taVariableCopyFromHost_ptds(TAvariable  hVar,
                                       const void* src,
                                       size_t      size,
                                       size_t      offset);

TAresult TA_API taVariableCopyFromHostAsync(TAvariable  hVar,
                                       const void* src,
                                       size_t      size,
                                       size_t      offset,
                                       TAstream    stream);

TAresult TA_API taVariableCopyFromHostAsync_ptsz(TAvariable  hVar,
                                            const void* src,
                                            size_t      size,
                                            size_t      offset,
                                            TAstream    stream);

TAresult TA_API taVariableCopyToDevice(TAdeviceptr dst,
                                  TAvariable  hVar,
                                  size_t      size,
                                  size_t      offset);

TAresult TA_API taVariableCopyToDevice_ptds(TAdeviceptr dst,
                                       TAvariable  hVar,
                                       size_t      size,
                                       size_t      offset);

TAresult TA_API taVariableCopyToDeviceAsync(TAdeviceptr dst,
                                       TAvariable  hVar,
                                       size_t      size,
                                       size_t      offset,
                                       TAstream    stream);

TAresult TA_API taVariableCopyToDeviceAsync_ptsz(TAdeviceptr dst,
                                            TAvariable  hVar,
                                            size_t      size,
                                            size_t      offset,
                                            TAstream    stream);

TAresult TA_API taVariableCopyToHost(void*      dst,
                                TAvariable hVar,
                                size_t     size,
                                size_t     offset);

TAresult TA_API taVariableCopyToHost_ptds(void*      dst,
                                     TAvariable hVar,
                                     size_t     size,
                                     size_t     offset);

TAresult TA_API taVariableCopyToHostAsync(void*      dst,
                                     TAvariable hVar,
                                     size_t     size,
                                     size_t     offset,
                                     TAstream   stream);

TAresult TA_API taVariableCopyToHostAsync_ptsz(void*      dst,
                                          TAvariable hVar,
                                          size_t     size,
                                          size_t     offset,
                                          TAstream   stream);

/**
 * @brief Enqueue A raw SCP command packet onto stream.
 *
 * @param stream The stream.
 * @param regs   The SCP command packet.
 * @param size   The size of the command packet in byte.
 * @return int
 * @warning A raw SCP command packet needs four bytes aligned.
 * The \p size must be integral multiple of 4.
 ****************************************************************/
TAresult TA_API taEnqueueCommand(TAstream stream, void* regs, size_t size);

TAresult TA_API taEnqueueCommand_ptsz(TAstream stream, void* regs, size_t size);

/**
 * @brief Launch a kernel function.
 *
 * @param func
 * @param gridX
 * @param gridY
 * @param gridZ
 * @param blockX
 * @param blockY
 * @param blockZ
 * @param sharedMemBytes
 * @param stream
 * @param funcParams
 * @param extra
 * @code {.cpp}
 *
 * @endcode
 * @param extra
 * @return int
 * ::TANG_SUCCESS
 * ::TANG_ERROR_OUT_OF_MEMORY
 * ::TANG_ERROR_NOT_IMPLEMENTED
 * @sa taModuleGetFunction
 * @sa TA_STREAM_LEGACY
 ********************************************************/
TAresult TA_API taLaunchFunction(TAfunction   func,
                            unsigned int gridX,
                            unsigned int gridY,
                            unsigned int gridZ,
                            unsigned int blockX,
                            unsigned int blockY,
                            unsigned int blockZ,
                            unsigned int sharedMemBytes,
                            TAstream     stream,
                            void**       funcParams,
                            void**       extra);

TAresult TA_API taLaunchFunction_ptsz(TAfunction   func,
                                 unsigned int gridX,
                                 unsigned int gridY,
                                 unsigned int gridZ,
                                 unsigned int blockX,
                                 unsigned int blockY,
                                 unsigned int blockZ,
                                 unsigned int sharedMemBytes,
                                 TAstream     stream,
                                 void**       funcParams,
                                 void**       extra);

TAresult TA_API taLaunchKernel(TAfunction   func,
                          unsigned int gridX,
                          unsigned int gridY,
                          unsigned int gridZ,
                          unsigned int blockX,
                          unsigned int blockY,
                          unsigned int blockZ,
                          unsigned int sharedMemBytes,
                          TAstream     stream,
                          void**       funcParams,
                          void**       extra);

TAresult TA_API taLaunchKernel_ptsz(TAfunction   func,
                               unsigned int gridX,
                               unsigned int gridY,
                               unsigned int gridZ,
                               unsigned int blockX,
                               unsigned int blockY,
                               unsigned int blockZ,
                               unsigned int sharedMemBytes,
                               TAstream     stream,
                               void**       funcParams,
                               void**       extra);

//!< fn(usrData)
TAresult TA_API taLaunchHostFunc(TAstream hStream, TAhostFn fn, void* userData);
TAresult TA_API taLaunchHostFunc_ptsz(TAstream hStream, TAhostFn fn, void* userData);

typedef void (*TAstreamCallback)(TAstream hStream,
                                 TAresult status,
                                 void*    userData);

//!< callback(hStream, status, userData);
TAresult TA_API taStreamAddCallback(TAstream         hStream,
                               TAstreamCallback callback,
                               void*            userData,
                               unsigned int     flags);

TAresult TA_API taStreamAddCallback_ptsz(TAstream         hStream,
                                    TAstreamCallback callback,
                                    void*            userData,
                                    unsigned int     flags);

//!< proxy(proxy_data, func, func_data, error)
TAresult TA_API taLaunchHostFuncProxy(TAstream stream,
                                 void*    proxy,
                                 void*    proxy_data,
                                 void*    func,
                                 void*    func_data);

TAresult TA_API taLaunchHostFuncProxy_ptsz(TAstream stream,
                                      void*    proxy,
                                      void*    proxy_data,
                                      void*    func,
                                      void*    func_data);

TAresult TA_API taOccupancyMaxActiveBlocksPerMultiprocessor(int*       numBlocks,
                                                       TAfunction func,
                                                       int        blockSize,
                                                       size_t dynamicSMemSize);
TAresult TA_API taProfilerStart();
TAresult TA_API taProfilerStop();

/**
 * @brief Gets an interprocess communication memory handle from device memory
 * allocated by tangMalloc or cuMemAlloc
 * 
 * @param pHandle 
 * @param dptr 
 * @return
 * ::TANG_SUCCESS
 * ::TANG_ERROR_INVALID
 * ::TANG_ERROR_OUT_OF_MEMORY
 * @sa
 * ::taIpcOpenMemHandle
 * ::taIpcCloseMemHandle
 */
TAresult TA_API taIpcGetMemHandle(TAipcMemHandle* pHandle, TAdeviceptr dptr);

/**
 * @brief Opens an interprocess communication memory handle exported from another
 * process and map it into the current context and returns a device pointer.
 * 
 * @param pdptr 
 * @param handle 
 * @param flags 
 * @return
 * ::TANG_SUCCESS
 * ::TANG_ERROR_IPC_MEM_DESTROIED
 * ::TANG_ERROR_OUT_OF_MEMORY
 * @sa 
 * ::taIpcGetMemHandle
 * ::taIpcCloseMemHandle
 */
TAresult TA_API taIpcOpenMemHandle(TAdeviceptr   *pdptr,
                              TAipcMemHandle handle,
                              unsigned int   flags);

/**
 * @brief Unmap the memory got from taIpcOpenMemHandle.
 * 
 * @param dptr 
 * @return int 
 */
TAresult TA_API taIpcCloseMemHandle(TAdeviceptr dptr);

/**
 * @brief Gets an interprocess event handle. The event must be created with
 * ::TA_EVENT_INTERPROCESS flag set.
 * 
 * @param pHandle 
 * @param event 
 * @return int 
 */
TAresult TA_API taIpcGetEventHandle(TAipcEventHandle* pHandle, TAevent event);

/**
 * @brief Opens an interprocess event handle for the calling process.
 *
 * Opens an interprocess event handle exported from another process with
 * ::taIpcGetEventHandle.
 * 
 * Use ::taEventDestroy to free the event.
 * @param phEvent 
 * @param handle 
 * @return int
 * ::TANG_SUCCESS
 * ::TANG_ERROR_OUT_MEMORY
 */
TAresult TA_API taIpcOpenEventHandle(TAevent* phEvent, TAipcEventHandle handle);

/**
 * @brief launch engine collectives witch stream.
 *
 * @param devId which ptpu
 * @param collType engine collectives type
 * @param devAddr HBM address
 * @param size params length
 * @param stream for sync
 * @return int
 */
TAresult TA_API taStreamEngineCollAssign(int devId, int collType, uint64_t devAddr, int size, TAstream stream);

// enum TAqueryInfoType_enum {
//   TA_QUERY_INFO_MEMORY_USAGE = 0x01,
// };
// typedef enum TAqueryInfoType_enum TAqueryInfoType;
//
// TAresult TA_API taCtxQueryMemoryUsage(int64_t* pTotal, TAcontext context);

TAresult TA_API taGetExportTable(void** pTable, void* args);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // _TANG_H_

