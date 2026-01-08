/*
Copyright declaration.
*/
#ifndef _TANG_RT_DRIVER_TYPES_H_
#define _TANG_RT_DRIVER_TYPES_H_
#include "tang_rt/vector_types.h"

#define tangHostAllocDefault 0x00 /**< Default page-locked allocation flag */
#define tangHostAllocPortable \
  0x01 /**< Pinned memory accessible by all TANG contexts */
#define tangHostAllocMapped        0x02 /**< Map allocation into device space */
#define tangHostAllocWriteCombined 0x04 /**< Write-combined memory */
#define tangHostAllocMapDeviceMemory \
  0x100 /**< Allocate device memory and map it to userspace */

#define tangHostRegisterDefault \
  0x00 /**< Default host memory registration flag */
#define tangHostRegisterPortable \
  0x01 /**< Pinned memory accessible by all TANG contexts */
#define tangHostRegisterMapped \
  0x02 /**< Map registered memory into device space */
#define tangHostRegisterIoMemory 0x04 /**< Memory-mapped I/O space */
#define tangHostRegisterReadOnly 0x08 /**< Memory-mapped read-only */

/**< Default behavior */
#define tangOccupancyDefault 0x00
/**< Assume global caching is enabled and cannot be automatically turned off */
#define tangOccupancyDisableCachingOverride 0x01
/*
 * TANG error types
 */
// Developer note - when updating these, update the tangGetErrorString
// functions.

/**
 * Not consider Cooperative Launch, Peer Access, Pitch, Texture, Graph, JIT,
 * Managed Memory, Multi-thread(compute mode) right now
 */

enum tangError {
  /**
   * Some of the tangError are not supported,
   * such as tangErrorInvalidPitchValue, tangErrorInvalidHostPointer, etc
   * If we develop tools to convert tang program to tang,
   * tangTangErrorToTangError should redirect them to tangErrorUnknown.
   * Need check.
   */

  /**
   * The API call returned with no errors. In the case of query calls, this
   * also means that the operation being queried is complete (see
   * ::tangEventQuery() and ::tangStreamQuery()).
   * Need check.
   */
  tangSuccess = 0,

  /**
   * This indicates that one or more of the parameters passed to the API call
   * is not within an acceptable range of values.
   */
  tangErrorInvalidValue = 1,

  /**
   * The API call failed because it was unable to allocate enough memory to
   * perform the requested operation.
   * In cuda: cudaErrorMemoryAllocation
   * In hip: hipErrorOutOfMemory
   */
  tangErrorMemoryAllocation = 2,

  /**
   * The API call failed because the TANG driver and runtime could not be
   * initialized.
   * In cuda: cudaErrorInitializationError
   * In hip: hipErrorNotInitialized
   */
  tangErrorInitializationError = 3,

  /**
   * This indicates that a TANG Runtime API call cannot be executed because
   * it is being called during process shut down, at a point in time after
   * Tang runtime has been deinitialized.
   * In cuda: cudaErrorCudartUnloading
   * In hip: hipErrorDeinitialized
   * hip declares it but not use.
   * Maybe hip will not meet such a point.
   * Need check.
   */
  tangErrorDeinitialized = 4,

  /**
   * This indicates that the device is removed.
   */
  tangErrorDeviceRemoved = 5,

  /**
   * This indicated that the device is reset.
   */
  tangErrorDeviceReset = 6,

  /**
   * This operation is not allowed
   */
  tangErrorNotPermitted = 7,

  /**
   * The specified file or directory is not found
   */
  tangErrorNoSuchFile = 8,

  /**
   * This indicates that a kernel launch is requesting resources that can
   * never be satisfied by the current device. Requesting more shared memory
   * per block than the device supports will trigger this error, as will
   * requesting too many threads or blocks. See ::tangDeviceProp for more
   * device limitations.
   */
  tangErrorInvalidConfiguration = 9,

  /**
   * @brief Null pointer is passed as argument but it is disallowed.
   */
  tangErrorNullPointer = 10,

  tangErrorOutOfResources = 11,

  /**
   * This indicates that the symbol name/identifier passed to the API call
   * is not a valid name or identifier.
   */
  tangErrorInvalidSymbol = 13,

  /**
   * This indicates that at least one device pointer passed to the API call is
   * not a valid device pointer.
   * Note: This error is deprecated from CUDA 10.1,
   * but hip still use it
   */
  tangErrorInvalidDevicePointer = 17,

  /**
   * This indicates that the direction of the memcpy passed to the API call is
   * not one of the types specified by ::tangMemcpyKind.
   * hip declares it but not use.
   * But it seems useful.
   * Need check.
   */
  tangErrorInvalidMemcpyDirection = 21,

  /**
   * This indicates that the installed TANG driver version is mismatch with the
   * runtime version.
   * hip declares it but not use.
   * But it seems useful.
   * Need check.
   */
  tangErrorInsufficientDriver = 35,

  /**
   * The device function being invoked (usually via ::tangLaunchKernel()) was
   * not previously configured via the ::tangConfigureCall() function
   * nor provided with ".kernel_info" in ELF.
   */
  tangErrorMissingConfiguration = 52,

  /**
   * The requested device function does not exist or is not compiled for the
   * proper device architecture.
   */
  tangErrorInvalidDeviceFunction = 98,

  /**
   * This error indicates the attempted operation is not implemented.
   */
  tangErrorNotImplemented = 99,

  /**
   * This indicates that no TANG-capable devices were detected by the installed
   * TANG driver. Call to tangGetDeviceCount returned 0 devices.
   */
  tangErrorNoDevice = 100,

  /**
   * This indicates that the device ordinal supplied by the user does not
   * correspond to a valid TANG device.
   * DeviceID must be in range 0...#compute-devices.
   */
  tangErrorInvalidDevice = 101,

  /**
   * This indicates that the device kernel image is invalid.
   * In cuda: cudaErrorInvalidKernelImage
   * In hip: hipErrorInvalidImage
   */
  tangErrorInvalidKernelImage = 200,

  /**
   * This most frequently indicates that there is no context bound to the
   * current thread. This can also be returned if the context passed to an
   * API call is not a valid handle (such as a context that has had
   * ::taCtxDestroy() invoked on it).
   * In cuda: cudaErrorDeviceUninitialized
   * In hip：hipErrorInvalidContext
   */
  tangErrorInvalidContext = 201,

  /**
   * This indicates that there is no kernel image available that is suitable
   * for the device. This can occur when a user specifies code generation
   * options for a particular TANG source file that do not include the
   * corresponding device configuration.
   * In cuda: cudaErrorNoKernelImageForDevice
   * In hip: hipErrorNoBinaryForGpu
   * Need check.
   */
  tangErrorNoKernelImageForDevice = 209,

  /**
   * This indicates that the ::tangLimit passed to the API call is not
   * supported by the active device.
   */
  tangErrorUnsupportedLimit = 215,

  /**
   * This indicates that a call tried to access an exclusive-thread device that
   * is already in use by a different thread.
   * In cuda: cudaErrorDeviceAlreadyInUse
   * In hip: hipErrorContextAlreadyInUse
   */
  tangErrorContextAlreadyInUse = 216,

  /**
   * A PTX compilation failed. The runtime may fall back to compiling PTX if
   * an application does not contain a suitable binary for the current device.
   * In cuda: cudaErrorInvalidPtx
   * In hip: hipErrorInvalidKernelFile
   */
  tangErrorInvalidKernelFile = 218,

  /**
   * When launch kernel, unable to find the corresponding fatbinary.
   */
  tangErrorFatBinaryNotFound = 300,

  /**
   * This indicates that the file specified was not found.
   */
  tangErrorFileNotFound = 301,

  /**
   * This indicates that a link to a shared object failed to resolve.
   */
  tangErrorSharedObjectSymbolNotFound = 302,

  /**
   * This indicates that initialization of a shared object failed.
   */
  tangErrorSharedObjectInitFailed = 303,

  /**
   * This error indicates that an OS call failed.
   * hip declares it but not use.
   * But it seems useful.
   * Need check.
   */
  tangErrorOperatingSystem = 304,

  tangErrorIllegalState = 305,

  tangErrorStreamCaptureUnsupported = 306,

  tangErrorStreamCaptureInvalidated = 307,

  tangErrorStreamCaptureMerge = 308,

  tangErrorStreamCaptureUnmatched = 309,

  tangErrorStreamCaptureUnjoined = 310,

  tangErrorStreamCaptureIsolation = 311,

  tangErrorStreamCaptureImplicit = 312,

  tangErrorStreamCaptureWrongThread = 313,

  tangErrorCapturedEvent = 314,

  /**
   * This indicates that a resource handle passed to the API call was not
   * valid. Resource handles are opaque types like ::tangStream_t and
   * ::tangEvent_t.
   * In cuda: cudaErrorInvalidResourceHandle
   * In hip: hipErrorInvalidHandle
   */
  tangErrorInvalidResourceHandle = 400,

  /**
   * This indicates that a named symbol was not found. Examples of symbols
   * are global/constant variable names, texture names, and surface names.
   * In cuda: cudaErrorSymbolNotFound
   * In hip: tangErrorNotFound
   */
  tangErrorSymbolNotFound = 500,

  /**
   * This indicates that asynchronous operations issued previously have not
   * completed yet. This result is not actually an error, but must be indicated
   * differently than ::tangSuccess (which indicates completion). Calls that
   * may return this value include ::tangEventQuery() and ::tangStreamQuery().
   */
  tangErrorNotReady = 600,

  /**
   * The device encountered a load or store instruction on an invalid memory
   * address. This leaves the process in an inconsistent state and any further
   * TANG work will return the same error. To continue using TANG, the process
   * must be terminated and relaunched. hip declares it but not use. But it
   * seems useful. Need check.
   */
  tangErrorIllegalAddress = 700,

  /**
   * This indicates that a launch did not occur because it did not have
   * appropriate resources. Although this error is similar to
   * ::tangErrorInvalidConfiguration, this error usually indicates that the
   * user has attempted to pass too many arguments to the device kernel, or the
   * kernel launch specifies too many threads for the kernel's register count.
   */
  tangErrorLaunchOutOfResources = 701,

  /**
   * This indicates that the device kernel took too long to execute. This can
   * only occur if timeouts are enabled - see the device property
   * \ref ::tangDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
   * for more information.
   * This leaves the process in an inconsistent state and any further TANG work
   * will return the same error. To continue using TANG, the process must be
   * terminated and relaunched. hip declares it but not use. But it seems
   * useful. Need check.
   */
  tangErrorLaunchTimeOut = 702,

  tangErrorPeerAccessAlreadyEnabled = 704,
  tangErrorPeerAccessNotEnabled     = 705,

  /**
   * This error indicates that the memory range passed to ::tangHostRegister()
   * has already been registered.
   * hip declares it but not use.
   * But it seems useful.
   * Need check.
   */
  tangErrorHostMemoryAlreadyRegistered = 712,

  /**
   * This error indicates that the pointer passed to ::tangHostUnregister()
   * does not correspond to any currently registered memory region.
   * hip declares it but not use.
   * But it seems useful.
   * Need check.
   */
  tangErrorHostMemoryNotRegistered = 713,

  /**
   * An exception occurred on the device while executing a kernel. Common
   * causes include dereferencing an invalid device pointer and accessing
   * out of bounds shared memory. Less common cases can be system specific -
   * more information about these cases can be found in the system specific user
   * guide. This leaves the process in an inconsistent state and any further
   * TANG work will return the same error. To continue using TANG, the process
   * must be terminated and relaunched.
   */
  tangErrorLaunchFailure = 719,

  /**
   * This error indicates the attempted operation is not supported
   * on the current system or device.
   */
  tangErrorNotSupported = 801,

  /**
   * This indicates that an unknown internal error has occurred.
   */
  tangErrorUnknown = 999,

  /**
   * @brief context is destroyed or in destroying in kernel
   *
   */
  tangErrorContextIsDestroyed = 3000,

  /**
   * @brief context is not valid in kernel
   *
   */
  tangErrorContextInvalid = 3001,

  /**
   * @brief stream is destroyed or in destroying in kernel
   *
   */
  tangErrorStreamIsDestroyed = 3002,

  /**
   * @brief stream is not valid in kernel
   *
   */
  tangErrorStreamInvalid = 3003,

  /**
   * @brief event is destroyed or in destroying in kernel
   *
   */
  tangErrorEventIsDestroyed = 3004,

  /**
   * @brief event is not valid in kernel
   *
   */
  tangErrorEventInvalid = 3005,

  /**
   * @brief device memory is not enough for current operation
   *
   */
  tangErrorDeviceOutOfMemory = 3006,

  /**
   * @brief device memory is not found
   *
   */
  tangErrorDeviceMemoryNotFound = 3007,

  /**
   * @brief pcie fatal error occured
   *
   */
  tangErrorPcieFatal = 3012,

  /**
   * @brief pcie non-fatal unrecovered error occured
   *
   */
  tangErrorPcieNonFatalUnrecovered = 3013,

  /**
   * @brief no more event exist
   *
   */
  tangErrorScpEventNotExist = 3014,

  /**
   * @brief record event failed
   *
   */
  tangErrorSCPEventRecordFailed = 3015,

  /**
   * @brief scp packet crc check failed
   *
   */
  tangErrorSCPCrcPacketFailed = 3016,

  /**
   * @brief scp dispatch send failed
   *
   */
  tangErrorSCPDispSendFailed = 3017,

  /**
   * @brief sq write sequence error
   *
   */
  tangErrorSCPSqWriteFailed = 3018,

  /**
   * @brief udrc pcie xdma packet invalid
   *
   */
  tangErrorUdrcPcieDmaPacketInvalid = 3019,

  /**
   * @brief udrc mp dma packet invalid
   *
   */
  tangErrorUdrcMpDmaPacketInvalid = 3020,

  /**
   * @brief udrc reg packet invalid
   *
   */
  tangErrorUdrcRegPacketInvalid = 3021,

  /**
   * @brief udrc reg access invalid
   *
   */
  tangErrorUdrcRegAcessInvalid = 3022,

  /**
   * @brief aiss cluster is not configured
   *
   */
  tangErrorAissClusterUsrNotAllocated = 3023,

  /**
   * @brief barrier is destroyed or in destroying in kernel
   *
   */
  tangErrorBarrierDestroyed = 3024,

  /**
   * @brief barrier is not valid in kernel
   *
   */
  tangErrorBarrierInvalid = 3025,

  /**
   * @brief one obj is destroyed or in destroying in kernel
   *
   */
  tangErrorDestroyed = 3026,

  /**
   * @brief xdma C2H align mismath
   *
   */
  tangErrorXdmaC2HAlignMismatch = 3300,

  /**
   * @brief xdma C2H invalid magic stopped
   *
   */
  tangErrorXdmaC2HInvalidMagicStopped = 3301,

  /**
   * @brief xdma C2H invalid Len
   *
   */
  tangErrorXdmaC2HInvalidLen = 3302,

  /**
   * @brief xdma C2H decode error
   *
   */
  tangErrorXdmaC2HDecode = 3303,

  /**
   * @brief xdma C2H slave
   *
   */
  tangErrorXdmaC2HSlave = 3304,

  /**
   * @brief xdma C2H descriptor unsupport request
   *
   */
  tangErrorXdmaC2HDescUnsupportRequest = 3305,

  /**
   * @brief xdma C2H descriptor completer abort
   *
   */
  tangErrorXdmaC2HDescCompleterAbort = 3306,

  /**
   * @brief xdma C2H descriptor parity
   *
   */
  tangErrorXdmaC2HDescParity = 3307,

  /**
   * @brief xdma C2H descriptor header ep
   *
   */
  tangErrorXdmaC2HDescHeaderEp = 3308,

  /**
   * @brief xdma C2H descriptor unexpected comp
   *
   */
  tangErrorXdmaC2HDescUnexpectedComp = 3309,

  /**
   * @brief xdma C2H timeout
   *
   */
  tangErrorXdmaC2HTimeout = 3310,

  /**
   * @brief xdma C2H unknown
   *
   */
  tangErrorXdmaC2HUnknown = 3311,

  /**
   * @brief xdma H2C align mismatch
   *
   */
  tangErrorXdmaH2CAlignMismatch = 3350,

  /**
   * @brief xdma H2C invalid magic stopped
   *
   */
  tangErrorXdmaH2CInvaildMagicStopped = 3351,

  /**
   * @brief xdma H2C invalid len
   *
   */
  tangErrorXdmaH2CInvalidLen = 3352,

  /**
   * @brief xdma H2C read unsupport request
   *
   */
  tangErrorXdmaH2CReadUnSupportRequest = 3353,

  /**
   * @brief xdma H2C read completer abort
   *
   */
  tangErrorXdmaH2CReadCompleterAbort = 3354,

  /**
   * @brief xdma H2C read parity
   *
   */
  tangErrorXdmaH2CReadParity = 3355,

  /**
   * @brief xdma H2C read header ep
   *
   */
  tangErrorXdmaH2CReadHeaderEp = 3356,

  /**
   * @brief xdma H2C read unexpected comp
   *
   */
  tangErrorXdmaH2CReadUnExpectedComp = 3357,

  /**
   * @brief xdma H2C decode error
   *
   */
  tangErrorXdmaH2CDecode = 3358,

  /**
   * @brief xdma H2C slave
   *
   */
  tangErrorXdmaH2CSlave = 3359,

  /**
   * @brief xdma H2C descriptor unsupport request
   *
   */
  tangErrorXdmaH2CDescUnSupportRequest = 3360,

  /**
   * @brief xdma H2C descriptor completer abort
   *
   */
  tangErrorXdmaH2CDescCompleterAbort = 3361,

  /**
   * @brief xdma H2C descriptor parity
   *
   */
  tangErrorXdmaH2CDescParity = 3362,

  /**
   * @brief xdma H2C descriptor header ep
   *
   */
  tangErrorXdmaH2CDescHeaderEp = 3363,

  /**
   * @brief xdma H2C descriptor unexpected com
   *
   */
  tangErrorXdmaH2CDescUnExpectedComp = 3364,

  /**
   * @brief xdma H2C timeout
   *
   */
  tangErrorXdmaH2CTimeout = 3365,

  /**
   * @brief xdma H2C unknown
   *
   */
  tangErrorXdmaH2CUnknown = 3366,

  /**
   * This indicates that the IOCTL operation of TANG driver is failed.
   * Added by TANG. hip and tang do not use.
   * Need check, avoid to use the save number with cuda and hip.
   */
  tangErrorDriverIoctlFailed = 10000,
};

/**
 * TANG memory copy types
 */
typedef enum tangMemcpyKind {
  tangMemcpyHostToHost     = 0, /**< Host   -> Host */
  tangMemcpyHostToDevice   = 1, /**< Host   -> Device */
  tangMemcpyDeviceToHost   = 2, /**< Device -> Host */
  tangMemcpyDeviceToDevice = 3, /**< Device -> Device */
                                /**
                                 * Direction of the transfer is inferred from the pointer values.
                                 * Requires unified virtual addressing, thus tang doesn't support.
                                 */
                                // tangMemcpyDefault             =   4
} tangMemcpyKind;

enum tangMemoryType {
  tangMemoryTypeUnregistered = 0, /**< Unregistered memory */
  tangMemoryTypeHost         = 1, /**< Host memory */
  tangMemoryTypeDevice       = 2, /**< Device memory */
  tangMemoryTypeManaged      = 3, /**< Managed memory */
};

struct tangPointerAttributes {
  enum tangMemoryType type;

  int device;

  void* devicePointer;

  void* hostPointer;
};

/**
 * TANG function attributes
 */
typedef struct tangFuncAttributes {
  /**
   * The size in bytes of statically-allocated shared memory per block
   * required by this function. This does not include dynamically-allocated
   * shared memory requested by the user at runtime.
   */
  size_t sharedSizeBytes;

  /**
   * The size in bytes of user-allocated constant memory required by this
   * function.
   */
  // PT devices use global memory to perform as constant memory,
  // and constant memory belongs to the module, rather than the function
  size_t constSizeBytes;

  /**
   * The size in bytes of local memory used by each thread of this function.
   */
  size_t localSizeBytes;

  /**
   * The maximum number of threads per block, beyond which a launch of the
   * function would fail. This number depends on both the function and the
   * device on which the function is currently loaded.
   */
  int maxThreadsPerBlock;

  /**
   * The number of registers used by each thread of this function.
   */
  int numRegs;

  /**
   * The PTX virtual architecture version for which the function was
   * compiled. This value is the major PTX version * 10 + the minor PTX
   * version, so a PTX version 1.3 function would return the value 13.
   */
  int ptxVersion;

  /**
   * The binary architecture version for which the function was compiled.
   * This value is the major binary version * 10 + the minor binary version,
   * so a binary version 1.3 function would return the value 13.
   */
  int binaryVersion;

  /**
   * The attribute to indicate whether the function has been compiled with
   * user specified option "-Xptxas --dlcm=ca" set.
   */
  int cacheModeCA;

  /**
   * The maximum size in bytes of dynamic shared memory per block for
   * this function. Any launch must have a dynamic shared memory size
   * smaller than this value.
   */
  int maxDynamicSharedSizeBytes;

  /**
   * On devices where the L1 cache and shared memory use the same hardware
   * resources, this sets the shared memory carveout preference, in percent of
   * the maximum shared memory. Refer to
   * ::tangDevAttrSHARED_MEM_PER_MULTIPROCESSOR. This is only a hint, and the
   * driver can choose a different ratio if required to execute the function.
   * See ::tangFuncSetAttribute
   *
   * PT devices do not suppport to config L1 cache/shared memory.
   */
  int preferredShmemCarveout;
} tangFuncAttributes;

/**
 * TANG function attributes that can be set using ::tangFuncSetAttribute
 */
typedef enum tangFuncAttribute {
  tangFuncAttributeMaxDynamicSharedMemorySize =
    8, /**< Maximum dynamic shared memory size */
  tangFuncAttributePreferredSharedMemoryCarveout =
    9, /**< Preferred shared memory-L1 cache split */
  tangFuncAttributeMax
} tangFuncAttribute;

/**
 * TANG function cache configurations
 * @warning On PT2 devices, L1 cache and shared memory are separated,
 * thus these hints and controls are ignored.
 */
typedef enum tangFuncCache {
  tangFuncCachePreferNone,  ///< no preference for shared memory or L1 (default)
  tangFuncCachePreferShared,  ///< prefer larger shared memory and smaller L1
                              ///< cache
  tangFuncCachePreferL1,  ///< prefer larger L1 cache and smaller shared memory
  tangFuncCachePreferEqual,  ///< prefer equal size L1 cache and shared memory
} tangFuncCache;

/**
 * TANG shared memory configuration
 * @warning On PT2 devices, shard memory bank size is fix to 4-bytes,
 * thus these hints and controls are ignored.
 */
typedef enum tangSharedMemConfig {
  tangSharedMemBankSizeDefault,   ///< The compiler selects a device-specific
                                  ///< value for the banking.
  tangSharedMemBankSizeFourByte,  ///< Shared mem is banked at 4-bytes intervals
                                  ///< and performs best when adjacent threads
                                  ///< access data 4 bytes apart.
  tangSharedMemBankSizeEightByte  ///< Shared mem is banked at 8-byte intervals
                                  ///< and performs best when adjacent threads
                                  ///< access data 4 bytes apart.
} tangSharedMemConfig;

/**
 * TANG Limits
 */
enum tangLimit {
  tangLimitStackSize      = 0x00, /**< GPU thread stack size */
  tangLimitPrintfFifoSize = 0x01, /**< GPU printf FIFO size */
  tangLimitMallocHeapSize = 0x02, /**< GPU malloc heap size */
  tangLimitDevRuntimeSyncDepth =
    0x03, /**< GPU device runtime synchronize depth */
  tangLimitDevRuntimePendingLaunchCount =
    0x04, /**< GPU device runtime pending launch count */
  tangLimitMaxL2FetchGranularity =
    0x05 /**< A value between 0 and 128 that indicates the
// maximum fetch granularity of L2 (in Bytes). This is a hint */
};

enum tangEventFlags_e {
  tangEventDefault       = 0x00,
  tangEventDisableTiming = 0x02,
  tangEventInterprocess  = 0x04,
};

enum tangEventSyncFlags_e {
  tangEventSyncDefault = 0x00,

  //!< Block until the event is recorded and done.
  tangEventSyncRecordedAndDone = 0x01,
};

enum tangEventRecordFlags_e {
  //!< The default recording mode.
  tangEventRecordDefault = 0x00,

  //!< Always use the hardware event.
  tangEventRecordHW = 0x0100,

  //!< Always use the software event.
  tangEventRecordSW = 0x0200,

  //!< Allow to blockinig the calling thread is resource is not available.
  tangEventRecordBlockingAllowed = 0x0400,
};

/**
 * TANG Memory Advise values
 */
// enum tangMemoryAdvise {};

/**
 * TANG range attributes
 */
// enum tangMemRangeAttribute {};

typedef enum tangStreamCaptureMode_e {
  tangStreamCaptureModeGlobal      = 0,
  tangStreamCaptureModeThreadLocal = 1,
  tangStreamCaptureModeRelaxed     = 2,
} tangStreamCaptureMode;

typedef enum tangStreamCaptureStatus_e {
  tangStreamCaptureStatusNone        = 0,
  tangStreamCaptureStatusActive      = 1,
  tangStreamCaptureStatusInvalidated = 2,
} tangStreamCaptureStatus;

/**
 * TANG device attribute enum
 */
typedef enum tangDeviceAttr {
  tangDevAttrMaxSharedMemPerBlock = 0,        //!< sharedMemPerBlock
  tangDevAttrMaxRegsPerBlock,                 //!< regsPerBlock
  tangDevAttrWarpSize,                        //!< warpSize
  tangDevAttrMemPitch,                        //!< memPitch
  tangDevAttrMaxThreadsPerBlock,              //!< maxThreadsPerBlock
  tangDevAttrMaxBlockDimX,                    //!< maxThreadsDimX
  tangDevAttrMaxBlockDimY,                    //!< maxThreadsDimY
  tangDevAttrMaxBlockDimZ,                    //!< maxThreadsDimZ
  tangDevAttrMaxGridDimX,                     //!< maxGridSizeX
  tangDevAttrMaxGridDimY,                     //!< maxGridSizeY
  tangDevAttrMaxGridDimZ,                     //!< maxGridSizeZ
  tangDevAttrClockRate,                       //!< clockRate
  tangDevAttrTotalConstantMemory,             //!< totalConstMem
  tangDevAttrMultiProcessorCount,             //!< multiProcessorCount
  tangDevAttrMaxBlocksPerMultiProcessor,      //!< maxBlocksPerMultiProcessor
  tangDevAttrAsyncEngineCount,                //!< asyncEngineCount
  tangDevAttrMemoryClockRate,                 //!< memoryClockRate
  tangDevAttrGlobalMemoryBusWidth,            //!< memoryBusWidth
  tangDevAttrL2CacheSize,                     //!< l2CacheSize
  tangDevAttrMaxThreadsPerMultiProcessor,     //!< maxThreadsPerMultiProcessor
  tangDevAttrGlobalL1CacheSupported,          //!< globalL1CacheSupported
  tangDevAttrLocalL1CacheSupported,           //!< localL1CacheSupported
  tangDevAttrMaxSharedMemoryPerMultiprocessor,//!< sharedMemPerMultiprocessor
  tangDevAttrMaxRegistersPerMultiprocessor,   //!< regsPerMultiprocessor
  tangDevAttrStreamPrioritiesSupported,       //!< streamPrioritiesSupported
  tangDevAttrConcurrentKernels,               //!< concurrentKernels
  tangDevAttrComputePreemptionSupported,      //!< computePreemptionSupported
  tangDevAttrKernelExecTimeout,               //!< kernelExecTimeoutEnabled
  tangDevAttrEccEnabled,                      //!< ECCEnabled
  tangDevAttrMaxAccessPolicyWindowSize,       //!< accessPolicyMaxWindowSize
  tangDevAttrTccDriver,                       //!< tccDriver
  tangDevAttrSingleToDoublePrecisionPerfRatio,//!< singleToDoublePrecisionPerfRatio
  tangDevAttrCooperativeLaunch,               //!< cooperativeLaunch
  tangDevAttrCooperativeMultiDeviceLaunch,    //!< cooperativeMultiDeviceLaunch
  tangDevAttrMaxPersistingL2CacheSize,        //!< persistingL2CacheMaxSize
  tangDevAttrCanMapHostMemory,                //!< canMapHostMemory
  tangDevAttrUnifiedAddressing,               //!< unifiedAddressing
  tangDevAttrManagedMemory,                   //!< managedMemory
  tangDevAttrConcurrentManagedAccess,         //!< concurrentManagedAccess
  tangDevAttrDirectManagedMemAccessFromHost,  //!< directManagedMemAccessFromHost
  tangDevAttrPageableMemoryAccess,            //!< pageableMemoryAccess
  tangDevAttrPageableMemoryAccessUsesHostPageTables,  //!< pageableMemoryAccessUsesHostPageTables
  tangDevAttrCanUseHostPointerForRegisteredMem,       //!< canUseHostPointerForRegisteredMem
  tangDevAttrHostNativeAtomicSupported,       //!< hostNativeAtomicSupported
  tangDevAttrCanFlushRemoteWrites,            //!< canFlushRemoteWrites
  tangDevAttrGpuOverlap,                      //!< gpuOverlap
  tangDevAttrIntegrated,                      //!< integrated
  tangDevAttrMaxSharedMemoryPerBlockOptin,    //!< maxSharedMemoryPerBlockOptin
  tangDevAttrGPUDirectRDMASupported,          //!< gpuDirectRDMASupported
  tangDevAttrGPUDirectRDMAFlushWritesOptions, //!< gpuDirectRDMAFlushWritesOptions
  tangDevAttrGPUDirectRDMAWritesOrdering,     //!< gpuDirectRDMAWritesOrdering
  tangDevAttrComputeCapabilityMajor,          //!< major
  tangDevAttrComputeCapabilityMinor,          //!< minor
  tangDevAttrPciBusId,                        //!< pciBusID
  tangDevAttrPciDeviceId,                     //!< pciDeviceID
  tangDevAttrPciDomainId,                     //!< pciDomainID
  tangDevAttrIsMultiGpuBoard,                 //!< isMultiGpuBoard
  tangDevAttrMultiGpuBoardGroupID,            //!< multiGpuBoardGroupID
  tangDevAttrComputeMode,                     //!< computeMode
  tangDevAttrReservedSharedMemoryPerBlock,    //!< reservedSharedMemoryPerBlock
  tangDevAttrSparseTangArraySupported,        //!< sparseTangArraySupported
  tangDevAttrHostRegisterSupported,           //!< hostRegisterSupported
  tangDevAttrHostRegisterReadOnlySupported,   //!< hostRegisterReadOnlySupported
  tangDevAttrMemoryPoolsSupported,            //!< memoryPoolsSupported
  tangDevAttrMemoryPoolSupportedHandleTypes,  //!< memoryPoolSupportedHandleTypes
  tangDevAttrMax
} tangDeviceAttr;

#ifndef TANGRT_DEVICE_P2P_ATTR_ENUM
#define TANGRT_DEVICE_P2P_ATTR_ENUM
/**
 * TANG Device P2P attributes
 */
enum tangDeviceP2PAttr {
  tangDevP2PAttrPerformanceRank          = 1,
  tangDevP2PAttrAccessSupported          = 2,
  tangDevP2PAttrNativeAtomicSupported    = 3,
  tangDevP2PAttrTangArrayAccessSupported = 4,
};
#endif  // TANGRT_DEVICE_P2P_ATTR_ENUM

/**
 *
 * TANG device properties
 * Inconsistent with cudaDeviceProp
 * Most field unused.
 * Need check.
 */
typedef struct tangDeviceProp {
  char name[256]; ///< Device name.
  char uuid[16];  ///< a 16-byte unique identifier
  uint64_t totalGlobalMem;  ///< size of global memory region (in bytes).
  int sharedMemPerBlock;  ///< the maximum amount of shared memory
                               ///< available to a thread block in bytes.
  int regsPerBlock;   ///< the maximum number of 32-bit registers available to a
                      ///< thread block.
  int      warpSize;  ///< the warp size in threads.
  int memPitch;  ///< the maximum pitch in bytes allowed by the memory copy
                      ///< functions
  int maxThreadsPerBlock;  ///< the maximum number of threads per block.
  int maxThreadsDim[3];  ///< Max number of threads in each dimension (XYZ) of a
                         ///< block.
  int      maxGridSize[3];  ///< Max grid dimensions (XYZ).
  int      clockRate;  ///< Max clock frequency of the multiProcessors in khz.
  int totalConstMem;  ///< the total amount of constant memory available on
                           ///< the device in bytes.
  int multiProcessorCount;  ///< Number of multi-processors (compute units).
  int maxBlocksPerMultiProcessor;  ///< the number of multiprocessors on the
                                   ///< device
  int asyncEngineCount;            ///<
  int memoryClockRate;  ///< Max global memory clock frequency in khz.
  int memoryBusWidth;   ///< Global memory bus width in bits.
  int l2CacheSize;      ///< L2 cache size.
  int maxThreadsPerMultiProcessor;  ///< Maximum resident threads per
                                    ///< multi-processor.
  int globalL1CacheSupported;       ///< whether the device supports caching of
                                    ///< globals in L1 cache
  int localL1CacheSupported;  ///< whether the device supports caching of locals
                              ///< in L1 cache
  int sharedMemPerMultiprocessor;  ///< Maximum Shared Memory Per Multiprocessor.
  int regsPerMultiprocessor;  ///< the maximum amount of shared memory available
                              ///< to a multiprocessor in bytes
  int streamPrioritiesSupported;  ///< whether the device supports stream
                                  ///< priorities
  int concurrentKernels;  ///< Device can possibly execute multiple kernels
                          ///< concurrently.
  int computePreemptionSupported;  ///< whether the device supports Compute
                                   ///< Preemption
  int kernelExecTimeoutEnabled;  ///< Run time limit for kernels executed on the
                                 ///< device
  int ECCEnabled;                ///< Device has ECC support enabled
  int accessPolicyMaxWindowSize;  ///< the maximum value of
                                  ///< tangAccessPolicyWindow::num_bytes
  int tccDriver;  ///< whether device is a Tesla device using TCC driver
  int singleToDoublePrecisionPerfRatio;  ///< the ratio of single precision
                                         ///< performance (in floating-point
                                         ///< operations per second) to double
                                         ///< precision performance
  int cooperativeLaunch;  ///< whether the device supports launching cooperative
                          ///< kernels via tangLaunchCooperativeKernel
  int cooperativeMultiDeviceLaunch;  ///< whether the device supports launching
                                     ///< cooperative kernels via
                                     ///< tangLaunchCooperativeKernelMultiDevice
  int persistingL2CacheMaxSize;  ///< L2 cache's maximum persisting lines size
                                 ///< in bytes
  int canMapHostMemory;          ///< Check whether TANG can map host memory
  int unifiedAddressing;  ///< whether the device shares a unified address space
                          ///< with the host and 0 otherwise
  int managedMemory;  ///< whether the device supports allocating managed memory
                      ///< on this system, or 0 if it is not supported
  int concurrentManagedAccess;  ///< whether the device can coherently access
                                ///< managed memory concurrently with the CPU
  int directManagedMemAccessFromHost;  ///< whether the host can directly access
                                       ///< managed memory on the device without
                                       ///< migration
  int pageableMemoryAccess;  ///< whether the device supports coherently
                             ///< accessing pageable memory without calling
                             ///< tangHostRegister on it
  int pageableMemoryAccessUsesHostPageTables;  ///< whether the device accesses
                                               ///< pageable memory via the
                                               ///< host's page tables
  int canUseHostPointerForRegisteredMem;  ///< whether the device can access
                                          ///< host registered memory at the
                                          ///< same virtual address as the CPU

  int hostNativeAtomicSupported;  ///< Link between the device and the host
                                  ///< supports native atomic operations
  int canFlushRemoteWrites;  ///< Device supports flushing of outstanding remote
                             ///< writes
  int gpuOverlap;  ///< Device can possibly copy memory and execute a kernel
                   ///< concurrently
  int integrated;  ///< Device is integrated with host memory
  int maxSharedMemoryPerBlockOptin;     ///< The maximum optin shared memory per
                                        ///< block. This value may vary by chip.
                                        ///< See ::tangFuncSetAttribute
  int gpuDirectRDMASupported;           ///< Device supports GPUDirect RDMA APIs
  int gpuDirectRDMAFlushWritesOptions;  ///< The returned attribute shall be
                                        ///< interpreted as a bitmask, where the
                                        ///< individual bits are listed in the
                                        ///< ::tangFlushGPUDirectRDMAWritesOptions
                                        ///< enum
  int gpuDirectRDMAWritesOrdering;  ///< GPUDirect RDMA writes to the device do
                                    ///< not need to be flushed for consumers
                                    ///< within the scope indicated by the
                                    ///< returned attribute. See
                                    ///< ::tangGPUDirectRDMAWritesOrdering for
                                    ///< the numerical values returned here.
  int  major;     ///< the major revision numbers defining the device's compute
                  ///< capability
  int minor;      ///< the minor revision numbers defining the device's compute
                  ///< capability
  int  pciBusID;  ///< PCI Bus ID.
  int  pciDeviceID;           ///< PCI Device ID.
  int  pciDomainID;           ///< PCI Domain ID
  int  isMultiGpuBoard;       ///< whether device is on a multi-GPU board.
  int  multiGpuBoardGroupID;  ///< a unique identifier for a group of devices
                              ///< associated with the same board
  int computeMode;  ///< the compute mode that the device is currently in
  int reservedSharedMemoryPerBlock;  ///< Shared memory reserved by TANG driver
                                     ///< per block in bytes
  int sparseTangArraySupported;  ///< Device supports sparse arrays and sparse
                                 ///< mipmapped arrays
  int hostRegisterSupported;  ///< Device supports host memory registration via
                              ///< ::tangHostRegister
  int hostRegisterReadOnlySupported;  ///< Device supports using the
                                      ///< ::tangHostRegister flag
                                      ///< tangHostRegisterReadOnly to register
                                      ///< memory that must be mapped as
                                      ///< read-only to the GPU
  int memoryPoolsSupported;  ///< Device supports using the ::tangMallocAsync
                             ///< and ::tangMemPool family of APIs
  int memoryPoolSupportedHandleTypes;  ///< Handle types supported with mempool
                                       ///< based IPC
} __attribute__((packed)) tangDeviceProp;

/**
 * TANG launch parameters
 */
/*
struct __device_builtin__ tangLaunchParams {
    void* func;             ///< Device function symbol
    dim3 gridDim;           ///< Grid dimentions
    dim3 blockDim;          ///< Block dimentions
    void **args;            ///< Arguments
    int sharedMem;          ///< Shared memory
    tangStream_t stream;    ///< Stream identifier
};
*/
/*******************************************************************************
 *                                                                              *
 *  SHORTHAND TYPE DEFINITION USED BY RUNTIME API *
 *                                                                              *
 *******************************************************************************/

/**
 * TANG Error types
 */
typedef enum tangError tangError_t;

/**
 * @brief TANG Device
 * @sa TAdevice
 */
typedef struct TAdevice_s* tangDevice_t;

/**
 * @brief TANG context
 * @sa TAcontext
 */
typedef struct TActx_s* tangContext_t;

/**
 * @brief TANG stream
 * @sa TAstream
 */
typedef struct TAstream_s* tangStream_t;

/**
 * @brief TANG event
 * @sa TAevent
 */
typedef struct TAevent_s* tangEvent_t;

/**
 * @brief TANG function
 * @sa TAfunction
 */
typedef struct TAfunc_s* tangFunction_t;

/**
 * @brief TANG graph & executable graph handle
 * @sa tangStreamBeginCapture
 * @sa tangStreamEndCapture
 * @sa tangGraphLaunch
 * @sa tangGraphInstantiate
 */
typedef struct TAgraph_s*     tangGraph_t;
typedef struct TAgraphExec_s* tangGraphExec_t;
typedef struct TAgraphNode_s* tangGraphNode_t;

typedef void (*tangHostFn_t)(void* userData);

typedef struct tangHostNodeParams_s {
  tangHostFn_t fn;
  void*        userData;
} tangHostNodeParams;

typedef struct tangKernelNodeParams_s {
  void* func;     /**< Kernel to launch */
  dim3  gridDim;  /**< Grid dimensions */
  dim3  blockDim; /**< Block dimensions */

  /**< Dynamic shared memory size per thread block in bytes */
  unsigned int sharedMemBytes;

  /**< Kernel parameters */
  void** kernelParams;
  void** extra;
} tangKernelNodeParams;

typedef struct tangGraphInfo_s {
  int nr_nodes;
} tangGraphInfo;

typedef struct tangEventTimestamp_s {
  uint64_t comp;
  uint64_t comp_sw;
  uint64_t create;
  uint64_t enqueue;
  uint64_t writeq_beg;
  uint64_t writeq_end;
} tangEventTimestamp;

struct tangLanuchParams {
  void*        func;
  dim3         gridDim;
  dim3         blockDim;
  void**       args;
  size_t       sharedMemBytes;
  tangStream_t stream;
};

/**
 * @brief tangFatbinaryWrapper
 * TANGCC will provides the following asm code on x86_64 platform.
 * @code{.s}
 * __tang_fatbin_wrapper:
 * 	.long	0                       # 0x0, version, 4 bytes
 * 	.zero	4                       # padding space, 4 bytes
 * 	.quad	.L_Z9vectorAddPfS_S_.11 # fatbin
 * 	.zero	160                     # data[20]
 * 	.size	__tang_fatbin_wrapper, 176
 * @endcode
 */
struct __tangFatbinaryWrapper {
  int version;
  const void* fatbin;
  // The TANGCC does not reserve space for size.
  // The size will be parsed from fatbin
  // unsigned long size;
  struct {
    uintptr_t data[20];
  } dso;
};

#define TANG_IPC_HANDLE_SIZE     64U
#define TANG_IPC_MEM_HANDLE_SIZE 64U

#define tangIpcMemLazyEnablePeerAccess 0x01

typedef struct tangIpcMemHandle_s {
  unsigned long reserved[TANG_IPC_MEM_HANDLE_SIZE / sizeof(unsigned long)];
} tangIpcMemHandle_t;

typedef struct tangIpcEventHandle_s {
  unsigned long reserved[TANG_IPC_HANDLE_SIZE / sizeof(unsigned long)];
} tangIpcEventHandle_t;

#endif  //! _TANG_RT_DRIVER_TYPES_H_
