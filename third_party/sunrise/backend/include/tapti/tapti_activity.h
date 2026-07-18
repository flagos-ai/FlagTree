#ifndef _TAPTI_ACTIVITY_HPP_
#define _TAPTI_ACTIVITY_HPP_

#include <stdint.h>
#include <stddef.h>

#include "tapti_callbacks.h"

/**
 * \brief The kind of a memory copy, indicating the source and
 * destination targets of the copy.
 *
 * Each kind represents the source and destination targets of a memory
 * copy. Targets are host, device, and array.
 */
typedef enum {
  /**
   * The memory copy kind is not known.
   */
  TAPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN = 0,

  /**
   * A host to device memory copy.
   */
  TAPTI_ACTIVITY_MEMCPY_KIND_HTOD    = 1,

  /**
   * A device to host memory copy.
   */
  TAPTI_ACTIVITY_MEMCPY_KIND_DTOH    = 2,

  /**
   * A device to device memory copy on the same device.
   */
  TAPTI_ACTIVITY_MEMCPY_KIND_DTOD    = 3,

  /**
   * A host to host memory copy.
   */
  TAPTI_ACTIVITY_MEMCPY_KIND_HTOH    = 4,

  /**
   * A peer to peer memory copy across different devices.
   */
  TAPTI_ACTIVITY_MEMCPY_KIND_PTOP    = 5,

  TAPTI_ACTIVITY_MEMCPY_KIND_FORCE_INT = 0x7fffffff
} TApti_ActivityMemcpyKind;

typedef enum {
  TAPTI_EXTERNAL_CORRELATION_KIND_INVALID     = 0,
  TAPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN     = 1,
  TAPTI_EXTERNAL_CORRELATION_KIND_OPENACC     = 2,
  TAPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0     = 3,
  TAPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1     = 4,
  TAPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2     = 5,
  TAPTI_EXTERNAL_CORRELATION_KIND_SIZE,
  TAPTI_EXTERNAL_CORRELATION_KIND_FORCE_INT   = 0x7fffffff
} TApti_ExternalCorrelationKind;

#define ACTIVITY_RECORD_ALIGNMENT 8
#define PACKED_ALIGNMENT __attribute__ ((__packed__)) __attribute__ ((aligned (ACTIVITY_RECORD_ALIGNMENT)))

/**
 * \brief The kinds of activity records.
 *
 * Each activity record kind represents information about a GPU or an
 * activity occurring on a CPU or GPU. Each kind is associated with a
 * activity record structure that holds the information associated
 * with the kind.
 * \see TApti_Activity
 * \see TApti_ActivityAPI
 * \see TApti_ActivityExternalCorrelation
 * \see TApti_ActivityKernel
 * \see TApti_ActivityMemcpy
 * \see TApti_ActivityMemcpyPtoP
 * \see TApti_ActivityMemset
 */
 typedef enum {
  /**
   * The activity record is invalid.
   */
  TAPTI_ACTIVITY_KIND_INVALID                  = 0,

  /**
   * A host<->host, host<->device, or device<->device memory copy. The
   * corresponding activity record structure is \ref
   */
  TAPTI_ACTIVITY_KIND_MEMCPY                   = 1,

  /**
   * A memory set executing on the GPU. The corresponding activity
   * record structure is \ref TApti_ActivityMemset.
   */
  TAPTI_ACTIVITY_KIND_MEMSET                   = 2,

  /**
   * A kernel executing on the GPU. This activity kind may significantly change
   * the overall performance characteristics of the application because all
   * kernel executions are serialized on the GPU. Other activity kind for kernel
   * TAPTI_ACTIVITY_KIND_CONCURRENT_KERNEL doesn't break kernel concurrency.
   * The corresponding activity record structure is \ref TApti_ActivityKernel.
   */
  TAPTI_ACTIVITY_KIND_KERNEL                   = 3,

  /**
   * A TANG driver API function execution. The corresponding activity
   * record structure is \ref TApti_ActivityAPI.
   */
  TAPTI_ACTIVITY_KIND_DRIVER                   = 4,

  /**
   * A TANG runtime API function execution. The corresponding activity
   * record structure is \ref TApti_ActivityAPI.
   */
  TAPTI_ACTIVITY_KIND_RUNTIME                  = 5,

  /**
   * Records for correlation of different programming APIs. The
   * corresponding activity record structure is \ref
   * TApti_ActivityExternalCorrelation.
   */
  TAPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION     = 6,

  TAPTI_ACTIVITY_KIND_COUNT,

  TAPTI_ACTIVITY_KIND_FORCE_INT                = 0x7fffffff
} TApti_ActivityKind;

/**
 * \brief The kinds of memory accessed by a memory operation/copy.
 *
 * Each kind represents the type of the memory
 * accessed by a memory operation/copy.
 */
typedef enum {
  /**
   * The memory kind is unknown.
   */
  TAPTI_ACTIVITY_MEMORY_KIND_UNKNOWN            = 0,

  /**
   * The memory is pageable.
   */
  TAPTI_ACTIVITY_MEMORY_KIND_PAGEABLE           = 1,

  /**
   * The memory is pinned.
   */
  TAPTI_ACTIVITY_MEMORY_KIND_PINNED             = 2,

  /**
   * The memory is on the device.
   */
  TAPTI_ACTIVITY_MEMORY_KIND_DEVICE             = 3,

  /**
   * The memory is an array.
   */
  TAPTI_ACTIVITY_MEMORY_KIND_ARRAY              = 4,

  /**
   * The memory is managed
   */
  TAPTI_ACTIVITY_MEMORY_KIND_MANAGED            = 5,

  /**
   * The memory is device static
   */
  TAPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC      = 6,

  /**
   * The memory is managed static
   */
  TAPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC     = 7,

  TAPTI_ACTIVITY_MEMORY_KIND_FORCE_INT          = 0x7fffffff
} TApti_ActivityMemoryKind;


/**
 * \brief The base activity record.
 *
 * The activity API uses a TApti_Activity as a generic representation
 * for any activity. The 'kind' field is used to determine the
 * specific activity kind, and from that the TAPTI_Activity object can
 * be cast to the specific activity record type appropriate for that kind.
 *
 * Note that all activity record types are padded and aligned to
 * ensure that each member of the record is naturally aligned.
 *
 * \see TApti_ActivityKind
 */
typedef struct PACKED_ALIGNMENT {
  /**
   * The kind of this activity.
   */
  TApti_ActivityKind kind;
} TApti_Activity;

/**
 * \brief The activity record for correlation with external records
 *
 * This activity record correlates native TANG records (e.g. TANG Driver API,
 * kernels, memcpys, ...) with records from external APIs such as OpenACC.
 * (TAPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION).
 *
 * \see TApti_ActivityKind
 */
typedef struct PACKED_ALIGNMENT {
  TApti_ActivityKind kind;

  /**
   * The kind of external API this record correlated to.
   */
  TApti_ExternalCorrelationKind externalKind;

  /**
   * The correlation ID of the associated non-TANG API record.
   * The exact field in the associated external record depends
   * on that record's activity kind (\see externalKind).
   */
  uint64_t externalId;

  /**
   * The correlation ID of the associated TANG driver or runtime API record.
   */
  uint32_t correlationId;
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t reserved;
} TApti_ActivityExternalCorrelation;

/**
 * \brief The activity record for a driver or runtime API invocation.
 *
 * This activity record represents an invocation of a driver or
 * runtime API (TAPTI_ACTIVITY_KIND_DRIVER and
 * TAPTI_ACTIVITY_KIND_RUNTIME).
 */
 typedef struct PACKED_ALIGNMENT {
  TApti_ActivityKind kind;
  /**
   * The ID of the driver or runtime function.
   */
  TApti_CallbackId cbid;
  /**
   * The start timestamp for the function, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the function.
   */
  uint64_t start;

  /**
   * The end timestamp for the function, in ns. A value of 0 for both
   * the start and end timestamps indicates that timestamp information
   * could not be collected for the function.
   */
  uint64_t end;

  /**
   * The ID of the process where the driver or runtime TANG function
   * is executing.
   */
  uint32_t processId;

  /**
   * The ID of the thread where the driver or runtime TANG function is
   * executing.
   */
  uint32_t threadId;

  /**
   * The correlation ID of the driver or runtime TANG function. Each
   * function invocation is assigned a unique correlation ID that is
   * identical to the correlation ID in the memcpy, memset, or kernel
   * activity record that is associated with this function.
   */
  uint32_t correlationId;

  /**
   * The return value for the function. For a TANG driver function
   * with will be a TAresult value, and for a TANG runtime function
   * this will be a tangError_t value.
   */
  uint32_t returnValue;
} TApti_ActivityAPI;

/**
 * \brief The activity record for kernel. (deprecated)
 *
 * This activity record represents a kernel execution
 * (TAPTI_ACTIVITY_KIND_KERNEL and
 * TAPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) but is no longer generated
 * by TAPTI. Kernel activities are now reported using the
 * TApti_ActivityKernel9 activity record.
 */
 typedef struct PACKED_ALIGNMENT {
 /**
   * The activity record kind, must be TAPTI_ACTIVITY_KIND_KERNEL
   * or TAPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.
   */
  TApti_ActivityKind kind;

  /**
   * The cache configuration requested by the kernel. The value is one
   * of the CUfunc_cache enumeration values from cuda.h.
   */
  uint8_t cacheConfigRequested;

  /**
   * The cache configuration used for the kernel. The value is one of
   * the CUfunc_cache enumeration values from cuda.h.
   */
  uint8_t cacheConfigExecuted;

  /**
   * The number of registers required for each thread executing the
   * kernel.
   */
  uint16_t registersPerThread;

  /**
   * The start timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t start;

  /**
   * The end timestamp for the kernel execution, in ns. A value of 0
   * for both the start and end timestamps indicates that timestamp
   * information could not be collected for the kernel.
   */
  uint64_t end;

  /**
   * The ID of the device where the kernel is executing.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the kernel is executing.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the kernel is executing.
   */
  uint32_t streamId;

  /**
   * The X-dimension grid size for the kernel.
   */
  int32_t gridX;

  /**
   * The Y-dimension grid size for the kernel.
   */
  int32_t gridY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t gridZ;

  /**
   * The X-dimension block size for the kernel.
   */
  int32_t blockX;

  /**
   * The Y-dimension block size for the kernel.
   */
  int32_t blockY;

  /**
   * The Z-dimension grid size for the kernel.
   */
  int32_t blockZ;

  /**
   * The static shared memory allocated for the kernel, in bytes.
   */
  int32_t staticSharedMemory;

  /**
   * The dynamic shared memory reserved for the kernel, in bytes.
   */
  int32_t dynamicSharedMemory;

  /**
   * The amount of local memory reserved for each thread, in bytes.
   */
  uint32_t localMemoryPerThread;

  /**
   * The total amount of local memory reserved for the kernel, in
   * bytes.
   */
  uint32_t localMemoryTotal;

  /**
   * The correlation ID of the kernel. Each kernel execution is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the kernel.
   */
  uint32_t correlationId;

  /**
   * The runtime correlation ID of the kernel. Each kernel execution
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the kernel.
   */
  uint32_t runtimeCorrelationId;

  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;

  /**
   * The name of the kernel. This name is shared across all activity
   * records representing the same kernel, and so should not be
   * modified.
   */
  const char *name;

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} TApti_ActivityKernel;

/**
 * \brief The activity record for memory copies. (deprecated)
 *
 * This activity record represents a memory copy
 * (TAPTI_ACTIVITY_KIND_MEMCPY).
 */
 typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be TAPTI_ACTIVITY_KIND_MEMCPY.
   */
  TApti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size. \see TApti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size. \see TApti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size. \see TApti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see TApti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory copy is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver API activity record that launched
   * the memory copy.
   */
  uint32_t correlationId;

  /**
   * The runtime correlation ID of the memory copy. Each memory copy
   * is assigned a unique runtime correlation ID that is identical to
   * the correlation ID in the runtime API activity record that
   * launched the memory copy.
   */
  uint32_t runtimeCorrelationId;

#ifdef TAptiLP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} TApti_ActivityMemcpy;

/**
 * \brief The activity record for peer-to-peer memory copies.
 *
 * This activity record represents a peer-to-peer memory copy
 * (TAPTI_ACTIVITY_KIND_MEMCPY2) but is no longer generated
 * by TAPTI. Peer-to-peer memory copy activities are now reported using the
 * TApti_ActivityMemcpyPtoP2 activity record..
 */
 typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be TAPTI_ACTIVITY_KIND_MEMCPY2.
   */
  TApti_ActivityKind kind;

  /**
   * The kind of the memory copy, stored as a byte to reduce record
   * size.  \see TApti_ActivityMemcpyKind
   */
  uint8_t copyKind;

  /**
   * The source memory kind read by the memory copy, stored as a byte
   * to reduce record size.  \see TApti_ActivityMemoryKind
   */
  uint8_t srcKind;

  /**
   * The destination memory kind read by the memory copy, stored as a
   * byte to reduce record size.  \see TApti_ActivityMemoryKind
   */
  uint8_t dstKind;

  /**
   * The flags associated with the memory copy. \see
   * TApti_ActivityFlag
   */
  uint8_t flags;

  /**
   * The number of bytes transferred by the memory copy.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory copy, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory copy.
   */
  uint64_t end;

  /**
  * The ID of the device where the memory copy is occurring.
  */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory copy is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory copy is occurring.
   */
  uint32_t streamId;

  /**
   * The ID of the device where memory is being copied from.
   */
  uint32_t srcDeviceId;

  /**
   * The ID of the context owning the memory being copied from.
   */
  uint32_t srcContextId;

  /**
   * The ID of the device where memory is being copied to.
   */
  uint32_t dstDeviceId;

  /**
   * The ID of the context owning the memory being copied to.
   */
  uint32_t dstContextId;

  /**
   * The correlation ID of the memory copy. Each memory copy is
   * assigned a unique correlation ID that is identical to the
   * correlation ID in the driver and runtime API activity record that
   * launched the memory copy.
   */
  uint32_t correlationId;

#ifndef TAptiLP64
  /**
   * Undefined. Reserved for internal use.
   */
  uint32_t pad;
#endif

  /**
   * Undefined. Reserved for internal use.
   */
  void *reserved0;
} TApti_ActivityMemcpyPtoP;

/**
 * \brief The activity record for memset. (deprecated)
 *
 * This activity record represents a memory set operation
 * (TAPTI_ACTIVITY_KIND_MEMSET).
 */
 typedef struct PACKED_ALIGNMENT {
  /**
   * The activity record kind, must be TAPTI_ACTIVITY_KIND_MEMSET.
   */
  TApti_ActivityKind kind;

  /**
   * The value being assigned to memory by the memory set.
   */
  uint32_t value;

  /**
   * The number of bytes being set by the memory set.
   */
  uint64_t bytes;

  /**
   * The start timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t start;

  /**
   * The end timestamp for the memory set, in ns. A value of 0 for
   * both the start and end timestamps indicates that timestamp
   * information could not be collected for the memory set.
   */
  uint64_t end;

  /**
   * The ID of the device where the memory set is occurring.
   */
  uint32_t deviceId;

  /**
   * The ID of the context where the memory set is occurring.
   */
  uint32_t contextId;

  /**
   * The ID of the stream where the memory set is occurring.
   */
  uint32_t streamId;

  /**
   * The correlation ID of the memory set. Each memory set is assigned
   * a unique correlation ID that is identical to the correlation ID
   * in the driver API activity record that launched the memory set.
   */
  uint32_t correlationId;

  /**
   * The flags associated with the memset. \see TApti_ActivityFlag
   */
  uint16_t flags;

  /**
   * The memory kind of the memory set \see TApti_ActivityMemoryKind
   */
  uint16_t memoryKind;


} TApti_ActivityMemset;

#ifdef __cplusplus
extern "C" {
#endif  //! __cplusplus

#if defined(_MSC_VER)
#define TAPTI_DEPRECATED __declspec(deprecated)
#define TAPTI_API_EXPORT __declspec(dllexport)
#define TAPTI_API_IMPORT __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#define TAPTI_DEPRECATED __attribute__((deprecated))
#define TAPTI_API_EXPORT __attribute__((visibility("default")))
#define TAPTI_API_IMPORT __attribute__((visibility("default")))
#else
#define TAPTI_DEPRECATED
#define TAPTI_API_EXPORT
#define TAPTI_API_IMPORT
#endif  //! UNKNOWN COMPILER

#if defined(tapti_shared_EXPORTS)
#define TAPTI_API TAPTI_API_EXPORT
#else
#define TAPTI_API TAPTI_API_IMPORT
#endif  //! For user
	
/**
 * \brief Enable collection of a specific kind of activity record.
 *
 * Enable collection of a specific kind of activity record. Multiple
 * kinds can be enabled by calling this function multiple times. By
 * default all activity kinds are disabled for collection.
 *
 * \param kind The kind of activity record to collect
 *
 * \retval TAPTI_SUCCESS
 * \retval TAPTI_ERROR_NOT_INITIALIZED
 * \retval TAPTI_ERROR_NOT_COMPATIBLE if the activity kind cannot be enabled
 * \retval TAPTI_ERROR_INVALID_KIND if the activity kind is not supported
 */
TAptiResult TAPTI_API taptiActivityEnable(TApti_ActivityKind kind);

/**
 * \brief Disable collection of a specific kind of activity record.
 *
 * Disable collection of a specific kind of activity record. Multiple
 * kinds can be disabled by calling this function multiple times. By
 * default all activity kinds are disabled for collection.
 *
 * \param kind The kind of activity record to stop collecting
 *
 * \retval TAPTI_SUCCESS
 * \retval TAPTI_ERROR_NOT_INITIALIZED
 * \retval TAPTI_ERROR_INVALID_KIND if the activity kind is not supported
 */
TAptiResult TAPTI_API taptiActivityDisable(TApti_ActivityKind kind);

/**
 * \brief Iterate over the activity records in a buffer.
 *
 * This is a helper function to iterate over the activity records in a
 * buffer. A buffer of activity records is typically obtained by
 * receiving a TApti_BuffersCallbackCompleteFunc callback.
 *
 * An example of typical usage:
 * \code
 * TApti_Activity *record = NULL;
 * TAptiResult status = TAPTI_SUCCESS;
 *   do {
 *      status = taptiActivityGetNextRecord(buffer, validSize, &record);
 *      if(status == TAPTI_SUCCESS) {
 *           // Use record here...
 *      }
 *      else if (status == TAPTI_ERROR_MAX_LIMIT_REACHED)
 *          break;
 *      else {
 *          goto Error;
 *      }
 *    } while (1);
 * \endcode
 *
 * \param buffer The buffer containing activity records
 * \param record Inputs the previous record returned by
 * taptiActivityGetNextRecord and returns the next activity record
 * from the buffer. If input value is NULL, returns the first activity
 * record in the buffer. Records of kind TAPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
 * may contain invalid (0) timestamps, indicating that no timing information could
 * be collected for lack of device memory.
 * \param validBufferSizeBytes The number of valid bytes in the buffer.
 *
 * \retval TAPTI_SUCCESS
 * \retval TAPTI_ERROR_NOT_INITIALIZED
 * \retval TAPTI_ERROR_MAX_LIMIT_REACHED if no more records in the buffer
 * \retval TAPTI_ERROR_INVALID_PARAMETER if \p buffer is NULL.
 */
TAptiResult TAPTI_API taptiActivityGetNextRecord(uint8_t* buffer, size_t validBufferSizeBytes, TApti_Activity **record);

/**
 * \brief Push an external correlation id for the calling thread
 *
 * This function notifies TAPTI that the calling thread is entering an external API region.
 * When a TAPTI activity API record is created while within an external API region and
 * TAPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION is enabled, the activity API record will
 * be preceeded by a TApti_ActivityExternalCorrelation record for each \ref TApti_ExternalCorrelationKind.
 *
 * \param kind The kind of external API activities should be correlated with.
 * \param id External correlation id.
 *
 * \retval TAPTI_SUCCESS
 * \retval TAPTI_ERROR_INVALID_PARAMETER The external API kind is invalid
 */
TAptiResult TAPTI_API taptiActivityPushExternalCorrelationId(TApti_ExternalCorrelationKind kind, uint64_t id);

/**
 * \brief Pop an external correlation id for the calling thread
 *
 * This function notifies TAPTI that the calling thread is leaving an external API region.
 *
 * \param kind The kind of external API activities should be correlated with.
 * \param lastId If the function returns successful, contains the last external correlation id for this \p kind, can be NULL.
 *
 * \retval TAPTI_SUCCESS
 * \retval TAPTI_ERROR_INVALID_PARAMETER The external API kind is invalid.
 * \retval TAPTI_ERROR_QUEUE_EMPTY No external id is currently associated with \p kind.
 */
TAptiResult TAPTI_API taptiActivityPopExternalCorrelationId(TApti_ExternalCorrelationKind kind, uint64_t *lastId);

/**
 * \brief Request to deliver activity records via the buffer completion callback.
 *
 * This function returns the activity records associated with all contexts/streams
 * (and the global buffers not associated with any stream) to the TAPTI client
 * using the callback registered in taptiActivityRegisterCallbacks.
 *
 * This is a blocking call but it doesn't issue any TANG synchronization calls
 * implicitly thus it's not guaranteed that all activities are completed on the
 * underlying devices. Activity record is considered as completed if it has all
 * the information filled up including the timestamps if any. It is the client's
 * responsibility to issue necessary TANG synchronization calls before calling
 * this function if all activity records with complete information are expected
 * to be delivered.
 *
 * Behavior of the function based on the input flag:
 * - ::For default flush i.e. when flag is set as 0, it returns all the
 * activity buffers which have all the activity records completed, buffers need not
 * to be full though. It doesn't return buffers which have one or more incomplete
 * records. Default flush can be done at a regular interval in a separate thread.
 * - ::For forced flush i.e. when flag TAPTI_ACTIVITY_FLAG_FLUSH_FORCED is passed
 * to the function, it returns all the activity buffers including the ones which have
 * one or more incomplete activity records. It's suggested for clients to do the
 * force flush before the termination of the profiling session to allow remaining
 * buffers to be delivered. In general, it can be done in the at-exit handler.
 *
 * Before calling this function, the buffer handling callback api must be activated
 * by calling taptiActivityRegisterCallbacks.
 *
 * \param flag The flag can be set to indicate a forced flush. See TApti_ActivityFlag
 *
 * \retval TAPTI_SUCCESS
 * \retval TAPTI_ERROR_NOT_INITIALIZED
 * \retval TAPTI_ERROR_INVALID_OPERATION if not preceeded by a
 * successful call to taptiActivityRegisterCallbacks
 * \retval TAPTI_ERROR_UNKNOWN an internal error occurred
 *
 * \see taptiActivityFlushPeriod
 */
TAptiResult TAPTI_API taptiActivityFlushAll(void);

/**
 * \brief Function type for callback used by TAPTI to request an empty
 * buffer for storing activity records.
 *
 * This callback function signals the TAPTI client that an activity
 * buffer is needed by TAPTI. The activity buffer is used by TAPTI to
 * store activity records. The callback function can decline the
 * request by setting \p *buffer to NULL. In this case TAPTI may drop
 * activity records.
 *
 * \param buffer Returns the new buffer. If set to NULL then no buffer
 * is returned.
 * \param size Returns the size of the returned buffer.
 * \param maxNumRecords Returns the maximum number of records that
 * should be placed in the buffer. If 0 then the buffer is filled with
 * as many records as possible. If > 0 the buffer is filled with at
 * most that many records before it is returned.
 */
typedef void (*TApti_BuffersCallbackRequestFunc)(uint8_t **buffer, size_t *size, size_t *maxNumRecords);

/**
 * \brief Function type for callback used by TAPTI to return a buffer
 * of activity records.
 *
 * This callback function returns to the TAPTI client a buffer
 * containing activity records.  The buffer contains \p validSize
 * bytes of activity records which should be read using
 * taptiActivityGetNextRecord. The number of dropped records can be
 * read using taptiActivityGetNumDroppedRecords. After this call TAPTI
 * relinquished ownership of the buffer and will not use it
 * anymore. The client may return the buffer to TAPTI using the
 * TApti_BuffersCallbackRequestFunc callback.

 * \param buffer The activity record buffer.
 * \param size The total size of the buffer in bytes as set in
 * TApti_BuffersCallbackRequestFunc.
 * \param validSize The number of valid bytes in the buffer.
 */
typedef void (*TApti_BuffersCallbackCompleteFunc)(uint8_t* buffer, size_t size, size_t validSize);

/**
 * \brief Registers callback functions with TAPTI for activity buffer
 * handling.
 *
 * This function registers two callback functions to be used in asynchronous
 * buffer handling. If registered, activity record buffers are handled using
 * asynchronous requested/completed callbacks from TAPTI.
 *
 * Registering these callbacks prevents the client from using TAPTI's
 * blocking enqueue/dequeue functions.
 *
 * \param funcBufferRequested callback which is invoked when an empty
 * buffer is requested by TAPTI
 * \param funcBufferCompleted callback which is invoked when a buffer
 * containing activity records is available from TAPTI
 *
 * \retval TAPTI_SUCCESS
 * \retval TAPTI_ERROR_INVALID_PARAMETER if either \p
 * funcBufferRequested or \p funcBufferCompleted is NULL
 */
TAptiResult TAPTI_API taptiActivityRegisterCallbacks(TApti_BuffersCallbackRequestFunc funcBufferRequested,
        TApti_BuffersCallbackCompleteFunc funcBufferCompleted);

TAptiResult TAPTI_API taptiActivityPostProcess(void);

#ifdef __cplusplus
}
#endif  //! __cplusplus

#endif  // _TAPTI_ACTIVITY_HPP_

