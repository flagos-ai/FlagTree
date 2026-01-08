#ifndef _TAPTI_RESULT_HPP_
#define _TAPTI_RESULT_HPP_

/**
 * \brief TAPTI result codes.
 *
 * Error and result codes returned by TAPTI functions.
 */
typedef enum {
    /**
     * No error.
     */
    TAPTI_SUCCESS                                       = 0,
    /**
     * One or more of the parameters is invalid.
     */
    TAPTI_ERROR_INVALID_PARAMETER                       = 1,
    /**
     * The device does not correspond to a valid TANG device.
     */
    TAPTI_ERROR_INVALID_DEVICE                          = 2,
    /**
     * The context is NULL or not valid.
     */
    TAPTI_ERROR_INVALID_CONTEXT                         = 3,
    /**
     * The event domain id is invalid.
     */
    TAPTI_ERROR_INVALID_EVENT_DOMAIN_ID                 = 4,
    /**
     * The event id is invalid.
     */
    TAPTI_ERROR_INVALID_EVENT_ID                        = 5,
    /**
     * The event name is invalid.
     */
    TAPTI_ERROR_INVALID_EVENT_NAME                      = 6,
    /**
     * The current operation cannot be performed due to dependency on
     * other factors.
     */
    TAPTI_ERROR_INVALID_OPERATION                       = 7,
    /**
     * Unable to allocate enough memory to perform the requested
     * operation.
     */
    TAPTI_ERROR_OUT_OF_MEMORY                           = 8,
    /**
     * An error occurred on the performance monitoring hardware.
     */
    TAPTI_ERROR_HARDWARE                                = 9,
    /**
     * The output buffer size is not sufficient to return all
     * requested data.
     */
    TAPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT           = 10,
    /**
     * API is not implemented.
     */
    TAPTI_ERROR_API_NOT_IMPLEMENTED                     = 11,
    /**
     * The maximum limit is reached.
     */
    TAPTI_ERROR_MAX_LIMIT_REACHED                       = 12,
    /**
     * The object is not yet ready to perform the requested operation.
     */
    TAPTI_ERROR_NOT_READY                               = 13,
    /**
     * The current operation is not compatible with the current state
     * of the object
     */
    TAPTI_ERROR_NOT_COMPATIBLE                          = 14,
    /**
     * TAPTI is unable to initialize its connection to the TANG
     * driver.
     */
    TAPTI_ERROR_NOT_INITIALIZED                         = 15,
    /**
     * The metric id is invalid.
     */
    TAPTI_ERROR_INVALID_METRIC_ID                        = 16,
    /**
     * The metric name is invalid.
     */
    TAPTI_ERROR_INVALID_METRIC_NAME                      = 17,
    /**
     * The queue is empty.
     */
    TAPTI_ERROR_QUEUE_EMPTY                              = 18,
    /**
     * Invalid handle (internal?).
     */
    TAPTI_ERROR_INVALID_HANDLE                           = 19,
    /**
     * Invalid stream.
     */
    TAPTI_ERROR_INVALID_STREAM                           = 20,
    /**
     * Invalid kind.
     */
    TAPTI_ERROR_INVALID_KIND                             = 21,
    /**
     * Invalid event value.
     */
    TAPTI_ERROR_INVALID_EVENT_VALUE                      = 22,
    /**
     * TAPTI is disabled due to conflicts with other enabled profilers
     */
    TAPTI_ERROR_DISABLED                                 = 23,
    /**
     * Invalid module.
     */
    TAPTI_ERROR_INVALID_MODULE                           = 24,
    /**
     * Invalid metric value.
     */
    TAPTI_ERROR_INVALID_METRIC_VALUE                     = 25,
    /**
     * The performance monitoring hardware is in use by other client.
     */
    TAPTI_ERROR_HARDWARE_BUSY                            = 26,
    /**
     * The attempted operation is not supported on the current
     * system or device.
     */
    TAPTI_ERROR_NOT_SUPPORTED                            = 27,
    /**
     * Unified memory profiling is not supported on the system.
     * Potential reason could be unsupported OS or architecture.
     */
    TAPTI_ERROR_UM_PROFILING_NOT_SUPPORTED               = 28,
    /**
     * Unified memory profiling is not supported on the device
     */
    TAPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE     = 29,
    /**
     * Unified memory profiling is not supported on a multi-GPU
     * configuration without P2P support between any pair of devices
     */
    TAPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES = 30,
    /**
     * Profiling on virtualized GPU is not supported.
     */
    TAPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED         = 33,
    /**
     * User doesn't have sufficient privileges which are required to
     * start the profiling session.
     * One possible reason for this may be that the NVIDIA driver or your system
     * administrator may have restricted access to the NVIDIA GPU performance counters.
     * To learn how to resolve this issue and find more information, please visit
     * https://developer.nvidia.com/TAPTI_ERROR_INSUFFICIENT_PRIVILEGES
     */
    TAPTI_ERROR_INSUFFICIENT_PRIVILEGES                  = 35,
    /**
     * Legacy TAPTI Profiling API i.e. event API from the header TAPTI_events.h and
     * metric API from the header TAPTI_metrics.h are not compatible with the
     * Profiling API in the header TAPTI_profiler_target.h and Perfworks metrics API
     * in the headers nvperf_host.h and nvperf_target.h.
     */
    TAPTI_ERROR_OLD_PROFILER_API_INITIALIZED             = 36,
    /**
     * Missing definition of the OpenACC API routine in the linked OpenACC library.
     *
     * One possible reason is that OpenACC library is linked statically in the
     * user application, which might not have the definition of all the OpenACC
     * API routines needed for the OpenACC profiling, as compiler might ignore
     * definitions for the functions not used in the application. This issue
     * can be mitigated by linking the OpenACC library dynamically.
     */
    TAPTI_ERROR_OPENACC_UNDEFINED_ROUTINE                = 37,
    /**
     * Legacy TAPTI Profiling API i.e. event API from the header TAPTI_events.h and
     * metric API from the header TAPTI_metrics.h are not supported on devices with
     * compute capability 7.5 and higher (i.e. Turing and later GPU architectures).
     * These API will be deprecated in a future TANG release. These are replaced by
     * Profiling API in the header TAPTI_profiler_target.h and Perfworks metrics API
     * in the headers nvperf_host.h and nvperf_target.h.
     */
    TAPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED            = 38,
    /**
     * TAPTI doesn't allow multiple callback subscribers. Only a single subscriber
     * can be registered at a time.
     * Same error code is used when application is launched using NVIDIA tools
     * like nvprof, Visual Profiler, Nsight Systems, Nsight Compute, cuda-gdb and
     * cuda-memcheck.
     */
    TAPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED       = 39,
    /**
     * Profiling on virtualized GPU is not allowed by hypervisor.
     */
    TAPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES = 40,
    /**
     * Profiling and tracing are not allowed when confidential computing mode
     * is enabled.
     */
    TAPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED = 41,
    /**
     * An unknown internal error has occurred.
     */
    TAPTI_ERROR_UNKNOWN                                  = 999,
    TAPTI_ERROR_FORCE_INT                                = 0x7fffffff
} TAptiResult;

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
 * \brief Get the descriptive string for a TAptiResult.
 *
 * Return the descriptive string for a TAptiResult in \p *str.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param result The result to get the string for
 * \param str Returns the string
 *
 * \retval TAPTI_SUCCESS on success
 * \retval TAPTI_ERROR_INVALID_PARAMETER if \p str is NULL or \p
 * result is not a valid TAptiResult
 */
TAptiResult TAPTI_API taptiGetResultString(TAptiResult result, const char **str);

#ifdef __cplusplus
}
#endif  //! __cplusplus

#endif  // _TAPTI_RESULT_HPP_
