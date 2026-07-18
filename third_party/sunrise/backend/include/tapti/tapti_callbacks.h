#ifndef __TAPTI_CALLBACKS_HPP__
#define __TAPTI_CALLBACKS_HPP__

#include <stdint.h>
#include "tapti_result.h"

/**
 * \brief Callback domains.
 *
 * Callback domains. Each domain represents callback points for a
 * group of related API functions or TANG driver activity.
 */
typedef enum {
  /**
   * Invalid domain.
   */
  TAPTI_CB_DOMAIN_INVALID           = 0,
  /**
   * Domain containing callback points for all driver API functions.
   */
  TAPTI_CB_DOMAIN_DRIVER_API        = 1,
  /**
   * Domain containing callback points for all runtime API
   * functions.
   */
  TAPTI_CB_DOMAIN_RUNTIME_API       = 2,
  TAPTI_CB_DOMAIN_SIZE,

  TAPTI_CB_DOMAIN_FORCE_INT         = 0x7fffffff
}TApti_CallbackDomain;

/**
 * \brief An ID for a driver API, runtime API, resource or
 * synchronization callback.
 *
 * An ID for a driver API, runtime API, resource or synchronization
 * callback. Within a driver API callback this should be interpreted
 * as a tapti_driver_api_trace_cbid value.
 * Within a runtime API callback this should be interpreted as a
 * TAPTI_runtime_api_trace_cbid value.
 * Within a resource API callback this should be interpreted as a
 * ref TAPTI_CallbackIdResource value.
 * Within a synchronize API callback this should be interpreted as a
 * ref TAPTI_CallbackIdSync value.
 */
typedef uint32_t TApti_CallbackId;

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
 * \brief Get the name of a callback for a specific domain and callback ID.
 *
 * Returns a pointer to the name c_string in \p **name.
 *
 * \note \b Names are available only for the DRIVER and RUNTIME domains.
 *
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 * \param name Returns pointer to the name string on success, NULL otherwise
 *
 * \retval TAPTI_SUCCESS on success
 * \retval TAPTI_ERROR_INVALID_PARAMETER if \p name is NULL, or if
 * \p domain or \p cbid is invalid.
 */
TAptiResult TAPTI_API taptiGetCallbackName(TApti_CallbackDomain domain,
                                          uint32_t cbid,
                                          const char **name);
/**
 * \brief Get the TAPTI timestamp.
 *
 * Returns a timestamp normalized to correspond with the start and end
 * timestamps reported in the TAPTI activity records. The timestamp is
 * reported in nanoseconds.
 *
 * \param timestamp Returns the TAPTI timestamp
 *
 * \retval TAPTI_SUCCESS
 * \retval TAPTI_ERROR_INVALID_PARAMETER if \p timestamp is NULL
 */
TAptiResult TAPTI_API taptiGetTimestamp(uint64_t *timestamp);

#ifdef __cplusplus
}
#endif  //! __cplusplus

#endif  // __TAPTI_CALLBACKS_HPP__

