/*
Copyright declaration.
*/
#ifndef _TANG_RUNTIME_API_H_
#define _TANG_RUNTIME_API_H_

#include "tang_rt/driver_types.h"
#include "tang_rt/host_defines.h"
#include "tang_rt/vector_types.h"

/**
 * @brief Flags that can be used with tangStreamCreateWithFlags
 * @{
 */
#define tangStreamDefault 0x00  ///< Default stream creation flags
#define tangStreamNonBlocking \
  0x01  ///< Stream does not implicitly synchronize with null stream

//! Flags that can be used with tangEventCreateWithFlags:
#define tangEventDefault 0x0  ///< Default flags
#define tangEventBlockingSync \
  0x1  ///< Waiting will yield CPU.  Power-friendly and usage-friendly but may
       ///< increase latency.
#define tangEventDisableTiming \
  0x2  ///< Disable event's capability to record timing information.  May
       ///< improve performance.
#define tangEventInterprocess \
  0x4  ///< Event can support IPC.  @warning - not supported right now.

//! Flags that can be used with tangStreamWaitEvent:
#define tangEventWaitDefault 0x00  ///< Default stream creation flags
#define tangEventWaitExternal \
  0x01  ///< Event is captured in the graph as an external event node when
        ///< performing stream capture. @warning - not supported right now.

/**
 * @brief enum values that can be used with tangStreamCreateWithPriority and
 * tangDeviceGetStreamPriorityRange
 * @{
 */
enum stream_priority {
  priority_high   = -2,
  priority_middle = -1,
  priority_normal = 0,
  priority_low    = 1
};

#ifdef __cplusplus
extern "C" {
#endif  //! __cplusplus

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Version Management
 *  @{
 */

/**
 * @brief Returns the TANG SDK Runtime version.
 *
 * @param [out] runtimeVersion
 *
 * @returns #tangSuccess, #tangErrorInavlidValue
 *
 * @warning The TANG SDK runtime version does not correspond to an exact CUDA
 * SDK runtime revision.
 *
 * @see tangDriverGetVersion
 */
tangError_t TANGRT_API_PUBLIC tangRuntimeGetVersion(int* runtimeVersion);

/**
 * @brief Returns the TANG SDK Driver version.
 *
 * @param [out] driverVersion
 *
 * @returns #tangSuccess, #tangErrorInavlidValue
 *
 * @warning The TANG SDK driver veriosn does not correspond to an exact CUDA SDK
 * driver revision.
 *
 * @see tangRuntimeGetVersion
 */
tangError_t TANGRT_API_PUBLIC tangDriverGetVersion(int* driverVersion);

// end doxygen Error
/**
 * @}
 */

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Error Handling
 *  @{
 */

/**
 * @brief Return last error returned by any TANG runtime API call and resets the
 * stored error code to #tangSuccess
 *
 * @returns return code from last TANG called from the active host thread
 *
 * Returns the last error that has been returned by any of the runtime calls in
 * the same host thread, and then resets the saved error to #tangSuccess.
 *
 * @see tangGetErrorString, tangGetLastError, tangPeakAtLastError, tangError_t
 */
tangError_t TANGRT_API_PUBLIC tangGetLastError(void);

/**
 * @brief Return last error returned by any TANG runtime API call.
 *
 * @return #tangSuccess
 *
 * Returns the last error that has been returned by any of the runtime calls in
 * the same host thread. Unlike tangGetLastError, this function does not reset
 * the saved error code.
 *
 * @see tangGetErrorString, tangGetLastError, tangPeakAtLastError, tangError_t
 */
tangError_t TANGRT_API_PUBLIC tangPeekAtLastError(void);

/**
 * @brief Return name of the specified error code in text form.
 *
 * @param tang_error Error code to convert to name.
 * @return const char pointer to the NULL-terminated error name
 *
 * @see tangGetErrorString, tangGetLastError, tangPeakAtLastError, tangError_t
 */
TANGRT_API_PUBLIC const char* tangGetErrorName(tangError_t tang_error);

/**
 * @brief Return handy text string message to explain the error which occurred
 *
 * @param tangError Error code to convert to string.
 * @return const char pointer to the NULL-terminated error string
 *
 * @warning : on HCC, this function returns the name of the error (same as
 * tangGetErrorName)
 *
 * @see tangGetErrorName, tangGetLastError, tangPeakAtLastError, tangError_t
 */
TANGRT_API_PUBLIC const char* tangGetErrorString(tangError_t tangError);

// end doxygen Error
/**
 * @}
 */

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Device Management
 *  @{
 */

/**
 * @brief Return number of compute-capable devices.
 *
 * @param [output] count Returns number of compute-capable devices.
 *
 * @returns #tangSuccess, #tangErrorNoDevice
 *
 *
 * Returns in @p *count the number of devices that have ability to run compute
 * commands.  If there are no such devices, then @ref tangGetDeviceCount will
 * return #tangErrorNoDevice. If 1 or more devices can be found, then
 * tangGetDeviceCount returns #tangSuccess.
 */
tangError_t TANGRT_API_PUBLIC tangGetDeviceCount(int* count);

/**
 * @brief Set default device to be used for subsequent tang API calls from this
 * thread.
 *
 * @param[in] deviceId Valid device in range 0...tangGetDeviceCount().
 *
 * Sets @p device as the default device for the calling host thread.  Valid
 * device id's are 0... (tangGetDeviceCount()-1).
 *
 * Many TANG APIs implicitly use the "default device" :
 *
 * - Any device memory subsequently allocated from this host thread (using
 * tangMalloc) will be allocated on device.
 * - Any streams or events created from this host thread will be associated with
 * device.
 * - Any kernels launched from this host thread (using tangLaunchKernel) will be
 * executed on device (unless a specific stream is specified, in which case the
 * device associated with that stream will be used).
 *
 * This function may be called from any host thread.  Multiple host threads may
 * use the same device. This function does no synchronization with the previous
 * or new device, and has very little runtime overhead. Applications can use
 * tangSetDevice to quickly switch the default device before making a TANG
 * runtime call which uses the default device.
 *
 * The default device is stored in thread-local-storage for each thread.
 * Thread-pool implementations may inherit the default device of the previous
 * thread.  A good practice is to always call tangSetDevice at the start of TANG
 * coding sequency to establish a known standard device.
 *
 * @returns #tangSuccess, #tangErrorInvalidDevice, #tangErrorDeviceAlreadyInUse
 *
 * @see tangGetDevice, tangGetDeviceCount
 */
tangError_t TANGRT_API_PUBLIC tangSetDevice(int deviceId);

/**
 * @brief Return the default device id for the calling host thread.
 *
 * @param [out] device *device is written with the default device
 *
 * TANG maintains an default device for each thread using thread-local-storage.
 * This device is used implicitly for TANG runtime APIs called by this thread.
 * tangGetDevice returns in * @p device the default device for the calling host
 * thread.
 *
 * @returns #tangSuccess, #tangErrorInvalidDevice, #tangErrorInvalidValue
 *
 * @see tangSetDevice, tangGetDevicesizeBytes
 */
tangError_t TANGRT_API_PUBLIC tangGetDevice(int* deviceId);

/**
 * @brief Waits on all active streams on current device
 *
 * When this command is invoked, the host thread gets blocked until all the
 * commands associated with streams associated with the device. TANG does not
 * support multiple blocking modes (yet!).
 *
 * @returns #tangSuccess
 *
 * @see tangSetDevice, tangDeviceReset
 */
tangError_t TANGRT_API_PUBLIC tangDeviceSynchronize(void);

/**
 * @brief Returns device properties.
 *
 * @param [out] props written with device properties
 * @param [in]  deviceId which device to query for information
 *
 * @return #tangSuccess, #tangErrorInvalidDevice
 * @bug HCC always returns 0 for maxThreadsPerMultiProcessor
 * @bug HCC always returns 0 for regsPerBlock
 * @bug HCC always returns 0 for l2CacheSize
 *
 * Populates tangGetDeviceProperties with information for the specified device.
 */
TANGRT_API_PUBLIC tangError_t tangGetDeviceProperties(tangDeviceProp* props,
                                                      int             deviceId);

/**
 * @brief Query for a specific device attribute.
 *
 * @param [out] value pointer to value to return
 * @param [in] attr attribute to query
 * @param [in] deviceId which device to query for information
 *
 * @returns #tangSuccess, #tangErrorInvalidDevice, #tangErrorInvalidValue
 */
TANGRT_API_PUBLIC tangError_t tangDeviceGetAttribute(int*           value,
                                                     tangDeviceAttr attr,
                                                     int            deviceId);

/**
 * @brief tangDeviceGetPeerPointer.
 * @via port to convert addr access a peer device's memory.
 *
 * @param[in ] s2 device index
 * @param[in ] ptlink used port index.
 * @param[in ] memory address alloc in peer device;
 * @param[out ] the pointer conver peerAddr to accessAddr;
 * @return #tangSuccess, #tangErrorInvalidValue
 */
TANGRT_API_PUBLIC tangError_t tangDeviceGetPeerPointer(int    srcDevice,
                                                       int    port,
                                                       void*  peerAddr,
                                                       void** accessAddr);

/**
 * @brief tangDeviceGetP2PAttribute.
 * @Queries attributes of the link between two devices.
 *
 * @param[out ] returned value of the requested attribute
 * @param[in ] the supported attributes.
 * @param[in ] source device of the target link.
 * @param[in ] destination device of the target link.
 * @return #tangSuccess, #tangErrorInvalidValue, #tangErrorInvalidDevice
 *
 */
TANGRT_API_PUBLIC tangError_t tangDeviceGetP2PAttribute(int*              value,
                                                        tangDeviceP2PAttr attr,
                                                        int srcDevice,
                                                        int dstDevice);

/**
 * @brief tangDeviceCanAccessPeer.
 * @Queries if a device may directly access a peer device's memory.
 *
 * @param[out ] canAccessPeer return value, 1: success, 0: false.
 * @param[in ] device local device id.
 * @param[in ] peerDevice remote device id;
 * @return #tangSuccess, #tangErrorInvalidDevice
 *
 */
TANGRT_API_PUBLIC tangError_t tangDeviceCanAccessPeer(int* canAccessPeer,
                                                      int  device,
                                                      int  peerDevice);

/**
 * @brief tangDeviceEnablePeerAccess.
 * @Enables direct access to memory allocations on a peer device.
 *
 * @param[in ] peerDevice remote device id.
 * @param[in ] flags set 0;
 * @return #tangSuccess, #tangErrorInvalidValue, #tangErrorInvalidDevice,
 * #tangErrorPeerAccessAlreadyEnabled
 *
 */
TANGRT_API_PUBLIC tangError_t tangDeviceEnablePeerAccess(int peerDevice,
                                                         unsigned int flags);

/**
 * @brief tangMemcpyPeer.
 * @memory copy from a device to a peer device.
 *
 * @param[in ] dst, dst device memory point;
 * @param[in ] dstDevice, dst device id;
 * @param[in ] src, src device memory point;
 * @param[in ] srcDevice, dst device id;
 * @param[in ] size of memory copy in bytes;
 * @return #tangSuccess
 *
 */
TANGRT_API_PUBLIC tangError_t tangMemcpyPeer(void*       dst,
                                             int         dstDevice,
                                             const void* src,
                                             int         srcDevice,
                                             size_t      count);

/**
 * @brief tangMemcpyPeerAsync.
 * @memory copy from a device to a peer device.
 *
 * @param[in ] dst, dst device memory point;
 * @param[in ] dstDevice, dst device id;
 * @param[in ] src, src device memory point;
 * @param[in ] srcDevice, dst device id;
 * @param[in ] size of memory copy in bytes;
 * @param[in ] stream, used stream;
 * @return #tangSuccess
 *
 */
TANGRT_API_PUBLIC tangError_t tangMemcpyPeerAsync(void*        dst,
                                                  int          dstDevice,
                                                  const void*  src,
                                                  int          srcDevice,
                                                  size_t       count,
                                                  tangStream_t stream);

TANGRT_API_PUBLIC tangError_t tangMemcpyPeer_v2(void*       dst,
                                                int         dstDevice,
                                                const void* src,
                                                int         srcDevice,
                                                size_t      count);

TANGRT_API_PUBLIC tangError_t tangMemcpyPeer_v2_ptds(void*       dst,
                                                     int         dstDevice,
                                                     const void* src,
                                                     int         srcDevice,
                                                     size_t      count);

TANGRT_API_PUBLIC tangError_t tangMemcpyPeerAsync_v2(void*        dst,
                                                     int          dstDevice,
                                                     const void*  src,
                                                     int          srcDevice,
                                                     size_t       count,
                                                     tangStream_t stream);

TANGRT_API_PUBLIC tangError_t tangMemcpyPeerAsync_v2_ptsz(void*       dst,
                                                          int         dstDevice,
                                                          const void* src,
                                                          int         srcDevice,
                                                          size_t      count,
                                                          tangStream_t stream);

/**
 * @brief tangDeviceDisablePeerAccess.
 * @Disables direct access to memory allocations on a peer device.
 *
 * @param[in ] peerDevice remote device id;
 * @return #tangSuccess, #tangErrorInvalidDevice, #tangErrorPeerAccessNotEnabled
 *
 */
TANGRT_API_PUBLIC tangError_t tangDeviceDisablePeerAccess(int peerDevice);

/**
 * @brief Get Resource limits of current device
 *
 * @param [out] pValue
 * @param [in]  limit
 *
 * @returns #tangSuccess, #tangErrorUnsupportedLimit, #tangErrorInvalidValue
 * Note: Currently, only tangLimitMallocHeapSize is available
 *
 */
TANGRT_API_PUBLIC tangError_t tangDeviceGetLimit(size_t*        pValue,
                                                 enum tangLimit limit);

TANGRT_API_PUBLIC tangError_t tangDeviceSetLimit(enum tangLimit limit,
                                                 size_t         value);

TANGRT_API_PUBLIC tangError_t tangDeviceReset(void);

/**
 * @brief Returns a handle to a compute device.
 * @param [out] device handle
 * @param [in] PCI Bus ID
 *
 * @returns #tangSuccess, #tangErrorInavlidDevice, #tangErrorInvalidValue
 */
TANGRT_API_PUBLIC tangError_t tangDeviceGetByPCIBusId(int*        device,
                                                      const char* pciBusId);

/**
 * @brief Returns a PCI Bus Id string for the device.
 *
 * @param [out] pciBusId - PCI Bus ID
 * @param [in] len - Maximum length of pciBusId name string
 * @param [in] deviceId - device handle
 * @returns #tangSuccess, #tangErrorInavlidDevice, #tangErrorInvalidValue
 */
TANGRT_API_PUBLIC tangError_t tangDeviceGetPCIBusId(char* pciBusId,
                                                    int   len,
                                                    int   deviceId);

/**
 * @brief Set L1/Shared cache partition.
 *
 * @param [in] config
 *
 * @returns #tangSuccess, #tangErrorNotInitialized
 * Note: On PT2 devices, L1 cache and shared memory are separated.
 * Thus these hints and controls are ignored on those architectures.
 *
 */
TANGRT_API_PUBLIC tangError_t tangDeviceSetCacheConfig(tangFuncCache config);

/**
 * @brief Set Cache configuration for a specific function
 *
 * @param [in] config
 *
 * @returns #tangSuccess, #tangErrorNotInitialized
 * Note: On PT2 devices, L1 cache and shared memory are separated.
 * Thus these hints and controls are ignored on those architectures.
 *
 */
TANGRT_API_PUBLIC tangError_t tangDeviceGetCacheConfig(tangFuncCache* config);

/**
 * @brief The bank width of shared memory on current device is set
 *
 * @param [in] config
 *
 * @returns #tangSuccess, #tangErrorInvalidValue, #tangErrorNotInitialized
 *
 * Note: On PT2 devices, shard memory bank size is fix to 4-bytes.
 * Thus these hints and controls are ignored on those architectures.
 *
 */
TANGRT_API_PUBLIC tangError_t
tangDeviceSetSharedMemConfig(tangSharedMemConfig config);

/**
 * @brief Returns bank width of shared memory for current device
 *
 * @param [out] config
 *
 * @returns #tangSuccess, #tangErrorInvalidValue, #tangErrorNotInitialized
 *
 * Note: On PT2 devices, shard memory bank size is fix to 4-bytes.
 * Thus these hints and controls are ignored on those architectures.
 *
 */
TANGRT_API_PUBLIC tangError_t
tangDeviceGetSharedMemConfig(tangSharedMemConfig* config);

/**
 * @brief Returns numerical values that correspond to the least and greatest
 * stream priority.
 *
 * @param[out] leastPriority pointer in which value corresponding to least
 * priority is returned.
 * @param[out] greatestPriority pointer in which value corresponding to greatest
 * priority is returned.
 *
 * Returns in *leastPriority and *greatestPriority the numerical values that
 * correspond to the least and greatest stream priority respectively. Stream
 * priorities follow a convention where lower numbers imply greater priorities.
 * The range of meaningful stream priorities is given by * [*greatestPriority,
 * *leastPriority]. If the user attempts to create a stream with a priority
 * value that is outside the the meaningful range as specified by this API, the
 * priority is automatically clamped to within the valid range.
 */
TANGRT_API_PUBLIC tangError_t
tangDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);

/**
 * @brief Set a list of devices that can be used for TANG.
 *
 * @param[in] List of devices to try.
 * @param[in] Number of devices in specified list.
 *
 * Sets a list of devices for TANG execution in priority order using device_arr.
 * The parameter len specifies the number of elements in the list. TANG will try
 * devices from the list sequentially until it finds one that works. If this
 * function is not called, or if it is called with a len of 0, then TANG will go
 * back to its default behavior of trying devices sequentially from a default
 * list containing all of the available TANG devices in the system. If a
 * specified device ID in the list does not exist, this function will return
 * tangErrorInvalidDevice. If len is not 0 and device_arr is NULL or if len
 * exceeds the number of devices in the system, then tangErrorInvalidValue is
 * returned.
 *
 * @return #tangSuccess, #tangErrorInvalidValue, #tangErrorInvalidDevice
 *
 */
TANGRT_API_PUBLIC tangError_t tangSetValidDevices(int* device_arr, int len);

/**
 * @brief Select compute-device which best matches criteria.
 *
 * @param[out] device Device with best match
 * @param[in] properties Desired device properties
 *
 * @return #tangSuccess, #tangErrorInvalidValue
 *
 */
TANGRT_API_PUBLIC tangError_t
tangChooseDevice(int* device, const tangDeviceProp* properties);

// end doxygen Device
/**
 * @}
 */

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Memory Management
 *  @{
 *
 */

/**
 *  @brief Allocate memory on the default accelerator
 *
 *  @param[out] pptr Pointer to the allocated memory
 *  @param[in]  size Requested memory size
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and tangSuccess
 * is returned.
 *
 *  @return #tangSuccess, #tangErrorOutOfMemory, #tangErrorInvalidValue (bad
 * context, null *ptr)
 *
 *  @see tangMallocPitch, tangFree, tangMallocArray, tangFreeArray,
 * tangMalloc3D, tangMalloc3DArray, tangHostFree, tangHostMalloc
 */
tangError_t TANGRT_API_PUBLIC tangMalloc(void** pptr, size_t sizeBytes);

/**
 * @brief Allocate memory.
 *
 * @param pptr
 * @param sizeBytes
 * @param hStream
 * @return tangError_t
 */
tangError_t TANGRT_API_PUBLIC tangMallocAsync(void**       pptr,
                                              size_t       sizeBytes,
                                              tangStream_t hStream);

tangError_t TANGRT_API_PUBLIC tangMallocAsync_ptsz(void**       pptr,
                                                   size_t       sizeBytes,
                                                   tangStream_t hStream);

/**
 *  @brief Free memory allocated by the hcc tang memory allocation API.
 *  This API performs an implicit tangDeviceSynchronize() call.
 *  If pointer is NULL, the tang runtime is initialized and tangSuccess is
 * returned.
 *
 *  @param[in] ptr Pointer to memory to be freed
 *  @return #tangSuccess
 *  @return #tangErrorInvalidDevicePointer (if pointer is invalid, including
 * host pointers allocated with tangHostMalloc)
 *
 *  @see tangMalloc, tangMallocPitch, tangMallocArray, tangFreeArray,
 * tangHostFree, tangMalloc3D, tangMalloc3DArray, tangHostMalloc
 */
tangError_t TANGRT_API_PUBLIC tangFree(void* ptr);

/**
 * @brief Free memory block async.
 *
 * @param ptr
 * @param hStream
 * @return tangError_t
 */
tangError_t TANGRT_API_PUBLIC tangFreeAsync(void* ptr, tangStream_t hStream);
tangError_t TANGRT_API_PUBLIC tangFreeAsync_ptsz(void*        ptr,
                                                 tangStream_t hStream);

/**
 *  @brief Allocate page locked host memory
 *
 *  @param[out] pptr Pointer to the allocated page locked host memory
 *  @param[in]  sizeBytes Requested memory size
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and tangSuccess
 * is returned.
 *
 *  @return #tangSuccess, #tangErrorOutOfMemory
 *
 */
tangError_t TANGRT_API_PUBLIC tangMallocHost(void** pptr, size_t sizeBytes);

/**
 *  @brief Allocate page locked host memory
 *
 *  @param[out] pptr Pointer to the allocated page locked host memory
 *  @param[in]  sizeBytes Requested memory size
 *  @param[in]  flags See below.
 *
 *  flags:
 *  - #tangHostAllocDefault       Memory is page locked.
 *  - #tangHostAllocPortable      Memory is considered registered by all
 * contexts.
 *  - #tangHostAllocMapped        Map the allocation into the address space for
 * the current device.
 *  - #tangHostAllocWriteCombined Allocates the memory as write-combined (WC).
 *  TANG does not support IOMMU on device side, so flags of tangHostAllocMapped
 *  and tangHostAllocWriteCombined will always return false.
 *
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and tangSuccess
 * is returned.
 *
 *  @return #tangSuccess, #tangErrorOutOfMemory
 */
tangError_t TANGRT_API_PUBLIC tangHostAlloc(void**       pptr,
                                            size_t       sizeBytes,
                                            unsigned int flags);

/**
 *  @brief Passes back the device pointer corresponding to the mapped, pinned
 * host buffer allocated by tangHostAlloc(). Note: on PT2 devices, device
 * pointer of mapped host memory requires 4-byte aligned access (because of the
 * PCIE access mode). The start address assigned by tangHostAlloc is 4-byte
 * aligned by default, but further use of the "offset" over the device pointer
 * of mapped host memory should be careful. Access that is not 4-byte aligned
 * may result in incorrect calculations.
 *
 *  @param[out] pDevice Returned device pointer for mapped memory
 *  @param[in]  pHost Requested host pointer mapping
 *  @param[in]  flags Flags for extensions (must be 0 for now)
 *
 *  @return #tangSuccess, #tangErrorInvalidValue, #tangErrorMemoryAllocation
 */
tangError_t TANGRT_API_PUBLIC tangHostGetDevicePointer(void**       pDevice,
                                                       void*        pHost,
                                                       unsigned int flags);

/**
 *  @brief Passes back flags used to allocate pinned host memory allocated by
 * tangHostAlloc.
 *
 *  @param[out] pFlags Returned flags word
 *  @param[in]  pHost Host pointer
 *
 *  pFlags:
 *  - #tangHostAllocDefault       Memory is page locked.
 *  - #tangHostAllocPortable      Memory is considered registered by all
 * contexts.
 *  - #tangHostAllocMapped        Map the allocation into the address space for
 * the current device.
 *  - #tangHostAllocWriteCombined Allocates the memory as write-combined (WC).
 *  TANG does not support IOMMU on device side, so flags of tangHostAllocMapped
 *  and tangHostAllocWriteCombined are not supported.
 *
 *  @return #tangSuccess, #tangErrorOutOfMemory
 */
TANGRT_API_PUBLIC tangError_t tangHostGetFlags(unsigned int* pFlags,
                                               void*         pHost);

/**
 *  @brief Free the page locked host memory allocated by the tang host memory
 allocation API.
 *
 *  @param[in] ptr Pointer to memory to be freed
 *
 *  @return #tangSuccess,
 *          #tangErrorInvalidValue (if pointer is invalid, including device
 pointers allocated with tangMalloc)
 */
tangError_t TANGRT_API_PUBLIC tangFreeHost(void* ptr);

/**
 *  @brief Register host memory as page locked memory.
 *
 *  @param[out] ptr Pointer to host memory to be registered.
 *  @param[in] sizeBytes Size of the host memory
 *  @param[in] flags See below.
 *
 *  flags:
 *  - #tangHostRegisterDefault  Memory is page locked.
 *  - #tangHostRegisterPortable Memory is considered registered by all contexts.
 *  - #tangHostRegisterMapped   Map the allocation into the address space for
 * the current device.
 *  - #tangHostRegisterIoMemory The passed memory pointer is treated as pointing
 * to some memory-mapped I/O space.
 *  - #tangHostRegisterReadOnly The passed memory pointer is treated as pointing
 * to memory that is considered read-only by the device. TANG does not support
 * IOMMU on device side, so flags of tangHostRegisterMapped and
 * tangHostRegisterIoMemory and tangHostRegisterReadOnly will always return
 * false.
 *
 *  @return #tangSuccess, #tangErrorOutOfMemory
 *
 *  @see tangHostUnregister, tangHostGetFlags, tangHostGetDevicePointer
 */
tangError_t TANGRT_API_PUBLIC tangHostRegister(void*        ptr,
                                               size_t       sizeBytes,
                                               unsigned int flags);

/**
 *  @brief Un-register host pointer
 *
 *  @param[in] ptr Host pointer previously registered
 *  @return Error code
 *
 *  @see tangHostRegister
 */
tangError_t TANGRT_API_PUBLIC tangHostUnregister(void* ptr);

/**
 *  @brief Copy data from src to dst.
 *
 *  It supports memory from host to device,
 *  device to host, device to device and host to host
 *  The src and dst must not overlap.
 *
 *  For tangMemcpy, the copy is always performed by the current device (set by
tangSetDevice).
 *  For multi-gpu or peer-to-peer configurations, it is recommended to set the
current device to the
 *  device where the src data is physically located. For optimal peer-to-peer
copies, the copy
 * device must be able to access the src and dst pointers (by calling
tangDeviceEnablePeerAccess
 * with copy agent as the current device and src/dest as the peerDevice
argument.  if this is not
 * done, the tangMemcpy will still work, but will perform the copy using a
staging buffer on the
 * host. Calling tangMemcpy with dst and src pointers that do not match the
tangMemcpyKind results
 * in undefined behavior.
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]  src Data being copy from
 *  @param[in]  sizeBytes Data size in bytes
 *  @param[in]  kind Memory copy type
 *  @return #tangSuccess, #tangErrorInvalidValue,
#tangErrorInvalidMemcpyDirection , #tangErrorDriverIoctlFailed
 *
 *  @see tangArrayCreate, tangArrayDestroy, tangArrayGetDescriptor,
tangMemAlloc, tangMemAllocHost,
 * tangMemAllocPitch, tangMemcpy2D, tangMemcpy2DAsync, tangMemcpy2DUnaligned,
tangMemcpyAtoA,
 * tangMemcpyAtoD, tangMemcpyAtoH, tangMemcpyAtoHAsync, tangMemcpyDtoA,
tangMemcpyDtoD,
 * tangMemcpyDtoDAsync, tangMemcpyDtoH, tangMemcpyDtoHAsync, tangMemcpyHtoA,
tangMemcpyHtoAAsync,
 * tangMemcpyHtoDAsync, tangMemFree, tangMemFreeHost, tangMemGetAddressRange,
tangMemGetInfo,
 * tangMemHostAlloc, tangMemHostGetDevicePointer
 */
tangError_t TANGRT_API_PUBLIC tangMemcpy(void*          dst,
                                         const void*    src,
                                         size_t         sizeBytes,
                                         tangMemcpyKind kind);

tangError_t TANGRT_API_PUBLIC tangMemcpy_ptds(void*          dst,
                                              const void*    src,
                                              size_t         sizeBytes,
                                              tangMemcpyKind kind);

/**
 *  @brief Copy data from src to dst asynchronously.
 *
 *  It supports memory from host to device,
 *  device to host, device to device and host to host
 *  The src and dst must not overlap.
 *
 *  For tangMemcpyAsync, the copy is always performed by the current device (set
 * by tangSetDevice). For multi-gpu or peer-to-peer configurations, it is
 * recommended to set the current device to the device where the src data is
 * physically located. For optimal peer-to-peer copies, the copy device must be
 * able to access the src and dst pointers (by calling
 * tangDeviceEnablePeerAccess with copy agent as the current device and src/dest
 * as the peerDevice argument.  if this is not done, the tangMemcpyAsync will
 * still work, but will perform the copy using a staging buffer on the host.
 * Calling tangMemcpy with dst and src pointers that do not match the
 * tangMemcpyKind results in undefined behavior.
 *
 *  @param[out]  dst Data being copy to
 *  @param[in]  src Data being copy from
 *  @param[in]  sizeBytes Data size in bytes
 *  @param[in]  kind Memory copy type
 *  @param[in]  stream Stream to execute copy on
 *  @return #tangSuccess, #tangErrorInvalidValue,
 * #tangErrorInvalidMemcpyDirection, #tangErrorDriverIoctlFailed
 *
 *  @see tangArrayCreate, tangArrayDestroy, tangArrayGetDescriptor,
 * tangMemAlloc, tangMemAllocHost, tangMemAllocPitch, tangMemcpy2D,
 * tangMemcpy2DAsync, tangMemcpy2DUnaligned, tangMemcpyAtoA, tangMemcpyAtoD,
 * tangMemcpyAtoH, tangMemcpyAtoHAsync, tangMemcpyDtoA, tangMemcpyDtoD,
 * tangMemcpyDtoDAsync, tangMemcpyDtoH, tangMemcpyDtoHAsync, tangMemcpyHtoA,
 * tangMemcpyHtoAAsync, tangMemcpyHtoDAsync, tangMemFree, tangMemFreeHost,
 * tangMemGetAddressRange, tangMemGetInfo, tangMemHostAlloc,
 * tangMemHostGetDevicePointer
 */
tangError_t TANGRT_API_PUBLIC
tangMemcpyAsync(void*              dst,
                const void*        src,
                size_t             sizeBytes,
                tangMemcpyKind     kind,
                tangStream_t strem __dparm(nullptr));

tangError_t TANGRT_API_PUBLIC
tangMemcpyAsync_ptsz(void*              dst,
                     const void*        src,
                     size_t             sizeBytes,
                     tangMemcpyKind     kind,
                     tangStream_t strem __dparm(nullptr));

/**
 *  @brief Fills the first sizeBytes bytes of the memory area pointed to by dest
 * with the constant byte value value.
 *
 *  @param[out] dst Data being filled
 *  @param[in]  constant value to be set
 *  @param[in]  sizeBytes Data size in bytes
 *  @return #tangSuccess, #tangErrorInvalidValue, #tangErrorNotInitialized
 */
tangError_t TANGRT_API_PUBLIC tangMemset(void*  dst,
                                         int    value,
                                         size_t sizeBytes);

tangError_t TANGRT_API_PUBLIC tangMemset_ptds(void*  dst,
                                              int    value,
                                              size_t sizeBytes);

/**
 *  @brief Fills the first sizeBytes bytes of the memory area pointed to by dev
 * with the constant byte value value.
 *
 *  tangMemsetAsync() is asynchronous with respect to the host, so the call may
 * return before the memset is complete. The operation can optionally be
 * associated to a stream by passing a non-zero stream argument. If stream is
 * non-zero, the operation may overlap with operations in other streams.
 *
 *  @param[out] dst Pointer to device memory
 *  @param[in]  value - Value to set for each byte of specified memory
 *  @param[in]  sizeBytes - Size in bytes to set
 *  @param[in]  stream - Stream identifier
 *  @return #tangSuccess, #tangErrorInvalidValue, #tangErrorMemoryFree
 */
tangError_t TANGRT_API_PUBLIC
tangMemsetAsync(void*               dst,
                int                 value,
                size_t              sizeBytes,
                tangStream_t stream __dparm(nullptr));

tangError_t TANGRT_API_PUBLIC
tangMemsetAsync_ptsz(void*              dst,
                     int                value,
                     size_t             sizeBytes,
                     tangStream_t strem __dparm(nullptr));

tangError_t TANGRT_API_PUBLIC
tangMemcpyFromSymbol(void*               dst,
                     const void*         symbol,
                     size_t              count,
                     size_t offset       __dparm(0),
                     tangMemcpyKind kind __dparm(tangMemcpyDeviceToHost));

tangError_t TANGRT_API_PUBLIC
tangMemcpyFromSymbol_ptds(void*               dst,
                          const void*         symbol,
                          size_t              count,
                          size_t offset       __dparm(0),
                          tangMemcpyKind kind __dparm(tangMemcpyDeviceToHost));

tangError_t TANGRT_API_PUBLIC
tangMemcpyToSymbol(const void*         symbol,
                   const void*         src,
                   size_t              count,
                   size_t offset       __dparm(0),
                   tangMemcpyKind kind __dparm(tangMemcpyHostToDevice));

tangError_t TANGRT_API_PUBLIC
tangMemcpyToSymbol_ptds(const void*         symbol,
                        const void*         src,
                        size_t              count,
                        size_t offset       __dparm(0),
                        tangMemcpyKind kind __dparm(tangMemcpyHostToDevice));

tangError_t TANGRT_API_PUBLIC
tangMemcpyToSymbolAsync(const void*         symbol,
                        const void*         src,
                        size_t              count,
                        size_t              offset,
                        tangMemcpyKind      kind,
                        tangStream_t stream __dparm(nullptr));

tangError_t TANGRT_API_PUBLIC
tangMemcpyToSymbolAsync_ptsz(const void*         symbol,
                             const void*         src,
                             size_t              count,
                             size_t              offset,
                             tangMemcpyKind      kind,
                             tangStream_t stream __dparm(nullptr));

tangError_t TANGRT_API_PUBLIC
tangMemcpyFromSymbolAsync(void*               dst,
                          const void*         symbol,
                          size_t              count,
                          size_t              offset,
                          tangMemcpyKind      kind,
                          tangStream_t stream __dparm(nullptr));

tangError_t TANGRT_API_PUBLIC
tangMemcpyFromSymbolAsync_ptsz(void*               dst,
                               const void*         symbol,
                               size_t              count,
                               size_t              offset,
                               tangMemcpyKind      kind,
                               tangStream_t stream __dparm(nullptr));

/**
 * @brief Query memory info.
 * Return snapshot of free memory, and total allocatable memory on the device.
 *
 * Returns in *free a snapshot of the current free memory.
 * @returns #tangSuccess, #tangErrorInvalidDevice, #tangErrorInvalidValue
 * @warning On HCC, the free memory only accounts for memory allocated by this
 *process and may be optimistic.
 **/
tangError_t TANGRT_API_PUBLIC tangMemGetInfo(size_t* free, size_t* total);

/**
 * @brief Finds the address associated with a TANG symbol.
 *
 * @param[out] devPtr Device pointer associated with symbol
 * @param[in]  symbol Device symbol address
 * @return #tangSuccess, #tangErrorInvalidValue
 **/
tangError_t TANGRT_API_PUBLIC tangGetSymbolAddress(void**      devPtr,
                                                   const void* symbol);

/**
 * @brief Finds the size of the object associated with a TANG symbol.
 *
 * @param[out] size Size of object associated with symbol
 * @param[in]  symbol Device symbol address
 * @return #tangSuccess, #tangErrorInvalidValue
 **/
tangError_t TANGRT_API_PUBLIC tangGetSymbolSize(size_t*     size,
                                                const void* symbol);

// doxygen end Memory
/**
 * @}
 */

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Stream Management
 *  @{
 */

/**
 * @brief Create an asynchronous stream.
 *
 * @param[in, out] stream Valid pointer to tangStream_t.  This function writes
 * the memory with the newly created stream.
 * @return #tangSuccess, #tangErrorInvalidValue
 *
 * Create a new asynchronous stream.  @p stream returns an opaque handle that
 * can be used to reference the newly created stream in subsequent tangStream*
 * commands.  The stream is allocated on the heap and will remain allocated even
 * if the handle goes out-of-scope.  To release the memory used by the stream,
 * applicaiton must call tangStreamDestroy.
 *
 * @return #tangSuccess, #tangErrorInvalidValue
 *
 * @see tangStreamCreateWithFlags, tangStreamCreateWithPriority,
 * tangStreamSynchronize, tangStreamWaitEvent, tangStreamDestroy
 */
tangError_t TANGRT_API_PUBLIC tangStreamCreate(tangStream_t* stream);

/**
 * @brief communicate to c2c.
 *
 * @param[in, out] stream Pointer to new stream
 * @param[in ] cmd is that command packets
 * @param[in ] cmd_count is that command count
 * @param[in ] device_addr is that hbm addr
 * @return #tangSuccess, #tangErrorInvalidValue
 *
 * send command to c2c module, it can be used by ptlink.
 *
 */
tangError_t TANGRT_API_PUBLIC tangStreamC2Ctransfers(tangStream_t stream,
                                                     uint32_t*    cmd,
                                                     uint32_t     cmd_count,
                                                     uint64_t     device_addr,
                                                     uint32_t     mem_size);

/**
 * @brief Create an asynchronous stream.
 *
 * @param[in, out] stream Pointer to new stream
 * @param[in ] flags to control stream creation.
 * @return #tangSuccess, #tangErrorInvalidValue
 *
 * Create a new asynchronous stream.  @p stream returns an opaque handle that
 * can be used to reference the newly created stream in subsequent tangStream*
 * commands.  The stream is allocated on the heap and will remain allocated even
 * if the handle goes out-of-scope.  To release the memory used by the stream,
 * applicaiton must call tangStreamDestroy. Flags controls behavior of the
 * stream.  See #tangStreamDefault, #tangStreamNonBlocking.
 *
 * @see tangStreamCreate, tangStreamCreateWithPriority, tangStreamSynchronize,
 * tangStreamWaitEvent, tangStreamDestroy
 */
tangError_t TANGRT_API_PUBLIC tangStreamCreateWithFlags(tangStream_t* stream,
                                                        unsigned int  flags __dparm(tangStreamDefault));

/**
 * @brief Create an asynchronous stream with the specified priority.
 *
 * @param[in, out] stream Pointer to new stream
 * @param[in ] flags to control stream creation.
 * @param[in ] priority of the stream. Lower numbers represent higher
 * priorities.
 * @return #tangSuccess, #tangErrorInvalidValue
 *
 * Create a new asynchronous stream with the specified priority.  @p stream
 * returns an opaque handle that can be used to reference the newly created
 * stream in subsequent tangStream* commands.  The stream is allocated on the
 * heap and will remain allocated even if the handle goes out-of-scope. To
 * release the memory used by the stream, applicaiton must call
 * tangStreamDestroy. Flags controls behavior of the stream.  See
 * #tangStreamDefault, #tangStreamNonBlocking.
 *
 * @see tangStreamCreate, tangStreamSynchronize, tangStreamWaitEvent,
 * tangStreamDestroy
 */
tangError_t TANGRT_API_PUBLIC tangStreamCreateWithPriority(tangStream_t* stream,
                                                           unsigned int  flags __dparm(tangStreamDefault),
                                                           int priority __dparm(priority_normal));

/**
 * @brief Returns numerical values that correspond to the least and greatest
 * stream priority.
 *
 * @param[in, out] leastPriority pointer in which value corresponding to least
 * priority is returned.
 * @param[in, out] greatestPriority pointer in which value corresponding to
 * greatest priority is returned.
 *
 * Returns in *leastPriority and *greatestPriority the numerical values that
 * correspond to the least and greatest stream priority respectively. Stream
 * priorities follow a convention where lower numbers imply greater priorities.
 * The range of meaningful stream priorities is given by
 * [*greatestPriority, *leastPriority]. If the user attempts to create a stream
 * with a priority value that is outside the the meaningful range as specified
 * by this API, the priority is automatically clamped to within the valid range.
 */
tangError_t tangDeviceGetStreamPriorityRange(int* leastPriority,
                                             int* greatestPriority);

/**
 * @brief Query the priority of a stream.
 *
 * @param[in] hStream stream to be queried
 * @param[in,out] priority Pointer to an unsigned integer in which the stream's
 * priority is returned
 * @return #tangSuccess, #tangErrorInvalidValue
 *
 * Query the priority of a stream. The priority is returned in in priority.
 *
 * @see tangStreamCreateWithFlags
 */
tangError_t TANGRT_API_PUBLIC tangStreamGetPriority(tangStream_t hStream,
                                                    int*         priority);

tangError_t TANGRT_API_PUBLIC tangStreamGetPriority_ptsz(tangStream_t stream,
                                                         int*         priority);

/**
 * @brief Destroys the specified stream.
 *
 * @param[in, out] stream Valid pointer to tangStream_t.  This function writes
 * the memory with the newly created stream.
 * @return #tangSuccess #tangErrorInvalidHandle
 *
 * Destroys the specified stream.
 *
 * If commands are still executing on the specified stream, some may complete
 * execution before the queue is deleted.
 *
 * The queue may be destroyed while some commands are still inflight, or may
 * wait for all commands queued to the stream before destroying it.
 *
 * @see tangStreamCreate, tangStreamCreateWithFlags,
 * tangStreamCreateWithPriority, tangStreamQuery, tangStreamWaitEvent,
 * tangStreamSynchronize
 */
tangError_t TANGRT_API_PUBLIC tangStreamDestroy(tangStream_t stream);

/**
 * @brief Wait for all commands in stream to complete.
 *
 * @param[in] stream stream identifier.
 *
 * @return #tangSuccess, #tangErrorInvalidHandle
 *
 * This command is host-synchronous : the host will block until the specified
 * stream is empty.
 *
 * This command follows standard null-stream semantics.  Specifically,
 * specifying the null stream will cause the command to wait for other streams
 * on the same device to complete all pending operations.
 *
 * This command honors the tangDeviceLaunchBlocking flag, which controls whether
 * the wait is active or blocking.
 *
 * @see tangStreamCreate, tangStreamCreateWithFlags,
 * tangStreamCreateWithPriority, tangStreamWaitEvent, tangStreamDestroy
 *
 */
tangError_t TANGRT_API_PUBLIC tangStreamSynchronize(tangStream_t stream);

tangError_t TANGRT_API_PUBLIC tangStreamSynchronize_ptsz(tangStream_t stream);

/**
 * @brief Check if a stream has completed all its commands
 *
 * @param stream
 * @return tangError_t
 * tangSuccess
 * tangErrorNotReady
 */
tangError_t TANGRT_API_PUBLIC tangStreamQuery(tangStream_t stream);
tangError_t TANGRT_API_PUBLIC tangStreamQuery_ptsz(tangStream_t stream);

tangError_t TANGRT_API_PUBLIC
tangStreamBeginCapture(tangStream_t stream, tangStreamCaptureMode mode);

tangError_t TANGRT_API_PUBLIC
tangStreamBeginCapture_ptsz(tangStream_t               stream,
                            tangStreamCaptureMode mode);

tangError_t TANGRT_API_PUBLIC tangStreamEndCapture(tangStream_t stream,
                                                   tangGraph_t* pGraph);

tangError_t TANGRT_API_PUBLIC tangStreamEndCapture_ptsz(tangStream_t stream,
                                                        tangGraph_t* pGraph);

tangError_t TANGRT_API_PUBLIC
tangStreamIsCapturing(tangStream_t                  stream,
                      tangStreamCaptureStatus* pStatus);

tangError_t TANGRT_API_PUBLIC
tangStreamIsCapturing_ptsz(tangStream_t                  stream,
                           tangStreamCaptureStatus* pStatus);

tangError_t TANGRT_API_PUBLIC
tangStreamGetCaptureInfo(tangStream_t                  hStream,
                         tangStreamCaptureStatus* pStatus,
                         unsigned long long* pId       __dparm(0),
                         tangGraph_t* pGraph           __dparm(0),
                         const tangGraphNode_t** deps  __dparm(0),
                         size_t* numDeps               __dparm(0));

tangError_t TANGRT_API_PUBLIC
tangStreamGetCaptureInfo_ptsz(tangStream_t                  hStream,
                              tangStreamCaptureStatus* pStatus,
                              unsigned long long* pId       __dparm(0),
                              tangGraph_t* pGraph           __dparm(0),
                              const tangGraphNode_t** deps  __dparm(0),
                              size_t* numDeps               __dparm(0));

tangError_t TANGRT_API_PUBLIC
tangThreadExchangeStreamCaptureMode(tangStreamCaptureMode* mode);

tangError_t TANGRT_API_PUBLIC tangGraphInstantiate(tangGraphExec_t* pGraphExec,
                                                   tangGraph_t      graph,
                                                   void*,
                                                   void*,
                                                   unsigned long long);

tangError_t TANGRT_API_PUBLIC tangGraphLaunch(tangGraphExec_t graphExec,
                                              tangStream_t    stream);

tangError_t TANGRT_API_PUBLIC tangGraphLaunch_ptsz(tangGraphExec_t graphExec,
                                                   tangStream_t    stream);
tangError_t TANGRT_API_PUBLIC
tangGraphInstantiateWithFlags(tangGraphExec_t*   pGraphExec,
                              tangGraph_t        graph,
                              unsigned long long flags);

tangError_t TANGRT_API_PUBLIC tangGraphDestroy(tangGraph_t graph);
tangError_t TANGRT_API_PUBLIC tangGraphExecDestroy(tangGraphExec_t graphExec);

tangError_t TANGRT_API_PUBLIC tangGraphGetInfo(tangGraph_t    graph,
                                               tangGraphInfo* pInfo);

tangError_t TANGRT_API_PUBLIC tangGraphCreate(tangGraph_t* pGraph,
                                              unsigned int flags);

tangError_t TANGRT_API_PUBLIC
tangGraphAddHostNode(tangGraphNode_t*          pGraphNode,
                     tangGraph_t               graph,
                     const tangGraphNode_t*    dependencies,
                     size_t                    numDependencies,
                     const tangHostNodeParams* nodeParams);

tangError_t TANGRT_API_PUBLIC
tangGraphAddKernelNode(tangGraphNode_t*            pGraphNode,
                       tangGraph_t                 graph,
                       const tangGraphNode_t*      dependencies,
                       size_t                      numDependencies,
                       const tangKernelNodeParams* nodeParams);

/**
 * @brief Make the specified compute stream wait for an event
 *
 * @param[in] stream stream to make wait.
 * @param[in] event event to wait on
 * @param[in] flag control operation
 *
 * @return #tangSuccess, #tangErrorInvalidHandle
 *
 * This function inserts a wait operation into the specified stream.
 * All future work submitted to @p stream will wait until @p event reports
 * completion before beginning execution.
 *
 * This function only waits for commands in the current stream to complete.
 * Notably,, this function does not impliciy wait for commands in the default
 * stream to complete, even if the specified stream is created with
 * tangStreamNonBlocking = 0.
 *
 * @see tangStreamCreate, tangStreamCreateWithFlags,
 * tangStreamCreateWithPriority, tangStreamSynchronize, tangStreamDestroy
 */
tangError_t TANGRT_API_PUBLIC tangStreamWaitEvent(tangStream_t      stream,
                                                  tangEvent_t       event,
                                                  unsigned int flag __dparm(0));

tangError_t TANGRT_API_PUBLIC
tangStreamWaitEvent_ptsz(tangStream_t      stream,
                         tangEvent_t       event,
                         unsigned int flag __dparm(0));
/**
 * @brief Return flags associated with this stream.
 *
 * @param[in] stream stream to be queried
 * @param[in,out] flag Pointer to an unsigned integer in which the stream's
 * flags are returned
 * @return #tangSuccess, #tangErrorInvalidValue, #tangErrorInvalidHandle
 *
 * @returns #tangSuccess #tangErrorInvalidValue #tangErrorInvalidHandle
 *
 * Return flags associated with this stream in *@p flag.
 *
 * @see tangStreamCreateWithFlags
 */
tangError_t TANGRT_API_PUBLIC tangStreamGetFlags(tangStream_t  stream,
                                                 unsigned int* flags);

tangError_t TANGRT_API_PUBLIC tangStreamGetFlags_ptsz(tangStream_t  stream,
                                                      unsigned int* flags);

tangError_t TANGRT_API_PUBLIC tangStreamGetId(tangStream_t stream,
                                              int*         pId);

tangError_t TANGRT_API_PUBLIC tangStreamGetId_ptsz(tangStream_t stream,
                                                   int*         pId);

typedef void (*tangStreamCallback_t)(tangStream_t stream,
                                     tangError_t  status,
                                     void*        userData);

tangError_t TANGRT_API_PUBLIC
tangStreamAddCallback(tangStream_t         stream,
                      tangStreamCallback_t callback,
                      void*                userData,
                      unsigned int         flags);

tangError_t TANGRT_API_PUBLIC
tangStreamAddCallback_ptsz(tangStream_t         stream,
                           tangStreamCallback_t callback,
                           void*                userData,
                           unsigned int         flags);

tangError_t TANGRT_API_PUBLIC tangLaunchHostFunc(tangStream_t stream,
                                                 tangHostFn_t fn,
                                                 void*        userData);

tangError_t TANGRT_API_PUBLIC tangLaunchHostFunc_ptsz(tangStream_t stream,
                                                      tangHostFn_t fn,
                                                      void*        userData);

tangError_t TANGRT_API_PUBLIC tangProfilerStart();
tangError_t TANGRT_API_PUBLIC tangProfilerStop();
// end doxygen Stream
/**
 * @}
 */

tangError_t TANGRT_API_PUBLIC tangEngineCollAssign(int devId,
                                                   int collType,
                                                   uint64_t devAddr,
                                                   int length,
                                                   tangStream_t stream);


/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Event Management
 *  @{
 */

/**
 * @brief Create an event with the specified flags
 *
 * @param[in,out] event Returns the newly created event.
 * @param[in] flag     Flag to control event behavior.  Valid values are
 #tangEventDefault, #tangEventBlockingSync, #tangEventDisableTiming,
 #tangEventInterprocess

 * #tangEventDefault : Default flag.  The event will use active synchronization
 and will support timing.  Blocking synchronization provides lowest possible
 latency at the expense of dedicating a CPU to poll on the event.
 * #tangEventBlockingSync : The event will use blocking synchronization : if
 tangEventSynchronize is called on this event, the thread will block until the
 event completes.  This can increase latency for the synchroniation but can
 result in lower power and more resources for other CPU threads.
 * #tangEventDisableTiming : Disable recording of timing information.  On ROCM
 platform, timing information is always recorded and this flag has no
 performance benefit.

 * @warning tangEventInterprocess support is under development.  Use of this
 flag will return an error.
 *
 * @returns #tangSuccess, #tangErrorNotInitialized, #tangErrorInvalidValue,
 #tangErrorLaunchFailure, #tangErrorOutOfMemory
 *
 * @see tangEventCreate, tangEventSynchronize, tangEventDestroy,
 tangEventElapsedTime
 */
tangError_t TANGRT_API_PUBLIC tangEventCreateWithFlags(tangEvent_t* event,
                                                       unsigned     flag);

/**
 *  Create an event
 *
 * @param[in,out] event Returns the newly created event.
 *
 * @returns #tangSuccess, #tangErrorNotInitialized, #tangErrorInvalidValue,
 * #tangErrorLaunchFailure, #tangErrorOutOfMemory
 *
 * @see tangEventCreateWithFlags, tangEventRecord, tangEventQuery,
 * tangEventSynchronize, tangEventDestroy, tangEventElapsedTime
 */
tangError_t TANGRT_API_PUBLIC tangEventCreate(tangEvent_t* event);

/**
 * @brief Record an event in the specified stream.
 *
 * @param[in] event event to record.
 * @param[in] stream stream in which to record event.
 * @returns #tangSuccess, #tangErrorInvalidValue, #tangErrorNotInitialized,
 * #tangErrorInvalidHandle, #tangErrorLaunchFailure
 *
 * tangEventQuery() or tangEventSynchronize() must be used to determine when the
 * event transitions from "recording" (after tangEventRecord() is called) to
 * "recorded" (when timestamps are set, if requested).
 *
 * Events which are recorded in a non-NULL stream will transition to
 * from recording to "recorded" state when they reach the head of
 * the specified stream, after all previous
 * commands in that stream have completed executing.
 *
 * If tangEventRecord() has been previously called on this event, then this call
 * will overwrite any existing state in event.
 *
 * If this function is called on an event that is currently being recorded,
 * results are undefined
 * - either outstanding recording may save state into the event, and the order
 * is not guaranteed.
 *
 * @see tangEventCreate, tangEventCreateWithFlags, tangEventQuery,
 * tangEventSynchronize, tangEventDestroy, tangEventElapsedTime
 *
 */
tangError_t TANGRT_API_PUBLIC
tangEventRecord(tangEvent_t event, tangStream_t stream __dparm(nullptr));

tangError_t TANGRT_API_PUBLIC
tangEventRecord_ptsz(tangEvent_t event, tangStream_t stream __dparm(nullptr));

tangError_t TANGRT_API_PUBLIC
tangEventRecordWithFlags(tangEvent_t         event,
                         tangStream_t stream __dparm(nullptr),
                         unsigned int flags  __dparm(0));

tangError_t TANGRT_API_PUBLIC
tangEventRecordWithFlags_ptsz(tangEvent_t         event,
                              tangStream_t stream __dparm(nullptr),
                              unsigned int flags  __dparm(0));
/**
 *  @brief Destroy the specified event.
 *
 *  @param[in] event Event to destroy.
 *  @returns #tangSuccess, #tangErrorNotInitialized, #tangErrorInvalidValue,
 * #tangErrorLaunchFailure
 *
 *  Releases memory associated with the event.  If the event is recording but
 * has not completed recording when tangEventDestroy() is called, the function
 * will return immediately and the completion_future resources will be released
 * later, when the tangDevice is synchronized.
 *
 * @see tangEventCreate, tangEventCreateWithFlags, tangEventQuery,
 * tangEventSynchronize, tangEventRecord, tangEventElapsedTime
 *
 * @returns #tangSuccess
 */
tangError_t TANGRT_API_PUBLIC tangEventDestroy(tangEvent_t event);

/**
 *  @brief Wait for an event to complete.
 *
 *  This function will block until the event is ready, waiting for all previous
 * work in the stream specified when event was recorded with tangEventRecord().
 *
 *  If tangEventRecord() has not been called on @p event, this function returns
 * immediately.
 *
 *  TODO-hcc - This function needs to support tangEventBlockingSync parameter.
 *
 *  @param[in] event Event on which to wait.
 *  @returns #tangSuccess, #tangErrorInvalidValue, #tangErrorNotInitialized,
 * #tangErrorInvalidHandle, #tangErrorLaunchFailure
 *
 *  @see tangEventCreate, tangEventCreateWithFlags, tangEventQuery,
 * tangEventDestroy, tangEventRecord, tangEventElapsedTime
 */
tangError_t TANGRT_API_PUBLIC tangEventSynchronize(tangEvent_t event);
tangError_t TANGRT_API_PUBLIC tangEventSynchronizeWithFlags(tangEvent_t  event,
                                                            unsigned int flags);

/**
 * @brief Return the elapsed time between two events.
 *
 * @param[out] ms : Return time between start and stop in ms.
 * @param[in]   start : Start event.
 * @param[in]   stop  : Stop event.
 * @returns #tangSuccess, #tangErrorInvalidValue, #tangErrorNotReady,
 * #tangErrorInvalidHandle, #tangErrorNotInitialized, #tangErrorLaunchFailure
 *
 * Computes the elapsed time between two events. Time is computed in ms, with
 * a resolution of approximately 1 us.
 *
 * Events which are recorded in a NULL stream will block until all commands
 * on all other streams complete execution, and then record the timestamp.
 *
 * Events which are recorded in a non-NULL stream will record their timestamp
 * when they reach the head of the specified stream, after all previous
 * commands in that stream have completed executing.  Thus the time that
 * the event recorded may be significantly after the host calls
 * tangEventRecord().
 *
 * If tangEventRecord() has not been called on either event, then
 * #tangErrorInvalidHandle is returned. If tangEventRecord() has been called on
 * both events, but the timestamp has not yet been recorded on one or both
 * events (that is, tangEventQuery() would return #tangErrorNotReady on at least
 * one of the events), then #tangErrorNotReady is returned.
 *
 * @see tangEventCreate, tangEventCreateWithFlags, tangEventQuery,
 * tangEventDestroy, tangEventRecord, tangEventSynchronize
 */
tangError_t TANGRT_API_PUBLIC tangEventElapsedTime(float*      ms,
                                                   tangEvent_t start,
                                                   tangEvent_t stop);

/**
 * @brief Query event status
 *
 * @param[in] event Event to query.
 * @returns #tangSuccess, #tangErrorNotReady, #tangErrorInvalidHandle,
 * #tangErrorInvalidValue, #tangErrorNotInitialized, #tangErrorLaunchFailure
 *
 * Query the status of the specified event.  This function will return
 * #tangErrorNotReady if all commands in the appropriate stream (specified to
 * tangEventRecord()) have completed.  If that work has not completed, or if
 * tangEventRecord() was not called on the event, then #tangSuccess is returned.
 *
 * @see tangEventCreate, tangEventCreateWithFlags, tangEventRecord,
 * tangEventDestroy, tangEventSynchronize, tangEventElapsedTime
 */
tangError_t TANGRT_API_PUBLIC tangEventQuery(tangEvent_t event);

tangError_t TANGRT_API_PUBLIC tangEventQueryTimestamp(tangEvent_t         event,
                                                      tangEventTimestamp* ts);
// end doxygen Event
/**
 * @}
 */

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Excution Control
 *  @{
 */

/**
 * @brief Find out attributes for a given function.
 *
 * @param [out] attr
 * @param [in] func
 *
 * @returns tangSuccess, tangErrorInvalidValue, tangErrorInvalidDeviceFunction
 *
 * NOTE: runtime only has tangFuncGetAttributes API, has no tangFuncGetAttribute
 * API. while user mode driver only has taFuncGetAttribute API, has no
 * taFuncGetAttributes API.
 */

tangError_t TANGRT_API_PUBLIC tangFuncGetAttributes(tangFuncAttributes* attr,
                                                    const void*         func);

tangError_t TANGRT_API_PUBLIC tangGetFuncBySymbol(tangFunction_t *hFunc,
                                                  const void *symbol);

tangError_t TANGRT_API_PUBLIC
tangPointerGetAttributes(struct tangPointerAttributes* attributes,
                         const void*                   ptr);

/**
 * @brief Set attribute for a specific function
 *
 * @param [in] func;
 * @param [in] attr;
 * @param [in] value;
 *
 * @returns #tangSuccess, #tangErrorInvalidDeviceFunction,
 * #tangErrorInvalidValue
 *
 * Note: PT devices do not support shared cache banking, and the hint is
 * ignored.
 *
 * NOTE: runtime tangFuncSetAttribute API only supports two types of
 * tangFuncAttribute. while user mode driver taFuncSetAttribute API supports
 * more types of TAfunction_attribute.
 *
 */
tangError_t TANGRT_API_PUBLIC tangFuncSetAttribute(const void*       func,
                                                   tangFuncAttribute attr,
                                                   int               value);

/**
 * @brief Set Cache configuration for a specific function
 *
 * @param [in] func;
 * @param [in] config;
 *
 * @returns #tangSuccess, #tangErrorNotInitialized
 * Note: PT devices do not support reconfigurable cache. This hint is ignored.
 *
 */
tangError_t TANGRT_API_PUBLIC tangFuncSetCacheConfig(const void*   func,
                                                     tangFuncCache config);

/**
 * @brief Set shared memory configuation for a specific function
 *
 * @param [in] func
 * @param [in] config
 *
 * @returns #tangSuccess, #tangErrorInvalidValue,
 * #tangErrorInvalidDeviceFunction
 *
 * Note: PT devices do not support shared cache banking, and the hint is
 * ignored.
 *
 */
tangError_t TANGRT_API_PUBLIC
tangFuncSetSharedMemConfig(const void* func, tangSharedMemConfig config);

/**
 * @brief Converts a double argument to be executed on a device.
 *
 * @param[in][out] d Double to convert.
 * @returns #tangSuccess, #tangErrorInvalidValue
 *
 * Note: PT2 devices do not support double both on hardware and software
 * simulation.
 *
 */
TANGRT_API_PUBLIC tangError_t tangSetDoubleForDevice(double* d);

/**
 * @brief Converts a double argument after execution on a device.
 *
 * @param[in][out] d Double to convert.
 * @returns #tangSuccess, #tangErrorInvalidValue
 *
 * Note: PT2 devices do not support double both on hardware and software
 * simulation.
 *
 */
TANGRT_API_PUBLIC tangError_t tangSetDoubleForHost(double* d);

// doxygen end Execution Control
/**
 * @}
 */

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Occupancy
 *  @{
 *
 */

/**
 * @brief Returns occupancy for a device function.
 *
 * @param[out] numBlocks Returned occupancy
 * @param[in] func Kernel function for which occupancy is calculated
 * @param[in] blockSize Block size the kernel is intended to be launched with
 * @param[in] dynamicSMemSize Per-block dynamic shared memory usage intended, in
 * bytes
 * @returns #tangSuccess, #tangErrorInvalidValue
 *
 */
TANGRT_API_PUBLIC tangError_t
tangOccupancyMaxActiveBlocksPerMultiprocessor(int*        numBlocks,
                                              const void* func,
                                              int         blockSize,
                                              size_t      dynamicSMemSize);

/**
 * @brief Returns occupancy for a device function with the specified flags.
 *
 * @param[out] numBlocks Returned occupancy
 * @param[in] func Kernel function for which occupancy is calculated
 * @param[in] blockSize Block size the kernel is intended to be launched with
 * @param[in] dynamicSMemSize Per-block dynamic shared memory usage intended,
 * in bytes
 * @param[in] flags Requested behavior for the occupancy calculator
 * @returns #tangSuccess, #tangErrorInvalidValue
 *
 */
TANGRT_API_PUBLIC tangError_t
tangOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int*        numBlocks,
                                                       const void* func,
                                                       int         blockSize,
                                                       size_t dynamicSMemSize,
                                                       unsigned int flags);

TANGRT_API_PUBLIC tangError_t tangIpcGetMemHandle(tangIpcMemHandle_t* pHandle,
                                                  void*               devPtr);

TANGRT_API_PUBLIC tangError_t tangIpcOpenMemHandle(void**             devPtr,
                                                   tangIpcMemHandle_t handle,
                                                   unsigned int       flags);

TANGRT_API_PUBLIC tangError_t tangIpcCloseMemHandle(void* devPtr);

TANGRT_API_PUBLIC tangError_t
tangIpcGetEventHandle(tangIpcEventHandle_t* pHandle, tangEvent_t event);

TANGRT_API_PUBLIC tangError_t
tangIpcOpenEventHandle(tangEvent_t* phEvent, tangIpcEventHandle_t handle);

// private api
TANGRT_API_PUBLIC tangError_t tangGetExportTable(void** pExportedTable,
                                                 void*  args);

// end doxygen Occupancy
/**
 * @}
 */

#ifdef __cplusplus
}
#endif  //! __cplusplus

#ifdef __cplusplus
template <typename T>
inline tangError_t tangMalloc(T** pptr, size_t sizeBytes) {
  return ::tangMalloc((void**)pptr, sizeBytes);
}
template <typename T>
inline tangError_t tangMallocHost(T** pptr, size_t sizeBytes) {
  return ::tangMallocHost((void**)pptr, sizeBytes);
}
template <typename T>
inline tangError_t tangMallocAsync(T** pptr, size_t sizeBytes, tangStream_t stream) {
  return ::tangMallocAsync((void**)pptr, sizeBytes, stream);
}
template <typename T>
inline tangError_t tangMallocAsync_ptsz(T** pptr, size_t sizeBytes, tangStream_t stream) {
  return ::tangMallocAsync_ptsz((void**)pptr, sizeBytes, stream);
}
template <typename T>
inline tangError_t tangHostAlloc(T**          pptr,
                                 size_t       sizeBytes,
                                 unsigned int flags) {
  return ::tangHostAlloc((void**)pptr, sizeBytes, flags);
}
template <typename T>
inline tangError_t tangHostGetDevicePointer(T**          pDevice,
                                            void*        pHost,
                                            unsigned int flags) {
  return ::tangHostGetDevicePointer((void**)pDevice, pHost, flags);
}
template <typename T>
inline tangError_t tangIpcOpenMemHandle(T**                pDevPtr,
                                        tangIpcMemHandle_t handle,
                                        unsigned int       flags) {
  return ::tangIpcOpenMemHandle((void**)pDevPtr, handle, flags);
}
#endif  // __cplusplus

#if defined(__TANGRT_API_PER_THREAD_DEFAULT_STREAM)
#define tangMemset                __TANGRT_API_PTDS(tangMemset)
#define tangMemsetAsync           __TANGRT_API_PTSZ(tangMemsetAsync)
#define tangMemcpy                __TANGRT_API_PTDS(tangMemcpy)
#define tangMemcpyAsync           __TANGRT_API_PTSZ(tangMemcpyAsync)
#define tangMallocAsync           __TANGRT_API_PTSZ(tangMallocAsync)
#define tangFreeAsync             __TANGRT_API_PTSZ(tangFreeAsync)
#define tangStreamSynchronize     __TANGRT_API_PTSZ(tangStreamSynchronize)
#define tangStreamQuery           __TANGRT_API_PTSZ(tangStreamQuery)
#define tangStreamWaitEvent       __TANGRT_API_PTSZ(tangStreamWaitEvent)
#define tangStreamC2Ctransfers    __TANGRT_API_PTSZ(tangStreamC2Ctransfers)
#define tangStreamGetFlags        __TANGRT_API_PTSZ(tangStreamGetFlags)
#define tangStreamGetId           __TANGRT_API_PTSZ(tangStreamGetId)
#define tangStreamGetPriority     __TANGRT_API_PTSZ(tangStreamGetPriority)
#define tangStreamAddCallback     __TANGRT_API_PTSZ(tangStreamAddCallback)
#define tangStreamBeginCapture    __TANGRT_API_PTSZ(tangStreamBeginCapture)
#define tangStreamEndCapture      __TANGRT_API_PTSZ(tangStreamEndCapture)
#define tangStreamIsCapturing     __TANGRT_API_PTSZ(tangStreamIsCapturing)
#define tangStreamGetCaptureInfo  __TANGRT_API_PTSZ(tangStreamGetCaptureInfo)
#define tangGraphLaunch           __TANGRT_API_PTSZ(tangGraphLaunch)
#define tangLaunchHostFunc        __TANGRT_API_PTSZ(tangLaunchHostFunc)
#define tangEventRecord           __TANGRT_API_PTSZ(tangEventRecord)
#define tangEventRecordWithFlags  __TANGRT_API_PTSZ(tangEventRecordWithFlags)
#define tangMemcpyFromSymbol      __TANGRT_API_PTDS(tangMemcpyFromSymbol)
#define tangMemcpyFromSymbolAsync __TANGRT_API_PTSZ(tangMemcpyFromSymbolAsync)
#define tangMemcpyToSymbol        __TANGRT_API_PTDS(tangMemcpyToSymbol)
#define tangMemcpyToSymbolAsync   __TANGRT_API_PTSZ(tangMemcpyToSymbolAsync)
//#define tangDeviceCanAccessPeer   __TANGRT_API_PTSZ(tangDeviceCanAccessPeer)
//#define tangDeviceEnablePeerAccess
//__TANGRT_API_PTSZ(tangDeviceEnablePeerAccess) #define
//tangDeviceDisablePeerAccess __TANGRT_API_PTSZ(tangDeviceDisablePeerAccess)
#endif  //! __TANGRT_API_PER_THREAD_DEFAULT_STREAM
// end doxygen Events
/**
 * @}
 */

#ifdef __cplusplus

// template is only available in cxx.
// And these template APIs may cause ambiguous.
// If you don't want to use these api, please define TANGRT_DISABLE_SYMBOL_TEMPLATE_API
// before #include <tang_runtime.h> or #include <tang_runtime_api.h>
#if defined(__cplusplus) && !defined(TANGRT_DISABLE_SYMBOL_TEMPLATE_API)

template <typename T>
inline tangError_t tangGetSymbolAddress(void** devPtr, const T& symbol) {
  return ::tangGetSymbolAddress((void**)devPtr, (const void*)&symbol);
}

template <typename T>
inline tangError_t tangGetSymbolSize(size_t* size, const T& symbol) {
  return ::tangGetSymbolSize(size, (const void*)&symbol);
}

template <typename T>
inline tangError_t tangMemcpyToSymbol(
  const T&            symbol,
  const void*         src,
  size_t              count,
  size_t              offset = 0,
  enum tangMemcpyKind kind   = tangMemcpyHostToDevice) {
  return ::tangMemcpyToSymbol((const void*)&symbol, src, count, offset, kind);
}

template <typename T>
inline tangError_t tangMemcpyToSymbolAsync(
  const T&            symbol,
  const void*         src,
  size_t              count,
  size_t              offset = 0,
  enum tangMemcpyKind kind   = tangMemcpyHostToDevice,
  tangStream_t        stream = 0) {
  return ::tangMemcpyToSymbolAsync((const void*)&symbol,
                                   src,
                                   count,
                                   offset,
                                   kind,
                                   stream);
}

template <typename T>
inline tangError_t tangMemcpyFromSymbol(
  void*               dst,
  const T&            symbol,
  size_t              count,
  size_t              offset = 0,
  enum tangMemcpyKind kind   = tangMemcpyDeviceToHost) {
  return ::tangMemcpyFromSymbol(dst, (const void*)&symbol, count, offset, kind);
}

template <typename T>
inline tangError_t tangMemcpyFromSymbolAsync(
  void*               dst,
  const T&            symbol,
  size_t              count,
  size_t              offset = 0,
  enum tangMemcpyKind kind   = tangMemcpyDeviceToHost,
  tangStream_t        stream = 0) {
  return ::tangMemcpyFromSymbolAsync(dst,
                                     (const void*)&symbol,
                                     count,
                                     offset,
                                     kind,
                                     stream);
}
#endif  //!< TANGRT_SYMBOL_FORCE_CXX_API

#endif  // __cplusplus

#endif  //! _TANG_RUNTIME_API_H_
