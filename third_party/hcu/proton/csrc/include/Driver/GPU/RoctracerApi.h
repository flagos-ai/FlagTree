// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef PROTON_DRIVER_GPU_ROCTRACER_API_H_
#define PROTON_DRIVER_GPU_ROCTRACER_API_H_

#include "Driver/Dispatch.h"
#include "roctracer/roctracer.h"

namespace proton {

namespace roctracer {

struct ExternLibRoctracer : public ExternLibBase {
  using RetType = roctracer_status_t;
  static constexpr const char *name = "libroctracer64.so";
  static constexpr const char *symbolName = "roctracer_start";
  static constexpr const char *pathEnv{};
  static constexpr RetType success = ROCTRACER_STATUS_SUCCESS;
  static inline void *lib = nullptr;
};

template <bool CheckSuccess>
roctracer_status_t setProperties(roctracer_domain_t domain, void *properties);

template <bool CheckSuccess>
roctracer_status_t getTimestamp(roctracer_timestamp_t *timestamp);

void start();

void stop();

//
// Callbacks
//

template <bool CheckSuccess>
roctracer_status_t enableDomainCallback(activity_domain_t domain,
                                        activity_rtapi_callback_t callback,
                                        void *arg);

template <bool CheckSuccess>
roctracer_status_t disableDomainCallback(activity_domain_t domain);

template <bool CheckSuccess>
roctracer_status_t enableOpCallback(activity_domain_t domain, uint32_t op,
                                    activity_rtapi_callback_t callback,
                                    void *arg);

template <bool CheckSuccess>
roctracer_status_t disableOpCallback(activity_domain_t domain, uint32_t op);

//
// Activity
//

template <bool CheckSuccess>
roctracer_status_t openPool(const roctracer_properties_t *properties);

template <bool CheckSuccess> roctracer_status_t closePool();

template <bool CheckSuccess>
roctracer_status_t enableOpActivity(activity_domain_t domain, uint32_t op);

template <bool CheckSuccess>
roctracer_status_t enableDomainActivity(activity_domain_t domain);

template <bool CheckSuccess>
roctracer_status_t disableOpActivity(activity_domain_t domain, uint32_t op);

template <bool CheckSuccess>
roctracer_status_t disableDomainActivity(activity_domain_t domain);

template <bool CheckSuccess> roctracer_status_t flushActivity();

template <bool CheckSuccess>
roctracer_status_t getNextRecord(const activity_record_t *record,
                                 const activity_record_t **next);

char *getOpString(uint32_t domain, uint32_t op, uint32_t kind);

//
// External correlation
//

template <bool CheckSuccess>
roctracer_status_t
activityPushExternalCorrelationId(activity_correlation_id_t id);

template <bool CheckSuccess>
roctracer_status_t
activityPopExternalCorrelationId(activity_correlation_id_t *last_id);

} // namespace roctracer

} // namespace proton

#endif // PROTON_DRIVER_GPU_ROCTRACER_API_H_
