// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "Driver/GPU/CuptiApi.h"
#include "Device.h"
#include "Driver/Dispatch.h"

namespace proton {

namespace cupti {

DEFINE_DISPATCH(ExternLibCupti, getVersion, cuptiGetVersion, uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, getContextId, cuptiGetContextId, CUcontext,
                uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, activityRegisterCallbacks,
                cuptiActivityRegisterCallbacks,
                CUpti_BuffersCallbackRequestFunc,
                CUpti_BuffersCallbackCompleteFunc)

DEFINE_DISPATCH(ExternLibCupti, subscribe, cuptiSubscribe,
                CUpti_SubscriberHandle *, CUpti_CallbackFunc, void *)

DEFINE_DISPATCH(ExternLibCupti, enableDomain, cuptiEnableDomain, uint32_t,
                CUpti_SubscriberHandle, CUpti_CallbackDomain)

DEFINE_DISPATCH(ExternLibCupti, enableCallback, cuptiEnableCallback, uint32_t,
                CUpti_SubscriberHandle, CUpti_CallbackDomain, CUpti_CallbackId);

DEFINE_DISPATCH(ExternLibCupti, activityEnable, cuptiActivityEnable,
                CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityDisable, cuptiActivityDisable,
                CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityEnableContext,
                cuptiActivityEnableContext, CUcontext, CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityDisableContext,
                cuptiActivityDisableContext, CUcontext, CUpti_ActivityKind)

DEFINE_DISPATCH(ExternLibCupti, activityFlushAll, cuptiActivityFlushAll,
                uint32_t)

DEFINE_DISPATCH(ExternLibCupti, activityGetNextRecord,
                cuptiActivityGetNextRecord, uint8_t *, size_t,
                CUpti_Activity **)

DEFINE_DISPATCH(ExternLibCupti, activityPushExternalCorrelationId,
                cuptiActivityPushExternalCorrelationId,
                CUpti_ExternalCorrelationKind, uint64_t)

DEFINE_DISPATCH(ExternLibCupti, activityPopExternalCorrelationId,
                cuptiActivityPopExternalCorrelationId,
                CUpti_ExternalCorrelationKind, uint64_t *)

DEFINE_DISPATCH(ExternLibCupti, activitySetAttribute, cuptiActivitySetAttribute,
                CUpti_ActivityAttribute, size_t *, void *)

DEFINE_DISPATCH(ExternLibCupti, activityEnableHWTrace,
                cuptiActivityEnableHWTrace, uint8_t)

DEFINE_DISPATCH(ExternLibCupti, unsubscribe, cuptiUnsubscribe,
                CUpti_SubscriberHandle)

DEFINE_DISPATCH(ExternLibCupti, finalize, cuptiFinalize)

DEFINE_DISPATCH(ExternLibCupti, getGraphExecId, cuptiGetGraphExecId,
                CUgraphExec, uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, getGraphId, cuptiGetGraphId, CUgraph,
                uint32_t *);

DEFINE_DISPATCH(ExternLibCupti, getGraphNodeId, cuptiGetGraphNodeId,
                CUgraphNode, uint64_t *);

DEFINE_DISPATCH(ExternLibCupti, getCubinCrc, cuptiGetCubinCrc,
                CUpti_GetCubinCrcParams *);

DEFINE_DISPATCH(ExternLibCupti, getSassToSourceCorrelation,
                cuptiGetSassToSourceCorrelation,
                CUpti_GetSassToSourceCorrelationParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingGetNumStallReasons,
                cuptiPCSamplingGetNumStallReasons,
                CUpti_PCSamplingGetNumStallReasonsParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingGetStallReasons,
                cuptiPCSamplingGetStallReasons,
                CUpti_PCSamplingGetStallReasonsParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingSetConfigurationAttribute,
                cuptiPCSamplingSetConfigurationAttribute,
                CUpti_PCSamplingConfigurationInfoParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingEnable, cuptiPCSamplingEnable,
                CUpti_PCSamplingEnableParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingDisable, cuptiPCSamplingDisable,
                CUpti_PCSamplingDisableParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingGetData, cuptiPCSamplingGetData,
                CUpti_PCSamplingGetDataParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingStart, cuptiPCSamplingStart,
                CUpti_PCSamplingStartParams *);

DEFINE_DISPATCH(ExternLibCupti, pcSamplingStop, cuptiPCSamplingStop,
                CUpti_PCSamplingStopParams *);

} // namespace cupti

} // namespace proton
