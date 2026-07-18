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

#ifndef PROTON_ATTR_DEFS
#define PROTON_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

def MetricTypeAttr : I32EnumAttr<
  "MetricType", "The type of metric to be profiled",
  [
    I32EnumAttrCase<"CYCLE", 0, "cycle">,
  ]> {
  let cppNamespace = "::mlir::triton::proton";
  let description = [{
    Attribute to indicate the metric to be profiled.
    The following metrics are supported:
    - CYCLE: Cycle count metric.
  }];
}

def SamplingStrategyAttr : I32EnumAttr<
  "SamplingStrategy", "The strategy for sampling the profiling data",
  [
    I32EnumAttrCase<"NONE", 0, "none">,
    I32EnumAttrCase<"SELECTIVE", 1, "selective">,
  ]> {
  let cppNamespace = "::mlir::triton::proton";
  let description = [{
    Attribute to indicate the sampling strategy for profiling.
    The following sampling strategies are supported:
    - NONE: No sampling.
    - SELECTIVE: Manually select a couple of instances to profile.
  }];
}

def ModeAttr : I32EnumAttr<
  "Mode", "The mode of profiling",
  [
    I32EnumAttrCase<"DEFAULT", 0, "default">,
    I32EnumAttrCase<"MMA", 1, "mma">,
  ]> {
  let cppNamespace = "::mlir::triton::proton";
  let description = [{
    Attribute to indicate the mode of profiling, which specifies passes and instructions to monitor.
  }];
}

#endif // PROTON_ATTR_DEFS
