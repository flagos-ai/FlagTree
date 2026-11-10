/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// triton_gcu300_core.h -- C ABI for the GCU300 in-process MLIR opt pipeline.
//
// Replaces the triton-gcu300-opt subprocess: the heavy MLIR/LLVM backend
// lives in lib_triton_gcu300_core.so (Python-independent), while a thin
// per-Python-version pybind11 wrapper links against it.

#ifndef TRITON_GCU300_CORE_H
#define TRITON_GCU300_CORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const char *data;
  size_t len;
} Gcu300String;

// Opaque pipeline handle for building and running pass pipelines.
typedef struct Gcu300Pipeline_ *Gcu300Pipeline;

// Pipeline lifecycle
Gcu300Pipeline gcu300_pipeline_create(void);
void gcu300_pipeline_destroy(Gcu300Pipeline p);

// Add a pass to the pipeline with optional key=value options.
// options_str format: "key1=value1,key2=value2" or empty string for no options.
void gcu300_pipeline_add_pass(Gcu300Pipeline p, const char *pass_name,
                              const char *options_str);

// Run the configured pipeline on MLIR text input.
// Returns the processed MLIR text. On error, data is NULL and
// gcu300_pipeline_last_error() gives the message.
Gcu300String gcu300_pipeline_run(Gcu300Pipeline p, const char *input,
                                 size_t input_len);

// Get last error message (empty string if none).
const char *gcu300_pipeline_last_error(Gcu300Pipeline p);

// Legacy interface: Run the MLIR opt pipeline with CLI-style args.
// Kept for backward compatibility. New code should use Pipeline API.
Gcu300String gcu300_run_opt(const char *input, size_t input_len,
                            const char *const *args, int num_args);

// Free a Gcu300String.
void gcu300_string_free(Gcu300String s);

// Full MlirOptMain wrapper -- registers all GCU300 dialects and passes,
// then forwards argc/argv to MlirOptMain.  Allows the opt binary to be
// a thin C wrapper with no LLVM/MLIR link dependency.
int gcu300_opt_main(int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif // TRITON_GCU300_CORE_H
