#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAGTREE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATE=`date +%Y%m%d%H%M%S`
LOG_DIR="logs/${DATE}"
mkdir -p ${LOG_DIR}
export TRITON_CACHE_DIR=".triton/${DATE}"

TIMEOUT=7200
EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}
iluvatar_tle_enabled()
{
    case "${FLAGTREE_ILUVATAR_TLE:-}" in
        1|ON|on|true|TRUE) return 0 ;;
        *) return 1 ;;
    esac
}
export CUDA_VISIBLE_DEVICES=0

for pkg in pytest hypothesis absl-py scipy lit filecheck pytest-forked; do
    pip3 list "$pkg" | grep "$pkg" || pip3 install "$pkg"
done
ln -sf "$(command -v filecheck)" "$PWD/bin/FileCheck"

# Preload libgomp.so on arm to prevent TLS allocation errors: "ImportError: /lib64/libgomp.so.1: cannot allocate memory in static TLS block"
if [[ "$(uname -m)" == "aarch64" ]]; then
    libgomp_path=$(find /usr/lib /usr/lib64 /lib /lib64 -type f -name 'libgomp.so*' 2>/dev/null | head -n 1)
    if [[ -n "$libgomp_path" ]]; then
        export LD_PRELOAD="$libgomp_path${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
fi

UMD_CUDAMODULELOADING=0 timeout ${TIMEOUT} pytest -v python/test/unit/language/test_core.py -o junit_suite_name="test_core" --junitxml=${LOG_DIR}_xml/___test_core_mr.xml 2>&1 | tee ${LOG_DIR}/test_core.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_annotations.py -o junit_suite_name="test_annotations" --junitxml=${LOG_DIR}_xml/___test_annotations.xml 2>&1 | tee ${LOG_DIR}/test_annotations.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_block_pointer.py -o junit_suite_name="test_block_pointer" --junitxml=${LOG_DIR}_xml/___test_block_pointer.xml 2>&1 | tee ${LOG_DIR}/test_block_pointer.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_compile_errors.py -o junit_suite_name="test_compile_errors" --junitxml=${LOG_DIR}_xml/___test_compile_errors.xml 2>&1 | tee ${LOG_DIR}/test_compile_errors.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_compile_only.py -o junit_suite_name="test_compile_only" --junitxml=${LOG_DIR}_xml/___test_compile_only.xml 2>&1 | tee ${LOG_DIR}/test_compile_only.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_conversions.py -o junit_suite_name="test_conversions" --junitxml=${LOG_DIR}_xml/___test_conversions.xml 2>&1 | tee ${LOG_DIR}/test_conversions.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_decorator.py -o junit_suite_name="test_decorator" --junitxml=${LOG_DIR}_xml/___test_decorator.xml 2>&1 | tee ${LOG_DIR}/test_decorator.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_frontend.py -o junit_suite_name="test_frontend" --junitxml=${LOG_DIR}_xml/___test_frontend.xml 2>&1 | tee ${LOG_DIR}/test_frontend.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_libdevice.py -o junit_suite_name="test_libdevice" --junitxml=${LOG_DIR}_xml/___test_libdevice.xml 2>&1 | tee ${LOG_DIR}/test_libdevice.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_iluvatar_math_fp16_bf16.py -o junit_suite_name="test_iluvatar_math_fp16_bf16" --junitxml=${LOG_DIR}_xml/___test_iluvatar_math_fp16_bf16.xml 2>&1 | tee ${LOG_DIR}/test_iluvatar_math_fp16_bf16.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_iluvatar_int8_dot_pipeline.py -o junit_suite_name="test_iluvatar_int8_dot_pipeline" --junitxml=${LOG_DIR}_xml/___test_iluvatar_int8_dot_pipeline.xml 2>&1 | tee ${LOG_DIR}/test_iluvatar_int8_dot_pipeline.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_line_info.py -o junit_suite_name="test_line_info" --junitxml=${LOG_DIR}_xml/___test_line_info.xml 2>&1 | tee ${LOG_DIR}/test_line_info.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_matmul.py -o junit_suite_name="test_matmul" --junitxml=${LOG_DIR}_xml/___test_matmul.xml 2>&1 | tee ${LOG_DIR}/test_matmul.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_module.py -o junit_suite_name="test_module" --junitxml=${LOG_DIR}_xml/___test_module.xml 2>&1 | tee ${LOG_DIR}/test_module.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_mxfp.py -o junit_suite_name="test_mxfp" --junitxml=${LOG_DIR}_xml/___test_mxfp.xml 2>&1 | tee ${LOG_DIR}/test_mxfp.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_pipeliner.py -o junit_suite_name="test_pipeliner.py" --junitxml=${LOG_DIR}_xml/___test_pipeliner.xml 2>&1 | tee ${LOG_DIR}/test_pipeliner.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_random.py -o junit_suite_name="test_random" --junitxml=${LOG_DIR}_xml/___test_random.xml 2>&1 | tee ${LOG_DIR}/test_random.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_reproducer.py -o junit_suite_name="test_reproducer" --junitxml=${LOG_DIR}_xml/___test_reproducer.xml 2>&1 | tee ${LOG_DIR}/test_reproducer.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_standard.py -o junit_suite_name="test_standard" --junitxml=${LOG_DIR}_xml/___test_standard.xml 2>&1 | tee ${LOG_DIR}/test_standard.log; check_status
timeout ${TIMEOUT} pytest -v python/test/unit/language/test_subprocess.py -o junit_suite_name="test_subprocess" --junitxml=${LOG_DIR}_xml/___test_subprocess.xml 2>&1 | tee ${LOG_DIR}/test_subprocess.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_tensor_descriptor.py -o junit_suite_name="test_tensor_descriptor" --junitxml=${LOG_DIR}_xml/___test_tensor_descriptor.xml 2>&1 | tee ${LOG_DIR}/test_tensor_descriptor.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_tuple.py -o junit_suite_name="test_tuple" --junitxml=${LOG_DIR}_xml/___test_tuple.xml 2>&1 | tee ${LOG_DIR}/test_tuple.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/language/test_warp_specialization.py -o junit_suite_name="test_warp_specialization" --junitxml=${LOG_DIR}_xml/___test_warp_specialization.xml 2>&1 | tee ${LOG_DIR}/test_warp_specialization.log; check_status
timeout ${TIMEOUT} pytest -v python/test/unit/operators/test_blocksparse.py -o junit_suite_name="test_blocksparse" --junitxml=${LOG_DIR}_xml/___test_blocksparse.xml 2>&1 | tee ${LOG_DIR}/test_blocksparse.log; check_status
timeout ${TIMEOUT} pytest -v python/test/unit/operators/test_cross_entropy.py -o junit_suite_name="test_cross_entropy" --junitxml=${LOG_DIR}_xml/___test_cross_entropy.xml 2>&1 | tee ${LOG_DIR}/test_cross_entropy.log; check_status
timeout ${TIMEOUT} pytest -v python/test/unit/operators/test_dot_trans.py -o junit_suite_name="test_dot_trans" --junitxml=${LOG_DIR}_xml/___test_dot_trans.xml 2>&1 | tee ${LOG_DIR}/test_dot_trans.log; check_status
timeout ${TIMEOUT} pytest -v python/test/unit/operators/test_flash_attention.py -o junit_suite_name="test_flash_attention" --junitxml=${LOG_DIR}_xml/___test_flash_attention.xml 2>&1 | tee ${LOG_DIR}/test_flash_attention.log; check_status
timeout ${TIMEOUT} pytest -v python/test/unit/operators/test_inductor.py -o junit_suite_name="test_inductor" --junitxml=${LOG_DIR}_xml/___test_inductor.xml 2>&1 | tee ${LOG_DIR}/test_inductor.log; check_status
timeout ${TIMEOUT} pytest -v python/test/unit/operators/test_matmul.py -o junit_suite_name="test_matmul" --junitxml=${LOG_DIR}_xml/___test_matmul.xml 2>&1 | tee ${LOG_DIR}/test_matmul.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/operators/test_sme.py -o junit_suite_name="test_sme" --junitxml=${LOG_DIR}_xml/___test_sme.xml 2>&1 | tee ${LOG_DIR}/test_sme.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/runtime/test_autotuner.py -o junit_suite_name="test_autotuner" --junitxml=${LOG_DIR}_xml/___test_autotuner.xml 2>&1 | tee ${LOG_DIR}/test_autotuner.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/runtime/test_bindings.py -o junit_suite_name="test_bindings" --junitxml=${LOG_DIR}_xml/___test_bindings.xml 2>&1 | tee ${LOG_DIR}/test_bindings.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/runtime/test_blaslt.py -o junit_suite_name="test_blaslt" --junitxml=${LOG_DIR}_xml/___test_blaslt.xml 2>&1 | tee ${LOG_DIR}/test_blaslt.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/runtime/test_build.py -o junit_suite_name="test_build" --junitxml=${LOG_DIR}_xml/___test_build.xml 2>&1 | tee ${LOG_DIR}/test_build.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/runtime/test_cache.py -o junit_suite_name="test_cache" --junitxml=${LOG_DIR}_xml/___test_cache.xml 2>&1 | tee ${LOG_DIR}/test_cache.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/runtime/test_compilation_listener.py -o junit_suite_name="test_compilation_listener" --junitxml=${LOG_DIR}_xml/___test_compilation_listener.xml 2>&1 | tee ${LOG_DIR}/test_compilation_listener.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/runtime/test_driver.py -o junit_suite_name="test_driver" --junitxml=${LOG_DIR}_xml/___test_driver.xml 2>&1 | tee ${LOG_DIR}/test_driver.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/runtime/test_launch.py -o junit_suite_name="test_launch" --junitxml=${LOG_DIR}_xml/___test_launch.xml 2>&1 | tee ${LOG_DIR}/test_launch.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/runtime/test_specialization.py -o junit_suite_name="test_specialization" --junitxml=${LOG_DIR}_xml/___test_specialization.xml 2>&1 | tee ${LOG_DIR}/test_specialization.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/runtime/test_subproc.py -o junit_suite_name="test_subproc" --junitxml=${LOG_DIR}_xml/___test_subproc.xml 2>&1 | tee ${LOG_DIR}/test_subproc.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/runtime/test_iluvatar_loop_unroll_warning.py -o junit_suite_name="test_iluvatar_loop_unroll_warning" --junitxml=${LOG_DIR}_xml/___test_iluvatar_loop_unroll_warning.xml 2>&1 | tee ${LOG_DIR}/test_iluvatar_loop_unroll_warning.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/instrumentation/test_gpuhello.py -o junit_suite_name="test_gpuhello" --junitxml=${LOG_DIR}_xml/___test_gpuhello.xml 2>&1 | tee ${LOG_DIR}/test_gpuhello.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/tools/test_aot.py -o junit_suite_name="test_aot" --junitxml=${LOG_DIR}_xml/___test_aot.xml 2>&1 | tee ${LOG_DIR}/test_aot.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/tools/test_disasm.py -o junit_suite_name="test_disasm" --junitxml=${LOG_DIR}_xml/___test_disasm.xml 2>&1 | tee ${LOG_DIR}/test_disasm.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/tools/test_irsource.py -o junit_suite_name="test_irsource" --junitxml=${LOG_DIR}_xml/___test_irsource.xml 2>&1 | tee ${LOG_DIR}/test_irsource.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/tools/test_linear_layout.py -o junit_suite_name="test_linear_layout" --junitxml=${LOG_DIR}_xml/___test_linear_layout.xml 2>&1 | tee ${LOG_DIR}/test_linear_layout.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/tools/test_triton_to_gluon.py -o junit_suite_name="test_triton_to_gluon" --junitxml=${LOG_DIR}_xml/___test_triton_to_gluon.xml 2>&1 | tee ${LOG_DIR}/test_triton_to_gluon.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/test_debug_dump.py -o junit_suite_name="test_debug_dump" --junitxml=${LOG_DIR}_xml/___test_debug_dump.xml 2>&1 | tee ${LOG_DIR}/test_debug_dump.log; check_status
timeout ${TIMEOUT} pytest -v python/test/unit/test_debug.py -o junit_suite_name="test_debug" --junitxml=${LOG_DIR}_xml/___test_debug.xml 2>&1 | tee ${LOG_DIR}/test_debug.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/test_debug_info.py -o junit_suite_name="test_debug_info" --junitxml=${LOG_DIR}_xml/___test_debug_info.xml 2>&1 | tee ${LOG_DIR}/test_debug_info.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/test_filecheck.py -o junit_suite_name="test_filecheck" --junitxml=${LOG_DIR}_xml/___test_filecheck.xml 2>&1 | tee ${LOG_DIR}/test_filecheck.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/test_knobs.py -o junit_suite_name="test_knobs" --junitxml=${LOG_DIR}_xml/___test_knobs.xml 2>&1 | tee ${LOG_DIR}/test_knobs.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/test_link.py -o junit_suite_name="test_link" --junitxml=${LOG_DIR}_xml/___test_link.xml 2>&1 | tee ${LOG_DIR}/test_link.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/test_perf_warning.py -o junit_suite_name="test_perf_warning" --junitxml=${LOG_DIR}_xml/___test_perf_warning.xml 2>&1 | tee ${LOG_DIR}/test_perf_warning.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/regression/test_cast_matmul.py -o junit_suite_name="test_cast_matmul" --junitxml=${LOG_DIR}_xml/___test_cast_matmul.xml 2>&1 | tee ${LOG_DIR}/test_cast_matmul.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/regression/test_cast_matmul.py -o junit_suite_name="test_cast_matmul" --junitxml=${LOG_DIR}_xml/___test_cast_matmul.xml 2>&1 | tee ${LOG_DIR}/test_cast_matmul.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/test_functional_regressions.py -o junit_suite_name="test_functional_regressions" --junitxml=${LOG_DIR}_xml/___test_functional_regressions.xml 2>&1 | tee ${LOG_DIR}/test_functional_regressions.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/integrations/vllm/chunk_o/test_chunk_fwd_kernel_o.py -o junit_suite_name="test_chunk_o" --junitxml=${LOG_DIR}_xml/___test_chunk_o.xml 2>&1 | tee ${LOG_DIR}/test_chunk_o.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/integrations/vllm/wy_fast/test_recompute_w_u.py -o junit_suite_name="test_recompute_w_u" --junitxml=${LOG_DIR}_xml/___test_recompute_w_u.xml 2>&1 | tee ${LOG_DIR}/test_recompute_w_u.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/integrations/fbgemm/test_jagged_flash_attention_bwd_basic_min.py -o junit_suite_name="test_jagged_flash_attention_bwd_basic_min" --junitxml=${LOG_DIR}_xml/___test_jagged_flash_attention_bwd_basic_min.xml 2>&1 | tee ${LOG_DIR}/test_jagged_flash_attention_bwd_basic_min.log; check_status
# PUNICA_TEST_LEVEL=quick timeout ${TIMEOUT} pytest -v python/test/unit/integrations/vllm/punica_lora/test_punica_ops.py -o junit_suite_name="test_punica_lora" --junitxml=${LOG_DIR}_xml/___test_punica_lora.xml 2>&1 | tee ${LOG_DIR}/test_punica_lora.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/integrations/sglang/flash_mla/test_flash_mla_ut.py -o junit_suite_name="test_flash_mla" --junitxml=${LOG_DIR}_xml/___test_flash_mla.xml 2>&1 | tee ${LOG_DIR}/test_flash_mla.log; check_status
# timeout ${TIMEOUT} pytest -v python/test/unit/integrations/inductor/test_bucketize_matmul.py -o junit_suite_name="test_bucketize_matmul" --junitxml=${LOG_DIR}_xml/___test_bucketize_matmul.xml 2>&1 | tee ${LOG_DIR}/test_bucketize_matmul.log; check_status

if iluvatar_tle_enabled; then
    timeout ${TIMEOUT} pytest -v ${FLAGTREE_ROOT}/python/test/tle/integration/test_tle_local_store.py -o junit_suite_name="test_tle_local_store" --junitxml=${LOG_DIR}_xml/___test_tle_local_store.xml 2>&1 | tee ${LOG_DIR}/test_tle_local_store.log; check_status
    timeout ${TIMEOUT} pytest -v ${FLAGTREE_ROOT}/python/test/tle/unit/test_tle_gpu_local_ptr.py -o junit_suite_name="test_tle_gpu_local_ptr" --junitxml=${LOG_DIR}_xml/___test_tle_gpu_local_ptr.xml 2>&1 | tee ${LOG_DIR}/test_tle_gpu_local_ptr.log; check_status
    timeout ${TIMEOUT} pytest -v ${FLAGTREE_ROOT}/python/test/tle/unit/test_extract_tile_static_index.py -o junit_suite_name="test_extract_tile_static_index" --junitxml=${LOG_DIR}_xml/___test_extract_tile_static_index.xml 2>&1 | tee ${LOG_DIR}/test_extract_tile_static_index.log; check_status
    timeout ${TIMEOUT} pytest -v ${FLAGTREE_ROOT}/python/test/tle/unit/test_extract_tile_dynamic_index.py -o junit_suite_name="test_extract_tile_dynamic_index" --junitxml=${LOG_DIR}_xml/___test_extract_tile_dynamic_index.xml 2>&1 | tee ${LOG_DIR}/test_extract_tile_dynamic_index.log; check_status
    timeout ${TIMEOUT} pytest -v ${FLAGTREE_ROOT}/python/test/tle/unit/test_insert_tile_static_index.py -o junit_suite_name="test_insert_tile_static_index" --junitxml=${LOG_DIR}_xml/___test_insert_tile_static_index.xml 2>&1 | tee ${LOG_DIR}/test_insert_tile_static_index.log; check_status
    timeout ${TIMEOUT} pytest -v ${FLAGTREE_ROOT}/python/test/tle/unit/test_insert_tile_dynamic_index.py -o junit_suite_name="test_insert_tile_dynamic_index" --junitxml=${LOG_DIR}_xml/___test_insert_tile_dynamic_index.xml 2>&1 | tee ${LOG_DIR}/test_insert_tile_dynamic_index.log; check_status
fi

timeout ${TIMEOUT} python3 util_auto_analysis.py ${LOG_DIR}; check_status

# Just for local test. CI will download from http://sw.iluvatar.ai/download/corex/daily_packages/ivcore11/x86_64/latest/.cache/sdk/
if [[ -d python/build ]]; then
    opt_path=$(find python/build -type f -name triton-opt | head -n 1)
    if [[ -n "$opt_path" ]]; then
        dir_path=$(dirname "$(realpath "$opt_path")")
        export PATH="$PATH:$dir_path"
    fi
fi
timeout ${TIMEOUT} lit test/Conversion/iluvatar/ -v; check_status

DATE_END=`date +%Y%m%d%H%M%S`
echo "Total Times: $DATE ---> $DATE_END"
rm -rf ${TRITON_CACHE_DIR}
exit $EXIT_STATUS
