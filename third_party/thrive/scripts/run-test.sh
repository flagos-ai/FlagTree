#!/bin/bash
set -ex

show_help() {
    cat <<'EOF'
Usage: $0 [OPTIONS]

OPTIONS:
  language       Run all standalone language tests + all core_ groups.
  runtime        Run only runtime tests (unit/runtime).
  core_<num>     Run specific core test group (e.g., core_1, core_3-5, core_1,10,14).
                 Supports comma-separated lists and ranges.
  --help, -h     Show this help message.
  (no option)    Run all tests (language + core + runtime).

Core Groups:
  core_1         Arithmetic & Logical Operations
  core_2         Tensor Memory Operations
  core_3         Atomic Operations
  core_4         Control Flow
  core_5         Matrix Multiplication / Dot
  core_6         Reduction & Scan
  core_7         Tensor Shape Operations
  core_8         Hardware Features
  core_9         Constexpr
  core_10        Inline Assembly
  core_11        Function Calls & Inlining
  core_12        Data Types & Encoding
  core_13        Math Functions
  core_14        Miscellaneous
EOF
    exit 0
}

run_all() {
    run_language
    run_runtime
}

run_language() {
    pytest unit/language/test_annotations.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_block_pointer.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_compile_errors.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_compile_only.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_conversions.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_decorator.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_frontend.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_libdevice.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_line_info.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_matmul.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_module.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_mxfp.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_pipeliner.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_random.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_reproducer.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_standard.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_subprocess.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_tensor_descriptor.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_tuple.py --device thrive -n 1 -v -s --timeout=600
    pytest unit/language/test_warp_specialization.py --device thrive -n 1 -v -s --timeout=600

    run_core 1
    run_core 2
    run_core 3
    run_core 4
    run_core 5
    run_core 6
    run_core 7
    run_core 8
    run_core 9
    run_core 10
    run_core 11
    run_core 12
    run_core 13
    run_core 14
}

run_runtime() {
    pytest unit/runtime --device thrive -n 1 -v -s --timeout=600
}

run_core() {
    case "$1" in
        1) pytest unit/language/test_core.py -k "bin_op or floordiv or bitwise_op or shift_op or compare_op or unary_op or test_abs or addptr or umulhi or max_returns_zero or max_min_with_nan or propagate_nan" --device thrive -n 1 -v -s --timeout=600 ;;
        2) pytest unit/language/test_core.py -k "masked_load or load_cache or store_cache or store_eviction or store_constant or load_store_same or vectorization or strided_load or strided_store or indirect_load or indirect_store or zero_strided or load_scope" --device thrive -n 1 -v -s --timeout=600 ;;
        3) pytest unit/language/test_core.py -k "atomic" --device thrive -n 1 -v -s --timeout=600 ;;
        4) pytest unit/language/test_core.py -k "test_if or test_while or for_iv or static_range or tl_range or disable_licm or nested_if or constexpr_if" --device thrive -n 1 -v -s --timeout=600 ;;
        5) pytest unit/language/test_core.py -k "test_dot or scaled_dot or clamp" --device thrive -n 1 -v -s --timeout=600 ;;
        6) pytest unit/language/test_core.py -k "reduce or sum_dtype or scan or histogram or cumsum or generic_reduction or chained_reductions or side_effectful" --device thrive -n 1 -v -s --timeout=600 ;;
        7) pytest unit/language/test_core.py -k "broadcast or slice or expand_dims or shapes_as_params or transpose or trans_ or reshape or permute" --device thrive -n 1 -v -s --timeout=600 ;;
        8) pytest unit/language/test_core.py -k "num_warps or num_ctas or num_programs or num_threads or globaltimer or smid or maxnreg or unroll_attr or thread_locality or fp_fusion or reflect_ftz or override_arch" --device thrive -n 1 -v -s --timeout=600 ;;
        9) pytest unit/language/test_core.py -k "test_constexpr or test_const or value_specialization" --device thrive -n 1 -v -s --timeout=600 ;;
        10) pytest unit/language/test_core.py -k "inline_asm" --device thrive -n 1 -v -s --timeout=600 ;;
        11) pytest unit/language/test_core.py -k "test_call or noinline or jit_function_arg or map_elementwise" --device thrive -n 1 -v -s --timeout=600 ;;
        12) pytest unit/language/test_core.py -k "scalar_overflow or dtype_codegen or test_dtype or unsigned_name_mangling or ptx_cast or test_cast or convert_float16 or abs_fp8" --device thrive -n 1 -v -s --timeout=600 ;;
        13) pytest unit/language/test_core.py -k "math_op or math_erf or math_fma or math_divide or precise_math" --device thrive -n 1 -v -s --timeout=600 ;;
        14) pytest unit/language/test_core.py -k "test_arange or test_full or test_where or test_gather or test_join or test_interleave or test_split or pointer_arguments or test_assume or poison_return or test_default or test_aliasing or short_circuiting or test_unsplat or tensor_member or temp_var or no_rematerialization" --device thrive -n 1 -v -s --timeout=600 ;;
        *) echo "Unknown core test number: $1"; return 1 ;;
    esac
}

parse_and_run_core() {
    local input="$1"
    local expanded=()
    local IFS=','
    for part in $input; do
        if echo "$part" | grep -q '-'; then
            local start=${part%-*}
            local end=${part#*-}
            for ((i=start; i<=end; i++)); do
                expanded+=($i)
            done
        else
            expanded+=($part)
        fi
    done

    for num in "${expanded[@]}"; do
        run_core "$num"
    done
}

main() {
    local mode="${1:-all}"

    if [[ "$mode" == "--help" || "$mode" == "-h" ]]; then
        show_help
    fi

    local start_seconds=$SECONDS

    if [ "$mode" = "all" ]; then
        echo "====================================="
        echo "Running all tests..."
        echo "====================================="
    fi

    case "$mode" in
        all) run_all ;;
        language) run_language ;;
        runtime) run_runtime ;;
        core_*)
            local nums="${mode#core_}"
            parse_and_run_core "$nums" ;;
        *)
            echo "Invalid option: $mode"
            show_help ;;
    esac

    local end_seconds=$SECONDS
    local duration=$((end_seconds - start_seconds))
    local hours=$((duration / 3600))
    local minutes=$(( (duration % 3600) / 60 ))
    local seconds=$((duration % 60))

    echo "====================================="
    echo "Execution completed."
    printf "Total time: %02d:%02d:%02d\n" $hours $minutes $seconds
    echo "====================================="
}

main "$@"
(End of file - total 119 lines)
