# Copyright (c) 2025  XCoreSigma Inc. All rights reserved.

#!/usr/bin/env python3
"""
Simple verification script for TLE TritonToTritonGPU conversion test.

This script manually checks if the TLE operations are correctly converted
by examining the output of triton-opt conversion.
"""

import subprocess
import sys
import os

def run_triton_conversion(test_file, target="cuda:80", num_warps=4):
    """Run triton-opt conversion and return output"""
    # Find Triton root directory (go up from script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    triton_root = os.path.abspath(os.path.join(script_dir, "../../../.."))

    # Relative path to test file from triton root
    rel_test_file = os.path.relpath(test_file, triton_root)

    cmd = [
        "./build/cmake.linux-x86_64-cpython-3.10/bin/triton-opt",
        rel_test_file,
        "-split-input-file",
        f"-convert-triton-to-tritongpu=target={target} num-warps={num_warps}"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=triton_root)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def verify_tle_operations(output):
    """Verify that TLE operations are present in the output"""
    checks = [
        ("ttg.local_alloc", "local_alloc operation"),
        ("ttg.local_load", "local_load operation"),
        ("!ttg.memdesc<", "memory descriptor type"),
        ("#shared, #smem", "shared memory attribute"),
        ("mutable>", "mutable attribute")
    ]

    results = []
    for pattern, description in checks:
        found = pattern in output
        results.append((description, found))
        print(f"✓ {description}: {'Present' if found else 'Missing'}")

    return all(result[1] for result in results)

def main():
    """Main function"""
    # Get script directory and find test file relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(script_dir, "test_tle_tritontoTritonGPU_alloc.mlir")

    print("🧪 TLE TritonToTritonGPU Conversion Test")
    print("=" * 50)

    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return 1

    print(f"📁 Test file: {test_file}")

    # Run conversion
    print("🔄 Running triton-opt conversion...")
    success, stdout, stderr = run_triton_conversion(test_file)

    if not success:
        print(f"❌ Conversion failed:")
        print(f"Stderr: {stderr}")
        return 1

    print("✅ Conversion successful")

    # Verify operations
    print("\n🔍 Verifying TLE operations...")
    all_present = verify_tle_operations(stdout)

    # Show key operations found
    print("\n📋 Key operations found:")
    local_alloc_count = stdout.count("ttg.local_alloc")
    local_load_count = stdout.count("ttg.local_load")
    memdesc_count = stdout.count("!ttg.memdesc<")

    print(f"  - ttg.local_alloc: {local_alloc_count} instances")
    print(f"  - ttg.local_load: {local_load_count} instances")
    print(f"  - memory descriptors: {memdesc_count} instances")

    # Show sample lines
    print("\n📝 Sample output lines:")
    lines = stdout.split('\n')
    for i, line in enumerate(lines):
        if 'ttg.local_alloc' in line or 'ttg.local_load' in line:
            print(f"  {i+1:4d}: {line.strip()}")
            if i >= 5:  # Limit output
                break

    if all_present:
        print("\n🎉 All required TLE operations are present!")
        print("✅ Test passed")
        return 0
    else:
        print("\n❌ Some TLE operations are missing")
        print("❌ Test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())