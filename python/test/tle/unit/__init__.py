# Copyright (c) 2025  XCoreSigma Inc. All rights reserved.

"""
TLE 测试运行器

提供简化的测试运行接口，用于开发和调试TLE功能
"""

import sys
import os

# 添加TLE模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def run_tle_tests():
    """运行所有TLE测试"""
    import pytest

    # 运行测试
    test_file = os.path.join(os.path.dirname(__file__), 'test_tle.py')
    result = pytest.main([test_file, "-v", "--tb=short"])

    return result == 0  # 返回是否全部通过

if __name__ == "__main__":
    success = run_tle_tests()
    if success:
        print("✅ 所有TLE测试通过！")
        sys.exit(0)
    else:
        print("❌ 部分测试失败")
        sys.exit(1)