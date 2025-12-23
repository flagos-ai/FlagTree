#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
update:
    1. 增加了ResultsCollectorPlugin下全局case_level、current_suite_name和docstring的设置，方便在测试函数体内进行参数控制
    2. 修复了conftest多次加载导致生成额外的空json文件的BUG
"""
import datetime
import inspect
import json
import os
import platform

import pytest

LEVEL_SCORE_DICT = {"level_0": 20, "level_1": 5, "level_2": 1, "L0": 20, "L1": 5, "L2": 1}


def score(value):
    """
    装饰器，用于给测试用例分配score。
    """

    def decorator(function):
        function.score = value
        return function

    return decorator


class ResultsCollectorPlugin:
    current_case_score = 0
    current_case_level = None
    current_suite_name = ''
    docstring = ''

    def __init__(self, config):
        self.config = config
        self.score_summary = {}
        self.total_case_score = 0
        self.total_case_num = 0
        self.passed_case_score = 0
        self.passed_case_num = 0
        self.result_dir = './log_result'
        self.result_file = '___test-summary.json'

    def pytest_runtest_setup(self, item):
        # case执行前current_case_score置0
        ResultsCollectorPlugin.current_case_score = 0
        ResultsCollectorPlugin.current_case_level = None
        ResultsCollectorPlugin.current_suite_name = ""
        ResultsCollectorPlugin.docstring = ""
        if hasattr(self.config.option, 'result_dir'):
            self.result_dir = self.config.option.result_dir
        if hasattr(self.config.option, 'result_file'):
            self.result_file = self.config.option.result_file

    @classmethod
    def get_test_level(cls, item):
        """获取测试项的级别"""
        # print(f"DEBUG: Checking item {item.nodeid}")

        # 优先检查通过pytest.mark添加的标记
        if hasattr(item, 'iter_markers'):
            for mark in item.iter_markers():
                # print(f"DEBUG: Found mark {mark.name}")
                if mark.name in LEVEL_SCORE_DICT:
                    # print(f"DEBUG: Matched level {mark.name}")
                    return mark.name

        # 检查函数级别的pytestmark属性
        level_marks = []
        if hasattr(item.function, 'pytestmark'):
            level_marks = [x.name for x in item.function.pytestmark if x.name in LEVEL_SCORE_DICT]
            # print(f"DEBUG: Function level marks: {level_marks}")

        # 检查类级别的pytestmark属性
        if len(level_marks) == 0 and hasattr(item, "cls") and hasattr(item.cls, 'pytestmark'):
            level_marks = [x.name for x in item.cls.pytestmark if x.name in LEVEL_SCORE_DICT]
            # print(f"DEBUG: Class level marks: {level_marks}")

        # 检查全局设置
        if cls.current_case_level and cls.current_case_level in LEVEL_SCORE_DICT:
            level_marks.append(cls.current_case_level)
            # print(f"DEBUG: Global level marks: {level_marks}")

        # 返回找到的第一个有效级别或默认级别
        if len(level_marks) > 0:
            result = level_marks[0]
            # print(f"DEBUG: Returning level {result}")
            return result

        # 默认返回level_0
        # print("DEBUG: Returning default level_0")
        return 'level_0'

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        start_time = datetime.datetime.now().timestamp() * 1000
        outcome = yield
        if (outcome.excinfo is not None and outcome.excinfo[0].__name__ == 'Skipped'):
            # 代码块内调用skip方法也不计入总case内
            return
        end_time = datetime.datetime.now().timestamp() * 1000
        test_level = self.get_test_level(item)
        case_score = LEVEL_SCORE_DICT.get(test_level)

        if self.current_case_score != 0:
            # 在测试case内设置current_case_score
            case_score = self.current_case_score

        if hasattr(item.function, 'score'):
            # score装饰器优先级最高，不建议使用
            case_score = item.function.score

        docstring = inspect.getdoc(item.obj)
        pos = item.nodeid.find('::test_')
        test_suite_name = item.nodeid[:pos]
        test_name = ResultsCollectorPlugin.current_suite_name + "::" + item.nodeid[pos + 2:]
        test_suit_info = self.score_summary.get(test_suite_name, {})
        test_suit_info[test_name] = {
            "case_score": case_score, "description": docstring if docstring else ResultsCollectorPlugin.docstring,
            "duration": end_time - start_time, "end_time": end_time, "start_time": start_time, "test_level": test_level,
            "test_name": item.nodeid, "test_result": "Unknow", "test_score": 0.0
        }

        if outcome.excinfo is None:
            self.total_case_score += case_score
            self.total_case_num += 1
            self.passed_case_score += case_score
            self.passed_case_num += 1
            test_suit_info[test_name]['test_score'] = case_score
            test_suit_info[test_name]['test_result'] = 'Passed'
        elif outcome.excinfo[0].__name__ == 'XFailed':
            self.total_case_score += case_score
            self.total_case_num += 1
            test_suit_info[test_name]['test_result'] = 'XFailed'
        else:
            self.total_case_score += case_score
            self.total_case_num += 1
            test_suit_info[test_name]['test_result'] = 'Failed'
        self.score_summary[test_suite_name] = test_suit_info

    def pytest_sessionfinish(self, session):
        self.generate_result_file()
        # 判断环境是否为Windows系统
        if platform.system() == 'Windows':
            # pytest_terminal_summary钩子函数的打印，在不同版本的python
            # 包下可能不会被调用，因此生成报告逻辑放在pytest_sessionfinish下，
            # 下边的打印逻辑根据实际需求进行变更
            terminalreporter = session.config.pluginmanager.get_plugin('terminalreporter')
            terminalreporter.write_sep("=", "Test Summary")
            terminalreporter.write_line(f"Score: {self.passed_case_score}"
                                        f" / {self.total_case_score}")

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        # 使用这个钩子函数打印是因为可以打印在结尾，pytest_sessionfinish信息打印在执行结束后和错误信息打印之间
        tw = terminalreporter._tw
        tw.line("")
        print_lines = gen_print_result(
            ["[Test Summary with Scores]", f" Score: {self.passed_case_score} /"
             f" {self.total_case_score} "])
        for line in print_lines:
            tw.line(line)

    def generate_result_file(self):
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if self.total_case_num == 0:
            return

        result_path = os.path.join(self.result_dir, self.result_file)

        # 如果结果文件已存在，则合并数据
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

                # 合并score_summary数据
                for suite, cases in self.score_summary.items():
                    if suite in existing_data.get('score_summary', {}):
                        # 如果suite已存在，合并其中的测试用例
                        existing_data['score_summary'][suite].update(cases)
                    else:
                        # 如果suite不存在，直接添加
                        existing_data.setdefault('score_summary', {})[suite] = cases

                # 累加统计信息
                existing_data['total_case_score'] += self.total_case_score
                existing_data['passed_case_score'] += self.passed_case_score
                existing_data['total_case_num'] += self.total_case_num
                existing_data['passed_case_num'] += self.passed_case_num

                final_data = existing_data
            except Exception as e:
                print(f"Warning: Failed to merge results, overwriting: {e}")
                final_data = self._get_result_data()
        else:
            # 如果文件不存在，直接使用当前数据
            final_data = self._get_result_data()

        # 写入合并后的数据
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)

    def _get_result_data(self):
        """获取当前会话的结果数据"""
        return {
            'total_case_score': self.total_case_score, 'passed_case_score': self.passed_case_score, 'total_case_num':
            self.total_case_num, 'passed_case_num': self.passed_case_num, 'score_summary': self.score_summary
        }


def gen_print_result(str_list):
    max_length = max([len(x) for x in str_list])
    str_end = (f"================="
               f"{'':=^{max_length}}"
               f"==================")
    line_list = []
    for line in str_list:
        fmt_line = (f"================="
                    f"{line:=^{max_length}}"
                    f"==================")
        line_list.append(fmt_line)
    line_list.append(str_end)
    return line_list
