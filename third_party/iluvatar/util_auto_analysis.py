# -*- coding: utf-8 -*-

import os
import re
import json
from sys import argv


def check_log(file_path):
    cookie_dict = {
        'res': [],
        'Elapsed time': [],
        'max diff': [],
        'max_diff_rate': [],
    }
    res = []
    lines = open(file_path, 'r', encoding='utf-8', errors='ignore').readlines()

    for line in lines:
        if re.match(r'^\=.*\=$', line) and (line.__contains__('passed') or line.__contains__('failed')
                                            or line.__contains__('skipped')):
            cookie_dict['res'].append(line.strip().strip("="))
    res.extend(cookie_dict['res'])
    return res


if __name__ == '__main__':
    try:
        base_path = argv[1]
        fname = base_path.replace(os.sep, '_').replace('logs_', '')

        if fname.endswith('_'):
            fname = fname[:-1]
        index = 0
        res_dict = {}
        final_dict = {}
        log_files = os.listdir(base_path)
        for file_name in log_files:
            res = check_log(os.path.join(base_path, file_name))
            if len(res) == 0:
                index += 1
                res = ['other bug']
            res_dict[file_name] = res[0]

        # with open("expectPassedNum.json",'r') as load_f:
        #     expect_passed_num = json.load(load_f)

        if not os.path.exists('logs/analysis'):
            os.mkdir('logs/analysis')
        f = open('logs/analysis/' + 'analysis_' + fname + '.txt', 'w')

        files_num = len(log_files)
        passed_num = failed_num = exit_code = 0
        f.write('*' * 60 + ' Failed Test Case ' + '*' * 60 + '\n')
        for key in res_dict.keys():
            if ' failed' in str(res_dict[key]) or str(
                    res_dict[key]
            ) == 'other bug':  # or str(expect_passed_num[key]) + ' passed' not in str(res_dict[key]):
                failed_num += 1
                exit_code = 1
                f.write(key + ':' + str(res_dict[key]) + '\n')
            else:
                passed_num += 1
        f.write('*' * 60 + ' Summary Info ' + '*' * 64 + '\n')
        f.write('Total Tests: ' + str(files_num) + ',  PASSED: ' + str(passed_num) + ',  FAILED: ' + str(failed_num) +
                '\n')

        with open('logs/analysis/' + 'analysis_' + fname + '.txt', 'r') as f:
            for line in f:
                print(line, end='')

        exit(exit_code)

    except IndexError as e:
        print('Error! Must provide a log directory name')
