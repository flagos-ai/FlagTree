inductor_skip_list=(
  "test_check_accuracy.py"
  "test_debug_msg.py"
  "test_embedding.py"
  "test_force_fallback.py"
  "test_foreach_add.py"
  "test_geometric.py"
  "test_lazy_register.py"
)

TEST_inductor="${WORKSPACE}/ascend/examples/inductor_cases"
# 定义统计文件路径
SUMMARY_FILE="${WORKSPACE}/ascend/examples/summary.txt"

# install daily torch_npu
current_date=$(date +%Y%m%d)
wget https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v2.6.0/${current_date}.3/pytorch_v2.6.0_py311.tar.gz
tar -zxvf pytorch_v2.6.0_py311.tar.gz
pip install *.dev${current_date}-cp311-cp311-manylinux_2_28_aarch64.whl

# remove inductor and triton cache
if [ -d /tmp/torchinductor_* ];then
  rm -rf /tmp/torchinductor_*
fi

if [ -d ~/.triton/dump ];then
  rm -rf ~/.triton/dump
fi

if [ -d ~/.triton/cache ];then
  rm -rf ~/.triton/cache
fi

cd ${TEST_inductor}
git init
git remote add origin http://gitee.com/ascend/pytorch.git
git config core.sparsecheckout true
echo "test/_inductor" >> .git/info/sparse-checkout
git pull origin v2.6.0:master
TEST_inductor_cases_path="${TEST_inductor}/test/_inductor"
cd ${TEST_inductor_cases_path}
export PYTHONPATH="${PYTHONPATH}:${TEST_inductor_cases_path}"

# 记录跳过的测试用例
echo -e "\n======= Inductor 测试跳过的用例 =======" >> $SUMMARY_FILE
for skip_case in ${inductor_skip_list[@]};
do
  if [ -e "${TEST_inductor_cases_path}/${skip_case}" ];then
    echo "跳过测试用例: ${skip_case}" | tee -a $SUMMARY_FILE
    mv ${skip_case} "${skip_case}_skip"
  fi
done

# 创建临时日志目录
LOG_DIR=$(mktemp -d)
INDUCTOR_CASE_LOG_FILE="$LOG_DIR/test_inductor_case_$(date +%Y%m%d).log"

# 记录测试开始时间
start_time=$(date +"%Y-%m-%d %H:%M:%S")

# 执行测试并生成JUnit报告
pytest -n 16 --dist=loadfile . \
  --junitxml="$LOG_DIR/results.xml" \
  2>&1 | tee "$INDUCTOR_CASE_LOG_FILE"

# 解析统计信息
# 使用Python解析JUnit XML报告
python3 -c "
import xml.etree.ElementTree as ET
import os

xml_file = '$LOG_DIR/results.xml'
if not os.path.exists(xml_file):
    print('JUnitXML report not found:', xml_file)
    exit(1)

tree = ET.parse(xml_file)
root = tree.getroot()

total_tests = 0
passed_tests = 0
failed_tests = 0
skipped_tests = 0
error_tests = 0

# 遍历所有testsuite
for testsuite in root.findall('testsuite'):
    total_tests += int(testsuite.get('tests', 0))
    skipped_tests += int(testsuite.get('skipped', 0))
    error_tests += int(testsuite.get('errors', 0))
    failed_tests += int(testsuite.get('failures', 0))

# 计算通过用例数
passed_tests = total_tests - skipped_tests - error_tests - failed_tests

# 输出统计信息
print(f'total_tests={total_tests}')
print(f'passed_tests={passed_tests}')
print(f'failed_tests={failed_tests}')
print(f'skipped_tests={skipped_tests}')
print(f'error_tests={error_tests}')
" > $LOG_DIR/stats.tmp

# 加载统计结果
source $LOG_DIR/stats.tmp
rm $LOG_DIR/stats.tmp

# 计算测试持续时间
end_time=$(date +"%Y-%m-%d %H:%M:%S")
duration=$(( $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) ))
duration_str=$(printf "%02dh %02dm %02ds" $((duration/3600)) $(((duration%3600)/60)) $((duration%60)))

# 计算通过率
if [ $total_tests -gt 0 ]; then
    pass_rate=$(( 100 * passed_tests / total_tests ))
else
    pass_rate=0
fi

# 生成统计信息摘要
stats_summary="
inductor 测试用例结果摘要:
------------------------
开始时间:       $start_time
结束时间:       $end_time
总耗时:         $duration_str
------------------------
总用例数:       $total_tests
成功用例:       $passed_tests
失败用例:       $failed_tests
跳过用例:       $skipped_tests
错误用例:       $error_tests
------------------------
通过率:         ${pass_rate}% (成功/总数)
并行度:         16个进程
------------------------
"

# 输出统计信息到控制台
echo "$stats_summary"

# 追加统计信息到summary.txt
echo "$stats_summary" >> $SUMMARY_FILE

# 保存原始日志文件
cp "$INDUCTOR_CASE_LOG_FILE" "/home/daily_log/"

# 清理临时文件
rm -rf "$LOG_DIR"

echo "测试统计信息已追加到: $SUMMARY_FILE"