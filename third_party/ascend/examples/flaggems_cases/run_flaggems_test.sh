
TEST_flaggems="${WORKSPACE}/ascend/examples/flaggems_cases"
cd ${TEST_flaggems}
git init
git clone https://gitee.com/leopold0801/flaggems.git
cd flaggems
git checkout 4f3f548
mv ../op_test_run.sh ./
ls -al
bash op_test_run.sh tests fullop 8 32
