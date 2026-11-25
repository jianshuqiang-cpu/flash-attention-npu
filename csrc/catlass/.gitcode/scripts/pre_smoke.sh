#!/bin/bash
set -ex

pip3 install ml_dtypes expecttest pybind11-stubgen pytest pytest-xdist
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

bash -x tests/run_all_test.sh
ret=$?
if [ $ret -ne 0 ]; then
    echo "run catlass_testcase fail"
    exit 1
fi

exit $ret