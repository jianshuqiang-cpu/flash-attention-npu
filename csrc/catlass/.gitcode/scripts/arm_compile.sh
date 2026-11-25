#!/bin/bash
set -ex

cd "${WORKSPACE}"
pip3 install ml_dtypes expecttest pybind11-stubgen pytest pytest-xdist
source /home/jenkins/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ASCEND_HOME_PATH}/$(uname -i)-linux/devlib
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH

bash -x tests/test_compile.sh
ret=$?
if [ $ret -ne 0 ]; then
    echo "compile catlass fail"
    exit 1
fi

exit $ret