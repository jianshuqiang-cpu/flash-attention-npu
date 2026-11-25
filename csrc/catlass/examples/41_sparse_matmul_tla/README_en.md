# SparseMatmulTla Example Readme

## Code Organization

```text
├── 41_sparse_matmul_tla
│   ├── CMakeLists.txt     #CMake build file
│   ├── README.md
│   ├── sparse_gen_data.py
│   └── sparse_matmul_tla.cpp # Main file
```

## Example

- After obtaining the code, compile the operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).

- Run `sparse_gen_data.py` to generate a test sample. The test sample needs to be input from the command line. After the command is executed, the `input` and `output` directories are generated in the specified path, including the input data of the operator and the golden data used for precision verification.
- Then, execute the operator. Note that the input shape of the operator must match the shape of the data generated in the first step. In addition, this sample supports only the `int8_t` data type for the input of matrix A or B.

The following is a complete shell script example (run in the project directory):

```text
m=160
n=320
k=64
device=0

function build() {
    bash scripts/build.sh 41_sparse_matmul_tla
}

function gen_data() {
    cd examples/41_sparse_matmul_tla
    python3 sparse_gen_data.py $m $n $k
    echo "Data gen finished"
}

function run_kernel {
    echo 'Case: m=' $m ' k=' $k ' n=' $n
    cd ../../output/bin/
    cp -r ../../examples/41_sparse_matmul_tla/input .
    cp -r ../../examples/41_sparse_matmul_tla/output .
    ./41_sparse_matmul_tla $m $n $k $device
}

build
gen_data
run_kernel
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
