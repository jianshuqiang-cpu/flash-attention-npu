# Gemm/Kernel Class Template Overview

## API List

| Component                        | Description|
| :----------------------------------------------------------- | :------: |
| [basic_matmul](./basic_matmul.md)     |   Basic matrix multiplication of the Common template |
| [basic_matmul_tla_visitor](./basic_matmul_tla_visitor.md)     |   EVG GM workspace kernel entry point |
| [basic_matmul_tla_ub_visitor](./basic_matmul_tla_ub_visitor.md)     |   EVG UB workspace kernel entry point |

## API Breakdown

The namespace is `Catlass::Gemm::Kernel`. The class template contains the following core members.

| Type        | Name|Function|
| :----------------------------------------------------------- | :------: |------: |
| struct    |   Params  |Input arguments used when invoking the device kernel function through `<<< >>>`.|
| struct    |   Arguments  |Input arguments encapsulated after device execution|
| Static function   |   bool CanImplement  |`Arguments` validation interface|
| Static function   |   size_t GetWorkspaceSize  |Calculates the required workspace size based on `Arguments`|
| Static function   |   Params ToUnderlyingArguments  |Converts `Arguments` into kernel arguments (`Params`)|
| Function   |   void operator()\<AscendC::AIC\>  |Inputs `Params` and executes MMAD computations on the AIC|
| Function   |   void operator()\<AscendC::AIV\>  |Inputs `Params` and executes calculation on the AIV, such as prologue and epilogue|
