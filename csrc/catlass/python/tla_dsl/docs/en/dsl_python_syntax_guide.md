# Catlass DSL Syntax Constraints

## 1. Basic Concepts

### 1.1 @tla.kernel and @tla.jit

Functions decorated with `@tla.kernel` or `@tla.jit` are translated by the frontend into device code; other ordinary Python functions are not translated and run as plain Python at compile time. A module-level function can be called from a dynamic region, but its body must not contain device control flow (such as `tla.range`); otherwise no device instructions are generated as expected.

The two decorators differ only in role:

- `@tla.kernel`: device kernel entry point. Calling it returns a launcher that runs on the device.
- `@tla.jit`: a sub-function callable from a kernel.

Both go through the same frontend transformation and lowering path, so the syntax constraints in this document apply equally to both.

### 1.2 Host Functions and Compile-Time Evaluation

Host functions (ordinary module-level Python functions and builtins not explicitly handled by the frontend) can be written inside a kernel, but they do not become part of the compiled computation: they generate no device instructions and are evaluated as plain Python at lowering time. If the argument is a compile-time constant, the result is a compile-time constant and can be used as-is (const evaluation — for example `for i in range(4)` is unrolled at lowering time); if the argument is device data, the result has no device semantics. To declare a compile-time constant parameter, annotate it with `tla.Constexpr[T]` (e.g. `reverse: tla.Constexpr[bool]`); `tla.const_expr(...)` marks a condition as compile-time evaluable.

Support for builtins and Python modules can be viewed as a whitelist: only the builtins listed in the table below enter kernel compilation; the rest do not and are evaluated at lowering time under the host-function rules above. The APIs under the `tla` namespace are device operations and are not part of this list. Builtins require no `import`; see the [official Python builtins documentation](https://docs.python.org/3/library/functions.html#import__) for the complete list and semantics.

| Builtins that enter compilation | Support | How they enter compilation | Reference |
| --- | --- | --- | --- |
| `any()`/`all()`/`bool()` | Partially supported (single positional argument, no keywords) | Redirected to device implementations | `ast_preprocessor.py:35-41`, `tla_ast_decorators.py:533-550` |
| `min()`/`max()` | Partially supported (no keywords) | Redirected to device implementations | `ast_preprocessor.py:35-41`, `tla_ast_decorators.py:559-567` |
| `abs(x)` | Supported | Via `__abs__`, produces device operations | `typing.py:804-821` |
| `pow(x, y)` | Partially supported (float only, same as `**`) | Via `__pow__`, produces device operations | `typing.py:861` |
| `range()` | Partially supported (bounds must be compile-time constants) | Drives compile-time loop unrolling | `ast_preprocessor.py:305-474` |

**Example: compile-time constant expansion of a host function and the builtin `range`**

Original code:

```python
def host_square(x: int) -> int:
    return x * x

@tla.kernel
def const_expand_kernel() -> None:
    n = host_square(3)      # literal argument, evaluated to 9 at compile time
    for i in range(n):      # builtin range, unrolled at compile time
        tla.make_coord(i, 0)
```

After compile-time expansion:

```python
@tla.kernel
def const_expand_kernel() -> None:
    tla.make_coord(0, 0)
    tla.make_coord(1, 0)
    tla.make_coord(2, 0)
    tla.make_coord(3, 0)
    tla.make_coord(4, 0)
    tla.make_coord(5, 0)
    tla.make_coord(6, 0)
    tla.make_coord(7, 0)
    tla.make_coord(8, 0)
```

### 1.3 Expressions and Operators at a Glance

| Operator | Support |
| --- | --- |
| `+` `-` `*` `/` `//` `%`, unary minus `-x` | Supported |
| Bitwise `& \| ^ ~ << >>` | Supported |
| Comparisons `== != < <= > >=` | Supported |
| `is`/`is not`/`in`/`not in` | Partially supported (host values only) |
| `and or not` (short-circuit), conditional expression `x if c else y` | Supported |
| `**` (power) | Partially supported (float types only) |
| Subscript read `meta[i]` | Supported |

### 1.4 Dynamic vs. Static

Whether a value, branch, or loop is dynamic or static depends on whether it relies on device data that is known only at runtime.

| Category | Criterion | Examples |
| --- | --- | --- |
| Static (known at compile time) | Determined at compile time | Literals, `tla.Constexpr[T]` parameters, values of `tla.const_expr(...)` |
| Dynamic (known only at runtime) | Depends on device data | Tensor elements, `tla.arch.block_idx()`, dynamic-loop induction variables and their results |

This yields two kinds of control flow:

- **Dynamic branches/loops**: when the `if` condition, `for` bound, or `while` condition depends on device data, the code is compiled into device branch/loop instructions that execute on the device and may take different paths on each run; they are subject to the dynamic-region constraints (no early exit, carried state must keep a consistent structure, etc.).
- **Compile-time branches/unrolling**: when the condition or bound is known at compile time (`if True:`, `if tla.const_expr(...)`, `for i in range(n)` with a constant bound), the branch is selected or the loop body is unrolled at compile time, producing no device branch/loop instructions; they are not subject to the dynamic-region constraints and can participate in compile-time constant expansion.

Static vs. dynamic loops: use `range`/`range_constexpr` when the bound is known at compile time (compile-time unrolling); use `tla.range` when the bound depends on device data (dynamic loop).

### 1.5 Glossary

| Term | Meaning |
| --- | --- |
| Compile time | The stage in which the frontend converts a kernel into device instructions (the lowering stage, inside the host Python process) |
| Runtime | The stage in which the kernel actually executes on the device |
| Device data (dynamic value) | Values determined at runtime: tensors, `tla.arch.block_idx()`, loop induction variables and their results |
| Compile-time constant (static value) | Values determined at compile time: literals, `tla.Constexpr` parameters, `tla.const_expr(...)` |
| Carried variable | A local variable passed between branches or loops and maintained automatically by the frontend |
| Dynamic region | An `if`/`for`/`while` block whose condition or bound depends on device data |

## 2. Control-Flow Constraints

### 2.1 The Three Kinds of for

| Form | Semantics | When to use |
| --- | --- | --- |
| `for i in tla.range(...)` | Dynamic loop, compiled into device loop instructions | Bound is device data determined at runtime |
| `for i in range(...)` | Compile-time unrolling (host iteration) | Bound is a compile-time constant |
| `for i in tla.range_constexpr(...)` | Compile-time unrolling (explicit) | Same as above |

Constraints: the loop target must be a simple local variable name; the induction variable cannot be used after the loop; `tla.range` supports the `start`/`stop`/`step` three-argument form and negative steps.

### 2.2 Dynamic if and while

- When the condition depends on device data, `if`/`while` is compiled into device branches/loops; both branches, the loop body, and the condition region generate device code.
- Assignment targets inside a dynamic if/while body are restricted, by location:

| Statement | Inside dynamic if/while body | Inside dynamic for body |
| --- | --- | --- |
| Local-name assignment `x = v`, augmented assignment `x += 1` | Supported | Supported |
| Tuple/list unpacking `(a, b) = v` | Supported | Supported |
| Tensor subscript write `out[i] = v` | Not supported | Supported |
| Attribute assignment `obj.attr = v` | Not supported | Not supported |
| `del` | Not supported | Not supported |
| Calling a local function/closure defined inside the kernel | Not supported | Not supported |

State changes are done by rebinding local names; do not rely on mutating object attributes or container elements.

Note: the frontend does not validate attribute assignment, `del`, etc. inside a dynamic for body — the lack of a diagnostic does not mean it is supported.

### 2.3 Early Exit and else

`return`/`break`/`continue`/`raise` are not supported inside dynamic regions, and dynamic `for`/`while` do not support an `else` clause. Device loops have no early-exit instructions or call stack; to exit early, skip with a condition or record the result in a flag variable and read it after the loop.

### 2.4 Variable Scope and Carried State

- Variables used outside a region must be initialized before entering it.
- An induction variable cannot be used after the loop.
- Variables passed between branches or loops (carried variables) must keep the same structure and leaf types on every path; supported containers are tuple/list/dict/dataclass, as well as custom classes implementing `__extract_mlir_values__`/`__new_from_mlir_values__`.

### 2.5 Compile-Time Branches

When the condition is a literal `True`/`False` or `tla.const_expr(...)`, the `if` selects a branch at compile time and generates no device branch; both sides are still checked. Combine with `tla.Constexpr[T]` parameters for compile-time dispatch.

### 2.6 with Regions

The Huawei NPU has two compute regions; device operations must be placed in their corresponding region (entered with `with`):

- **Cube region**: hosts matrix operations (e.g. `tla.mmad`), entered with `with tla.cube(...)`.
- **Vector region**: hosts element-wise/vector operations, entered with `with tla.vector(...)`. Vector computation inside the region runs as a **Vector Function (VF)**; use `tla.vec.func(mode="simd")` to wrap the SIMD VF sub-function. See the [HiAscend documentation: Reg Vector Computation Overview](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/docs/api/SIMD-API/%E5%9F%BA%E7%A1%80API/Reg%E7%9F%A2%E9%87%8F%E8%AE%A1%E7%AE%97/%E6%A6%82%E8%BF%B0.md) for details.

Constraints:

- A `with` allows only one context manager; with multiple context managers there is no error, but the region is not generated.
- Temporary values defined inside the region cannot be used outside it.
- Rebinding of enclosing variables inside the region is handled automatically by the frontend (generating `nonlocal`); do not write it by hand.

## 3. Keywords at a Glance

| Keyword | Support | Notes |
| --- | --- | --- |
| `pass` | Supported | Emits nothing |
| `if`/`elif`/`else` | Supported | Dynamic branch or compile-time branch |
| `for`/`while` | Supported | Dynamic loop or compile-time unrolling |
| `with` | Partially supported (only `tla.cube`/`tla.vector`/`tla.vec.func`) | Device region; one context manager per with |
| `def` | Partially supported (root functions and module-level functions only) | Local functions defined inside a kernel cannot be called from a dynamic region |
| `lambda` | Partially supported (host evaluation only) | Evaluated as plain Python |
| `return`/`break`/`continue`/`raise` | Not supported | No early exit in dynamic regions |
| `del` | Partially supported (outside dynamic if/while bodies) | Assignment-target check |
| `global`/`nonlocal` | Not supported (hand-written) | `nonlocal` is generated by the frontend |
| `import` | Not supported | Loaded at compile time, not executed on the device |
| `class` | Not supported | Nested definitions |
| `assert`/`try`/`except`/`finally`/`match`/`case` | Not supported | No compile-time error, but the result cannot be relied on |
| `async`/`await` | Not supported | — |
| `yield`/`yield from` | Not supported | — |

## 4. Examples

The examples below are trimmed from `examples/end_to_end`; see the corresponding files for complete runnable versions.

### 4.1 Dynamic if: initialize before rebinding in the branch

Dynamic `if` branches are converted into standalone functions, so a variable must be initialized first and then rebound inside the branch.

```python
@tla.kernel
def dynamic_if_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i                     # initialize first
        if i == 0:
            coord = i + 1             # rebind inside the branch
        else:
            coord = i + 2             # the other branch assigns too
        tla.make_coord(coord, 0)      # used after the branch
```

Incorrect:

```python
    for i in tla.range(0, limit, 1):
        if i == 0:
            coord = i + 1             # defined only inside the branch
        tla.make_coord(coord, 0)      # SyntaxError: ... must be initialized before the if
```

### 4.2 Dynamic if: passing values between branches

Each branch assigns different values to the same set of variables, which are used after the branch.

```python
@tla.kernel
def select_kernel(limit: int) -> None:
    for i in tla.range(0, limit, 1):
        coord = i                     # initialize first
        offset = i + 1
        if i == 0:
            coord = i + 2             # branches assign different values
            offset = i + 3
        else:
            coord = i + 4
            offset = i + 5
        tla.make_coord(coord, offset) # used after the branch
```

### 4.3 Dynamic for: carrying state across iterations

Dynamic loops do not support early exit; variables rebound inside the loop are passed across iterations automatically, and the type must stay the same on every assignment.

```python
@tla.kernel
def carried_state_kernel(mem_src: tla.Tensor, mem_out: tla.Tensor) -> None:
    with tla.vector():
        src_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        out_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        src_ub = tla.make_tensor_like(src_ptr, mem_src, tla.arch.RowMajor)
        out_ub = tla.make_tensor_like(out_ptr, mem_out, tla.arch.RowMajor)
        tla.copy(src_ub, mem_src)
        with tla.vec.func(mode="simd"):
            value = src_ub.load()     # initialize first
            for _ in tla.range(2):    # dynamic loop
                value = tla.abs(value)  # carried across iterations
            out_ub.store(value)
        tla.copy(mem_out, out_ub)
```

Incorrect:

```python
    with tla.vec.func(mode="simd"):
        for _ in tla.range(2):
            break                     # SyntaxError: ... does not support return, break, continue, or raise
```

### 4.4 Dynamic for: tiled loops

Loop over tiles, processing one tile per iteration.

```python
@tla.kernel
def tile_loop_kernel(mem: tla.Tensor, out: tla.Tensor) -> None:
    with tla.vector():
        src_ptr = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
        dst_ptr = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
        src_ub = tla.make_tensor_like(src_ptr, mem, tla.arch.RowMajor)
        dst_ub = tla.make_tensor_like(dst_ptr, out, tla.arch.RowMajor)
        tla.copy(src_ub, mem)
        with tla.vec.func(mode="simd"):
            for i in tla.range(0, 4, 1):              # tiled loop
                src_tile = tla.tile_view(src_ub, tla.make_shape(64), tla.make_coord(i))
                dst_tile = tla.tile_view(dst_ub, tla.make_shape(64), tla.make_coord(i))
                dst_tile.store(tla.abs(src_tile.load()))
        tla.copy(out, dst_ub)
```

### 4.5 Compile-time branch

When the condition is known at compile time, the branch is selected at compile time and no device branch is generated.

```python
@tla.kernel
def constexpr_if_kernel(flag: tla.Constexpr[bool]) -> None:
    if tla.const_expr(flag):          # compile-time branch
        tla.make_coord(1, 0)
    else:
        tla.make_coord(2, 0)
```

### 4.6 with regions

Device operations are placed inside a `tla.vector()` region, with vector computation wrapped in `tla.vec.func(mode="simd")`.

```python
@tla.kernel
def vec_region_kernel(mem_a: tla.Tensor, mem_b: tla.Tensor, mem_c: tla.Tensor) -> None:
    with tla.vector():
        a_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        b_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        c_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
        a_ub = tla.make_tensor_like(a_ptr, mem_a, tla.arch.RowMajor)
        b_ub = tla.make_tensor_like(b_ptr, mem_b, tla.arch.RowMajor)
        c_ub = tla.make_tensor_like(c_ptr, mem_c, tla.arch.RowMajor)
        tla.copy(a_ub, mem_a)
        tla.copy(b_ub, mem_b)
        with tla.vec.func(mode="simd"):
            c_ub.store(a_ub.load() + b_ub.load())
        tla.copy(mem_c, c_ub)
        tla.pipe_barrier(tla.pipes.ALL)
```

### 4.7 struct-like args

Class instances created with `@dataclass` can be used as kernel arguments, and can also be created inside the kernel.

```python
from __future__ import annotations
from dataclasses import dataclass
import catlass.tla as tla

@dataclass(frozen=True, kw_only=True)
class TilingData:
    TILE_M: tla.Constexpr[int]    # compile-time constant, not in the ABI; takes no argument slot in the generated IR
    tiling_gm_out: tla.Tensor     # supports tla.Tensor arguments
    tiling_int16: tla.Int16       # Int16 scalar
    tiling_float: tla.Float32     # Float32 scalar
    tiling_int: int               # python int: compile-time int, runtime tla.Int32

@dataclass(frozen=True)
class Info:
    tile_m: int
    tile_n: int

def print_tiling(tiling: TilingData):
    tla.print("tiling_int={}", tiling.tiling_int)

@tla.kernel
def struct_arg_kernel(tiling: TilingData, out: tla.Tensor) -> None:
    out[0] = tiling.tiling_int16
    ptr = tla.allocate(tiling.TILE_M, ...)
    print_tiling(tiling)    # can be passed to other functions inside the kernel, just like a plain variable
    info = Info(tile_m=tiling.tiling_int, ...)  # can be instantiated on the kernel side


tiling = TilingData(TILE_M=128, ...)
artifact = tla.compile(struct_arg_kernel, tiling, out)
artifact(tiling, out)
```

**Usage scope**:
- Create an instance on the host side and pass it as a kernel argument, fields should be numeric and tensor
- Create an instance on the kernel side

**Customizable dataclass options**: only `frozen/kw_only`; the other options cannot be changed from their defaults. With `frozen=True` field values cannot be modified after instantiation; with `kw_only=True` fields must be passed by keyword, e.g. `tiling=TilingData(TILE_M=128, tiling_int16=128, ...)`, not `tiling=TilingData(128, 128, ...)`.

**`tla.Constexpr[...]` fields**:

- Treated as compile-time constants: read-only inside the kernel; usable in `tla.allocate`, `tla.range_constexpr`, etc.
- Do not enter the kernel ABI / IR: `tla.func` has no corresponding block argument.

**Supported argument types**:

- `tla.*` scalars: `tla.Bool`, `tla.Int8/16/32/64`, `tla.UInt8/16/32/64`, `tla.Float16/32`, `tla.BFloat16`.
- Plain Python scalars: `bool`→`i1`, `int`→`i32`, `float`→`f32`.
- `tla.Tensor`.

**Note**: the actual field type is determined by the **field value** (consistent with Python's dynamic semantics); constructing the dataclass does not coerce values to the annotated type.

### 4.8 Error Reference

| Error message (keyword) | Cause | Fix |
| --- | --- | --- |
| `does not support return, break, continue, or raise` | Early exit inside a dynamic region | Skip with a condition, or record the result in a flag variable |
| `does not support for-else`/`while-else` | Dynamic loop with else | Remove else, move the logic into the loop body |
| `requires a simple local name target` | for target such as `pair[0]` | Use a single local variable name |
| `induction variables cannot be used after the loop` | Using the induction variable after the loop | Store the needed value in a variable initialized beforehand inside the loop |
| `must be initialized before the if/loop` | Variable defined only inside a branch/loop, used outside | Initialize it before the region |
| `only supports assignments to local names or tuples/lists` | Writing `out[i] =`, `obj.attr =` etc. inside an if/while body | Move the store out of the if/while body, or rebind a local name |
| `calling active local callable` | Calling a local function inside a dynamic region | Inline it, or move it to module level |
| `structure`/`expected i32` (TlaCoreAPIError) | Carried state structure or type differs across branches/iterations | Keep the structure and leaf types identical on every path |
| `'**' is only supported for float types` | Integer power | Use multiplication |

## 5. References

- Control-flow examples: `examples/end_to_end/basic_vadd/basic_vadd.py`, `examples/end_to_end/vector_ops/register_control_flow.py`
- Control-flow frontend implementation: `catlass/base_dsl/ast_preprocessor.py`, `catlass/tla_ast_decorators.py`
