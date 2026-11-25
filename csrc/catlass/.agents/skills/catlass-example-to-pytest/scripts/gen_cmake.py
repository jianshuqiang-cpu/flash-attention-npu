#!/usr/bin/env python3
"""Generate CMakeLists.txt for a JIT kernel directory.

Usage: python gen_cmake.py <nn> <name>
"""

import sys


CMAKE_TEMPLATE = """add_kernel(NAME {name}
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${{CMAKE_CURRENT_SOURCE_DIR}}/{name}.cpp
    TEMPLATE ${{CMAKE_CURRENT_SOURCE_DIR}}/{name}_impl.cpp)
"""


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <nn> <name>")
        sys.exit(1)

    nn = sys.argv[1]
    name = sys.argv[2]
    print(CMAKE_TEMPLATE.format(name=name))


if __name__ == "__main__":
    main()
