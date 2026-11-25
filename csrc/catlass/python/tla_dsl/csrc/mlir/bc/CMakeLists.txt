# Global properties used as file lists (modifiable across nested scopes).
set_property(GLOBAL PROPERTY c310_aic_op_list)
set_property(GLOBAL PROPERTY c310_aiv_op_list)
set_property(GLOBAL PROPERTY c310_mix_aic_op_list)
set_property(GLOBAL PROPERTY c310_mix_aiv_op_list)
set_property(GLOBAL PROPERTY all_host_op_list)
set_property(GLOBAL PROPERTY all_regbaseM300_op_list)
set_property(GLOBAL PROPERTY all_linked_bitcode_files)


# Map cce arch → numerical CATLASS arch (e.g. c310 → 3510).
function(get_catlass_arch cce_arch out_catlass_arch)
    if(cce_arch STREQUAL "c310")
        set(${out_catlass_arch} 3510 PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Unsupported cce arch: ${cce_arch}")
    endif()
endfunction()

# Map (cce_arch, core_type) → cce aicore arch flag value.
#   cce_arch:      "c310", ...
#   core_type:     "aic", "aiv", "mix_aic", "mix_aiv"
#   → aicore_arch: "dav-c310-cube", "dav-c310-vec", ...
function(get_cce_arch cce_arch core_type out_aicore_arch)
    if(cce_arch STREQUAL "c310")
        set(_cce c310)
    else()
        message(FATAL_ERROR "Unsupported cce arch: ${cce_arch}")
    endif()
    if(core_type MATCHES "^(aic|mix_aic)$")
        set(_core cube)
    elseif(core_type MATCHES "^(aiv|mix_aiv)$")
        set(_core vec)
    else()
        message(FATAL_ERROR "Unsupported core type ${core_type} for cce arch ${cce_arch}")
    endif()
    set(${out_aicore_arch} dav-${_cce}-${_core} PARENT_SCOPE)
endfunction()

# Compile a single .cpp to .bc.
#   arch:  cce arch (e.g. "c310"), or "DEBUG" for CPU trace mode.
function(compile_single_cpp_to_bc cpp_with_path core_type arch)
    string(REGEX MATCHALL "[0-9A-Za-z_]*.cpp" cpp_name ${cpp_with_path})

    set(is_debug FALSE)
    if(arch STREQUAL "DEBUG")
        set(is_debug TRUE)
    endif()

    # --- output naming and property registration ---
    if(core_type MATCHES "^(aic|aiv|mix_aic|mix_aiv)$")
        if(is_debug)
            string(REGEX REPLACE ".cpp" ".${core_type}.bc" bc_name ${cpp_name})
        else()
            string(REGEX REPLACE ".cpp" ".${core_type}.${arch}.bc" bc_name ${cpp_name})
            if(NOT cpp_with_path MATCHES ".*/print_tensor\\.cpp$")
                set_property(GLOBAL APPEND PROPERTY ${arch}_${core_type}_op_list
                    "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")
            endif()
        endif()
    elseif(core_type STREQUAL "host")
        string(REGEX REPLACE ".cpp" ".host.bc" bc_name ${cpp_name})
        set_property(GLOBAL APPEND PROPERTY all_host_op_list
            "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")
    elseif(core_type STREQUAL "regbaseM300")
        string(REGEX REPLACE ".cpp" ".regbaseM300.bc" bc_name ${cpp_name})
        set_property(GLOBAL APPEND PROPERTY all_regbaseM300_op_list
            "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")
    else()
        message(FATAL_ERROR "Unsupported core type: ${core_type}")
    endif()

    # --- compilation flags ---
    set(bisheng_args "-O2")
    if(NOT is_debug)
        list(APPEND bisheng_args -x cce)
        list(APPEND bisheng_args --cce-auto-sync=off)
        list(APPEND bisheng_args --cce-aicore-only)
        list(APPEND bisheng_args --cce-generic-addrspace=off)
    endif()
    list(APPEND bisheng_args ${cpp_with_path})
    list(APPEND bisheng_args -emit-llvm -c)
    list(APPEND bisheng_args -mllvm -disable-llvm-optzns)

    if(is_debug)
        set_source_files_properties(${cpp_with_path} PROPERTIES LANGUAGE c++)
        list(APPEND bisheng_args -DENABLE_CPU_TRACE_INTRINSIC)
    else()
        get_catlass_arch(${arch} _catlass_arch)
        list(APPEND bisheng_args -DCATLASS_ARCH=${_catlass_arch})
        list(APPEND bisheng_args -DTILING_KEY_VAR)
        list(APPEND bisheng_args -Wno-ignored-attributes)
        list(APPEND bisheng_args -I${ASCEND_HOME_PATH}/include)
        list(APPEND bisheng_args -I${ASCEND_HOME_PATH}/compiler/tikcpp)
        list(APPEND bisheng_args -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw)
        list(APPEND bisheng_args -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl)
        list(APPEND bisheng_args -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface)
        list(APPEND bisheng_args -I${CATLASS_DSL_CATLASS_INCLUDE_DIR})
        if(core_type MATCHES "^mix_")
            list(APPEND bisheng_args -cce-enable-mix)
            list(APPEND bisheng_args -mllvm -enable-mix=true)
        endif()
        if(core_type MATCHES "^(aic|aiv|mix_aic|mix_aiv)$")
            get_cce_arch(${arch} ${core_type} _aicore_arch)
            list(APPEND bisheng_args --cce-enable-print)
            if(cpp_with_path MATCHES ".*/print_tensor\\.cpp$")
                list(APPEND bisheng_args -DASCENDC_DUMP)
                list(APPEND bisheng_args -DONE_CORE_DUMP_SIZE=1048576)
            endif()
            list(APPEND bisheng_args --cce-aicore-arch=${_aicore_arch})
        elseif(core_type STREQUAL "regbaseM300")
            list(APPEND bisheng_args --cce-aicore-arch=dav-m300)
            string(FIND "${cpp_with_path}" "Debug.cpp" IS_PRINT)
            if(IS_PRINT GREATER_EQUAL 0)
                list(APPEND bisheng_args --cce-enable-print)
            endif()
        endif()
    endif()

    list(APPEND bisheng_args -std=c++17)
    list(APPEND bisheng_args -I ${INCLUDE_PATH})
    list(APPEND bisheng_args -o "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bc_name}
        COMMAND ${BISHENG_TOOL} ${bisheng_args}
        COMMAND_EXPAND_LISTS
        DEPENDS ${cpp_with_path}
    )
    set(local_deps ${local_deps} ${CMAKE_CURRENT_BINARY_DIR}/${bc_name} PARENT_SCOPE)
endfunction()

# Compile all .cpp files in CMAKE_CURRENT_SOURCE_DIR for the given core_type.
#   cce_arch: cce arch identifier (e.g. c310).
function(compile_cpp_to_bc core_type cce_arch)
    set(local_deps "")
    get_catlass_arch(${cce_arch} _arch)
    file(GLOB cpp_list "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
    foreach(cpp_with_path ${cpp_list})
        if(ENABLE_CPU_TRACE_INTRINSIC AND NOT core_type MATCHES "^mix_")
            compile_single_cpp_to_bc(${cpp_with_path} ${core_type} "DEBUG")
        else()
            compile_single_cpp_to_bc(${cpp_with_path} ${core_type} ${cce_arch})
        endif()
    endforeach()

    string(REGEX REPLACE "^${CMAKE_BINARY_DIR}/?" "" local_target_name
        "${CMAKE_CURRENT_BINARY_DIR}")
    string(REGEX REPLACE "[./ ]" "_" local_target_name
        "${local_target_name}_${core_type}_${_arch}_bitcode")
    add_custom_target(${local_target_name} DEPENDS ${local_deps})
    add_dependencies(bishengir_template_bitcode ${local_target_name})
endfunction()

# Macro: links a GLOBAL property list into a single .bc output.
macro(add_custom_llvm_link_command proplist outfile)
    get_property(tmp GLOBAL PROPERTY ${proplist})
    if(tmp)
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${outfile}
            COMMAND ${LLVM_LINK_TOOL} ${tmp}
            -o ${CMAKE_CURRENT_BINARY_DIR}/${outfile}
            COMMAND_EXPAND_LISTS
            DEPENDS ${tmp}
        )
        list(APPEND local_deps ${CMAKE_CURRENT_BINARY_DIR}/${outfile})
        set_property(GLOBAL APPEND PROPERTY all_linked_bitcode_files
            "${CMAKE_CURRENT_BINARY_DIR}/${outfile}")
    endif()
endmacro()

function(link_one_to_meta_op arch)
    set(local_deps "")
    foreach(type IN ITEMS aic aiv mix_aic mix_aiv)
        add_custom_llvm_link_command(${arch}_${type}_op_list "meta_op.${type}.${arch}.bc")
    endforeach()
    set(local_target_name "link_meta_op_${arch}_bitcode")
    add_custom_target(${local_target_name} DEPENDS ${local_deps})
    add_dependencies(bishengir_template_bitcode ${local_target_name})
endfunction()

function(link_all_to_meta_op)
    link_one_to_meta_op("c310")
    set(local_deps "")
    add_custom_llvm_link_command(all_host_op_list "host.bc")
    add_custom_llvm_link_command(all_regbaseM300_op_list "regbaseM300.bc")
    set(local_target_name "link_all_to_meta_op_bitcode")
    add_custom_target(${local_target_name} DEPENDS ${local_deps})
    add_dependencies(bishengir_template_bitcode ${local_target_name})
endfunction()

# --- Main ---
if(BISHENGIR_BUILD_TEMPLATE STREQUAL "ON")
    message("-- BISHENGIR_BUILD_TEMPLATE Value: ${BISHENGIR_BUILD_TEMPLATE}")
    if(NOT BISHENG_COMPILER_PATH)
        set(BISHENG_COMPILER_PATH "${ASCEND_HOME_PATH}/bin")
    endif()
    message("-- BISHENG_COMPILER_PATH Value: ${BISHENG_COMPILER_PATH}")

    if(NOT EXISTS "${CATLASS_DSL_CATLASS_INCLUDE_DIR}")
        message(FATAL_ERROR
            "Catlass headers are missing. Expected include tree at ${CATLASS_DSL_CATLASS_INCLUDE_DIR}.")
    endif()

    set(INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include")
    set(BISHENG_TOOL "${BISHENG_COMPILER_PATH}/ccec")
    set(LLVM_LINK_TOOL "${BISHENG_COMPILER_PATH}/llvm-link")
    if(NOT EXISTS "${BISHENG_TOOL}" OR NOT EXISTS "${LLVM_LINK_TOOL}")
        message(FATAL_ERROR "Cannot find ccec or llvm-link in ${BISHENG_COMPILER_PATH}")
    endif()

    add_custom_target(bishengir_template_bitcode ALL)
    add_subdirectory(Cube)
    add_subdirectory(Vector)

    link_all_to_meta_op()
endif()
