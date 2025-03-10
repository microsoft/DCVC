# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.7)
set(PROJECT_NAME MLCodec_CXX)
project(${PROJECT_NAME})

set(cxx_source
    ops.cpp
    )

set(include_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYBIND11_INCLUDE}
    )

pybind11_add_module(${PROJECT_NAME} ${cxx_source})

target_include_directories (${PROJECT_NAME} PUBLIC ${include_dirs})

# The post build argument is executed after make!
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND
            "${CMAKE_COMMAND}" -E copy
            "$<TARGET_FILE:${PROJECT_NAME}>"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../models/"
)
