# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.7)
set(PROJECT_NAME MLCodec_rans)
project(${PROJECT_NAME})

set(py_rans_source
    py_rans.h
    py_rans.cpp
    )

set(include_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYBIND11_INCLUDE}
    )

pybind11_add_module(${PROJECT_NAME} ${py_rans_source})

target_include_directories (${PROJECT_NAME} PUBLIC ${include_dirs})
target_link_libraries (${PROJECT_NAME} LINK_PUBLIC Rans)

# The post build argument is executed after make!
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND
            "${CMAKE_COMMAND}" -E copy
            "$<TARGET_FILE:${PROJECT_NAME}>"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../models/"
)
