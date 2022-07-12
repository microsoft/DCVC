# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.7)
set(PROJECT_NAME MLCodec_rans)
project(${PROJECT_NAME})

set(rans_source
    rans_interface.hpp
    rans_interface.cpp
    )

set(include_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYBIND11_INCLUDE}
    ${RYG_RANS_INCLUDE}
    )

pybind11_add_module(${PROJECT_NAME} ${rans_source})

target_include_directories (${PROJECT_NAME} PUBLIC ${include_dirs})

# The post build argument is executed after make!
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND
            "${CMAKE_COMMAND}" -E copy
            "$<TARGET_FILE:${PROJECT_NAME}>"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../entropy_models/"
)
