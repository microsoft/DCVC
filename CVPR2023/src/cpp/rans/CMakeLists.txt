# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.7)
set(PROJECT_NAME Rans)
project(${PROJECT_NAME})

set(rans_source
    rans.h
    rans.cpp
    )

set(include_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RYG_RANS_INCLUDE}
    )

if (NOT MSVC)
    add_compile_options(-fPIC)
endif()
add_library (${PROJECT_NAME} ${rans_source})
target_include_directories (${PROJECT_NAME} PUBLIC ${include_dirs})
