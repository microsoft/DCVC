# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

configure_file(CMakeLists.txt.in pybind11-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pybind11-download )
if(result)
    message(FATAL_ERROR "CMake step for pybind11 failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pybind11-download )
if(result)
    message(FATAL_ERROR "Build step for pybind11 failed: ${result}")
endif()

add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/pybind11-src/
                 ${CMAKE_CURRENT_BINARY_DIR}/pybind11-build/
                 EXCLUDE_FROM_ALL)

set(PYBIND11_INCLUDE 
    ${CMAKE_CURRENT_BINARY_DIR}/pybind11-src/include/ 
    CACHE INTERNAL "")
