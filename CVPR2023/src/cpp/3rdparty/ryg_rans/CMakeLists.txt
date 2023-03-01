# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

configure_file(CMakeLists.txt.in ryg_rans-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ryg_rans-download )
if(result)
    message(FATAL_ERROR "CMake step for ryg_rans failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ryg_rans-download )
if(result)
    message(FATAL_ERROR "Build step for ryg_rans failed: ${result}")
endif()

# add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/ryg_rans-src/
#                  ${CMAKE_CURRENT_BINARY_DIR}/ryg_rans-build
#                  EXCLUDE_FROM_ALL)

set(RYG_RANS_INCLUDE 
    ${CMAKE_CURRENT_BINARY_DIR}/ryg_rans-src/ 
    CACHE INTERNAL "")
