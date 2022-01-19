cmake_minimum_required(VERSION 3.18.0)
project(ncu-test VERSION 1.0.0 LANGUAGES C CXX CUDA)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)


find_package(CUDA)
add_compile_options(-lineinfo)
add_compile_options(-keep)
include(Apps)

add_subdirectory(test_cases)