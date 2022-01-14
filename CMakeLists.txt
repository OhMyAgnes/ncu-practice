cmake_minimum_required(VERSION 3.18.0)
project(ncu-test VERSION 1.0.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

find_package(CUDA)

add_subdirectory(case1)