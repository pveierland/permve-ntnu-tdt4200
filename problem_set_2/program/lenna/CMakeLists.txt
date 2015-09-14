cmake_minimum_required(VERSION 2.8)
project(permve-ntnu-tdt4200-problem-set-2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

FILE(GLOB LENNA_SOURCES "*.cu" "*.cpp" "*.c")
CUDA_ADD_EXECUTABLE(lenna LENNA_SOURCES)


