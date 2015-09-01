cmake_minimum_required(VERSION 2.8)
project(permve-ntnu-tdt4200-problem-set-1-benchmarks)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(log_double log_double.cpp)
target_link_libraries(log_double pthread)

add_executable(log_magic log_magic.cpp)
target_link_libraries(log_magic pthread)

