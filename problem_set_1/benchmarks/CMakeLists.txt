cmake_minimum_required(VERSION 2.8)
project(permve-ntnu-tdt4200-problem-set-1-benchmarks)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(log_base_two log_base_two.cpp)
target_link_libraries(log_base_two pthread)

add_executable(log_base_e log_base_e.cpp)
target_link_libraries(log_base_e pthread)

add_executable(sum_inverse_log_skip_multiples_of_two sum_inverse_log_skip_multiples_of_two.cpp)
target_link_libraries(sum_inverse_log_skip_multiples_of_two pthread)

