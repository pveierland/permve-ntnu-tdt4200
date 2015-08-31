cmake_minimum_required(VERSION 2.8)
project(permve-ntnu-tdt4200-problem-set-1-unit-tests)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
file(GLOB unit_test_sources *.cpp)

add_executable(runTests ${unit_test_sources})
target_link_libraries(runTests ${GTEST_BOTH_LIBRARIES} pthread)

