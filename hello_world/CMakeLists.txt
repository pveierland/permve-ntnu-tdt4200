cmake_minimum_required(VERSION 2.8)
project(para)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")

#SET(CMAKE C COMPILER mpicc)
#SET(CMAKE CXX COMPILER mpicxx)

find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
include_directories(${MPI_INCLUDE_PATH})

add_executable(para para.cpp)
target_link_libraries(para ${MPI_LIBRARIES})

enable_testing()
add_test(para ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${CMAKE_CURRENT_BINARY_DIR}/para)

