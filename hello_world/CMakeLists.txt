cmake_minimum_required(VERSION 2.8)
project(hello_world)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")

SET(CMAKE C COMPILER $ENV{MPICC})
SET(CMAKE_C_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} -I"$ENV{MPI_ROOT}"/include)
SET(CMAKE CXX COMPILER $ENV{MPICXX})
SET(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} -I"$ENV{MPI_ROOT}"/include)
SET(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} -L"$ENV{MPI_ROOT}"/lib -lmpi)

#find_package(MPI REQUIRED)
#set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
#set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
#include_directories(${MPI_INCLUDE_PATH})
#include_directories(/sw/sgi/mpt/mpt-2.06/include/)

add_executable(hello_world hello_world.cpp)
#target_link_libraries(hello_world ${MPI_LIBRARIES})

enable_testing()
add_test(hello_world ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${CMAKE_CURRENT_BINARY_DIR}/hello_world)

