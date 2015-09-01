cmake_minimum_required(VERSION 2.8)
project(permve-ntnu-tdt4200-problem-set-1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(DEFINED ENV{MPI_ROOT})
    # Workaround as FindPackage fails for MPI on Vilje
    set(CMAKE C COMPILER "$ENV{MPICC}")
    set(CMAKE_C_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} -I$ENV{MPI_ROOT}/include")
    set(CMAKE CXX COMPILER "$ENV{MPICXX}")
    set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} -I$ENV{MPI_ROOT}/include")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L$ENV{MPI_ROOT}/lib -lmpi++ -lmpi")
    # Intel library needed for _intel_fast_memset __intel_sse2_strlen etc.
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lirc")
else()
    find_package(MPI REQUIRED)
    set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
    set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
    include_directories(${MPI_INCLUDE_PATH})
endif()

set(Boost_USE_STATIC_LIBS    ON)
set(Boost_USE_MULTITHREADED  ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS mpi serialization)

include_directories(${Boost_INCLUDE_DIR})
add_executable(parallel computeMPI.cpp)
add_executable(parallel2 mach2.cpp)
target_link_libraries(parallel ${Boost_LIBRARIES} ${MPI_LIBRARIES})
target_link_libraries(parallel2 ${Boost_LIBRARIES} ${MPI_LIBRARIES})

