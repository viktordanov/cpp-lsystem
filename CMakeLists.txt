cmake_minimum_required(VERSION 3.28)
project(cuda_lsystem_cpp_debug)

set(CMAKE_CXX_STANDARD 20)

find_package(TBB REQUIRED)
add_executable(cuda_lsystem_cpp_debug
        main.cpp

        parse.h
        strings.h
        types.h
        lsystem.h
        distributions.h

        types.cpp
        distributions.cpp
        lsystem.cpp
        evolution.h
        evol_operators.h
        kl_divergence.h
        mem_pool.h
        grid_search.h
)
target_link_libraries(cuda_lsystem_cpp_debug PRIVATE TBB::tbb)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp -mavx")
