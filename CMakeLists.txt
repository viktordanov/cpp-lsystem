cmake_minimum_required(VERSION 3.28)
project(cuda_lsystem_cpp_debug)

set(CMAKE_CXX_STANDARD 20)

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
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")
