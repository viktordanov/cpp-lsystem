cmake_minimum_required(VERSION 3.28)
project(cuda_lsystem_cpp_debug)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure OpenMP
if(APPLE)
    # Use Homebrew's LLVM/Clang
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
    set(OpenMP_C "${CMAKE_C_COMPILER}")
    set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
    set(OpenMP_C_FLAGS "-fopenmp")
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()


# Find OpenMP package
find_package(OpenMP REQUIRED)

# Add the executable
add_executable(cuda_lsystem_cpp_debug
    distributions.h
    evol_operators.h
    evolution.h
    grid_search.h
    kl_divergence.h
    lsystem.h
    mem_pool.h
    parse.h
    strings.h
    types.h
    distributions.cpp
    lsystem.cpp
    main.cpp
    types.cpp
)

# Set compiler flags
target_compile_options(cuda_lsystem_cpp_debug PRIVATE -O3 ${OpenMP_CXX_FLAGS})

# Include directories
target_include_directories(cuda_lsystem_cpp_debug PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    /opt/homebrew/opt/libomp/include
)

# Link libraries
target_link_libraries(cuda_lsystem_cpp_debug PRIVATE 
    OpenMP::OpenMP_CXX
)

