cmake_minimum_required(VERSION 3.28)
project(cuda_lsystem_cpp_debug)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure OpenMP
if(APPLE)
    # Use Homebrew's LLVM/Clang
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang" CACHE STRING "" FORCE)
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++" CACHE STRING "" FORCE)
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
    src/main.cpp
    src/lsystem/lsystem.cpp
    src/lsystem/types.cpp
    src/optimization/distributions.cpp
    src/lsystem/lsystem.h
    src/lsystem/types.h
    src/lsystem/parse.h
    src/optimization/distributions.h
    src/optimization/evol_operators.h
    src/optimization/evolution.h
    src/optimization/grid_search.h
    src/optimization/kl_divergence.h
    src/optimization/mem_pool.h
    src/utils/strings.h
)

# Set compiler flags
target_compile_options(cuda_lsystem_cpp_debug PRIVATE -O3 ${OpenMP_CXX_FLAGS})

# Include directories
target_include_directories(cuda_lsystem_cpp_debug PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lsystem
    ${CMAKE_CURRENT_SOURCE_DIR}/src/optimization
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    /opt/homebrew/opt/libomp/include
)

# Link libraries
target_link_libraries(cuda_lsystem_cpp_debug PRIVATE 
    OpenMP::OpenMP_CXX
)

