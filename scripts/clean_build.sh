#!/bin/bash

# Clean CMake and Make build artifacts
rm -rf build/
rm -rf cmake-build-debug/
rm -rf CMakeFiles/
rm -f CMakeCache.txt
rm -f cmake_install.cmake
rm -f cuda_lsystem_cpp_debug
rm -f *.o *.obj *.exe *.out *.log

echo "Build artifacts cleaned." 