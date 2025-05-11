# CUDA L-System C++ Debug

## Project Structure

```
.
├── src/
│   ├── main.cpp
│   ├── lsystem/
│   ├── optimization/
│   └── utils/
├── data/
├── build/ (for out-of-source builds)
├── scripts/
├── CMakeLists.txt
├── Makefile
├── README.md
└── .gitignore
```

- **src/lsystem/**: L-system logic, types, and parsing
- **src/optimization/**: Evolutionary algorithms, grid search, distributions, memory pool, KL divergence
- **src/utils/**: Utility functions
- **data/**: Experiment result files (CSV)
- **build/**: Build artifacts (recommended for out-of-source builds)
- **scripts/**: Utility scripts (e.g., cleaning build files)

## Building

Recommended: use out-of-source builds:

```sh
mkdir -p build
cd build
cmake ..
make
```

## Cleaning Build Files

Use the provided script:

```sh
bash scripts/clean_build.sh
```
