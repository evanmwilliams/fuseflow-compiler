# The Sparse Abstract Machine for Machine Learning (SAMML)
## Prerequisites
- Initialize submodules in external directory (llvm-project, or-tools) 
```shell
git submodule update --init
```
- Install [cmake](https://github.com/Kitware/CMake?tab=readme-ov-file) or use pre-installed cmake.
- Build MLIR under external/llvm-project using instructions provided [here](https://mlir.llvm.org/getting_started/).
For Linux (Ubuntu) optimized for memory usage. May remove `--parallel 8` if you want to build single core (less memory)
```shell
cmake -G Ninja ../llvm \           
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_TARGETS_TO_BUILD="Native;" \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DLLVM_USE_SPLIT_DWARF=ON -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON && cmake --build . --target tools/mlir/test/check-mlir --parallel 8
```
- Build or-tools for C++ under external/or-tools using instructions provided [here](https://developers.google.com/optimization/install/cpp).
For Linux (Ubuntu) machines to reduce time and memory usage. May remove `-j` for single-core build.
```shell
cmake -S . -B build -DBUILD_DEPS=ON -DUSE_SCIP=OFF -DUSE_HIGHS=OFF -DUSE_COINOR=OFF && 
cmake --build build --config Release --target all -j -v
```
- Install [protobuf compiler and c++ runtime](https://github.com/protocolbuffers/protobuf/blob/v24.0/src/README.md)

## Building The Project And Testing
- Build using cmake
```shell
mkdir build && cd build
cmake .. && cmake --build . --parallel 8
```
- Run compiler flow using ``sam-opt`` and ``sam-translate`` binaries under ``tools/`` passing in input MLIR test to compile.
- Executing ``sam-opt`` binary compiles and lowers from a linalg/sparse_tensor program to the samml dialect
- Executing ``sam-translate`` binary with ``--emit-proto`` flag emits protobuf binary of samml output program
- Example mlir input programs provided under tests directory.
```shell
# Generated protobuf output program binary is stored under /tmp/op.bin 
./tools/sam-opt --linalg-to-sam ${MLIR_TEST} | ./tools/sam-translate --emit-proto
```
- Apply optimizations/transformations (parallelization/vectorization). (optional)
```shell
# Using parallelization pass, stream-level parameter specifies level to be parallelized and par-factor specifies the parallelization factor
# Can parallelize multiple levels by adding more levels separated by a space
./tools/sam-opt --linalg-to-sam "--stream-parallelizer="stream-level=0 par-factor=1"" ${MLIR_TEST} | ./tools/sam-translate --emit-proto
# Using vectorization pass, stream-shape parameter specifies vector length
./tools/sam-opt --linalg-to-sam "--stream-vectorizer="stream-shape=16"" ${MLIR_TEST} | ./tools/sam-translate --emit-proto
# Using vectorization pass with block sparse enabled
./tools/sam-opt --linalg-to-sam "--stream-vectorizer="stream-shape=16 enable-block-sparse"" ${MLIR_TEST} | ./tools/sam-translate --emit-proto
```

## Building and Testing in CLion
- Reload cmake project after installing all [prerequisites](Prerequisites)
- In right drop down should see `sam-opt` and `sam-translate`. Click or search for "Edit Configurations..."
- For the `sam-opt` "Program arguments:" add `--linalg-to-sam` along with the MLIR file you want to run. For example,
```--linalg-to-sam /home/username/Documents/repos/samml/tests/nested_matmuls.mlir```
- For the `sam-translate` "Program arguments:" add `--emit-proto` with the generated SAMML MLIR file you want to run. For example, 
```--emit-proto /home/user/Documents/repos/samml/tests/nested_matmuls_samml.mlir```
- Now you can bpress the Build `sam-opt` icon in CLion

