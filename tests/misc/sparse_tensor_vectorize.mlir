module {
  func.func @scale_dd(%arg0: f32, %arg1: tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "dense" ] }>>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %dim = tensor.dim %arg1, %c0 : tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "dense" ] }>>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "dense" ] }>>
    %0 = sparse_tensor.values %arg1 : tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "dense" ] }>> to memref<?xf32>
    %1 = bufferization.to_memref %arg2 : memref<?x?xf32>
    linalg.fill ins(%cst : f32) outs(%1 : memref<?x?xf32>)
    scf.parallel (%arg3) = (%c0) to (%dim) step (%c1) {
      scf.for %arg4 = %c0 to %dim_0 step %c1 {
        %3 = arith.muli %dim_0, %arg3 : index
        %4 = arith.addi %3, %arg4 : index
        %5 = memref.load %0[%4] : memref<?xf32>
        %6 = arith.mulf %5, %arg0 : f32
        memref.store %6, %1[%arg3, %arg4] : memref<?x?xf32>
      } {"Emitted from" = "linalg.generic"}
      scf.yield
    } {"Emitted from" = "linalg.generic"}
    %2 = bufferization.to_tensor %1 : memref<?x?xf32>
    return %2 : tensor<?x?xf32>
  }
  func.func @scale_ss(%arg0: f32, %arg1: tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed" ] }>>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = sparse_tensor.positions %arg1 {level = 0 : index} : tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed" ] }>> to memref<?xindex>
    %1 = sparse_tensor.coordinates %arg1 {level = 0 : index} : tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed" ] }>> to memref<?xindex>
    %2 = sparse_tensor.positions %arg1 {level = 1 : index} : tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed" ] }>> to memref<?xindex>
    %3 = sparse_tensor.coordinates %arg1 {level = 1 : index} : tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed" ] }>> to memref<?xindex>
    %4 = sparse_tensor.values %arg1 : tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed" ] }>> to memref<?xf32>
    %5 = bufferization.to_memref %arg2 : memref<?x?xf32>
    linalg.fill ins(%cst : f32) outs(%5 : memref<?x?xf32>)
    %6 = memref.load %0[%c0] : memref<?xindex>
    %7 = memref.load %0[%c1] : memref<?xindex>
    scf.for %arg3 = %6 to %7 step %c1 {
      %9 = memref.load %1[%arg3] : memref<?xindex>
      %10 = memref.load %2[%arg3] : memref<?xindex>
      %11 = arith.addi %arg3, %c1 : index
      %12 = memref.load %2[%11] : memref<?xindex>
      scf.for %arg4 = %10 to %12 step %c1 {
        %13 = memref.load %3[%arg4] : memref<?xindex>
        %14 = memref.load %4[%arg4] : memref<?xf32>
        %15 = arith.mulf %14, %arg0 : f32
        memref.store %15, %5[%9, %13] : memref<?x?xf32>
      } {"Emitted from" = "linalg.generic"}
    } {"Emitted from" = "linalg.generic"}
    %8 = bufferization.to_tensor %5 : memref<?x?xf32>
    return %8 : tensor<?x?xf32>
  }
  func.func @matvec(%arg0: tensor<16x32xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>>, %arg1: tensor<32xf32>, %arg2: tensor<16xf32>) -> tensor<16xf32> {
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = sparse_tensor.positions %arg0 {level = 1 : index} : tensor<16x32xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> to memref<?xindex>
    %1 = sparse_tensor.coordinates %arg0 {level = 1 : index} : tensor<16x32xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> to memref<?xindex>
    %2 = sparse_tensor.values %arg0 : tensor<16x32xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> to memref<?xf32>
    %3 = bufferization.to_memref %arg1 : memref<32xf32>
    %4 = bufferization.to_memref %arg2 : memref<16xf32>
    scf.parallel (%arg3) = (%c0) to (%c16) step (%c1) {
      %6 = memref.load %4[%arg3] : memref<16xf32>
      %7 = memref.load %0[%arg3] : memref<?xindex>
      %8 = arith.addi %arg3, %c1 : index
      %9 = memref.load %0[%8] : memref<?xindex>
      %10 = scf.for %arg4 = %7 to %9 step %c1 iter_args(%arg5 = %6) -> (f32) {
        %11 = memref.load %1[%arg4] : memref<?xindex>
        %12 = memref.load %2[%arg4] : memref<?xf32>
        %13 = memref.load %3[%11] : memref<32xf32>
        %14 = arith.mulf %12, %13 : f32
        %15 = arith.addf %14, %arg5 : f32
        scf.yield %15 : f32
      } {"Emitted from" = "linalg.generic"}
      memref.store %10, %4[%arg3] : memref<16xf32>
      scf.yield
    } {"Emitted from" = "linalg.generic"}
    %5 = bufferization.to_tensor %4 : memref<16xf32>
    return %5 : tensor<16xf32>
  }
}