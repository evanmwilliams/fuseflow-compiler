#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0) -> (d0)>
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
#SPARSE1D = #sparse_tensor.encoding<{
  map = (i) -> (i : compressed)
}>
// LUNA16 Autoencoder: 50176 (224x224x1) -> 256 -> 50176
module attributes {torch.debug_module_name = "SparseAutoencoder_LUNA16"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<50176xf32, #SPARSE1D>) -> tensor<50176xf32, #SPARSE1D> {
    // Encoder bias (hidden_dim)
    %cst = arith.constant dense_resource<__elided__> : tensor<256xf32, #SPARSE1D>
    %cst_0 = arith.constant 0.000000e+00 : f32
    // Decoder weights (hidden_dim x input_dim)
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<256x50176xf32, #SPARSE>
    // Decoder bias (input_dim)
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<50176xf32, #SPARSE1D>
    // Encoder weights (input_dim x hidden_dim)
    %cst_3 = arith.constant dense_resource<__elided__> : tensor<50176x256xf32, #SPARSE>

    // Encoder: transpose weights
    %0 = tensor.empty() : tensor<256x50176xf32, #SPARSE>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_3 : tensor<50176x256xf32, #SPARSE>) outs(%0 : tensor<256x50176xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<256x50176xf32, #SPARSE>

    // Encoder: matrix-vector multiply
    %2 = tensor.empty() : tensor<256xf32, #SPARSE1D>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<256xf32, #SPARSE1D>) -> tensor<256xf32, #SPARSE1D>
    %4 = linalg.matvec ins(%1, %arg0 : tensor<256x50176xf32, #SPARSE>, tensor<50176xf32, #SPARSE1D>) outs(%3 : tensor<256xf32, #SPARSE1D>) -> tensor<256xf32, #SPARSE1D>

    // Encoder: add bias
    %5 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%4, %cst : tensor<256xf32, #SPARSE1D>, tensor<256xf32, #SPARSE1D>) outs(%2 : tensor<256xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %13 = arith.addf %in, %in_4 : f32
      linalg.yield %13 : f32
    } -> tensor<256xf32, #SPARSE1D>

    // Encoder: ReLU activation
    %6 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%5 : tensor<256xf32, #SPARSE1D>) outs(%2 : tensor<256xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %out: f32):
      %13 = arith.cmpf ugt, %in, %cst_0 : f32
      %14 = arith.select %13, %in, %cst_0 : f32
      linalg.yield %14 : f32
    } -> tensor<256xf32, #SPARSE1D>

    // Decoder: transpose weights
    %7 = tensor.empty() : tensor<50176x256xf32, #SPARSE>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<256x50176xf32, #SPARSE>) outs(%7 : tensor<50176x256xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<50176x256xf32, #SPARSE>

    // Decoder: matrix-vector multiply
    %9 = tensor.empty() : tensor<50176xf32, #SPARSE1D>
    %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<50176xf32, #SPARSE1D>) -> tensor<50176xf32, #SPARSE1D>
    %11 = linalg.matvec ins(%8, %6 : tensor<50176x256xf32, #SPARSE>, tensor<256xf32, #SPARSE1D>) outs(%10 : tensor<50176xf32, #SPARSE1D>) -> tensor<50176xf32, #SPARSE1D>

    // Decoder: add bias
    %12 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%11, %cst_2 : tensor<50176xf32, #SPARSE1D>, tensor<50176xf32, #SPARSE1D>) outs(%9 : tensor<50176xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %13 = arith.addf %in, %in_4 : f32
      linalg.yield %13 : f32
    } -> tensor<50176xf32, #SPARSE1D>

    return %12 : tensor<50176xf32, #SPARSE1D>
  }
}
