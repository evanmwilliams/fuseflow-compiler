#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2)>
module attributes {torch.debug_module_name = "Encoder"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<8x512x512xf32>) -> tensor<8x512x512xf32> {
    %cst = arith.constant dense_resource<__elided__> : tensor<2048xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<2048x512xf32>
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_3 = arith.constant dense_resource<__elided__> : tensor<512x2048xf32>
    %0 = tensor.empty() : tensor<512x2048xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<2048x512xf32>) outs(%0 : tensor<512x2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<512x2048xf32>
    %2 = tensor.empty() : tensor<8x512x2048xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<512x2048xf32>) outs(%2 : tensor<8x512x2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x512x2048xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<8x512x2048xf32>) -> tensor<8x512x2048xf32>
    %5 = linalg.batch_matmul ins(%arg0, %3 : tensor<8x512x512xf32>, tensor<8x512x2048xf32>) outs(%4 : tensor<8x512x2048xf32>) -> tensor<8x512x2048xf32>
    %6 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %cst : tensor<8x512x2048xf32>, tensor<2048xf32>) outs(%2 : tensor<8x512x2048xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %16 = arith.addf %in, %in_4 : f32
      linalg.yield %16 : f32
    } -> tensor<8x512x2048xf32>
    %7 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6 : tensor<8x512x2048xf32>) outs(%2 : tensor<8x512x2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = arith.cmpf ugt, %in, %cst_0 : f32
      %17 = arith.select %16, %in, %cst_0 : f32
      linalg.yield %17 : f32
    } -> tensor<8x512x2048xf32>
    %8 = tensor.empty() : tensor<2048x512xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_3 : tensor<512x2048xf32>) outs(%8 : tensor<2048x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2048x512xf32>
    %10 = tensor.empty() : tensor<8x2048x512xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<2048x512xf32>) outs(%10 : tensor<8x2048x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x2048x512xf32>
    %12 = tensor.empty() : tensor<8x512x512xf32>
    %13 = linalg.fill ins(%cst_0 : f32) outs(%12 : tensor<8x512x512xf32>) -> tensor<8x512x512xf32>
    %14 = linalg.batch_matmul ins(%7, %11 : tensor<8x512x2048xf32>, tensor<8x2048x512xf32>) outs(%13 : tensor<8x512x512xf32>) -> tensor<8x512x512xf32>
    %15 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14, %cst_2 : tensor<8x512x512xf32>, tensor<512xf32>) outs(%12 : tensor<8x512x512xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %16 = arith.addf %in, %in_4 : f32
      linalg.yield %16 : f32
    } -> tensor<8x512x512xf32>
    return %15 : tensor<8x512x512xf32>
  }
}