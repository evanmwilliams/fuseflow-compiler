#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "SparseAutoencoder"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1767x50xf32>) -> tensor<1767x50xf32> {
    %cst = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<256x50xf32>
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<50xf32>
    %cst_3 = arith.constant dense_resource<__elided__> : tensor<50x256xf32>
    %0 = tensor.empty() : tensor<50x256xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<256x50xf32>) outs(%0 : tensor<50x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<50x256xf32>
    %2 = tensor.empty() : tensor<1767x256xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1767x256xf32>) -> tensor<1767x256xf32>
    %4 = linalg.matmul ins(%arg0, %1 : tensor<1767x50xf32>, tensor<50x256xf32>) outs(%3 : tensor<1767x256xf32>) -> tensor<1767x256xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst : tensor<1767x256xf32>, tensor<256xf32>) outs(%2 : tensor<1767x256xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %13 = arith.addf %in, %in_4 : f32
      linalg.yield %13 : f32
    } -> tensor<1767x256xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1767x256xf32>) outs(%2 : tensor<1767x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %13 = arith.cmpf ugt, %in, %cst_0 : f32
      %14 = arith.select %13, %in, %cst_0 : f32
      linalg.yield %14 : f32
    } -> tensor<1767x256xf32>
    %7 = tensor.empty() : tensor<256x50xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_3 : tensor<50x256xf32>) outs(%7 : tensor<256x50xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<256x50xf32>
    %9 = tensor.empty() : tensor<1767x50xf32>
    %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<1767x50xf32>) -> tensor<1767x50xf32>
    %11 = linalg.matmul ins(%6, %8 : tensor<1767x256xf32>, tensor<256x50xf32>) outs(%10 : tensor<1767x50xf32>) -> tensor<1767x50xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %cst_2 : tensor<1767x50xf32>, tensor<50xf32>) outs(%9 : tensor<1767x50xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %13 = arith.addf %in, %in_4 : f32
      linalg.yield %13 : f32
    } -> tensor<1767x50xf32>
    return %12 : tensor<1767x50xf32>
  }
}