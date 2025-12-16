#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "SimpleNN"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<10x784xf32>) -> tensor<10x128xf32> {
    %cst = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<784x128xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<784x128xf32>
    %2 = tensor.empty() : tensor<10x128xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<10x128xf32>) -> tensor<10x128xf32>
    %4 = linalg.matmul ins(%arg0, %cst_0 : tensor<10x784xf32>, tensor<784x128xf32>) outs(%3 : tensor<10x128xf32>) -> tensor<10x128xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst : tensor<10x128xf32>, tensor<128xf32>) outs(%2 : tensor<10x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.addf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<10x128xf32>
    return %5 : tensor<10x128xf32>
  }
}