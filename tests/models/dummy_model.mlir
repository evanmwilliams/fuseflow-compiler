#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0, 0)>
#map5 = affine_map<(d0, d1) -> (0, 0)>
module attributes {torch.debug_module_name = "SimpleNN"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x784xf32>) -> tensor<1x10xf32> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<[0.00904431194, -0.0562534407, -0.0540498868, -0.0748723894, -0.0307656825, -0.0435144082, -0.0101534948, 0.008740969, -0.0820712074, 0.0403052047]> : tensor<10xf32>
    // %cst_0 = arith.constant dense_resource<__elided__> : tensor<10x128xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<128x10xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<128x784xf32>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %cst_4 = arith.constant 0xFF800000 : f32
    %cst_5 = arith.constant dense_resource<__elided__> : tensor<784x128xf32>
    %0 = tensor.empty() : tensor<784x128xf32>
    // %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_2 : tensor<128x784xf32>) outs(%0 : tensor<784x128xf32>) {
    // ^bb0(%in: f32, %out: f32):
    //   linalg.yield %in : f32
    // } -> tensor<784x128xf32>
    %2 = tensor.empty() : tensor<1x128xf32>
    %3 = linalg.fill ins(%cst_3 : f32) outs(%2 : tensor<1x128xf32>) -> tensor<1x128xf32>
    %4 = linalg.matmul ins(%arg0, %cst_5 : tensor<1x784xf32>, tensor<784x128xf32>) outs(%3 : tensor<1x128xf32>) -> tensor<1x128xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst_1 : tensor<1x128xf32>, tensor<128xf32>) outs(%2 : tensor<1x128xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %23 = arith.addf %in, %in_5 : f32
      linalg.yield %23 : f32
    } -> tensor<1x128xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x128xf32>) outs(%2 : tensor<1x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %23 = arith.cmpf ugt, %in, %cst_3 : f32
      %24 = arith.select %23, %in, %cst_3 : f32
      linalg.yield %24 : f32
    } -> tensor<1x128xf32>
    %7 = tensor.empty() : tensor<128x10xf32>
    // %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<10x128xf32>) outs(%7 : tensor<128x10xf32>) {
    // ^bb0(%in: f32, %out: f32):
    //   linalg.yield %in : f32
    // } -> tensor<128x10xf32>
    %9 = tensor.empty() : tensor<1x10xf32>
    %10 = linalg.fill ins(%cst_3 : f32) outs(%9 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %11 = linalg.matmul ins(%6, %cst_0 : tensor<1x128xf32>, tensor<128x10xf32>) outs(%10 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %12 = linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %cst : tensor<1x10xf32>, tensor<10xf32>) outs(%9 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %23 = arith.addf %in, %in_5 : f32
      linalg.yield %23 : f32
    } -> tensor<1x10xf32>
    %13 = tensor.empty() : tensor<1x1xi64>
    %14 = linalg.fill ins(%c0_i64 : i64) outs(%13 : tensor<1x1xi64>) -> tensor<1x1xi64>
    %15 = tensor.empty() : tensor<1x1xf32>
    %16 = linalg.fill ins(%cst_4 : f32) outs(%15 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %17:2 = linalg.generic {indexing_maps = [#map, #map4, #map4], iterator_types = ["parallel", "reduction"]} ins(%12 : tensor<1x10xf32>) outs(%16, %14 : tensor<1x1xf32>, tensor<1x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_5: i64):
      %23 = linalg.index 1 : index
      %24 = arith.index_cast %23 : index to i64
      %25 = arith.maximumf %in, %out : f32
      %26 = arith.cmpf ogt, %in, %out : f32
      %27 = arith.select %26, %24, %out_5 : i64
      linalg.yield %25, %27 : f32, i64
    } -> (tensor<1x1xf32>, tensor<1x1xi64>)
    %18 = linalg.generic {indexing_maps = [#map2, #map5, #map], iterator_types = ["parallel", "parallel"]} ins(%12, %17#0 : tensor<1x10xf32>, tensor<1x1xf32>) outs(%9 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %23 = arith.subf %in, %in_5 : f32
      linalg.yield %23 : f32
    } -> tensor<1x10xf32>
    %19 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%18 : tensor<1x10xf32>) outs(%9 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %23 = math.exp %in : f32
      linalg.yield %23 : f32
    } -> tensor<1x10xf32>
    %20 = linalg.fill ins(%cst_3 : f32) outs(%15 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %21 = linalg.generic {indexing_maps = [#map, #map4], iterator_types = ["parallel", "reduction"]} ins(%19 : tensor<1x10xf32>) outs(%20 : tensor<1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %23 = arith.addf %in, %out : f32
      linalg.yield %23 : f32
    } -> tensor<1x1xf32>
    %22 = linalg.generic {indexing_maps = [#map2, #map5, #map], iterator_types = ["parallel", "parallel"]} ins(%19, %21 : tensor<1x10xf32>, tensor<1x1xf32>) outs(%9 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %23 = arith.divf %in, %in_5 : f32
      linalg.yield %23 : f32
    } -> tensor<1x10xf32>
    return %22 : tensor<1x10xf32>
  }
}