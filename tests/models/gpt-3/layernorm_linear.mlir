#map = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map3 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d1, d0)>
#map6 = affine_map<(d0, d1, d2) -> (d2)>
#map7 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<1x1024x768xf32>) -> tensor<1x1024x768xf32> {
    %cst = arith.constant dense_resource<torch_tensor_768_768_torch.float32> : tensor<768x768xf32>
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e-05 : f64
    %cst_3 = arith.constant 7.680000e+02 : f64
    %cst_4 = arith.constant 7.680000e+02 : f32
    %cst_5 = arith.constant dense_resource<torch_tensor_768_torch.float32> : tensor<768xf32>
    %0 = tensor.empty() : tensor<1x1024x768xf64>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1024x768xf32>) outs(%0 : tensor<1x1024x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %26 = arith.extf %in : f32 to f64
      linalg.yield %26 : f64
    } -> tensor<1x1024x768xf64>
    %2 = tensor.empty() : tensor<1x1024x1xf64>
    %3 = linalg.fill ins(%cst_0 : f64) outs(%2 : tensor<1x1024x1xf64>) -> tensor<1x1024x1xf64>
    %4 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1 : tensor<1x1024x768xf64>) outs(%3 : tensor<1x1024x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %26 = arith.addf %in, %out : f64
      linalg.yield %26 : f64
    } -> tensor<1x1024x1xf64>
    %5 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<1x1024x1xf64>) outs(%2 : tensor<1x1024x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %26 = arith.divf %in, %cst_3 : f64
      linalg.yield %26 : f64
    } -> tensor<1x1024x1xf64>
    %6 = linalg.generic {indexing_maps = [#map, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %5 : tensor<1x1024x768xf64>, tensor<1x1024x1xf64>) outs(%0 : tensor<1x1024x768xf64>) {
    ^bb0(%in: f64, %in_6: f64, %out: f64):
      %26 = arith.subf %in, %in_6 : f64
      linalg.yield %26 : f64
    } -> tensor<1x1024x768xf64>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %6 : tensor<1x1024x768xf64>, tensor<1x1024x768xf64>) outs(%0 : tensor<1x1024x768xf64>) {
    ^bb0(%in: f64, %in_6: f64, %out: f64):
      %26 = arith.mulf %in, %in_6 : f64
      linalg.yield %26 : f64
    } -> tensor<1x1024x768xf64>
    %8 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7 : tensor<1x1024x768xf64>) outs(%3 : tensor<1x1024x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %26 = arith.addf %in, %out : f64
      linalg.yield %26 : f64
    } -> tensor<1x1024x1xf64>
    %9 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<1x1024x1xf64>) outs(%2 : tensor<1x1024x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %26 = arith.divf %in, %cst_3 : f64
      linalg.yield %26 : f64
    } -> tensor<1x1024x1xf64>
    %10 = tensor.empty() : tensor<1x1024x1xf32>
    %11 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<1x1024x1xf64>) outs(%10 : tensor<1x1024x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %26 = arith.truncf %in : f64 to f32
      linalg.yield %26 : f32
    } -> tensor<1x1024x1xf32>
    %12 = linalg.fill ins(%cst_1 : f32) outs(%10 : tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %13 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1x1024x768xf32>) outs(%12 : tensor<1x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %26 = arith.addf %in, %out : f32
      linalg.yield %26 : f32
    } -> tensor<1x1024x1xf32>
    %14 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%13 : tensor<1x1024x1xf32>) outs(%10 : tensor<1x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %26 = arith.divf %in, %cst_4 : f32
      linalg.yield %26 : f32
    } -> tensor<1x1024x1xf32>
    %15 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11 : tensor<1x1024x1xf32>) outs(%10 : tensor<1x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %26 = arith.truncf %cst_2 : f64 to f32
      %27 = arith.addf %in, %26 : f32
      linalg.yield %27 : f32
    } -> tensor<1x1024x1xf32>
    %16 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15 : tensor<1x1024x1xf32>) outs(%10 : tensor<1x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %26 = math.rsqrt %in : f32
      linalg.yield %26 : f32
    } -> tensor<1x1024x1xf32>
    %17 = tensor.empty() : tensor<1x1024x768xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %14 : tensor<1x1024x768xf32>, tensor<1x1024x1xf32>) outs(%17 : tensor<1x1024x768xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %26 = arith.subf %in, %in_6 : f32
      linalg.yield %26 : f32
    } -> tensor<1x1024x768xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18, %16 : tensor<1x1024x768xf32>, tensor<1x1024x1xf32>) outs(%17 : tensor<1x1024x768xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %26 = arith.mulf %in, %in_6 : f32
      linalg.yield %26 : f32
    } -> tensor<1x1024x768xf32>
    %20 = tensor.empty() : tensor<768x768xf32>
    %21 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<768x768xf32>) outs(%20 : tensor<768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<768x768xf32>
    %22 = tensor.empty() : tensor<1x1024x768xf32>
    %23 = linalg.fill ins(%cst_1 : f32) outs(%22 : tensor<1x1024x768xf32>) -> tensor<1x1024x768xf32>
    %24 = linalg.generic {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%19, %21 : tensor<1x1024x768xf32>, tensor<768x768xf32>) outs(%17 : tensor<1x1024x768xf32>) {
        ^bb0(%in: f32, %in_6: f32, %out: f32):
          %26 = arith.mulf %in, %in_6 : f32
          %27 = arith.addf %26, %out : f32
          linalg.yield %27 : f32
        } -> tensor<1x1024x768xf32>
    %25 = linalg.generic {indexing_maps = [#map1, #map6, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %cst_5 : tensor<1x1024x768xf32>, tensor<768xf32>) outs(%22 : tensor<1x1024x768xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %26 = arith.addf %in, %in_6 : f32
      linalg.yield %26 : f32
    } -> tensor<1x1024x768xf32>
    return %25 : tensor<1x1024x768xf32>
  }
}
