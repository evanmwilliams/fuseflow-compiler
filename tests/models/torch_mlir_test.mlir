#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
#CSR = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
module attributes {torch.debug_module_name = "SimpleNN"} {
  // ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x784xf32, #CSR>) -> tensor<1x10xf32, #CSR> {
    %cst = arith.constant dense<[-0.0723748729, -5.407840e-02, 0.025628034, -0.0344499387, 0.00704824878, 0.0517944396, -0.0395646803, -0.0183494631, 0.0350681469, -0.012156236]> : tensor<10xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<10x128xf32, #CSR>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<128xf32>
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<128x784xf32, #CSR>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<784x128xf32, #CSR>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_2 : tensor<128x784xf32, #CSR>) outs(%0 : tensor<784x128xf32, #CSR>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<784x128xf32, #CSR>
    %2 = tensor.empty() : tensor<1x128xf32, #CSR>
    // %3 = linalg.fill ins(%cst_3 : f32) outs(%2 : tensor<1x128xf32, #CSR>) -> tensor<1x128xf32, #CSR>
    %4 = linalg.matmul ins(%arg0, %1 : tensor<1x784xf32, #CSR>, tensor<784x128xf32, #CSR>) outs(%2 : tensor<1x128xf32, #CSR>) -> tensor<1x128xf32, #CSR>
    %5 = linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst_1 : tensor<1x128xf32, #CSR>, tensor<128xf32>) outs(%2 : tensor<1x128xf32, #CSR>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %12 = arith.addf %in, %in_4 : f32
      linalg.yield %12 : f32
    } -> tensor<1x128xf32, #CSR>
    %6 = tensor.empty() : tensor<128x10xf32, #CSR>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<10x128xf32, #CSR>) outs(%6 : tensor<128x10xf32, #CSR>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x10xf32, #CSR>
    %8 = tensor.empty() : tensor<1x10xf32, #CSR>
    // %9 = linalg.fill ins(%cst_3 : f32) outs(%8 : tensor<1x10xf32, #CSR>) -> tensor<1x10xf32, #CSR>
    %10 = linalg.matmul ins(%5, %7 : tensor<1x128xf32, #CSR>, tensor<128x10xf32, #CSR>) outs(%8 : tensor<1x10xf32, #CSR>) -> tensor<1x10xf32, #CSR>
    %11 = linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%10, %cst : tensor<1x10xf32, #CSR>, tensor<10xf32>) outs(%8 : tensor<1x10xf32, #CSR>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %12 = arith.addf %in, %in_4 : f32
      linalg.yield %12 : f32
    } -> tensor<1x10xf32, #CSR>
    return %11 : tensor<1x10xf32, #CSR>
  }
}
















































// #map = affine_map<(d0, d1) -> (d0, d1)>
// #map1 = affine_map<(d0, d1) -> (d1, d0)>
// #map2 = affine_map<(d0, d1) -> (0, d1)>
// #map3 = affine_map<(d0, d1) -> (d1)>
// #CSR = #sparse_tensor.encoding<{
//   map = (i,j) -> (i : compressed, j : compressed)
// }>
// module attributes {torch.debug_module_name = "SimpleNN"} {
//   ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
//   func.func @forward(%arg0: tensor<1x784xf32, #CSR>) -> tensor<1x10xf32, #CSR> {
//     %cst = arith.constant dense<[0.0378403477, -0.0278361496, 0.0137024038, -0.0781007558, -0.0500908419, 0.0799108371, 0.0508317947, 0.0712327138, -0.0735698491, 0.0404269621]> : tensor<10xf32, #CSR>
//     %cst_0 = arith.constant dense_resource<__elided__> : tensor<10x128xf32, #CSR>
//     %cst_1 = arith.constant dense_resource<__elided__> : tensor<128xf32, #CSR>
//     %cst_2 = arith.constant dense_resource<__elided__> : tensor<128x784xf32, #CSR>
//     %cst_3 = arith.constant 0.000000e+00 : f32
//     %0 = tensor.empty() : tensor<784x128xf32, #CSR>
//     %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_2 : tensor<128x784xf32, #CSR>) outs(%0 : tensor<784x128xf32, #CSR>) {
//     ^bb0(%in: f32, %out: f32):
//       linalg.yield %in : f32
//     } -> tensor<784x128xf32, #CSR>
//     %2 = tensor.empty() : tensor<1x128xf32, #CSR>
//     %3 = linalg.fill ins(%cst_3 : f32) outs(%2 : tensor<1x128xf32, #CSR>) -> tensor<1x128xf32, #CSR>
//     %4 = linalg.matmul ins(%arg0, %1 : tensor<1x784xf32, #CSR>, tensor<784x128xf32, #CSR>) outs(%3 : tensor<1x128xf32, #CSR>) -> tensor<1x128xf32, #CSR>
//     %5 = linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst_1 : tensor<1x128xf32, #CSR>, tensor<128xf32, #CSR>) outs(%2 : tensor<1x128xf32, #CSR>) {
//     ^bb0(%in: f32, %in_4: f32, %out: f32):
//       %13 = arith.addf %in, %in_4 : f32
//       linalg.yield %13 : f32
//     } -> tensor<1x128xf32, #CSR>
//     %6 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x128xf32, #CSR>) outs(%2 : tensor<1x128xf32, #CSR>) {
//     ^bb0(%in: f32, %out: f32):
//       %13 = arith.cmpf ugt, %in, %cst_3 : f32
//       %14 = arith.select %13, %in, %cst_3 : f32
//       linalg.yield %14 : f32
//     } -> tensor<1x128xf32, #CSR>
//     %7 = tensor.empty() : tensor<128x10xf32, #CSR>
//     %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<10x128xf32, #CSR>) outs(%7 : tensor<128x10xf32, #CSR>) {
//     ^bb0(%in: f32, %out: f32):
//       linalg.yield %in : f32
//     } -> tensor<128x10xf32, #CSR>
//     %9 = tensor.empty() : tensor<1x10xf32, #CSR>
//     %10 = linalg.fill ins(%cst_3 : f32) outs(%9 : tensor<1x10xf32, #CSR>) -> tensor<1x10xf32, #CSR>
//     %11 = linalg.matmul ins(%5, %8 : tensor<1x128xf32, #CSR>, tensor<128x10xf32, #CSR>) outs(%10 : tensor<1x10xf32, #CSR>) -> tensor<1x10xf32, #CSR>
//     %12 = linalg.generic {indexing_maps = [#map2, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %cst : tensor<1x10xf32, #CSR>, tensor<10xf32, #CSR>) outs(%9 : tensor<1x10xf32, #CSR>) {
//     ^bb0(%in: f32, %in_4: f32, %out: f32):
//       %13 = arith.addf %in, %in_4 : f32
//       linalg.yield %13 : f32
//     } -> tensor<1x10xf32, #CSR>
//     return %12 : tensor<1x10xf32, #CSR>
//   }
// }