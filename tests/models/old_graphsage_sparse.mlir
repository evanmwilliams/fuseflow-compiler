#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
#SPARSE1D = #sparse_tensor.encoding<{
  map = (i) -> (i : compressed)
}>
module attributes {torch.debug_module_name = "CustomGraphSAGE"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1767x50xf32, #SPARSE>, %arg1: tensor<1767x1767xf32, #SPARSE>) -> tensor<1767x121xf32, #SPARSE> {
    %cst = arith.constant dense_resource<__elided__> : tensor<16x50xf32, #SPARSE>
    // %cst = arith.constant dense_resource<__elided__> : tensor<50x16xf32, #SPARSE>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<16xf32, #SPARSE1D>
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<16x50xf32, #SPARSE>
    // %cst_2 = arith.constant dense_resource<__elided__> : tensor<50x16xf32, #SPARSE>
    %cst_3 = arith.constant dense_resource<__elided__> : tensor<16xf32, #SPARSE1D>
    %cst_4 = arith.constant dense_resource<__elided__> : tensor<121x16xf32, #SPARSE>
    // %cst_4 = arith.constant dense_resource<__elided__> : tensor<16x121xf32, #SPARSE>
    %cst_5 = arith.constant dense_resource<__elided__> : tensor<121xf32, #SPARSE1D>
    // %cst_6 = arith.constant dense_resource<__elided__> : tensor<121x16xf32, #SPARSE>
    %cst_6 = arith.constant dense_resource<__elided__> : tensor<16x121xf32, #SPARSE>
    %cst_7 = arith.constant dense_resource<__elided__> : tensor<121xf32, #SPARSE1D>
    %0 = tensor.empty() : tensor<50x16xf32, #SPARSE>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<16x50xf32, #SPARSE>) outs(%0 : tensor<50x16xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<50x16xf32, #SPARSE>
    %2 = tensor.empty() : tensor<1767x16xf32, #SPARSE>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1767x16xf32, #SPARSE>) -> tensor<1767x16xf32, #SPARSE>
    %4 = linalg.matmul ins(%arg0, %1 : tensor<1767x50xf32, #SPARSE>, tensor<50x16xf32, #SPARSE>) outs(%3 : tensor<1767x16xf32, #SPARSE>) -> tensor<1767x16xf32, #SPARSE>
    // %4 = linalg.matmul ins(%arg0, %cst : tensor<1767x50xf32, #SPARSE>, tensor<50x16xf32, #SPARSE>) outs(%3 : tensor<1767x16xf32, #SPARSE>) -> tensor<1767x16xf32, #SPARSE>
    %5 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst_1 : tensor<1767x16xf32, #SPARSE>, tensor<16xf32, #SPARSE1D>) outs(%2 : tensor<1767x16xf32, #SPARSE>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %27 = arith.addf %in, %in_8 : f32
      linalg.yield %27 : f32
    } -> tensor<1767x16xf32, #SPARSE>
    %6 = tensor.empty() : tensor<1767x50xf32, #SPARSE>
    %7 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<1767x50xf32, #SPARSE>) -> tensor<1767x50xf32, #SPARSE>
    %8 = linalg.matmul ins(%arg1, %arg0 : tensor<1767x1767xf32, #SPARSE>, tensor<1767x50xf32, #SPARSE>) outs(%7 : tensor<1767x50xf32, #SPARSE>) -> tensor<1767x50xf32, #SPARSE>
    // %8 = linalg.matmul ins(%arg1, %arg0 : tensor<1767x1767xf32, #SPARSE>, tensor<1767x50xf32, #SPARSE>) outs(%7 : tensor<1767x50xf32, #SPARSE>) -> tensor<1767x50xf32, #SPARSE>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_2 : tensor<16x50xf32, #SPARSE>) outs(%0 : tensor<50x16xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<50x16xf32, #SPARSE>
    %10 = linalg.matmul ins(%8, %9 : tensor<1767x50xf32, #SPARSE>, tensor<50x16xf32, #SPARSE>) outs(%3 : tensor<1767x16xf32, #SPARSE>) -> tensor<1767x16xf32, #SPARSE>
    // %10 = linalg.matmul ins(%8, %cst_2 : tensor<1767x50xf32, #SPARSE>, tensor<50x16xf32, #SPARSE>) outs(%3 : tensor<1767x16xf32, #SPARSE>) -> tensor<1767x16xf32, #SPARSE>
    %11 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%10, %cst_3 : tensor<1767x16xf32, #SPARSE>, tensor<16xf32, #SPARSE1D>) outs(%2 : tensor<1767x16xf32, #SPARSE>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %27 = arith.addf %in, %in_8 : f32
      linalg.yield %27 : f32
    } -> tensor<1767x16xf32, #SPARSE>
    %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %11 : tensor<1767x16xf32, #SPARSE>, tensor<1767x16xf32, #SPARSE>) outs(%2 : tensor<1767x16xf32, #SPARSE>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %27 = arith.addf %in, %in_8 : f32
      linalg.yield %27 : f32
    } -> tensor<1767x16xf32, #SPARSE>
    %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<1767x16xf32, #SPARSE>) outs(%2 : tensor<1767x16xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      %27 = arith.cmpf ugt, %in, %cst_0 : f32
      %28 = arith.select %27, %in, %cst_0 : f32
      linalg.yield %28 : f32
    } -> tensor<1767x16xf32, #SPARSE>
    %14 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<1767x16xf32, #SPARSE>) outs(%2 : tensor<1767x16xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      %27 = arith.cmpf ugt, %in, %cst_0 : f32
      %28 = arith.select %27, %in, %cst_0 : f32
      linalg.yield %28 : f32
    } -> tensor<1767x16xf32, #SPARSE>

    // FIXME: manually cloning for now to avoid bug but need to fix
    %new_clone = arith.constant dense_resource<__elided__> : tensor<1767x16xf32, #SPARSE>

    %15 = tensor.empty() : tensor<16x121xf32, #SPARSE>
    %16 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_4 : tensor<121x16xf32, #SPARSE>) outs(%15 : tensor<16x121xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x121xf32, #SPARSE>
    %17 = tensor.empty() : tensor<1767x121xf32, #SPARSE>
    %18 = linalg.fill ins(%cst_0 : f32) outs(%17 : tensor<1767x121xf32, #SPARSE>) -> tensor<1767x121xf32, #SPARSE>
    %19 = linalg.matmul ins(%14, %16 : tensor<1767x16xf32, #SPARSE>, tensor<16x121xf32, #SPARSE>) outs(%18 : tensor<1767x121xf32, #SPARSE>) -> tensor<1767x121xf32, #SPARSE>
    // %19 = linalg.matmul ins(%14, %cst_4 : tensor<1767x16xf32, #SPARSE>, tensor<16x121xf32, #SPARSE>) outs(%18 : tensor<1767x121xf32, #SPARSE>) -> tensor<1767x121xf32, #SPARSE>
    %20 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%19, %cst_5 : tensor<1767x121xf32, #SPARSE>, tensor<121xf32, #SPARSE1D>) outs(%17 : tensor<1767x121xf32, #SPARSE>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %27 = arith.addf %in, %in_8 : f32
      linalg.yield %27 : f32
    } -> tensor<1767x121xf32, #SPARSE>
    // %21 = linalg.matmul ins(%arg1_clone, %new_clone : tensor<1767x1767xf32, #SPARSE>, tensor<1767x16xf32, #SPARSE>) outs(%3 : tensor<1767x16xf32, #SPARSE>) -> tensor<1767x16xf32, #SPARSE>
    %21 = linalg.matmul ins(%arg1, %14 : tensor<1767x1767xf32, #SPARSE>, tensor<1767x16xf32, #SPARSE>) outs(%3 : tensor<1767x16xf32, #SPARSE>) -> tensor<1767x16xf32, #SPARSE>
    // // %22 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_6 : tensor<121x16xf32, #SPARSE>) outs(%15 : tensor<16x121xf32, #SPARSE>) {
    // // ^bb0(%in: f32, %out: f32):
    // //   linalg.yield %in : f32
    // // } -> tensor<16x121xf32, #SPARSE>
    %23 = linalg.matmul ins(%21, %cst_6 : tensor<1767x16xf32, #SPARSE>, tensor<16x121xf32, #SPARSE>) outs(%18 : tensor<1767x121xf32, #SPARSE>) -> tensor<1767x121xf32, #SPARSE>
    %24 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%23, %cst_7 : tensor<1767x121xf32, #SPARSE>, tensor<121xf32, #SPARSE1D>) outs(%17 : tensor<1767x121xf32, #SPARSE>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %27 = arith.addf %in, %in_8 : f32
      linalg.yield %27 : f32
    } -> tensor<1767x121xf32, #SPARSE>
    %25 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%20, %24 : tensor<1767x121xf32, #SPARSE>, tensor<1767x121xf32, #SPARSE>) outs(%17 : tensor<1767x121xf32, #SPARSE>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %27 = arith.addf %in, %in_8 : f32
      linalg.yield %27 : f32
    } -> tensor<1767x121xf32, #SPARSE>
    %26 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%25 : tensor<1767x121xf32, #SPARSE>) outs(%17 : tensor<1767x121xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      %27 = arith.cmpf ugt, %in, %cst_0 : f32
      %28 = arith.select %27, %in, %cst_0 : f32
      linalg.yield %28 : f32
    } -> tensor<1767x121xf32, #SPARSE>
    return %26 : tensor<1767x121xf32, #SPARSE>
  }
}