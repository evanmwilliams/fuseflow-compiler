#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
module attributes {torch.debug_module_name = "GCN"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward() -> tensor<1767x121xf32, #SPARSE> {
    %9 = arith.constant dense_resource<__elided__> : tensor<1767x16xf32, #SPARSE>
    %11 = arith.constant dense_resource<__elided__> : tensor<16x121xf32, #SPARSE>
    %12 = tensor.empty() : tensor<1767x121xf32, #SPARSE>
    // %13 = linalg.fill ins(%cst_0 : f32) outs(%12 : tensor<1767x121xf32, #SPARSE>) -> tensor<1767x121xf32, #SPARSE>
    %14 = linalg.matmul ins(%9, %11 : tensor<1767x16xf32, #SPARSE>, tensor<16x121xf32, #SPARSE>) outs(%12 : tensor<1767x121xf32, #SPARSE>) -> tensor<1767x121xf32, #SPARSE>
    %cst = arith.constant dense_resource<__elided__> : tensor<1767x121xf32, #SPARSE>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<121x16xf32, #SPARSE>
    %15 = tensor.empty() : tensor<1767x1xi64>
    %16 = linalg.fill ins(%c0_i64 : i64) outs(%15 : tensor<1767x1xi64>) -> tensor<1767x1xi64>
    %17 = tensor.empty() : tensor<1767x1xf32, #SPARSE>
    %18 = linalg.fill ins(%cst_1 : f32) outs(%17 : tensor<1767x1xf32, #SPARSE>) -> tensor<1767x1xf32, #SPARSE>
    %19:2 = linalg.generic {indexing_maps = [#map, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%14 : tensor<1767x121xf32, #SPARSE>) outs(%18, %16 : tensor<1767x1xf32, #SPARSE>, tensor<1767x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_3: i64):
      %26 = linalg.index 1 : index
      %27 = arith.index_cast %26 : index to i64
      %28 = arith.maximumf %in, %out : f32
      %29 = arith.cmpf ogt, %in, %out : f32
      %30 = arith.select %29, %27, %out_3 : i64
      linalg.yield %28, %30 : f32, i64
    } -> (tensor<1767x1xf32, #SPARSE>, tensor<1767x1xi64>)
    %20 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%14, %19#0 : tensor<1767x121xf32, #SPARSE>, tensor<1767x1xf32, #SPARSE>) outs(%12 : tensor<1767x121xf32, #SPARSE>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %26 = arith.subf %in, %in_3 : f32
      linalg.yield %26 : f32
    } -> tensor<1767x121xf32, #SPARSE>
    %21 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%20 : tensor<1767x121xf32, #SPARSE>) outs(%12 : tensor<1767x121xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      %26 = math.exp %in : f32
      linalg.yield %26 : f32
    } -> tensor<1767x121xf32, #SPARSE>
    %22 = tensor.empty() : tensor<1767x1xf32, #SPARSE>
    %23 = linalg.fill ins(%cst_0 : f32) outs(%22 : tensor<1767x1xf32, #SPARSE>) -> tensor<1767x1xf32, #SPARSE>
    %24 = linalg.generic {indexing_maps = [#map, #map3], iterator_types = ["parallel", "reduction"]} ins(%21 : tensor<1767x121xf32, #SPARSE>) outs(%23 : tensor<1767x1xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      %26 = arith.addf %in, %out : f32
      linalg.yield %26 : f32
    } -> tensor<1767x1xf32, #SPARSE>
    %25 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%21, %24 : tensor<1767x121xf32, #SPARSE>, tensor<1767x1xf32, #SPARSE>) outs(%12 : tensor<1767x121xf32, #SPARSE>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %26 = arith.divf %in, %in_3 : f32
      linalg.yield %26 : f32
    } -> tensor<1767x121xf32, #SPARSE>
    return %25 : tensor<1767x121xf32, #SPARSE>
    // return %20 : tensor<1767x121xf32, #SPARSE>
  }
}