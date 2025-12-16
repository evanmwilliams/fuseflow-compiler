#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
#map4 = affine_map<(d0, d1) -> (d0, 0)>

#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
#SPARSE1D = #sparse_tensor.encoding<{
  map = (i) -> (i : dense)
}>

module attributes {torch.debug_module_name = "CustomGraphSAGE"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1767x50xf32>, %arg1: tensor<1767x1767xf32, #SPARSE>, %cst_2: tensor<16xf32>, %1: tensor<50x16xf32, #SPARSE>) -> tensor<1767x16xf32> {
  // func.func @forward(%arg0: tensor<1767x50xf32>, %arg1: tensor<1767x1767xf32, #SPARSE>) -> tensor<1767x16xf32> {
    %cst = arith.constant dense_resource<__elided__> : tensor<16xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<16x50xf32>
    // %cst_2 = arith.constant dense_resource<__elided__> : tensor<16xf32>
    %cst_3 = arith.constant dense_resource<__elided__> : tensor<16x50xf32>
    %cst_4 = arith.constant 0xFF800000 : f32

    // Begin Linear_self
    %0 = tensor.empty() : tensor<50x16xf32, #SPARSE>
    // %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<16x50xf32>) outs(%0 : tensor<50x16xf32, #SPARSE>) {
    // ^bb0(%in: f32, %out: f32):
      //linalg.yield %in : f32
    //} -> tensor<50x16xf32, #SPARSE>
    %2 = tensor.empty() : tensor<1767x16xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1767x16xf32>) -> tensor<1767x16xf32>
    // arg0: X, %1: W -> W*X, %1 = cst_1^T
    %4 = linalg.matmul ins(%arg0, %1 : tensor<1767x50xf32>, tensor<50x16xf32, #SPARSE>) outs(%3 : tensor<1767x16xf32>) -> tensor<1767x16xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst : tensor<1767x16xf32>, tensor<16xf32>) outs(%2 : tensor<1767x16xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %15 = arith.addf %in, %in_4 : f32
      linalg.yield %15 : f32
    } -> tensor<1767x16xf32>
    // End Linear_self

    // Begin Linear_neighbor
    %6 = tensor.empty() : tensor<1767x50xf32>
    %7 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<1767x50xf32>) -> tensor<1767x50xf32>
    // %arg1: D, %arg0: X, D*X
    %8 = linalg.matmul ins(%arg1, %arg0 : tensor<1767x1767xf32, #SPARSE>, tensor<1767x50xf32>) outs(%7 : tensor<1767x50xf32>) -> tensor<1767x50xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_3 : tensor<16x50xf32>) outs(%0 : tensor<50x16xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<50x16xf32, #SPARSE>
    // D*W*X, W=cst_3^T
    %10 = linalg.matmul ins(%8, %9 : tensor<1767x50xf32>, tensor<50x16xf32, #SPARSE>) outs(%3 : tensor<1767x16xf32>) -> tensor<1767x16xf32>
    // Bias=cst_2
    %11 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%10, %cst_2 : tensor<1767x16xf32>, tensor<16xf32>) outs(%2 : tensor<1767x16xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %15 = arith.addf %in, %in_4 : f32
      linalg.yield %15 : f32
    } -> tensor<1767x16xf32>
    // End Linear_neighbor

    // Add linear_self with linear_neighbor
    %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %11 : tensor<1767x16xf32>, tensor<1767x16xf32>) outs(%2 : tensor<1767x16xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %15 = arith.addf %in, %in_4 : f32
      linalg.yield %15 : f32
    } -> tensor<1767x16xf32>

    // Begin Relus
    %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<1767x16xf32>) outs(%2 : tensor<1767x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.cmpf ugt, %in, %cst_0 : f32
      %16 = arith.select %15, %in, %cst_0 : f32
      linalg.yield %16 : f32
    } -> tensor<1767x16xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<1767x16xf32>) outs(%2 : tensor<1767x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.cmpf ugt, %in, %cst_0 : f32
      %16 = arith.select %15, %in, %cst_0 : f32
      linalg.yield %16 : f32
    } -> tensor<1767x16xf32>

    %17 = tensor.empty() : tensor<1767x1xi64>
    %18 = linalg.fill ins(%c0_i64 : i64) outs(%17 : tensor<1767x1xi64>) -> tensor<1767x1xi64>
    %19 = tensor.empty() : tensor<1767x1xf32, #SPARSE>
    // %20 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<1767x1xf32, #SPARSE>) -> tensor<1767x1xf32, #SPARSE>
    %21:2 = linalg.generic {indexing_maps = [#map, #map3, #map3], iterator_types = ["parallel", "reduction"]} ins(%14 : tensor<1767x16xf32>) outs(%19, %18 : tensor<1767x1xf32, #SPARSE>, tensor<1767x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_5: i64):
      %29 = linalg.index 1 : index
      %30 = arith.index_cast %29 : index to i64
      %31 = arith.maximumf %in, %out : f32
      %32 = arith.cmpf ogt, %in, %out : f32
      %33 = arith.select %32, %30, %out_5 : i64
      linalg.yield %31, %33 : f32, i64
    } -> (tensor<1767x1xf32, #SPARSE>, tensor<1767x1xi64>)
    // %expanded = tensor.expand_shape %21#0 [[0, 1]] output_shape [1767, 1] : tensor<1767x1xf32, #SPARSE> into tensor<1767x1xf32, #SPARSE>
    %22 = linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel"]} ins(%14, %21#0 : tensor<1767x16xf32>, tensor<1767x1xf32, #SPARSE>) outs(%13 : tensor<1767x16xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.subf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x16xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%22 : tensor<1767x16xf32>) outs(%13 : tensor<1767x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = math.exp %in : f32
      linalg.yield %29 : f32
    } -> tensor<1767x16xf32>
    %24 = tensor.empty() : tensor<1767x1xf32, #SPARSE>
    %25 = linalg.fill ins(%cst_0 : f32) outs(%24 : tensor<1767x1xf32, #SPARSE>) -> tensor<1767x1xf32, #SPARSE>
    %26 = linalg.generic {indexing_maps = [#map, #map4], iterator_types = ["parallel", "reduction"]} ins(%23 : tensor<1767x16xf32>) outs(%25 : tensor<1767x1xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      %29 = arith.addf %in, %out : f32
      linalg.yield %29 : f32
    } -> tensor<1767x1xf32, #SPARSE>
    %27 = linalg.generic {indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel"]} ins(%26 : tensor<1767x1xf32, #SPARSE>) outs(%24 : tensor<1767x1xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      %29 = math.log %in : f32
      linalg.yield %29 : f32
    } -> tensor<1767x1xf32, #SPARSE>
    %28 = linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel"]} ins(%22, %26 : tensor<1767x16xf32>, tensor<1767x1xf32, #SPARSE>) outs(%13 : tensor<1767x16xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.subf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x16xf32>
    return %13 : tensor<1767x16xf32>
  }
}