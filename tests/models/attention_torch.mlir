#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module attributes {torch.debug_module_name = "MultiHeadAttention"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<8x512x512xf32>, %arg1: tensor<8x512x512xf32>, %arg2: tensor<8x512x512xf32>) -> tensor<?x?x512xf32> {
    %cst = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_3 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_4 = arith.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_5 = arith.constant dense_resource<__elided__> : tensor<512xf32>
    %cst_6 = arith.constant dense_resource<__elided__> : tensor<512x512xf32>
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant 0xFF800000 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_9 = arith.constant 8.000000e+00 : f32
    %0 = tensor.empty() : tensor<512x512xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_6 : tensor<512x512xf32>) outs(%0 : tensor<512x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<512x512xf32>
    %2 = tensor.empty() : tensor<8x512x512xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<512x512xf32>) outs(%2 : tensor<8x512x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x512x512xf32>
    %4 = linalg.fill ins(%cst_7 : f32) outs(%2 : tensor<8x512x512xf32>) -> tensor<8x512x512xf32>
    %5 = linalg.batch_matmul ins(%arg0, %3 : tensor<8x512x512xf32>, tensor<8x512x512xf32>) outs(%4 : tensor<8x512x512xf32>) -> tensor<8x512x512xf32>
    %6 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %cst_5 : tensor<8x512x512xf32>, tensor<512xf32>) outs(%2 : tensor<8x512x512xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %43 = arith.addf %in, %in_12 : f32
      linalg.yield %43 : f32
    } -> tensor<8x512x512xf32>
    %expanded = tensor.expand_shape %6 [[0], [1], [2, 3]] : tensor<8x512x512xf32> into tensor<8x512x8x64xf32>
    %7 = tensor.empty() : tensor<8x8x512x64xf32>
    %8 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<8x512x8x64xf32>) outs(%7 : tensor<8x8x512x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x8x512x64xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_4 : tensor<512x512xf32>) outs(%0 : tensor<512x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<512x512xf32>
    %10 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<512x512xf32>) outs(%2 : tensor<8x512x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x512x512xf32>
    %11 = linalg.batch_matmul ins(%arg1, %10 : tensor<8x512x512xf32>, tensor<8x512x512xf32>) outs(%4 : tensor<8x512x512xf32>) -> tensor<8x512x512xf32>
    %12 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11, %cst_3 : tensor<8x512x512xf32>, tensor<512xf32>) outs(%2 : tensor<8x512x512xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %43 = arith.addf %in, %in_12 : f32
      linalg.yield %43 : f32
    } -> tensor<8x512x512xf32>
    %expanded_10 = tensor.expand_shape %12 [[0], [1], [2, 3]] : tensor<8x512x512xf32> into tensor<8x512x8x64xf32>
    %13 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_10 : tensor<8x512x8x64xf32>) outs(%7 : tensor<8x8x512x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x8x512x64xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_2 : tensor<512x512xf32>) outs(%0 : tensor<512x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<512x512xf32>
    %15 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14 : tensor<512x512xf32>) outs(%2 : tensor<8x512x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x512x512xf32>
    %16 = linalg.batch_matmul ins(%arg2, %15 : tensor<8x512x512xf32>, tensor<8x512x512xf32>) outs(%4 : tensor<8x512x512xf32>) -> tensor<8x512x512xf32>
    %17 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16, %cst_1 : tensor<8x512x512xf32>, tensor<512xf32>) outs(%2 : tensor<8x512x512xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %43 = arith.addf %in, %in_12 : f32
      linalg.yield %43 : f32
    } -> tensor<8x512x512xf32>
    %expanded_11 = tensor.expand_shape %17 [[0], [1], [2, 3]] : tensor<8x512x512xf32> into tensor<8x512x8x64xf32>
    %18 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_11 : tensor<8x512x8x64xf32>) outs(%7 : tensor<8x8x512x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x8x512x64xf32>
    %19 = tensor.empty() : tensor<8x8x64x512xf32>
    %20 = linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13 : tensor<8x8x512x64xf32>) outs(%19 : tensor<8x8x64x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x8x64x512xf32>
    %21 = tensor.empty() : tensor<8x8x512x512xf32>
    %22 = linalg.fill ins(%cst_7 : f32) outs(%21 : tensor<8x8x512x512xf32>) -> tensor<8x8x512x512xf32>
    %23 = linalg.generic {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]} ins(%8, %20 : tensor<8x8x512x64xf32>, tensor<8x8x64x512xf32>) outs(%22 : tensor<8x8x512x512xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %43 = arith.mulf %in, %in_12 : f32
      %44 = arith.addf %43, %out : f32
      linalg.yield %44 : f32
    } -> tensor<8x8x512x512xf32>
    %24 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23 : tensor<8x8x512x512xf32>) outs(%21 : tensor<8x8x512x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %43 = arith.divf %in, %cst_9 : f32
      linalg.yield %43 : f32
    } -> tensor<8x8x512x512xf32>
    %25 = tensor.empty() : tensor<8x8x512x1xi64>
    %26 = linalg.fill ins(%c0_i64 : i64) outs(%25 : tensor<8x8x512x1xi64>) -> tensor<8x8x512x1xi64>
    %27 = tensor.empty() : tensor<8x8x512x1xf32>
    %28 = linalg.fill ins(%cst_8 : f32) outs(%27 : tensor<8x8x512x1xf32>) -> tensor<8x8x512x1xf32>
    %29:2 = linalg.generic {indexing_maps = [#map5, #map11, #map11], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%24 : tensor<8x8x512x512xf32>) outs(%28, %26 : tensor<8x8x512x1xf32>, tensor<8x8x512x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_12: i64):
      %43 = linalg.index 3 : index
      %44 = arith.index_cast %43 : index to i64
      %45 = arith.maximumf %in, %out : f32
      %46 = arith.cmpf ogt, %in, %out : f32
      %47 = arith.select %46, %44, %out_12 : i64
      linalg.yield %45, %47 : f32, i64
    } -> (tensor<8x8x512x1xf32>, tensor<8x8x512x1xi64>)
    %30 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24, %29#0 : tensor<8x8x512x512xf32>, tensor<8x8x512x1xf32>) outs(%21 : tensor<8x8x512x512xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %43 = arith.subf %in, %in_12 : f32
      linalg.yield %43 : f32
    } -> tensor<8x8x512x512xf32>
    %31 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30 : tensor<8x8x512x512xf32>) outs(%21 : tensor<8x8x512x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %43 = math.exp %in : f32
      linalg.yield %43 : f32
    } -> tensor<8x8x512x512xf32>
    %32 = linalg.fill ins(%cst_7 : f32) outs(%27 : tensor<8x8x512x1xf32>) -> tensor<8x8x512x1xf32>
    %33 = linalg.generic {indexing_maps = [#map5, #map11], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%31 : tensor<8x8x512x512xf32>) outs(%32 : tensor<8x8x512x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %43 = arith.addf %in, %out : f32
      linalg.yield %43 : f32
    } -> tensor<8x8x512x1xf32>
    %34 = linalg.generic {indexing_maps = [#map5, #map11, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %33 : tensor<8x8x512x512xf32>, tensor<8x8x512x1xf32>) outs(%21 : tensor<8x8x512x512xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %43 = arith.divf %in, %in_12 : f32
      linalg.yield %43 : f32
    } -> tensor<8x8x512x512xf32>
    %35 = linalg.fill ins(%cst_7 : f32) outs(%7 : tensor<8x8x512x64xf32>) -> tensor<8x8x512x64xf32>
    %36 = linalg.generic {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]} ins(%34, %18 : tensor<8x8x512x512xf32>, tensor<8x8x512x64xf32>) outs(%35 : tensor<8x8x512x64xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %43 = arith.mulf %in, %in_12 : f32
      %44 = arith.addf %43, %out : f32
      linalg.yield %44 : f32
    } -> tensor<8x8x512x64xf32>
    %37 = tensor.empty() : tensor<8x512x8x64xf32>
    %38 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36 : tensor<8x8x512x64xf32>) outs(%37 : tensor<8x512x8x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x512x8x64xf32>
    %collapsed = tensor.collapse_shape %38 [[0], [1], [2, 3]] : tensor<8x512x8x64xf32> into tensor<8x512x512xf32>
    %39 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<512x512xf32>) outs(%0 : tensor<512x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<512x512xf32>
    %40 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39 : tensor<512x512xf32>) outs(%2 : tensor<8x512x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x512x512xf32>
    %41 = linalg.batch_matmul ins(%collapsed, %40 : tensor<8x512x512xf32>, tensor<8x512x512xf32>) outs(%4 : tensor<8x512x512xf32>) -> tensor<8x512x512xf32>
    %42 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %cst : tensor<8x512x512xf32>, tensor<512xf32>) outs(%2 : tensor<8x512x512xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %43 = arith.addf %in, %in_12 : f32
      linalg.yield %43 : f32
    } -> tensor<8x512x512xf32>
    %cast = tensor.cast %42 : tensor<8x512x512xf32> to tensor<?x?x512xf32>
    return %cast : tensor<?x?x512xf32>
  }
}