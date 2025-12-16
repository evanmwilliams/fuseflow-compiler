#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
#map2 = affine_map<(d0, d1) -> (d1, d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant dense_resource<torch_tensor_128_128_torch.float32> : tensor<128x128xf32>
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant 0xFF800000 : f32
    %cst_3 = arith.constant 1.000000e-05 : f64
    %cst_4 = arith.constant 1.280000e+02 : f64
    %cst_5 = arith.constant 1.280000e+02 : f32
    %cst_6 = arith.constant dense_resource<torch_tensor_128_torch.float32> : tensor<128xf32>
    %cst_7 = arith.constant dense_resource<torch_tensor_128_128_torch.float32_1> : tensor<128x128xf32>
    %cst_8 = arith.constant dense_resource<torch_tensor_128_torch.float32_1> : tensor<128xf32>
    %cst_9 = arith.constant dense_resource<torch_tensor_128_128_torch.float32_2> : tensor<128x128xf32>
    %cst_10 = arith.constant dense_resource<torch_tensor_128_torch.float32_2> : tensor<128xf32>
    %cst_11 = arith.constant dense_resource<torch_tensor_128_128_torch.float32_3> : tensor<128x128xf32>
    %0 = tensor.empty() : tensor<128x128xf64>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%0 : tensor<128x128xf64>) {
    ^bb0(%in: f32, %out: f64):
      %41 = arith.extf %in : f32 to f64
      linalg.yield %41 : f64
    } -> tensor<128x128xf64>
    %2 = tensor.empty() : tensor<128x1xf64>
    %3 = linalg.fill ins(%cst_0 : f64) outs(%2 : tensor<128x1xf64>) -> tensor<128x1xf64>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%1 : tensor<128x128xf64>) outs(%3 : tensor<128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %41 = arith.addf %in, %out : f64
      linalg.yield %41 : f64
    } -> tensor<128x1xf64>
    %5 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<128x1xf64>) outs(%2 : tensor<128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %41 = arith.divf %in, %cst_4 : f64
      linalg.yield %41 : f64
    } -> tensor<128x1xf64>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %5 : tensor<128x128xf64>, tensor<128x1xf64>) outs(%0 : tensor<128x128xf64>) {
    ^bb0(%in: f64, %in_11: f64, %out: f64):
      %41 = arith.subf %in, %in_11 : f64
      linalg.yield %41 : f64
    } -> tensor<128x128xf64>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %6 : tensor<128x128xf64>, tensor<128x128xf64>) outs(%0 : tensor<128x128xf64>) {
    ^bb0(%in: f64, %in_11: f64, %out: f64):
      %41 = arith.mulf %in, %in_11 : f64
      linalg.yield %41 : f64
    } -> tensor<128x128xf64>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%7 : tensor<128x128xf64>) outs(%3 : tensor<128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %41 = arith.addf %in, %out : f64
      linalg.yield %41 : f64
    } -> tensor<128x1xf64>
    %9 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<128x1xf64>) outs(%2 : tensor<128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %41 = arith.divf %in, %cst_4 : f64
      linalg.yield %41 : f64
    } -> tensor<128x1xf64>
    %10 = tensor.empty() : tensor<128x1xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<128x1xf64>) outs(%10 : tensor<128x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %41 = arith.truncf %in : f64 to f32
      linalg.yield %41 : f32
    } -> tensor<128x1xf32>
    %12 = linalg.fill ins(%cst_1 : f32) outs(%10 : tensor<128x1xf32>) -> tensor<128x1xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<128x128xf32>) outs(%12 : tensor<128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %41 = arith.addf %in, %out : f32
      linalg.yield %41 : f32
    } -> tensor<128x1xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<128x1xf32>) outs(%10 : tensor<128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %41 = arith.divf %in, %cst_5 : f32
      linalg.yield %41 : f32
    } -> tensor<128x1xf32>
    %15 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<128x1xf32>) outs(%10 : tensor<128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %41 = arith.truncf %cst_3 : f64 to f32
      %42 = arith.addf %in, %41 : f32
      linalg.yield %42 : f32
    } -> tensor<128x1xf32>
    %16 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%15 : tensor<128x1xf32>) outs(%10 : tensor<128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %41 = math.rsqrt %in : f32
      linalg.yield %41 : f32
    } -> tensor<128x1xf32>
    %17 = tensor.empty() : tensor<128x128xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %14 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%17 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %41 = arith.subf %in, %in_11 : f32
      linalg.yield %41 : f32
    } -> tensor<128x128xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%18, %16 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%17 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %41 = arith.mulf %in, %in_11 : f32
      linalg.yield %41 : f32
    } -> tensor<128x128xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<128x128xf32>) outs(%17 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x128xf32>
    %21 = linalg.fill ins(%cst_1 : f32) outs(%17 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %22 = linalg.matmul ins(%19, %20 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%21 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%22, %cst_6 : tensor<128x128xf32>, tensor<128xf32>) outs(%17 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %41 = arith.addf %in, %in_11 : f32
      linalg.yield %41 : f32
    } -> tensor<128x128xf32>
    %24 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel"]} ins(%cst_7 : tensor<128x128xf32>) outs(%17 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x128xf32>
    %25 = linalg.matmul ins(%19, %24 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%21 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %26 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%25, %cst_8 : tensor<128x128xf32>, tensor<128xf32>) outs(%17 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %41 = arith.addf %in, %in_11 : f32
      linalg.yield %41 : f32
    } -> tensor<128x128xf32>
    %27 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel"]} ins(%cst_9 : tensor<128x128xf32>) outs(%17 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x128xf32>
    %28 = linalg.matmul ins(%19, %27 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%21 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%28, %cst_10 : tensor<128x128xf32>, tensor<128xf32>) outs(%17 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %41 = arith.addf %in, %in_11 : f32
      linalg.yield %41 : f32
    } -> tensor<128x128xf32>
    //%30 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%23, %26 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%17 : tensor<128x128xf32>) {
    //^bb0(%in: f32, %in_11: f32, %out: f32):
      //%41 = arith.addf %in, %in_11 : f32
      //linalg.yield %41 : f32
    //} -> tensor<128x128xf32>
    %30 = linalg.matmul ins(%22, %25 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%17 : tensor<128x128xf32>) -> tensor<128x128xf32>
    //%30 = linalg.add ins(%23, %26 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%17 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %31 = tensor.empty() : tensor<128x1xi64>
    %32 = linalg.fill ins(%c0_i64 : i64) outs(%31 : tensor<128x1xi64>) -> tensor<128x1xi64>
    %33 = tensor.empty() : tensor<128x1xf32>
    %34 = linalg.fill ins(%cst_2 : f32) outs(%33 : tensor<128x1xf32>) -> tensor<128x1xf32>
    %35:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%30 : tensor<128x128xf32>) outs(%34, %32 : tensor<128x1xf32>, tensor<128x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_11: i64):
      %41 = linalg.index 1 : index
      %42 = arith.index_cast %41 : index to i64
      %43 = arith.maximumf %in, %out : f32
      %44 = arith.cmpf ogt, %in, %out : f32
      %45 = arith.select %44, %42, %out_11 : i64
      linalg.yield %43, %45 : f32, i64
    } -> (tensor<128x1xf32>, tensor<128x1xi64>)
    //%expanded = tensor.expand_shape %35#0 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %36 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%30, %35#0 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%17 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %41 = arith.subf %in, %in_11 : f32
      linalg.yield %41 : f32
    } -> tensor<128x128xf32>
    %37 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%36 : tensor<128x128xf32>) outs(%17 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %41 = math.exp %in : f32
      linalg.yield %41 : f32
    } -> tensor<128x128xf32>
    %38 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%37 : tensor<128x128xf32>) outs(%12 : tensor<128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %41 = arith.addf %in, %out : f32
      linalg.yield %41 : f32
    } -> tensor<128x1xf32>
    %39 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%37, %38 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%17 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %41 = arith.divf %in, %in_11 : f32
      linalg.yield %41 : f32
    } -> tensor<128x128xf32>
    //%40 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%39, %29 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%17 : tensor<128x128xf32>) {
    //^bb0(%in: f32, %in_11: f32, %out: f32):
      //%41 = arith.addf %in, %in_11 : f32
      //linalg.yield %41 : f32
    //} -> tensor<128x128xf32>
    %40 = linalg.matmul ins(%39, %28 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%17 : tensor<128x128xf32>) -> tensor<128x128xf32>
    //%40 = linalg.add ins(%39, %29 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%17 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %41 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel"]} ins(%cst_11 : tensor<128x128xf32>) outs(%17 : tensor<128x128xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<128x128xf32>
    %42 = linalg.matmul ins(%40, %41 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%21 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %43 = linalg.add ins(%42, %arg0 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%21 : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %43 : tensor<128x128xf32>
  }
}