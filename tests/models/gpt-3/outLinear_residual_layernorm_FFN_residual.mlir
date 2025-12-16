// -----// IR Dump Before VerifyLinalgOnTensorsBackendContract (torch-verify-linalg-on-tensors-backend-contract) ('builtin.module' operation) //----- //
// Partially fused component: OutLinear + Residual + Layernorm + FFN + Residual
// (Does NOT include next decoder's layernorm and QKV projection)
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map6 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
#map7 = affine_map<(d0, d1, d2) -> (d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<1x1024x768xf32>, %arg1: tensor<1x1024x768xf32>) -> tensor<1x1024x768xf32> {
    %cst = arith.constant dense_resource<torch_tensor_768_768_torch.float32> : tensor<768x768xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f64
    %cst_2 = arith.constant 1.000000e-05 : f64
    %cst_3 = arith.constant 7.680000e+02 : f64
    %cst_4 = arith.constant 7.680000e+02 : f32
    %cst_5 = arith.constant dense_resource<torch_tensor_768_torch.float32> : tensor<768xf32>
    %cst_6 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32> : tensor<3072x768xf32>
    %cst_7 = arith.constant dense_resource<torch_tensor_3072_torch.float32> : tensor<3072xf32>
    %cst_8 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32> : tensor<768x3072xf32>
    %cst_9 = arith.constant dense_resource<torch_tensor_768_torch.float32_1> : tensor<768xf32>
    // OutLinear matmul: transpose weight
    %0 = tensor.empty() : tensor<768x768xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<768x768xf32>) outs(%0 : tensor<768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<768x768xf32>
    %2 = tensor.empty() : tensor<1x1024x768xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1x1024x768xf32>) -> tensor<1x1024x768xf32>
    // OutLinear matmul
    %4 = linalg.generic {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %1 : tensor<1x1024x768xf32>, tensor<768x768xf32>) outs(%2 : tensor<1x1024x768xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %26 = arith.mulf %in, %in_6 : f32
      %27 = arith.addf %26, %out : f32
      linalg.yield %27 : f32
    } -> tensor<1x1024x768xf32>
    // OutLinear bias
    %5 = linalg.generic {indexing_maps = [#map4, #map7, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %cst_5 : tensor<1x1024x768xf32>, tensor<768xf32>) outs(%2 : tensor<1x1024x768xf32>) {
    ^bb0(%in: f32, %in_18: f32, %out: f32):
      %57 = arith.addf %in, %in_18 : f32
      linalg.yield %57 : f32
    } -> tensor<1x1024x768xf32>
    // Residual 1: add with arg1
    %6 = tensor.empty() : tensor<1x1024x768xf32>
    %7 = linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %arg1 : tensor<1x1024x768xf32>, tensor<1x1024x768xf32>) outs(%6 : tensor<1x1024x768xf32>) {
    ^bb0(%in: f32, %in_18: f32, %out: f32):
      %57 = arith.addf %in, %in_18 : f32
      linalg.yield %57 : f32
    } -> tensor<1x1024x768xf32>
    // Layernorm: f32 -> f64 conversion
    %8 = tensor.empty() : tensor<1x1024x768xf64>
    %9 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<1x1024x768xf32>) outs(%8 : tensor<1x1024x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %57 = arith.extf %in : f32 to f64
      linalg.yield %57 : f64
    } -> tensor<1x1024x768xf64>
    // Layernorm: sum reduction for mean (f64)
    %10 = tensor.empty() : tensor<1x1024x1xf64>
    %11 = linalg.fill ins(%cst_1 : f64) outs(%10 : tensor<1x1024x1xf64>) -> tensor<1x1024x1xf64>
    %12 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%9 : tensor<1x1024x768xf64>) outs(%11 : tensor<1x1024x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %57 = arith.addf %in, %out : f64
      linalg.yield %57 : f64
    } -> tensor<1x1024x1xf64>
    // Layernorm: divide by dim for mean
    %13 = linalg.generic {indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12 : tensor<1x1024x1xf64>) outs(%10 : tensor<1x1024x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %57 = arith.divf %in, %cst_3 : f64
      linalg.yield %57 : f64
    } -> tensor<1x1024x1xf64>
    // Layernorm: subtract mean
    %14 = linalg.generic {indexing_maps = [#map3, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %13 : tensor<1x1024x768xf64>, tensor<1x1024x1xf64>) outs(%8 : tensor<1x1024x768xf64>) {
    ^bb0(%in: f64, %in_18: f64, %out: f64):
      %57 = arith.subf %in, %in_18 : f64
      linalg.yield %57 : f64
    } -> tensor<1x1024x768xf64>
    // Layernorm: square for variance
    %15 = linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14, %14 : tensor<1x1024x768xf64>, tensor<1x1024x768xf64>) outs(%8 : tensor<1x1024x768xf64>) {
    ^bb0(%in: f64, %in_18: f64, %out: f64):
      %57 = arith.mulf %in, %in_18 : f64
      linalg.yield %57 : f64
    } -> tensor<1x1024x768xf64>
    // Layernorm: sum squared differences
    %16 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : tensor<1x1024x768xf64>) outs(%11 : tensor<1x1024x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %57 = arith.addf %in, %out : f64
      linalg.yield %57 : f64
    } -> tensor<1x1024x1xf64>
    // Layernorm: divide for variance
    %17 = linalg.generic {indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16 : tensor<1x1024x1xf64>) outs(%10 : tensor<1x1024x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %57 = arith.divf %in, %cst_3 : f64
      linalg.yield %57 : f64
    } -> tensor<1x1024x1xf64>
    // Layernorm: truncate variance to f32
    %18 = tensor.empty() : tensor<1x1024x1xf32>
    %19 = linalg.generic {indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17 : tensor<1x1024x1xf64>) outs(%18 : tensor<1x1024x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %57 = arith.truncf %in : f64 to f32
      linalg.yield %57 : f32
    } -> tensor<1x1024x1xf32>
    // Layernorm: f32 mean calculation
    %20 = linalg.fill ins(%cst_0 : f32) outs(%18 : tensor<1x1024x1xf32>) -> tensor<1x1024x1xf32>
    %21 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%7 : tensor<1x1024x768xf32>) outs(%20 : tensor<1x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %57 = arith.addf %in, %out : f32
      linalg.yield %57 : f32
    } -> tensor<1x1024x1xf32>
    %22 = linalg.generic {indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%21 : tensor<1x1024x1xf32>) outs(%18 : tensor<1x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %57 = arith.divf %in, %cst_4 : f32
      linalg.yield %57 : f32
    } -> tensor<1x1024x1xf32>
    // Layernorm: add epsilon
    %23 = linalg.generic {indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19 : tensor<1x1024x1xf32>) outs(%18 : tensor<1x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %57 = arith.truncf %cst_2 : f64 to f32
      %58 = arith.addf %in, %57 : f32
      linalg.yield %58 : f32
    } -> tensor<1x1024x1xf32>
    // Layernorm: rsqrt
    %24 = linalg.generic {indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23 : tensor<1x1024x1xf32>) outs(%18 : tensor<1x1024x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %57 = math.rsqrt %in : f32
      linalg.yield %57 : f32
    } -> tensor<1x1024x1xf32>
    // Layernorm: subtract mean (f32)
    %25 = linalg.generic {indexing_maps = [#map3, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7, %22 : tensor<1x1024x768xf32>, tensor<1x1024x1xf32>) outs(%6 : tensor<1x1024x768xf32>) {
    ^bb0(%in: f32, %in_18: f32, %out: f32):
      %57 = arith.subf %in, %in_18 : f32
      linalg.yield %57 : f32
    } -> tensor<1x1024x768xf32>
    // Layernorm: multiply by rsqrt (normalize)
    %26 = linalg.generic {indexing_maps = [#map3, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %24 : tensor<1x1024x768xf32>, tensor<1x1024x1xf32>) outs(%6 : tensor<1x1024x768xf32>) {
    ^bb0(%in: f32, %in_18: f32, %out: f32):
      %57 = arith.mulf %in, %in_18 : f32
      linalg.yield %57 : f32
    } -> tensor<1x1024x768xf32>

    // Start FFN1: transpose weight
    %27 = tensor.empty() : tensor<768x3072xf32>
    %28 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_6 : tensor<3072x768xf32>) outs(%27 : tensor<768x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<768x3072xf32>
    %29 = tensor.empty() : tensor<1x1024x3072xf32>
    %30 = linalg.fill ins(%cst_0 : f32) outs(%29 : tensor<1x1024x3072xf32>) -> tensor<1x1024x3072xf32>
    // FFN1 matmul
    %31 = linalg.generic {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%26, %28 : tensor<1x1024x768xf32>, tensor<768x3072xf32>) outs(%29 : tensor<1x1024x3072xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %58 = arith.mulf %in, %in_6 : f32
      %59 = arith.addf %58, %out : f32
      linalg.yield %59 : f32
    } -> tensor<1x1024x3072xf32>
    // FFN1 bias
    %32 = linalg.generic {indexing_maps = [#map4, #map7, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %cst_7 : tensor<1x1024x3072xf32>, tensor<3072xf32>) outs(%29 : tensor<1x1024x3072xf32>) {
    ^bb0(%in: f32, %in_18: f32, %out: f32):
      %57 = arith.addf %in, %in_18 : f32
      linalg.yield %57 : f32
    } -> tensor<1x1024x3072xf32>
    // FFN1 ReLU
    %33 = tensor.empty() : tensor<1x1024x3072xf32>
    %34 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32 : tensor<1x1024x3072xf32>) outs(%33 : tensor<1x1024x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %57 = arith.cmpf ugt, %in, %cst_0 : f32
      %58 = arith.select %57, %in, %cst_0 : f32
      linalg.yield %58 : f32
    } -> tensor<1x1024x3072xf32>

    // Start FFN2: transpose weight
    %35 = tensor.empty() : tensor<3072x768xf32>
    %36 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_8 : tensor<768x3072xf32>) outs(%35 : tensor<3072x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3072x768xf32>
    // FFN2 matmul
    %37 = linalg.generic {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%34, %36 : tensor<1x1024x3072xf32>, tensor<3072x768xf32>) outs(%6 : tensor<1x1024x768xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %58 = arith.mulf %in, %in_6 : f32
      %59 = arith.addf %58, %out : f32
      linalg.yield %59 : f32
    } -> tensor<1x1024x768xf32>
    // FFN2 bias
    %38 = linalg.generic {indexing_maps = [#map4, #map7, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %cst_9 : tensor<1x1024x768xf32>, tensor<768xf32>) outs(%6 : tensor<1x1024x768xf32>) {
    ^bb0(%in: f32, %in_18: f32, %out: f32):
      %57 = arith.addf %in, %in_18 : f32
      linalg.yield %57 : f32
    } -> tensor<1x1024x768xf32>

    // Residual 2: add FFN output with pre-layernorm input (%7)
    %39 = linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %7 : tensor<1x1024x768xf32>, tensor<1x1024x768xf32>) outs(%6 : tensor<1x1024x768xf32>) {
    ^bb0(%in: f32, %in_18: f32, %out: f32):
      %57 = arith.addf %in, %in_18 : f32
      linalg.yield %57 : f32
    } -> tensor<1x1024x768xf32>

    // Return after Residual2 (does NOT include next decoder's layernorm and QKV projection)
    return %39 : tensor<1x1024x768xf32>
  }
}
