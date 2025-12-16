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
  //func.func @main(%arg0: tensor<1x1024x768xf32>, %arg1: tensor<1x1024x768xf32>, %cst : tensor<768x768xf32>) -> tensor<1x1024x768xf32> {
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
    %cst_10 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_1> : tensor<768x768xf32>
    %cst_11 = arith.constant dense_resource<torch_tensor_768_torch.float32_2> : tensor<768xf32>
    %0 = tensor.empty() : tensor<1x1024x768xf32>
    %1 = linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x1024x768xf32>, tensor<1x1024x768xf32>) outs(%0 : tensor<1x1024x768xf32>) {
    ^bb0(%in: f32, %in_18: f32, %out: f32):
      %57 = arith.addf %in, %in_18 : f32
      linalg.yield %57 : f32
    } -> tensor<1x1024x768xf32>
    return %1 : tensor<1x1024x768xf32>
  }
 }