#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
// Encoder Fused: SpMM + Bias + ReLU
// Input: (32, 50176), Weight: (50176, 256), Bias: (256,) -> Output: (32, 256)
module attributes {torch.debug_module_name = "Encoder_Fused"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%input: tensor<32x50176xf32, #SPARSE>) -> tensor<32x256xf32, #SPARSE> {
    %cst_encoder_bias = arith.constant dense_resource<__elided__> : tensor<256xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_encoder_weight = arith.constant dense_resource<__elided__> : tensor<50176x256xf32, #SPARSE>

    // SpMM: (32, 50176) @ (50176, 256) = (32, 256)
    %enc_out_empty = tensor.empty() : tensor<32x256xf32, #SPARSE>
    %enc_out_init = linalg.fill ins(%cst_0 : f32) outs(%enc_out_empty : tensor<32x256xf32, #SPARSE>) -> tensor<32x256xf32, #SPARSE>
    %enc_matmul = linalg.generic {
      indexing_maps = [#map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%input, %cst_encoder_weight : tensor<32x50176xf32, #SPARSE>, tensor<50176x256xf32, #SPARSE>)
      outs(%enc_out_init : tensor<32x256xf32, #SPARSE>) {
    ^bb0(%in_x: f32, %in_w: f32, %out: f32):
      %prod = arith.mulf %in_x, %in_w : f32
      %sum = arith.addf %out, %prod : f32
      linalg.yield %sum : f32
    } -> tensor<32x256xf32, #SPARSE>

    // Bias add (broadcast)
    %enc_bias_out = tensor.empty() : tensor<32x256xf32, #SPARSE>
    %enc_biased = linalg.generic {
      indexing_maps = [#map, #map1, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%enc_matmul, %cst_encoder_bias : tensor<32x256xf32, #SPARSE>, tensor<256xf32>)
      outs(%enc_bias_out : tensor<32x256xf32, #SPARSE>) {
    ^bb0(%in: f32, %bias: f32, %out: f32):
      %sum = arith.addf %in, %bias : f32
      linalg.yield %sum : f32
    } -> tensor<32x256xf32, #SPARSE>

    // ReLU
    %enc_relu_out = tensor.empty() : tensor<32x256xf32, #SPARSE>
    %enc_relu = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%enc_biased : tensor<32x256xf32, #SPARSE>)
      outs(%enc_relu_out : tensor<32x256xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      %cmp = arith.cmpf ugt, %in, %cst_0 : f32
      %relu = arith.select %cmp, %in, %cst_0 : f32
      linalg.yield %relu : f32
    } -> tensor<32x256xf32, #SPARSE>

    return %enc_relu : tensor<32x256xf32, #SPARSE>
  }
}
