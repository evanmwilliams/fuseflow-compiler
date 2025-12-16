#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
// LUNA16 Unfused 1: Encoder SpMM only
// Input: (32, 262144), Weight: (262144, 512) -> Output: (32, 512)
module attributes {torch.debug_module_name = "LUNA16_Unfused_1_Enc_SpMM"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<32x262144xf32, #SPARSE>) -> tensor<32x512xf32, #SPARSE> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_encoder_weight = arith.constant dense_resource<__elided__> : tensor<262144x512xf32, #SPARSE>

    // SpMM: (32, 262144) @ (262144, 512) = (32, 512)
    %enc_out_empty = tensor.empty() : tensor<32x512xf32, #SPARSE>
    %enc_out_init = linalg.fill ins(%cst_0 : f32) outs(%enc_out_empty : tensor<32x512xf32, #SPARSE>) -> tensor<32x512xf32, #SPARSE>
    %enc_matmul = linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%arg0, %cst_encoder_weight : tensor<32x262144xf32, #SPARSE>, tensor<262144x512xf32, #SPARSE>)
      outs(%enc_out_init : tensor<32x512xf32, #SPARSE>) {
    ^bb0(%in_x: f32, %in_w: f32, %out: f32):
      %prod = arith.mulf %in_x, %in_w : f32
      %sum = arith.addf %out, %prod : f32
      linalg.yield %sum : f32
    } -> tensor<32x512xf32, #SPARSE>

    return %enc_matmul : tensor<32x512xf32, #SPARSE>
  }
}
