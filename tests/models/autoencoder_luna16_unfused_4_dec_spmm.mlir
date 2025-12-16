#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
// LUNA16 Unfused 4: Decoder SpMM only
// Input: (32, 512), Weight: (512, 262144) -> Output: (32, 262144)
module attributes {torch.debug_module_name = "LUNA16_Unfused_4_Dec_SpMM"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%input: tensor<32x512xf32, #SPARSE>) -> tensor<32x262144xf32, #SPARSE> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_decoder_weight = arith.constant dense_resource<__elided__> : tensor<512x262144xf32, #SPARSE>

    // SpMM: (32, 512) @ (512, 262144) = (32, 262144)
    %dec_out_empty = tensor.empty() : tensor<32x262144xf32, #SPARSE>
    %dec_out_init = linalg.fill ins(%cst_0 : f32) outs(%dec_out_empty : tensor<32x262144xf32, #SPARSE>) -> tensor<32x262144xf32, #SPARSE>
    %dec_matmul = linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%input, %cst_decoder_weight : tensor<32x512xf32, #SPARSE>, tensor<512x262144xf32, #SPARSE>)
      outs(%dec_out_init : tensor<32x262144xf32, #SPARSE>) {
    ^bb0(%in_h: f32, %in_w: f32, %out: f32):
      %prod = arith.mulf %in_h, %in_w : f32
      %sum = arith.addf %out, %prod : f32
      linalg.yield %sum : f32
    } -> tensor<32x262144xf32, #SPARSE>

    return %dec_matmul : tensor<32x262144xf32, #SPARSE>
  }
}
