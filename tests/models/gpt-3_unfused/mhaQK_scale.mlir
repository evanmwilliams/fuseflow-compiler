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
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j,k,l) -> (i : dense, j : dense, k : compressed, l : compressed)
}>
module attributes {torch.debug_module_name = "MultiHeadAttention"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg1: tensor<1x12x4096x4096xf32, #SPARSE>, %arg2: tensor<1x12x64x4096xf32, #SPARSE>, %arg3: tensor<1x12x4096x64xf32, #SPARSE>) -> tensor<1x12x4096x4096xf32, #SPARSE> {
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant 0xFF800000 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_9 = arith.constant 8.000000e+00 : f32
    %7 = tensor.empty() : tensor<1x12x4096x64xf32, #SPARSE>
    %19 = tensor.empty() : tensor<1x12x64x4096xf32, #SPARSE>
    %21 = tensor.empty() : tensor<1x12x4096x4096xf32, #SPARSE>
    %22 = linalg.fill ins(%cst_7 : f32) outs(%21 : tensor<1x12x4096x4096xf32, #SPARSE>) -> tensor<1x12x4096x4096xf32, #SPARSE>
    %8 = tensor.empty() : tensor<1x12x4096x64xf32, #SPARSE>
    %25 = tensor.empty() : tensor<1x12x4096x1xi64>
    %26 = linalg.fill ins(%c0_i64 : i64) outs(%25 : tensor<1x12x4096x1xi64>) -> tensor<1x12x4096x1xi64>
    %27 = tensor.empty() : tensor<1x12x4096x1xf32, #SPARSE>
    %28 = linalg.fill ins(%cst_8 : f32) outs(%27 : tensor<1x12x4096x1xf32, #SPARSE>) -> tensor<1x12x4096x1xf32, #SPARSE>
    %24 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x12x4096x4096xf32, #SPARSE>) outs(%21 : tensor<1x12x4096x4096xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      %43 = arith.divf %in, %cst_9 : f32
      linalg.yield %43 : f32
    } -> tensor<1x12x4096x4096xf32, #SPARSE>
    return %24 : tensor<1x12x4096x4096xf32, #SPARSE>
  }
}
