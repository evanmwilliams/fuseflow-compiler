#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>

#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
#SPARSE1D = #sparse_tensor.encoding<{
  map = (i) -> (i : compressed)
}>

// Unfused implementation of GCN function above from torch-mlir
// FIXME (owhsu): All functions commented out below are NOT working
func.func @matmul1 (%arg0 : tensor<1767x1767xf32, #SPARSE>, %arg1 : tensor<1767x50xf32>, %argout : tensor<1767x50xf32>) -> tensor<1767x50xf32> {
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<1767x1767xf32, #SPARSE>, tensor<1767x50xf32>) outs(%argout : tensor<1767x50xf32>) -> tensor<1767x50xf32>
  return %2 : tensor<1767x50xf32>
}