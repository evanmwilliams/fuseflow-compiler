#spmm = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>, // A
    affine_map<(i,j,k) -> (k,j)>, // B
    affine_map<(i,j,k) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) += A(i,k) * B(k,j)"
}

#CSR = #sparse_tensor.encoding<{
  map = (i,j) -> (i : dense, j : compressed)
}>

module {
func.func @kernel_spmm(%arga: tensor<?x?xf64>,
                       %argb: tensor<?x?xf64>,
                       %argx: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic #spmm
      ins(%arga, %argb: tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%argx: tensor<?x?xf64>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %0 = arith.mulf %a, %b : f64
        %1 = arith.addf %x, %0 : f64
        linalg.yield %1 : f64
    } -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
}

}