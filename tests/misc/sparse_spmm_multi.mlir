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
                       %argc: tensor<?x?xf64>,
                       %argd: tensor<?x?xf64>,
                       %argx: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic #spmm
      ins(%arga, %argb: tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%argx: tensor<?x?xf64>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %0 = arith.mulf %a, %b : f64
        %1 = arith.addf %x, %0 : f64
        %2 = math.exp %1 : f64
        linalg.yield %2 : f64
    } -> tensor<?x?xf64>
    %1 = linalg.generic #spmm
      ins(%0, %argc: tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%argx: tensor<?x?xf64>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %1 = arith.mulf %a, %b : f64
        %2 = arith.addf %x, %1 : f64
        linalg.yield %2 : f64
    } -> tensor<?x?xf64>
    %2 = linalg.generic #spmm
      ins(%1, %argd: tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%argx: tensor<?x?xf64>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %2 = arith.mulf %a, %b : f64
        %3 = arith.addf %x, %2 : f64
        linalg.yield %3 : f64
    } -> tensor<?x?xf64>
    return %2 : tensor<?x?xf64>
}

}