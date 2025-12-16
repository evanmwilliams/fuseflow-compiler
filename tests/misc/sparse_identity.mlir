#identity = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // X (out)
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j)"
}

#CSR = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>

func.func @kernel_identity(%arga: tensor<?x?xf64, #CSR>,
                       %argx: tensor<?x?xf64, #CSR>) -> tensor<?x?xf64, #CSR> {
    %0 = linalg.generic #identity
      ins(%arga: tensor<?x?xf64, #CSR>)
      outs(%argx: tensor<?x?xf64, #CSR>) {
      ^bb(%a: f64, %x: f64):
        linalg.yield %a : f64
    } -> tensor<?x?xf64, #CSR>
    // return
    return %0 : tensor<?x?xf64, #CSR>
}