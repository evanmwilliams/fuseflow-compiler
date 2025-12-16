#spmm = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> (i,j)>, // B
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) += A(i,k) * B(k,j)"
}

#CSR = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>

func.func @kernel_spmm(%arga: tensor<?x?xf64, #CSR>,
                       %argb: tensor<?x?xf64, #CSR>,
                       %argx: tensor<?x?xf64, #CSR>) {
                      //  %argx: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic #spmm
      ins(%arga, %argb: tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>)
      outs(%argx: tensor<?x?xf64, #CSR>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %1 = arith.addf %a, %b : f64
        linalg.yield %1 : f64
    } -> tensor<?x?xf64, #CSR>
    return
    // return %0 : tensor<?x?xf64>
}