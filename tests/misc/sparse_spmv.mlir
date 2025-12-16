#spmv = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> (j)>, // B
    affine_map<(i,j) -> (i)>  // X (out)
  ],
  iterator_types = ["parallel", "reduction"],
  doc = "X(i) += A(i,j) * B(j)"
}

#CSR = #sparse_tensor.encoding<{
  map = (i,j) -> (i : dense, j : compressed)
}>

func.func @kernel_spmv(%arga: tensor<?x?xf64, #CSR>,
                       %argb: tensor<?xf64>,
                       %argx: tensor<?xf64>) {
                      //  %argx: tensor<64x64xf64, #CSR>) -> tensor<64x64xf64, #CSR> {
    %0 = linalg.generic #spmv
      ins(%arga, %argb: tensor<?x?xf64, #CSR>, tensor<?xf64>)
      outs(%argx: tensor<?xf64>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %0 = arith.mulf %a, %b : f64
        %1 = arith.addf %x, %0 : f64
        linalg.yield %1 : f64
    } -> tensor<?xf64>
    return
    // return %0 : tensor<64x64xf64, #CSR>
}

