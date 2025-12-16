#spmm = {
  indexing_maps = [
    affine_map<(i,k,j) -> (i,k)>, // A
    affine_map<(i,k,j) -> (k,j)>, // B
    affine_map<(i,k,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "reduction", "parallel"],
  doc = "X(i,j) += A(i,k) * B(k,j)"
}

#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>

func.func @kernel_spmm(%arga: tensor<?x?xf64, #SPARSE>,
                       %argb: tensor<?x?xf64, #SPARSE>,
                       %argx: tensor<?x?xf64, #SPARSE>) {
    %0 = linalg.generic #spmm
      ins(%arga, %argb: tensor<?x?xf64, #SPARSE>, tensor<?x?xf64, #SPARSE>)
      outs(%argx: tensor<?x?xf64, #SPARSE>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %0 = arith.mulf %a, %b : f64
        %1 = arith.addf %x, %0 : f64
        linalg.yield %1 : f64
    } -> tensor<?x?xf64, #SPARSE>
    return
}