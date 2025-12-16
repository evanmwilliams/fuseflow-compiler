#spmm = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>, // A
    affine_map<(i,j,k) -> (k,j)>, // B
    affine_map<(i,j,k) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "reduction", "parallel"],
  doc = "X(i,j) += A(i,k) * B(k,j)"
}

#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>

module {
func.func @kernel_spmm(%arga: tensor<?x?xf64, #SPARSE>,
                       %argb: tensor<?x?xf64, #SPARSE>,
                       %argc: tensor<?x?xf64, #SPARSE>,
                       %argd: tensor<?x?xf64, #SPARSE>,
                       %arge: tensor<?x?xf64, #SPARSE>,
                       %argf: tensor<?x?xf64, #SPARSE>,
                       %argx: tensor<?x?xf64, #SPARSE>) -> tensor<?x?xf64, #SPARSE> {
    %0 = linalg.matmul ins(%arga, %argb: tensor<?x?xf64, #SPARSE>, tensor<?x?xf64, #SPARSE>)
                       outs(%argx: tensor<?x?xf64, #SPARSE>) -> tensor<?x?xf64, #SPARSE>
    %1 = linalg.matmul ins(%argc, %argd: tensor<?x?xf64, #SPARSE>, tensor<?x?xf64, #SPARSE>)
                       outs(%argx: tensor<?x?xf64, #SPARSE>) -> tensor<?x?xf64, #SPARSE>
    %2 = linalg.add ins(%0, %1: tensor<?x?xf64, #SPARSE>, tensor<?x?xf64, #SPARSE>)
                       outs(%argx: tensor<?x?xf64, #SPARSE>) -> tensor<?x?xf64, #SPARSE>
    %3 = linalg.matmul ins(%2, %arge: tensor<?x?xf64, #SPARSE>, tensor<?x?xf64, #SPARSE>)
                       outs(%argx: tensor<?x?xf64, #SPARSE>) -> tensor<?x?xf64, #SPARSE>
    %4 = linalg.matmul ins(%3, %argf: tensor<?x?xf64, #SPARSE>, tensor<?x?xf64, #SPARSE>)
                       outs(%argx: tensor<?x?xf64, #SPARSE>) -> tensor<?x?xf64, #SPARSE>
    return %2 : tensor<?x?xf64, #SPARSE>
}

}