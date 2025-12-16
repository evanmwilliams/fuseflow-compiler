
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>

module {
func.func @kernel_spmm(%arga: tensor<?x?xf64, #SPARSE>,
                       %argb: tensor<?x?xf64, #SPARSE>,
                       %argc: tensor<?x?xf64, #SPARSE>,
                       %argx: tensor<?x?xf64, #SPARSE>) -> tensor<?x?xf64, #SPARSE> {
    %0 = linalg.matmul ins(%arga, %argb: tensor<?x?xf64, #SPARSE>, tensor<?x?xf64, #SPARSE>)
                       outs(%argx: tensor<?x?xf64, #SPARSE>) -> tensor<?x?xf64, #SPARSE>
    %1 = linalg.matmul ins(%argb, %0: tensor<?x?xf64, #SPARSE>, tensor<?x?xf64, #SPARSE>)
                       outs(%argx: tensor<?x?xf64, #SPARSE>) -> tensor<?x?xf64, #SPARSE>
    // %2 = linalg.matmul ins(%1, %0: tensor<?x?xf64, #SPARSE>, tensor<?x?xf64, #SPARSE>)
    //                    outs(%argx: tensor<?x?xf64, #SPARSE>) -> tensor<?x?xf64, #SPARSE>
    return %1 : tensor<?x?xf64, #SPARSE>
}

}