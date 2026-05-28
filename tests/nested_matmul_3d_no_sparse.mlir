module {
func.func @kernel_batch_spmm(%arga: tensor<8x128x128xf64>,
                              %argb: tensor<8x128x128xf64>,
                              %argc: tensor<8x128x128xf64>,
                              %argd: tensor<8x128x128xf64>,
                              %arge: tensor<8x128x128xf64>,
                              %argf: tensor<8x128x128xf64>,
                              %argx: tensor<8x128x128xf64>) -> tensor<8x128x128xf64> {
    %0 = linalg.batch_matmul ins(%arga, %argb: tensor<8x128x128xf64>, tensor<8x128x128xf64>)
                             outs(%argx: tensor<8x128x128xf64>) -> tensor<8x128x128xf64>
    %1 = linalg.batch_matmul ins(%0, %argc: tensor<8x128x128xf64>, tensor<8x128x128xf64>)
                             outs(%argx: tensor<8x128x128xf64>) -> tensor<8x128x128xf64>
    %2 = linalg.batch_matmul ins(%1, %argd: tensor<8x128x128xf64>, tensor<8x128x128xf64>)
                             outs(%argx: tensor<8x128x128xf64>) -> tensor<8x128x128xf64>
    %3 = linalg.batch_matmul ins(%2, %arge: tensor<8x128x128xf64>, tensor<8x128x128xf64>)
                             outs(%argx: tensor<8x128x128xf64>) -> tensor<8x128x128xf64>
    %4 = linalg.batch_matmul ins(%3, %argf: tensor<8x128x128xf64>, tensor<8x128x128xf64>)
                             outs(%argx: tensor<8x128x128xf64>) -> tensor<8x128x128xf64>
    return %1 : tensor<8x128x128xf64>
}

}
