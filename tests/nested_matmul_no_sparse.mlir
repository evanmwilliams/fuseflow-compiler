module {
func.func @kernel_spmm(%arga: tensor<128x128xf64>,
                       %argb: tensor<128x128xf64>,
                       %argc: tensor<128x128xf64>,
                       %argd: tensor<128x128xf64>,
                       %arge: tensor<128x128xf64>,
                       %argf: tensor<128x128xf64>,
                       %argx: tensor<128x128xf64>) -> tensor<128x128xf64> {
    %0 = linalg.matmul ins(%arga, %argb: tensor<128x128xf64>, tensor<128x128xf64>)
                       outs(%argx: tensor<128x128xf64>) -> tensor<128x128xf64>
    %1 = linalg.matmul ins(%0, %argc: tensor<128x128xf64>, tensor<128x128xf64>)
                       outs(%argx: tensor<128x128xf64>) -> tensor<128x128xf64>
    %2 = linalg.matmul ins(%1, %argd: tensor<128x128xf64>, tensor<128x128xf64>)
                       outs(%argx: tensor<128x128xf64>) -> tensor<128x128xf64>
    %3 = linalg.matmul ins(%2, %arge: tensor<128x128xf64>, tensor<128x128xf64>)
                       outs(%argx: tensor<128x128xf64>) -> tensor<128x128xf64>
    %4 = linalg.matmul ins(%3, %argf: tensor<128x128xf64>, tensor<128x128xf64>)
                       outs(%argx: tensor<128x128xf64>) -> tensor<128x128xf64>
    return %1 : tensor<128x128xf64>
}

}
