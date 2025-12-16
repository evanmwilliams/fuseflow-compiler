// MLIR test with two matrix multiplications, concatenation, and final matmul
module {
  func.func @two_matmuls_concat_final(%arg0: tensor<4x3xf32>, %arg1: tensor<3x2xf32>, %arg2: tensor<3x2xf32>, %arg3: tensor<2x5xf32>, %arg4: tensor<4x3xf32>) -> tensor<8x5xf32> {
//  func.func @two_matmuls_concat_final(%arg0: tensor<4x3xf32>, %arg1: tensor<3x2xf32>, %arg2: tensor<3x2xf32>, %arg3: tensor<2x5xf32>, %arg4: tensor<4x3xf32>) -> tensor<8x2xf32> {
    // Initialize output tensors for first two matmuls
    %c0 = arith.constant 0.0 : f32
    %init1 = tensor.empty() : tensor<4x2xf32>
    %init2 = tensor.empty() : tensor<4x2xf32>

    // First matrix multiplication: arg0 @ arg1
    %matmul1 = linalg.matmul ins(%arg0, %arg1 : tensor<4x3xf32>, tensor<3x2xf32>) outs(%init1 : tensor<4x2xf32>) -> tensor<4x2xf32>

    // Second matrix multiplication: arg0 @ arg2
    %matmul2 = linalg.matmul ins(%arg4, %arg2 : tensor<4x3xf32>, tensor<3x2xf32>) outs(%init2 : tensor<4x2xf32>) -> tensor<4x2xf32>

    // Concatenate the results along dimension 0 (rows)
    %concat = tensor.concat dim(0) %matmul1, %matmul2 : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<8x2xf32>

    // Initialize output tensor for final matmul
    %init3 = tensor.empty() : tensor<8x5xf32>

    // Final matrix multiplication: concat @ arg3
    %final_matmul = linalg.matmul ins(%concat, %arg3 : tensor<8x2xf32>, tensor<2x5xf32>) outs(%init3 : tensor<8x5xf32>) -> tensor<8x5xf32>

    return %final_matmul : tensor<8x5xf32>
//    return %concat : tensor<8x2xf32>
  }
}