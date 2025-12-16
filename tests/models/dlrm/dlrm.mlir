#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map5 = affine_map<(d0) -> (d0)>
#map6 = affine_map<(d0, d1) -> (0, 0)>
module {
  func.func @main(%arg0: tensor<1x4xf32>, %arg1: tensor<3x1xi64>, %arg2: tensor<3xi64>, %arg3: tensor<1xi64>, %arg4: tensor<1xi64>) -> tensor<1x1xf32> {
    %cst = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c4_i64 = arith.constant 4 : i64
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_2_3_torch.float32> : tensor<2x3xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_2_torch.float32> : tensor<2xf32>
    %c0_i64 = arith.constant 0 : i64
    %cst_5 = arith.constant dense_resource<torch_tensor_4_2_torch.float32> : tensor<4x2xf32>
    %c1_i64 = arith.constant 1 : i64
    %cst_6 = arith.constant dense_resource<torch_tensor_3_2_torch.float32> : tensor<3x2xf32>
    %cst_7 = arith.constant dense_resource<torch_tensor_2_2_torch.float32> : tensor<2x2xf32>
    %cst_8 = arith.constant dense_resource<torch_tensor_6_torch.int64> : tensor<6xi64>
    %cst_9 = arith.constant dense_resource<torch_tensor_6_torch.int64_1> : tensor<6xi64>
    %cst_10 = arith.constant dense_resource<torch_tensor_4_8_torch.float32> : tensor<4x8xf32>
    %cst_11 = arith.constant dense_resource<torch_tensor_4_torch.float32> : tensor<4xf32>
    %cst_12 = arith.constant dense_resource<torch_tensor_2_4_torch.float32> : tensor<2x4xf32>
    %cst_13 = arith.constant dense_resource<torch_tensor_2_torch.float32_1> : tensor<2xf32>
    %cst_14 = arith.constant dense_resource<torch_tensor_1_2_torch.float32> : tensor<1x2xf32>
    %cst_15 = arith.constant dense<-1.2125231> : tensor<1xf32>
    %c3_i64 = arith.constant 3 : i64
    %0 = tensor.empty() : tensor<4x3xf32>
    %transposed = linalg.transpose ins(%cst_2 : tensor<3x4xf32>) outs(%0 : tensor<4x3xf32>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<1x3xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %3 = linalg.matmul ins(%arg0, %transposed : tensor<1x4xf32>, tensor<4x3xf32>) outs(%2 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %cst : tensor<1x3xf32>, tensor<3xf32>) outs(%1 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_27: f32, %out: f32):
      %39 = arith.addf %in, %in_27 : f32
      linalg.yield %39 : f32
    } -> tensor<1x3xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<1x3xf32>) outs(%1 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %39 = arith.cmpf ugt, %in, %cst_0 : f32
      %40 = arith.select %39, %in, %cst_0 : f32
      linalg.yield %40 : f32
    } -> tensor<1x3xf32>
    %6 = tensor.empty() : tensor<3x2xf32>
    %transposed_16 = linalg.transpose ins(%cst_3 : tensor<2x3xf32>) outs(%6 : tensor<3x2xf32>) permutation = [1, 0] 
    %7 = tensor.empty() : tensor<1x2xf32>
    %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<1x2xf32>) -> tensor<1x2xf32>
    %9 = linalg.matmul ins(%5, %transposed_16 : tensor<1x3xf32>, tensor<3x2xf32>) outs(%8 : tensor<1x2xf32>) -> tensor<1x2xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%9, %cst_4 : tensor<1x2xf32>, tensor<2xf32>) outs(%7 : tensor<1x2xf32>) {
    ^bb0(%in: f32, %in_27: f32, %out: f32):
      %39 = arith.addf %in, %in_27 : f32
      linalg.yield %39 : f32
    } -> tensor<1x2xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<1x2xf32>) outs(%7 : tensor<1x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %39 = arith.cmpf ugt, %in, %cst_0 : f32
      %40 = arith.select %39, %in, %cst_0 : f32
      linalg.yield %40 : f32
    } -> tensor<1x2xf32>
    %extracted_slice = tensor.extract_slice %arg1[0, 0] [1, 1] [1, 1] : tensor<3x1xi64> to tensor<1x1xi64>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x1xi64> into tensor<1xi64>
    %12 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg2, %collapsed : tensor<3xi64>, tensor<1xi64>) outs(%8 : tensor<1x2xf32>) {
    ^bb0(%in: i64, %in_27: i64, %out: f32):
      %39 = linalg.index 1 : index
      %40 = arith.index_cast %39 : index to i64
      %41 = arith.cmpi slt, %in_27, %40 : i64
      %42 = arith.cmpi eq, %in_27, %40 : i64
      %43 = arith.ori %41, %42 : i1
      %44 = arith.cmpi slt, %40, %c3_i64 : i64
      %45 = arith.andi %43, %44 : i1
      %46 = arith.index_cast %in : i64 to index
      %47 = linalg.index 2 : index
      %extracted = tensor.extract %cst_5[%46, %47] : tensor<4x2xf32>
      %48 = arith.addf %extracted, %out : f32
      %49 = arith.select %45, %48, %out : f32
      linalg.yield %49 : f32
    } -> tensor<1x2xf32>
    %extracted_slice_17 = tensor.extract_slice %arg1[1, 0] [1, 1] [1, 1] : tensor<3x1xi64> to tensor<1x1xi64>
    %collapsed_18 = tensor.collapse_shape %extracted_slice_17 [[0, 1]] : tensor<1x1xi64> into tensor<1xi64>
    %13 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg3, %collapsed_18 : tensor<1xi64>, tensor<1xi64>) outs(%8 : tensor<1x2xf32>) {
    ^bb0(%in: i64, %in_27: i64, %out: f32):
      %39 = arith.cmpi slt, %in_27, %c0_i64 : i64
      %40 = arith.cmpi eq, %in_27, %c0_i64 : i64
      %41 = arith.ori %39, %40 : i1
      %42 = arith.index_cast %in : i64 to index
      %43 = linalg.index 2 : index
      %extracted = tensor.extract %cst_6[%42, %43] : tensor<3x2xf32>
      %44 = arith.addf %extracted, %out : f32
      %45 = arith.select %41, %44, %out : f32
      linalg.yield %45 : f32
    } -> tensor<1x2xf32>
    %extracted_slice_19 = tensor.extract_slice %arg1[2, 0] [1, 1] [1, 1] : tensor<3x1xi64> to tensor<1x1xi64>
    %collapsed_20 = tensor.collapse_shape %extracted_slice_19 [[0, 1]] : tensor<1x1xi64> into tensor<1xi64>
    %14 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg4, %collapsed_20 : tensor<1xi64>, tensor<1xi64>) outs(%8 : tensor<1x2xf32>) {
    ^bb0(%in: i64, %in_27: i64, %out: f32):
      %39 = arith.cmpi slt, %in_27, %c0_i64 : i64
      %40 = arith.cmpi eq, %in_27, %c0_i64 : i64
      %41 = arith.ori %39, %40 : i1
      %42 = arith.index_cast %in : i64 to index
      %43 = linalg.index 2 : index
      %extracted = tensor.extract %cst_7[%42, %43] : tensor<2x2xf32>
      %44 = arith.addf %extracted, %out : f32
      %45 = arith.select %41, %44, %out : f32
      linalg.yield %45 : f32
    } -> tensor<1x2xf32>
    %concat = tensor.concat dim(1) %11, %12, %13, %14 : (tensor<1x2xf32>, tensor<1x2xf32>, tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x8xf32>
    %expanded = tensor.expand_shape %concat [[0], [1, 2]] output_shape [1, 4, 2] : tensor<1x8xf32> into tensor<1x4x2xf32>
    %15 = tensor.empty() : tensor<1x2x4xf32>
    %transposed_21 = linalg.transpose ins(%expanded : tensor<1x4x2xf32>) outs(%15 : tensor<1x2x4xf32>) permutation = [0, 2, 1] 
    %16 = tensor.empty() : tensor<1x4x4xf32>
    %17 = linalg.fill ins(%cst_0 : f32) outs(%16 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %18 = linalg.batch_matmul ins(%expanded, %transposed_21 : tensor<1x4x2xf32>, tensor<1x2x4xf32>) outs(%17 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %19 = tensor.empty() : tensor<1xi64>
    %20 = linalg.generic {indexing_maps = [#map5], iterator_types = ["parallel"]} outs(%19 : tensor<1xi64>) {
    ^bb0(%out: i64):
      linalg.yield %c0_i64 : i64
    } -> tensor<1xi64>
    %expanded_22 = tensor.expand_shape %20 [[0, 1]] output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
    %21 = tensor.empty() : tensor<1x6xf32>
    %22 = linalg.generic {indexing_maps = [#map6, #map1, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_22, %cst_8, %cst_9 : tensor<1x1xi64>, tensor<6xi64>, tensor<6xi64>) outs(%21 : tensor<1x6xf32>) {
    ^bb0(%in: i64, %in_27: i64, %in_28: i64, %out: f32):
      %39 = arith.cmpi slt, %in, %c0_i64 : i64
      %40 = arith.addi %in, %c1_i64 : i64
      %41 = arith.select %39, %40, %in : i64
      %42 = arith.index_cast %41 : i64 to index
      %43 = arith.cmpi slt, %in_27, %c0_i64 : i64
      %44 = arith.addi %in_27, %c4_i64 : i64
      %45 = arith.select %43, %44, %in_27 : i64
      %46 = arith.index_cast %45 : i64 to index
      %47 = arith.cmpi slt, %in_28, %c0_i64 : i64
      %48 = arith.addi %in_28, %c4_i64 : i64
      %49 = arith.select %47, %48, %in_28 : i64
      %50 = arith.index_cast %49 : i64 to index
      %extracted = tensor.extract %18[%42, %46, %50] : tensor<1x4x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x6xf32>
    %concat_23 = tensor.concat dim(1) %11, %22 : (tensor<1x2xf32>, tensor<1x6xf32>) -> tensor<1x8xf32>
    %23 = tensor.empty() : tensor<8x4xf32>
    %transposed_24 = linalg.transpose ins(%cst_10 : tensor<4x8xf32>) outs(%23 : tensor<8x4xf32>) permutation = [1, 0] 
    %24 = tensor.empty() : tensor<1x4xf32>
    %25 = linalg.fill ins(%cst_0 : f32) outs(%24 : tensor<1x4xf32>) -> tensor<1x4xf32>
    %26 = linalg.matmul ins(%concat_23, %transposed_24 : tensor<1x8xf32>, tensor<8x4xf32>) outs(%25 : tensor<1x4xf32>) -> tensor<1x4xf32>
    %27 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%26, %cst_11 : tensor<1x4xf32>, tensor<4xf32>) outs(%24 : tensor<1x4xf32>) {
    ^bb0(%in: f32, %in_27: f32, %out: f32):
      %39 = arith.addf %in, %in_27 : f32
      linalg.yield %39 : f32
    } -> tensor<1x4xf32>
    %28 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%27 : tensor<1x4xf32>) outs(%24 : tensor<1x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %39 = arith.cmpf ugt, %in, %cst_0 : f32
      %40 = arith.select %39, %in, %cst_0 : f32
      linalg.yield %40 : f32
    } -> tensor<1x4xf32>
    %29 = tensor.empty() : tensor<4x2xf32>
    %transposed_25 = linalg.transpose ins(%cst_12 : tensor<2x4xf32>) outs(%29 : tensor<4x2xf32>) permutation = [1, 0] 
    %30 = linalg.matmul ins(%28, %transposed_25 : tensor<1x4xf32>, tensor<4x2xf32>) outs(%8 : tensor<1x2xf32>) -> tensor<1x2xf32>
    %31 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%30, %cst_13 : tensor<1x2xf32>, tensor<2xf32>) outs(%7 : tensor<1x2xf32>) {
    ^bb0(%in: f32, %in_27: f32, %out: f32):
      %39 = arith.addf %in, %in_27 : f32
      linalg.yield %39 : f32
    } -> tensor<1x2xf32>
    %32 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%31 : tensor<1x2xf32>) outs(%7 : tensor<1x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %39 = arith.cmpf ugt, %in, %cst_0 : f32
      %40 = arith.select %39, %in, %cst_0 : f32
      linalg.yield %40 : f32
    } -> tensor<1x2xf32>
    %33 = tensor.empty() : tensor<2x1xf32>
    %transposed_26 = linalg.transpose ins(%cst_14 : tensor<1x2xf32>) outs(%33 : tensor<2x1xf32>) permutation = [1, 0] 
    %34 = tensor.empty() : tensor<1x1xf32>
    %35 = linalg.fill ins(%cst_0 : f32) outs(%34 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %36 = linalg.matmul ins(%32, %transposed_26 : tensor<1x2xf32>, tensor<2x1xf32>) outs(%35 : tensor<1x1xf32>) -> tensor<1x1xf32>
    %37 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%36, %cst_15 : tensor<1x1xf32>, tensor<1xf32>) outs(%34 : tensor<1x1xf32>) {
    ^bb0(%in: f32, %in_27: f32, %out: f32):
      %39 = arith.addf %in, %in_27 : f32
      linalg.yield %39 : f32
    } -> tensor<1x1xf32>
    %38 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%37 : tensor<1x1xf32>) outs(%34 : tensor<1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %39 = arith.negf %in : f32
      %40 = math.exp %39 : f32
      %41 = arith.addf %40, %cst_1 : f32
      %42 = arith.divf %cst_1, %41 : f32
      linalg.yield %42 : f32
    } -> tensor<1x1xf32>
    return %38 : tensor<1x1xf32>
  }
}
