// -----// IR Dump Before VerifyLinalgOnTensorsBackendContract (torch-verify-linalg-on-tensors-backend-contract) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<16x784xf32>) -> tensor<16x784xf32> {
    %cst = arith.constant dense_resource<torch_tensor_256_784_torch.float32> : tensor<256x784xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant dense_resource<torch_tensor_256_torch.float32> : tensor<256xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_128_256_torch.float32> : tensor<128x256xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_128_torch.float32> : tensor<128xf32>
    %cst_5 = arith.constant dense_resource<torch_tensor_64_128_torch.float32> : tensor<64x128xf32>
    %cst_6 = arith.constant dense_resource<torch_tensor_64_torch.float32> : tensor<64xf32>
    %cst_7 = arith.constant dense_resource<torch_tensor_32_64_torch.float32> : tensor<32x64xf32>
    %cst_8 = arith.constant dense_resource<torch_tensor_32_torch.float32> : tensor<32xf32>
    %cst_9 = arith.constant dense_resource<torch_tensor_16_32_torch.float32> : tensor<16x32xf32>
    %cst_10 = arith.constant dense_resource<torch_tensor_16_torch.float32> : tensor<16xf32>
    %cst_11 = arith.constant dense_resource<torch_tensor_32_16_torch.float32> : tensor<32x16xf32>
    %cst_12 = arith.constant dense_resource<torch_tensor_32_torch.float32_1> : tensor<32xf32>
    %cst_13 = arith.constant dense_resource<torch_tensor_64_32_torch.float32> : tensor<64x32xf32>
    %cst_14 = arith.constant dense_resource<torch_tensor_64_torch.float32_1> : tensor<64xf32>
    %cst_15 = arith.constant dense_resource<torch_tensor_128_64_torch.float32> : tensor<128x64xf32>
    %cst_16 = arith.constant dense_resource<torch_tensor_128_torch.float32_1> : tensor<128xf32>
    %cst_17 = arith.constant dense_resource<torch_tensor_256_128_torch.float32> : tensor<256x128xf32>
    %cst_18 = arith.constant dense_resource<torch_tensor_256_torch.float32_1> : tensor<256xf32>
    %cst_19 = arith.constant dense_resource<torch_tensor_784_256_torch.float32> : tensor<784x256xf32>
    %cst_20 = arith.constant dense_resource<torch_tensor_784_torch.float32> : tensor<784xf32>
    %0 = tensor.empty() : tensor<784x256xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<256x784xf32>) outs(%0 : tensor<784x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<784x256xf32>
    %2 = tensor.empty() : tensor<16x256xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<16x256xf32>) -> tensor<16x256xf32>
    %4 = linalg.matmul ins(%arg0, %1 : tensor<16x784xf32>, tensor<784x256xf32>) outs(%3 : tensor<16x256xf32>) -> tensor<16x256xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst_2 : tensor<16x256xf32>, tensor<256xf32>) outs(%2 : tensor<16x256xf32>) {
    ^bb0(%in: f32, %in_21: f32, %out: f32):
      %62 = arith.addf %in, %in_21 : f32
      linalg.yield %62 : f32
    } -> tensor<16x256xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<16x256xf32>) outs(%2 : tensor<16x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<16x256xf32>
    %7 = tensor.empty() : tensor<256x128xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_3 : tensor<128x256xf32>) outs(%7 : tensor<256x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<256x128xf32>
    %9 = tensor.empty() : tensor<16x128xf32>
    %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %11 = linalg.matmul ins(%6, %8 : tensor<16x256xf32>, tensor<256x128xf32>) outs(%10 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %cst_4 : tensor<16x128xf32>, tensor<128xf32>) outs(%9 : tensor<16x128xf32>) {
    ^bb0(%in: f32, %in_21: f32, %out: f32):
      %62 = arith.addf %in, %in_21 : f32
      linalg.yield %62 : f32
    } -> tensor<16x128xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<16x128xf32>) outs(%9 : tensor<16x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<16x128xf32>
    %14 = tensor.empty() : tensor<128x64xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_5 : tensor<64x128xf32>) outs(%14 : tensor<128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x64xf32>
    %16 = tensor.empty() : tensor<16x64xf32>
    %17 = linalg.fill ins(%cst_0 : f32) outs(%16 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %18 = linalg.matmul ins(%13, %15 : tensor<16x128xf32>, tensor<128x64xf32>) outs(%17 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%18, %cst_6 : tensor<16x64xf32>, tensor<64xf32>) outs(%16 : tensor<16x64xf32>) {
    ^bb0(%in: f32, %in_21: f32, %out: f32):
      %62 = arith.addf %in, %in_21 : f32
      linalg.yield %62 : f32
    } -> tensor<16x64xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%19 : tensor<16x64xf32>) outs(%16 : tensor<16x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<16x64xf32>
    %21 = tensor.empty() : tensor<64x32xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_7 : tensor<32x64xf32>) outs(%21 : tensor<64x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x32xf32>
    %23 = tensor.empty() : tensor<16x32xf32>
    %24 = linalg.fill ins(%cst_0 : f32) outs(%23 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %25 = linalg.matmul ins(%20, %22 : tensor<16x64xf32>, tensor<64x32xf32>) outs(%24 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %26 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%25, %cst_8 : tensor<16x32xf32>, tensor<32xf32>) outs(%23 : tensor<16x32xf32>) {
    ^bb0(%in: f32, %in_21: f32, %out: f32):
      %62 = arith.addf %in, %in_21 : f32
      linalg.yield %62 : f32
    } -> tensor<16x32xf32>
    %27 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%26 : tensor<16x32xf32>) outs(%23 : tensor<16x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<16x32xf32>
    %28 = tensor.empty() : tensor<32x16xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_9 : tensor<16x32xf32>) outs(%28 : tensor<32x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x16xf32>
    %30 = tensor.empty() : tensor<16x16xf32>
    %31 = linalg.fill ins(%cst_0 : f32) outs(%30 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %32 = linalg.matmul ins(%27, %29 : tensor<16x32xf32>, tensor<32x16xf32>) outs(%31 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %33 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%32, %cst_10 : tensor<16x16xf32>, tensor<16xf32>) outs(%30 : tensor<16x16xf32>) {
    ^bb0(%in: f32, %in_21: f32, %out: f32):
      %62 = arith.addf %in, %in_21 : f32
      linalg.yield %62 : f32
    } -> tensor<16x16xf32>
    %34 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%33 : tensor<16x16xf32>) outs(%30 : tensor<16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<16x16xf32>
    %35 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_11 : tensor<32x16xf32>) outs(%23 : tensor<16x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x32xf32>
    %36 = linalg.matmul ins(%34, %35 : tensor<16x16xf32>, tensor<16x32xf32>) outs(%24 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %37 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%36, %cst_12 : tensor<16x32xf32>, tensor<32xf32>) outs(%23 : tensor<16x32xf32>) {
    ^bb0(%in: f32, %in_21: f32, %out: f32):
      %62 = arith.addf %in, %in_21 : f32
      linalg.yield %62 : f32
    } -> tensor<16x32xf32>
    %38 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%37 : tensor<16x32xf32>) outs(%23 : tensor<16x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<16x32xf32>
    %39 = tensor.empty() : tensor<32x64xf32>
    %40 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_13 : tensor<64x32xf32>) outs(%39 : tensor<32x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x64xf32>
    %41 = linalg.matmul ins(%38, %40 : tensor<16x32xf32>, tensor<32x64xf32>) outs(%17 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %42 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%41, %cst_14 : tensor<16x64xf32>, tensor<64xf32>) outs(%16 : tensor<16x64xf32>) {
    ^bb0(%in: f32, %in_21: f32, %out: f32):
      %62 = arith.addf %in, %in_21 : f32
      linalg.yield %62 : f32
    } -> tensor<16x64xf32>
    %43 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%42 : tensor<16x64xf32>) outs(%16 : tensor<16x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<16x64xf32>
    %44 = tensor.empty() : tensor<64x128xf32>
    %45 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_15 : tensor<128x64xf32>) outs(%44 : tensor<64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x128xf32>
    %46 = linalg.matmul ins(%43, %45 : tensor<16x64xf32>, tensor<64x128xf32>) outs(%10 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %47 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%46, %cst_16 : tensor<16x128xf32>, tensor<128xf32>) outs(%9 : tensor<16x128xf32>) {
    ^bb0(%in: f32, %in_21: f32, %out: f32):
      %62 = arith.addf %in, %in_21 : f32
      linalg.yield %62 : f32
    } -> tensor<16x128xf32>
    %48 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%47 : tensor<16x128xf32>) outs(%9 : tensor<16x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<16x128xf32>
    %49 = tensor.empty() : tensor<128x256xf32>
    %50 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_17 : tensor<256x128xf32>) outs(%49 : tensor<128x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x256xf32>
    %51 = linalg.matmul ins(%48, %50 : tensor<16x128xf32>, tensor<128x256xf32>) outs(%3 : tensor<16x256xf32>) -> tensor<16x256xf32>
    %52 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%51, %cst_18 : tensor<16x256xf32>, tensor<256xf32>) outs(%2 : tensor<16x256xf32>) {
    ^bb0(%in: f32, %in_21: f32, %out: f32):
      %62 = arith.addf %in, %in_21 : f32
      linalg.yield %62 : f32
    } -> tensor<16x256xf32>
    %53 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%52 : tensor<16x256xf32>) outs(%2 : tensor<16x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<16x256xf32>
    %54 = tensor.empty() : tensor<256x784xf32>
    %55 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_19 : tensor<784x256xf32>) outs(%54 : tensor<256x784xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<256x784xf32>
    %56 = tensor.empty() : tensor<16x784xf32>
    %57 = linalg.fill ins(%cst_0 : f32) outs(%56 : tensor<16x784xf32>) -> tensor<16x784xf32>
    %58 = linalg.matmul ins(%53, %55 : tensor<16x256xf32>, tensor<256x784xf32>) outs(%57 : tensor<16x784xf32>) -> tensor<16x784xf32>
    %59 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%58, %cst_20 : tensor<16x784xf32>, tensor<784xf32>) outs(%56 : tensor<16x784xf32>) {
    ^bb0(%in: f32, %in_21: f32, %out: f32):
      %62 = arith.addf %in, %in_21 : f32
      linalg.yield %62 : f32
    } -> tensor<16x784xf32>
    %60 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%59 : tensor<16x784xf32>) outs(%56 : tensor<16x784xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.cmpf ugt, %in, %cst_0 : f32
      %63 = arith.select %62, %in, %cst_0 : f32
      linalg.yield %63 : f32
    } -> tensor<16x784xf32>
    %61 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%60 : tensor<16x784xf32>) outs(%56 : tensor<16x784xf32>) {
    ^bb0(%in: f32, %out: f32):
      %62 = arith.negf %in : f32
      %63 = math.exp %62 : f32
      %64 = arith.addf %63, %cst_1 : f32
      %65 = arith.divf %cst_1, %64 : f32
      linalg.yield %65 : f32
    } -> tensor<16x784xf32>
    return %60 : tensor<16x784xf32>
  }
}
