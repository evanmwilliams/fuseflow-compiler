#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0) -> (d0)>
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
#SPARSE1D = #sparse_tensor.encoding<{
  map = (i) -> (i : compressed)
}>
module attributes {torch.debug_module_name = "SparseAutoencoder"} { 
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1767xf32, #SPARSE1D>) -> tensor<1767xf32, #SPARSE1D> {
    %cst = arith.constant dense_resource<__elided__> : tensor<512xf32, #SPARSE1D>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<512x1767xf32, #SPARSE>
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1767xf32, #SPARSE1D>
    %cst_3 = arith.constant dense_resource<__elided__> : tensor<1767x512xf32, #SPARSE>

    %cst_4 = arith.constant dense_resource<__elided__> : tensor<256xf32, #SPARSE1D>
    %cst_5 = arith.constant dense_resource<__elided__> : tensor<256x512xf32, #SPARSE>
    %cst_6 = arith.constant dense_resource<__elided__> : tensor<512xf32, #SPARSE1D>
    %cst_7 = arith.constant dense_resource<__elided__> : tensor<512x256xf32, #SPARSE>
    %cst_8 = arith.constant dense_resource<__elided__> : tensor<128xf32, #SPARSE1D>
    %cst_9 = arith.constant dense_resource<__elided__> : tensor<128x256xf32, #SPARSE>
    %cst_10 = arith.constant dense_resource<__elided__> : tensor<256xf32, #SPARSE1D>
    %cst_11 = arith.constant dense_resource<__elided__> : tensor<256x128xf32, #SPARSE>
    %cst_12 = arith.constant dense_resource<__elided__> : tensor<64xf32, #SPARSE1D>
    %cst_13 = arith.constant dense_resource<__elided__> : tensor<64x128xf32, #SPARSE>
    %cst_14 = arith.constant dense_resource<__elided__> : tensor<128xf32, #SPARSE1D>
    %cst_15 = arith.constant dense_resource<__elided__> : tensor<128x64xf32, #SPARSE>
    %cst_16 = arith.constant dense_resource<__elided__> : tensor<32xf32, #SPARSE1D>
    %cst_17 = arith.constant dense_resource<__elided__> : tensor<32x64xf32, #SPARSE>
    %cst_18 = arith.constant dense_resource<__elided__> : tensor<64xf32, #SPARSE1D>
    %cst_19 = arith.constant dense_resource<__elided__> : tensor<64x32xf32, #SPARSE>


    %0 = tensor.empty() : tensor<512x1767xf32, #SPARSE>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<512x1767xf32, #SPARSE>) outs(%0 : tensor<512x1767xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<512x1767xf32, #SPARSE>
    %2 = tensor.empty() : tensor<512xf32, #SPARSE1D>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<512xf32, #SPARSE1D>) -> tensor<512xf32, #SPARSE1D>
    %4 = linalg.matvec ins(%1, %arg0 : tensor<512x1767xf32, #SPARSE>, tensor<1767xf32, #SPARSE1D>) outs(%3 : tensor<512xf32, #SPARSE1D>) -> tensor<512xf32, #SPARSE1D>
    %5 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%4, %cst : tensor<512xf32, #SPARSE1D>, tensor<512xf32, #SPARSE1D>) outs(%2 : tensor<512xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %13 = arith.addf %in, %in_4 : f32
      linalg.yield %13 : f32
    } -> tensor<512xf32, #SPARSE1D>
    %6 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%5 : tensor<512xf32, #SPARSE1D>) outs(%2 : tensor<512xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %out: f32):
      %13 = arith.cmpf ugt, %in, %cst_0 : f32
      %14 = arith.select %13, %in, %cst_0 : f32
      linalg.yield %14 : f32
    } -> tensor<512xf32, #SPARSE1D>

    %20 = tensor.empty() : tensor<256x512xf32, #SPARSE>
    %21 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst_5 : tensor<256x512xf32, #SPARSE>) outs(%20 : tensor<256x512xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<256x512xf32, #SPARSE>
    %22 = tensor.empty() : tensor<256xf32, #SPARSE1D>
    %23 = linalg.fill ins(%cst_0 : f32) outs(%22 : tensor<256xf32, #SPARSE1D>) -> tensor<256xf32, #SPARSE1D>
    %24 = linalg.matvec ins(%21, %6 : tensor<256x512xf32, #SPARSE>, tensor<512xf32, #SPARSE1D>) outs(%23 : tensor<256xf32, #SPARSE1D>) -> tensor<256xf32, #SPARSE1D>
    %25 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%24, %cst_4 : tensor<256xf32, #SPARSE1D>, tensor<256xf32, #SPARSE1D>) outs(%22 : tensor<256xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %13 = arith.addf %in, %in_4 : f32
      linalg.yield %13 : f32
    } -> tensor<256xf32, #SPARSE1D>
    %26 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%25 : tensor<256xf32, #SPARSE1D>) outs(%22 : tensor<256xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %out: f32):
      %13 = arith.cmpf ugt, %in, %cst_0 : f32
      %14 = arith.select %13, %in, %cst_0 : f32
      linalg.yield %14 : f32
    } -> tensor<256xf32, #SPARSE1D>

    %27 = tensor.empty() : tensor<128x256xf32, #SPARSE>
    %28 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst_9 : tensor<128x256xf32, #SPARSE>) outs(%27 : tensor<128x256xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x256xf32, #SPARSE>
    %29 = tensor.empty() : tensor<128xf32, #SPARSE1D>
    %30 = linalg.fill ins(%cst_0 : f32) outs(%29 : tensor<128xf32, #SPARSE1D>) -> tensor<128xf32, #SPARSE1D>
    %31 = linalg.matvec ins(%28, %26 : tensor<128x256xf32, #SPARSE>, tensor<256xf32, #SPARSE1D>) outs(%30 : tensor<128xf32, #SPARSE1D>) -> tensor<128xf32, #SPARSE1D>
    %32 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%31, %cst_8 : tensor<128xf32, #SPARSE1D>, tensor<128xf32, #SPARSE1D>) outs(%29 : tensor<128xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %13 = arith.addf %in, %in_4 : f32
      linalg.yield %13 : f32
    } -> tensor<128xf32, #SPARSE1D>
    %33 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%32 : tensor<128xf32, #SPARSE1D>) outs(%29 : tensor<128xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %out: f32):
      %13 = arith.cmpf ugt, %in, %cst_0 : f32
      %14 = arith.select %13, %in, %cst_0 : f32
      linalg.yield %14 : f32
    } -> tensor<128xf32, #SPARSE1D>

    %34 = tensor.empty() : tensor<64x128xf32, #SPARSE>
    %35 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst_13 : tensor<64x128xf32, #SPARSE>) outs(%34 : tensor<64x128xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x128xf32, #SPARSE>
    %36 = tensor.empty() : tensor<64xf32, #SPARSE1D>
    %37 = linalg.fill ins(%cst_0 : f32) outs(%36 : tensor<64xf32, #SPARSE1D>) -> tensor<64xf32, #SPARSE1D>
    %38 = linalg.matvec ins(%35, %33 : tensor<64x128xf32, #SPARSE>, tensor<128xf32, #SPARSE1D>) outs(%37 : tensor<64xf32, #SPARSE1D>) -> tensor<64xf32, #SPARSE1D>
    %39 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%38, %cst_12 : tensor<64xf32, #SPARSE1D>, tensor<64xf32, #SPARSE1D>) outs(%36 : tensor<64xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %13 = arith.addf %in, %in_4 : f32
      linalg.yield %13 : f32
    } -> tensor<64xf32, #SPARSE1D>
    %40 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%39 : tensor<64xf32, #SPARSE1D>) outs(%36 : tensor<64xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %out: f32):
      %13 = arith.cmpf ugt, %in, %cst_0 : f32
      %14 = arith.select %13, %in, %cst_0 : f32
      linalg.yield %14 : f32
    } -> tensor<64xf32, #SPARSE1D>

    %41 = tensor.empty() : tensor<32x64xf32, #SPARSE>
    %42 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst_17 : tensor<32x64xf32, #SPARSE>) outs(%41 : tensor<32x64xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x64xf32, #SPARSE>
    %43 = tensor.empty() : tensor<32xf32, #SPARSE1D>
    %44 = linalg.fill ins(%cst_0 : f32) outs(%43 : tensor<32xf32, #SPARSE1D>) -> tensor<32xf32, #SPARSE1D>
    %45 = linalg.matvec ins(%42, %40 : tensor<32x64xf32, #SPARSE>, tensor<64xf32, #SPARSE1D>) outs(%44 : tensor<32xf32, #SPARSE1D>) -> tensor<32xf32, #SPARSE1D>
    %46 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%45, %cst_16 : tensor<32xf32, #SPARSE1D>, tensor<32xf32, #SPARSE1D>) outs(%43 : tensor<32xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %13 = arith.addf %in, %in_4 : f32
      linalg.yield %13 : f32
    } -> tensor<32xf32, #SPARSE1D>
    %47 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%46 : tensor<32xf32, #SPARSE1D>) outs(%43 : tensor<32xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %out: f32):
      %13 = arith.cmpf ugt, %in, %cst_0 : f32
      %14 = arith.select %13, %in, %cst_0 : f32
      linalg.yield %14 : f32
    } -> tensor<32xf32, #SPARSE1D>

    %48 = tensor.empty() : tensor<64x32xf32, #SPARSE>
    %49 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst_19 : tensor<64x32xf32, #SPARSE>) outs(%48 : tensor<64x32xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x32xf32, #SPARSE>
    %50 = tensor.empty() : tensor<64xf32, #SPARSE1D>
    %51 = linalg.fill ins(%cst_0 : f32) outs(%50 : tensor<64xf32, #SPARSE1D>) -> tensor<64xf32, #SPARSE1D>
    %52 = linalg.matvec ins(%48, %47 : tensor<64x32xf32, #SPARSE>, tensor<32xf32, #SPARSE1D>) outs(%51 : tensor<64xf32, #SPARSE1D>) -> tensor<64xf32, #SPARSE1D>
    %53 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%52, %cst_18 : tensor<64xf32, #SPARSE1D>, tensor<64xf32, #SPARSE1D>) outs(%50 : tensor<64xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %13 = arith.addf %in, %in_4 : f32
      linalg.yield %13 : f32
    } -> tensor<64xf32, #SPARSE1D>

    %54 = tensor.empty() : tensor<128x64xf32, #SPARSE>
    %55 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst_15 : tensor<128x64xf32, #SPARSE>) outs(%54 : tensor<128x64xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x64xf32, #SPARSE>
    %56 = tensor.empty() : tensor<128xf32, #SPARSE1D>
    %57 = linalg.fill ins(%cst_0 : f32) outs(%56 : tensor<128xf32, #SPARSE1D>) -> tensor<128xf32, #SPARSE1D>
    %58 = linalg.matvec ins(%54, %53 : tensor<128x64xf32, #SPARSE>, tensor<64xf32, #SPARSE1D>) outs(%57 : tensor<128xf32, #SPARSE1D>) -> tensor<128xf32, #SPARSE1D>
    %59 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%58, %cst_14 : tensor<128xf32, #SPARSE1D>, tensor<128xf32, #SPARSE1D>) outs(%56 : tensor<128xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %13 = arith.addf %in, %in_4 : f32
      linalg.yield %13 : f32
    } -> tensor<128xf32, #SPARSE1D>

    %60 = tensor.empty() : tensor<256x128xf32, #SPARSE>
    %61 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst_11 : tensor<256x128xf32, #SPARSE>) outs(%60 : tensor<256x128xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<256x128xf32, #SPARSE>
    %62 = tensor.empty() : tensor<256xf32, #SPARSE1D>
    %63 = linalg.fill ins(%cst_0 : f32) outs(%62 : tensor<256xf32, #SPARSE1D>) -> tensor<256xf32, #SPARSE1D>
    %64 = linalg.matvec ins(%60, %59 : tensor<256x128xf32, #SPARSE>, tensor<128xf32, #SPARSE1D>) outs(%63 : tensor<256xf32, #SPARSE1D>) -> tensor<256xf32, #SPARSE1D>
    %65 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%63, %cst_10 : tensor<256xf32, #SPARSE1D>, tensor<256xf32, #SPARSE1D>) outs(%62 : tensor<256xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %13 = arith.addf %in, %in_4 : f32
      linalg.yield %13 : f32
    } -> tensor<256xf32, #SPARSE1D>

    %66 = tensor.empty() : tensor<512x256xf32, #SPARSE>
    %67 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst_7 : tensor<512x256xf32, #SPARSE>) outs(%66 : tensor<512x256xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<512x256xf32, #SPARSE>
    %68 = tensor.empty() : tensor<512xf32, #SPARSE1D>
    %69 = linalg.fill ins(%cst_0 : f32) outs(%68 : tensor<512xf32, #SPARSE1D>) -> tensor<512xf32, #SPARSE1D>
    %70 = linalg.matvec ins(%66, %65 : tensor<512x256xf32, #SPARSE>, tensor<256xf32, #SPARSE1D>) outs(%68 : tensor<512xf32, #SPARSE1D>) -> tensor<512xf32, #SPARSE1D>
    %71 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%68, %cst_6 : tensor<512xf32, #SPARSE1D>, tensor<512xf32, #SPARSE1D>) outs(%68 : tensor<512xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %13 = arith.addf %in, %in_4 : f32
      linalg.yield %13 : f32
    } -> tensor<512xf32, #SPARSE1D>

    %7 = tensor.empty() : tensor<1767x512xf32, #SPARSE>
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst_3 : tensor<1767x512xf32, #SPARSE>) outs(%7 : tensor<1767x512xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1767x512xf32, #SPARSE>
    %9 = tensor.empty() : tensor<1767xf32, #SPARSE1D>
    %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<1767xf32, #SPARSE1D>) -> tensor<1767xf32, #SPARSE1D>
    %11 = linalg.matvec ins(%7, %71 : tensor<1767x512xf32, #SPARSE>, tensor<512xf32, #SPARSE1D>) outs(%10 : tensor<1767xf32, #SPARSE1D>) -> tensor<1767xf32, #SPARSE1D>
    %12 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%11, %cst_2 : tensor<1767xf32, #SPARSE1D>, tensor<1767xf32, #SPARSE1D>) outs(%9 : tensor<1767xf32, #SPARSE1D>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %13 = arith.addf %in, %in_4 : f32
      linalg.yield %13 : f32
    } -> tensor<1767xf32, #SPARSE1D>
    return %12 : tensor<1767xf32, #SPARSE1D>
  }
}