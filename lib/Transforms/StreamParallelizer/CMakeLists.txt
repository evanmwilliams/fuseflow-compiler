add_mlir_library(StreamParallelizer
        StreamParallelizer.cpp

        ${PROJECT_SOURCE_DIR}/lib/Transforms/StreamParallelizer
        ADDITIONAL_HEADER_DIRS

        DEPENDS
        StreamParallelizerPassesIncGen

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRSam
        MLIRArithDialect
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRFuncDialect
        MLIRFuncTransforms
        MLIRIR
        MLIRPass
        MLIRSCFDialect
        MLIRTensorDialect
        MLIRTransforms
#        ortools
)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name StreamParallelizer)
add_dependencies(mlir-headers MLIRSamOpsIncGen)
add_public_tablegen_target(StreamParallelizerPassesIncGen)
add_mlir_doc(StreamParallelizer StreamParallelizer StreamParallelizer/ -gen-pass-doc)