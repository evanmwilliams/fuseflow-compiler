add_mlir_library(StreamVectorizer
        StreamVectorizer.cpp

        ${PROJECT_SOURCE_DIR}/lib/Transforms/StreamVectorizer
        ADDITIONAL_HEADER_DIRS

        DEPENDS
        StreamVectorizerPassesIncGen

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
mlir_tablegen(Passes.h.inc -gen-pass-decls -name StreamVectorizer)
add_dependencies(mlir-headers MLIRSamOpsIncGen)
add_public_tablegen_target(StreamVectorizerPassesIncGen)
add_mlir_doc(StreamVectorizer StreamVectorizer StreamVectorizer/ -gen-pass-doc)