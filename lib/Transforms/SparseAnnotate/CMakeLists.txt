add_mlir_library(SparseAnnotate
        SparseAnnotate.cpp

        ${PROJECT_SOURCE_DIR}/lib/Transforms/SparseAnnotate
        ADDITIONAL_HEADER_DIRS

        DEPENDS
        SparseAnnotatePassesIncGen

        LINK_COMPONENTS
        Core
        Support

        LINK_LIBS PUBLIC
        MLIRSam
        MLIRIR
        MLIRPass
        MLIRSupport
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRSparseTensorDialect
)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SparseAnnotate)
add_dependencies(mlir-headers MLIRSamOpsIncGen)
add_public_tablegen_target(SparseAnnotatePassesIncGen)
add_mlir_doc(SparseAnnotate SparseAnnotate SparseAnnotate/ -gen-pass-doc)
