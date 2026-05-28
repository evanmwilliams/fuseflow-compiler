add_mlir_library(Interactive
        Interactive.cpp

        ${PROJECT_SOURCE_DIR}/lib/Transforms/Interactive
        ADDITIONAL_HEADER_DIRS

        DEPENDS
        InteractivePassesIncGen

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRSam
        MLIRIR
        MLIRPass
        MLIRSupport
        MLIRLinalgDialect
)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Interactive)
add_dependencies(mlir-headers MLIRSamOpsIncGen)
add_public_tablegen_target(InteractivePassesIncGen)
add_mlir_doc(Interactive Interactive Interactive/ -gen-pass-doc)
