add_mlir_library(MLIRSODAPTosa
        JoinMultipleTiles.cpp
        TosaTransposes.cpp
        NetworkHost.cpp
        OutlineAdditions.cpp
        OutlineConvolutions.cpp
        FoldConvAffine.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/sodap/Tosa

        DEPENDS
        SODAPTosaPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRArithDialect
        MLIRFuncDialect
        MLIRTensorDialect
        MLIRMemRefDialect
        MLIRLinalgDialect
        MLIRTosaDialect
        MLIRTosaTransforms
        MLIRSODAPDataflow
        MLIRTransformUtils
        )
