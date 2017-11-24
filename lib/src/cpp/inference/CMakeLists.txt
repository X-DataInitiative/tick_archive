add_library(tick_inference EXCLUDE_FROM_ALL
        ${PROJECT_SOURCE_DIR}/include/tick/inference/hawkes_conditional_law.h 
        hawkes_conditional_law.cpp
        hawkes_em.cpp 
        ${PROJECT_SOURCE_DIR}/include/tick/inference/hawkes_em.h
        ${PROJECT_SOURCE_DIR}/include/tick/inference/hawkes_adm4.h 
        hawkes_adm4.cpp
        hawkes_basis_kernels.cpp 
        ${PROJECT_SOURCE_DIR}/include/tick/inference/hawkes_basis_kernels.h
        ${PROJECT_SOURCE_DIR}/include/tick/inference/hawkes_sumgaussians.h 
        hawkes_sumgaussians.cpp)

target_link_libraries(tick_inference

        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_MODEL})
