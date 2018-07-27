add_library(tick_shared_benchmark shared_saga.h shared_saga.cpp)

target_link_libraries(tick_shared_benchmark
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_CRANDOM}
        ${TICK_LIB_BASE_MODEL}
        ${TICK_LIB_LINEAR_MODEL}
        ${TICK_LIB_PROX}
        ${TICK_LIB_SOLVER})

add_executable(tick_saga_sparse saga_random_order_gtest_sparse.cpp)

target_link_libraries(tick_saga_sparse
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_CRANDOM}
        ${TICK_LIB_BASE_MODEL}
        ${TICK_LIB_LINEAR_MODEL}
        ${TICK_LIB_PROX}
        ${TICK_LIB_SOLVER}
        tick_shared_benchmark
    )

add_executable(tick_asaga_sparse asaga_random_order_gtest_sparse.cpp)

target_link_libraries(tick_asaga_sparse
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_CRANDOM}
        ${TICK_LIB_BASE_MODEL}
        ${TICK_LIB_LINEAR_MODEL}
        ${TICK_LIB_PROX}
        ${TICK_LIB_SOLVER}
        tick_shared_benchmark
        )


add_executable(tick_half_asaga_sparse half_asaga_random_order_gtest_sparse.cpp)

target_link_libraries(tick_half_asaga_sparse
        ${TICK_LIB_BASE}
        ${TICK_LIB_ARRAY}
        ${TICK_LIB_CRANDOM}
        ${TICK_LIB_BASE_MODEL}
        ${TICK_LIB_LINEAR_MODEL}
        ${TICK_LIB_PROX}
        ${TICK_LIB_SOLVER}
        tick_shared_benchmark
        )
