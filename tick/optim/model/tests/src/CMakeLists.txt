add_executable(tick_test_model models_gtest.cpp hawkes_models_gtests.cpp)

target_link_libraries(tick_test_model
    ${TICKLIB_ARRAY}
    ${TICKLIB_BASE}
    ${TICKLIB_MODEL}
    ${TICK_TEST_LIBS}
    )
