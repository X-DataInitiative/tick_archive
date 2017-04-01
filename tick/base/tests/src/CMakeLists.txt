add_executable(tick_test_base utils_gtest.cpp)

target_link_libraries(tick_test_base
    ${TICKLIB_ARRAY}
    ${TICKLIB_BASE}

    ${TICK_TEST_LIBS}
    )
