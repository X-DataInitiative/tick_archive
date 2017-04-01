add_executable(tick_test_array array_gtest.cpp)
add_executable(tick_test_varray varray_gtest.cpp)

target_link_libraries(tick_test_array
    ${TICKLIB_ARRAY}
    ${TICK_TEST_LIBS}
    )

target_link_libraries(tick_test_varray
    ${TICKLIB_ARRAY}
    ${TICK_TEST_LIBS}
    )
