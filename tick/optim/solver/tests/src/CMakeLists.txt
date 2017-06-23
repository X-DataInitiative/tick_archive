include_directories(${CMAKE_SOURCE_DIR}/..)

add_executable(tick_test_sgd sgd_gtest.cpp)

target_link_libraries(tick_test_sgd ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${EXTRA_TEST_LIBS} base array model solver prox random)
